"""Exercise learned compact transport through serving and an exact splice reference."""
from __future__ import annotations

from concurrent.futures import Future
import itertools
import math

import torch

from ..common.hashing import sha256_hex
from ..model.native_model import native_identity
from ..model.semantic_message import SemanticStateMessage
from ..model.shared_state import RoleTokens, native_step, splice_block
from ..serve.engine import Engine, GenerationRequest, Sequence
from ..serve.store import TensorStore, encode_native_state, decode_native_state


def _error(actual, expected, *, rtol, atol):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('semantic transport reference tensor shape/dtype differs')
    a, b = actual.detach().float(), expected.detach().float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {'passed': finite and bool(torch.allclose(a, b, rtol=rtol, atol=atol)),
            'max_abs': float((a-b).abs().max()) if a.numel() else 0.,
            'finite': finite}


def _cache_errors(actual, expected, *, rtol, atol):
    if actual.lengths != expected.lengths or len(actual.states) != len(expected.states):
        raise ValueError('semantic transport reference cache positions/layout differ')
    result = []
    for index, (a, b) in enumerate(zip(actual.states, expected.states)):
        if type(a) is not type(b):
            raise ValueError('semantic transport reference cache layer kind differs')
        names = ('window',) if hasattr(a, 'window') else ('k', 'v')
        for name in names:
            result.append({'layer': index, 'tensor': name,
                           **_error(getattr(a, name), getattr(b, name), rtol=rtol, atol=atol)})
    return result


def qualify_semantic_transport(participants, channel, *, texts, rtol, atol):
    """Verify every real cross-participant compact write/read, without sampling.

    ``texts`` supplies complete admitted payloads selected by the curriculum.
    Both implementations use the same learned decoder; this checks the serving
    route against direct decoding/splicing, not whether the code is useful.
    """
    from ..serve import bind_native_semantic_channel
    if (len(participants) < 2 or set(participants) != set(channel.bindings)
            or set(texts) != set(participants)):
        raise ValueError('semantic transport qualification needs every declared participant and payload')
    if any(not isinstance(value, str) or not value for value in texts.values()):
        raise ValueError('semantic transport qualification needs nonempty actual payload text')
    if any(not math.isfinite(value) or value < 0 for value in (rtol, atol)):
        raise ValueError('semantic transport qualification tolerances must be finite and nonnegative')
    models = [participant.model for participant in participants.values()] + [channel.vae]
    modules = {id(module): module for model in models for module in model.modules()}
    modes = {key: module.training for key, module in modules.items()}
    devices = sorted({parameter.device.index for model in models for parameter in model.parameters()
                      if parameter.device.type == 'cuda'})
    engines, messages, directions = {}, {}, []
    try:
        for model in models:
            model.eval()
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            for name, participant in participants.items():
                identity = channel.bindings[name]
                if native_identity(participant.model, participant.config,
                                   checkpoint_sha256=identity['checkpoint_sha256']) != identity:
                    raise ValueError('semantic transport participant differs from actual exported model identity')
                device = next(participant.model.parameters()).device
                engine = Engine(participant.model.port, None, participant.tokenizer, TensorStore(),
                    'native-state/' + identity['architecture'], device=device,
                    shared_model=participant.model, role_tokens=RoleTokens(*participant.model.marker_ids),
                    model_config=participant.config, checkpoint_sha256=identity['checkpoint_sha256'],
                    embedding_weight_sha256=identity['embedding_weight_sha256'])
                engine.native_identity = identity
                bind_native_semantic_channel(engine, channel, name)
                written = engine.native_write_text(texts[name])
                block = engine.store.get_native_state(written['id']).block
                if not isinstance(block, SemanticStateMessage) or block.layers or not len(block.token_ids):
                    raise ValueError('semantic transport writer did not produce a nonempty compact learned message')
                wire = encode_native_state(block)
                decoded, identifier = decode_native_state(wire)
                if identifier != written['id'] or not isinstance(decoded, SemanticStateMessage) or decoded.layers:
                    raise ValueError('semantic transport wire round trip changed its compact artifact')
                engines[name], messages[name] = engine, (decoded, identifier, sha256_hex(wire), len(wire))
            for source, target in itertools.permutations(sorted(participants), 2):
                engine = engines[target]
                message, identifier, wire_sha, wire_bytes = messages[source]
                if engine.store.put_native_state(message).id != identifier:
                    raise ValueError('semantic transport target store changed the source artifact identity')
                sequence = Sequence(GenerationRequest(messages=[]), Future())
                engine._native_consume(sequence, identifier)
                observation = sequence.native_imports[-1]
                if observation['mode'] != 'learned-semantic-state-deltas':
                    raise ValueError('semantic transport consumer bypassed learned decoding')
                adapted, adaptation = channel.adapt(message, target_name=target,
                    target_model=engine.shared_model, target_tokenizer=engine.tokenizer,
                    target_identity=engine.native_identity, allow_nearest=True)
                def step(ids, cache=None):
                    return native_step(engine.shared_model,
                        torch.tensor([ids], dtype=torch.long, device=engine.device), cache)
                opened = step([engine.role_tokens.consume_open])
                reference_cache = splice_block(engine.shared_model, opened['cache'], adapted,
                    checkpoint_sha256=engine.checkpoint_sha256,
                    tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'],
                    embedding_weight_sha256=engine.embedding_weight_sha256)
                closed = step([engine.role_tokens.consume_close], reference_cache)
                logits = _error(sequence.logits, engine._native_logits(closed['h_final']), rtol=rtol, atol=atol)
                caches = _cache_errors(sequence.cache, closed['cache'], rtol=rtol, atol=atol)
                # A second consumed close marker is a valid ordinary query token;
                # it verifies that both attention and hybrid state survive reuse.
                next_actual = step([engine.role_tokens.consume_close], sequence.cache)
                next_expected = step([engine.role_tokens.consume_close], closed['cache'])
                next_logits = _error(engine._native_logits(next_actual['h_final']),
                    engine._native_logits(next_expected['h_final']), rtol=rtol, atol=atol)
                next_caches = _cache_errors(next_actual['cache'], next_expected['cache'], rtol=rtol, atol=atol)
                checks = [logits, next_logits, *caches, *next_caches]
                directions.append({'source': source, 'target': target, 'source_native_identity': channel.bindings[source],
                    'target_native_identity': channel.bindings[target], 'artifact_id': identifier,
                    'wire_sha256': wire_sha, 'wire_bytes': wire_bytes,
                    'wire_layer_tensors': 0, 'mode': observation['mode'], 'adaptation': adaptation,
                    'logits': logits, 'cache': caches, 'next_logits': next_logits, 'next_cache': next_caches,
                    'passed': all(check['passed'] for check in checks)})
    finally:
        for key, module in modules.items():
            module.training = modes[key]
    return {'schema': 'natlang.native-semantic-transport-certificate/1',
            'adapter_sha256': channel.adapter_sha256, 'rtol': rtol, 'atol': atol,
            'passed': bool(directions) and all(row['passed'] for row in directions),
            'directions': directions,
            'scope': 'Actual compact wire and learned consumer parity; task quality and Neuralese lift are not established.'}
