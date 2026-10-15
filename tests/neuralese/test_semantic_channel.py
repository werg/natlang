"""Learned cross-model decoding reaches the real native serving cache path."""
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.model.semantic_channel import SemanticChannel
from natlang_neuralese.model.shared_state import RoleTokens, native_step, splice_block
from natlang_neuralese.serve.engine import Engine
from natlang_neuralese.serve.store import TensorStore
from natlang_neuralese.train.native_semantic import parse_pair, producer
from test_native_semantic import setup_pair


def test_semantic_cross_model_import_uses_learned_native_delta_and_archives_rl_observation():
    participants, row, vae = setup_pair()
    channel = SemanticChannel(vae, {name: entry.identity for name, entry in participants.items()},
        {name: entry.tokenizer for name, entry in participants.items()}, adapter_sha256='a' * 64)
    source_example, view = parse_pair(row, participants)['lfm']
    with torch.no_grad():
        source = producer(participants['lfm'], source_example, view, supervise=False)[0]
        target = participants['mellum']
        adapted, receipt = channel.adapt(source, target_name='mellum', target_model=target.model,
            target_tokenizer=target.tokenizer, target_identity=target.identity)
    assert receipt['mode'] == 'learned-semantic-state-deltas'
    assert adapted.architecture == target.identity['architecture']
    assert adapted.token_ids.tolist() == [8, 9, 10]
    engine = object.__new__(Engine)
    engine.shared_model, engine.backbone, engine.tokenizer = target.model, target.model.port, target.tokenizer
    engine.model_config, engine.native_identity = target.config, target.identity
    engine.role_tokens = RoleTokens(*target.model.marker_ids)
    engine.checkpoint_sha256 = target.identity['checkpoint_sha256']
    engine.embedding_weight_sha256 = target.identity['embedding_weight_sha256']
    engine.semantic_channel, engine.semantic_participant = channel, 'mellum'
    engine.store = TensorStore()
    record = engine.store.put_native_state(source)
    seq = SimpleNamespace(cache=None, request=SimpleNamespace(rl_capture=True), native_rl_prompt=[],
                          native_own_position=0, native_imports=[])
    with torch.no_grad():
        engine._native_consume(seq, record.id)
        prefix = native_step(target.model, torch.tensor([[target.model.marker_ids[2]]]))
        cache = splice_block(target.model, prefix['cache'], adapted,
            **{name: value for name, value in target.identity.items() if name != 'architecture'})
        expected = native_step(target.model, torch.tensor([[target.model.marker_ids[3]]]), cache)
    torch.testing.assert_close(seq.logits, target.model.logits(expected['h_final'][:, -1]), rtol=0, atol=0)
    assert seq.native_imports[0]['mode'] == 'learned-semantic-state-deltas'
    assert seq.native_own_position == 2  # imported state is not fabricated own-token sampling
    observed = seq.native_rl_prompt[1]
    assert observed['kind'] == 'native_state' and observed['id'] != record.id
    assert engine.store.get_native_state(observed['id']).block.checkpoint_sha256 == engine.checkpoint_sha256
    with pytest.raises(ValueError, match='actual target'):
        channel.require_target('mellum', {**target.identity, 'checkpoint_sha256': 'f' * 64})


def test_unrelated_shape_compatible_source_cannot_use_trained_semantic_adapter():
    participants, row, vae = setup_pair()
    channel = SemanticChannel(vae, {name: entry.identity for name, entry in participants.items()},
        {name: entry.tokenizer for name, entry in participants.items()}, adapter_sha256='a' * 64)
    example, view = parse_pair(row, participants)['lfm']
    with torch.no_grad():
        block = producer(participants['lfm'], example, view, supervise=False)[0]
    block.checkpoint_sha256 = 'f' * 64
    with pytest.raises(ValueError, match='exactly bound'):
        channel.encode(block)
