"""The SGLang actor front: the fork's packed per-token record and the routing between SGLang and the native engine."""
import base64

import numpy as np
import pytest

from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.serve.sglang_actor import SglangEngine, decode_record
from natlang_neuralese.serve.chat import RequestError


def record(logp, ids, logps):
    # The fork's layout (layers/sampler.py natlang_capture_top_k): <f4 sampled logp | <i4 ids[K] | <f2 logps[K].
    raw = (np.array([logp], '<f4').tobytes() + np.array(ids, '<i4').tobytes() + np.array(logps, '<f2').tobytes())
    return base64.b64encode(raw).decode()


def test_decode_record_prefix_of_larger_k():
    text = record(-0.25, [7, 3, 9], [-0.25, -1.5, -3.0])
    logp, ids, logps = decode_record(text, 2)
    assert logp == -0.25 and ids.tolist() == [7, 3] and logps.astype(float).tolist() == [-0.25, -1.5]
    with pytest.raises(RequestError):
        decode_record(text, 4)


def test_routing_sends_neuralese_turns_to_native():
    plain = GenerationRequest(messages=[], temperature=1.0, rl_capture=True, rl_top_k=128, adapters=[])
    assert SglangEngine.sglang_unsupported(plain, [('tokens', [1, 2]), ('privileged', [3])]) is None
    # Imported native state is served on SGLang; a declared cross-adapter state mode is not.
    assert SglangEngine.sglang_unsupported(plain, [('tokens', [1]), ('native_state', 'b0')]) is None
    moded = GenerationRequest(messages=[], native_state_modes={'b0': 'same-base-adapter-state'})
    assert SglangEngine.sglang_unsupported(moded, [('native_state', 'b0')]) == 'native_state_modes'
    templated = GenerationRequest(messages=[], template={'call': 'f', 'value': 'write'})
    assert SglangEngine.sglang_unsupported(templated, [('tokens', [1])]) == 'template'


def test_recorded_collector_turns_stay_on_sglang():
    # The collector's graph recorder sets native_recording on every turn; it only asks for the capture.
    recorded = GenerationRequest(messages=[], temperature=1.0, rl_capture=True, native_recording=True)
    assert SglangEngine.sglang_unsupported(recorded, [('tokens', [1, 2])]) is None


@pytest.mark.parametrize('mode', ['text', 'text-twin'])
def test_privileged_text_guidance_is_captured_like_the_native_engine(tmp_path, mode):
    """JOLT+ text guidance on the SGLang path: same rendered prompt, insertion point and receipt as serve/engine.py.

    ``text-twin`` (a student under writer off): nothing enters the sampled prompt; the receipt records the same token
    ids and insertion point on both engines.
    """
    import threading
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.rl.privileged_context import REQUEST_SCHEMA, captured_block
    from natlang_neuralese.serve import load_native_engine
    native = engine_fixture(tmp_path, 'lfm')
    front = load_native_engine(str(tmp_path / 'native.pt'), device='cpu', dtype=native.backbone.embedding_weight.dtype,
                               engine_class=SglangEngine)
    front.sglang_banned = sorted(front.role_tokens.ids)
    front._sglang_lock, front.sglang_requests, front.sglang_tokens, front.native_fallbacks = threading.Lock(), 0, 0, 0
    front.native_fallback_reasons = {}
    request = {'schema': REQUEST_SCHEMA, 'invocation': {'name': 'run-1', 'role': 'root'}, 'mode': mode,
               'custody': {'schema': 'natlang.jolt-privileged-custody/1', 'case_id': 'c', 'split': 'train',
                           'world_sha256': 'd' * 64, 'protected_sha256': 'c' * 64},
               'fallback': 'writer failed', 'selection': {'implementation': 'nl', 'sha256': 'b' * 64},
               'facts': [{'id': 'expected_result:0', 'kind': 'expected_result', 'text': 'The verified result is t6.'}]}
    messages = [{'role': 'system', 'content': 't7'}, {'role': 'user', 'content': 't4 t5'}]
    make = lambda: GenerationRequest(messages=messages, temperature=.9, seed=5, max_tokens=4, rl_capture=True,
                                     rl_top_k=3, privileged_context=request, native_recording=True)
    expected = native.generate(make())['x_natlang_rl_capture']
    sampled = [s['token_id'] for s in expected['samples']]
    front._sglang_generate = lambda body: {'output_ids': sampled,
                                           'meta_info': {'natlang_top_k': [record(-1.0, [t, t + 1, t + 2], [-1.0, -2.0, -3.0])
                                                                           for t in sampled]}}
    actual = front.generate(make())['x_natlang_rl_capture']
    assert actual['prompt'] == expected['prompt'] and actual['privileged'] == expected['privileged']
    block = captured_block({'graph_turn': 'run-1#turn1', 'sampling': actual})
    assert block.insert_at == expected['privileged']['insert_at'] and block.token_ids
    if mode == 'text-twin':
        assert list(block.token_ids) == actual['privileged']['token_ids'] == expected['privileged']['token_ids']
        assert not any(segment.get('privileged') for segment in actual['prompt'])
    assert [s['token_id'] for s in actual['samples']] == sampled and actual['finish_reason'] == expected['finish_reason']


def test_streamed_reply_matches_whole_reply(tmp_path):
    """The collector streams: deltas concatenate to the reply text, and the capture equals the non-streamed one."""
    import threading
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.serve import load_native_engine
    native = engine_fixture(tmp_path, 'lfm')
    front = load_native_engine(str(tmp_path / 'native.pt'), device='cpu', dtype=native.backbone.embedding_weight.dtype,
                               engine_class=SglangEngine)
    front.sglang_banned = sorted(front.role_tokens.ids)
    front._sglang_lock, front.sglang_requests, front.sglang_tokens, front.native_fallbacks = threading.Lock(), 0, 0, 0
    front.native_fallback_reasons = {}
    words = front.tokenizer('t4 t5 t6 t7 t8', add_special_tokens=False)['input_ids']
    tokens = list(words) + [sorted(front.stop_ids)[0]]
    records = [record(-0.5, [t, t + 1], [-0.5, -1.5]) for t in tokens]
    front._sglang_generate = lambda body: {'output_ids': tokens, 'meta_info': {'natlang_top_k': records}}
    front._sglang_stream = lambda body: iter([{'output_ids': tokens[i:i + 2], 'meta_info': {'natlang_top_k': records[i:i + 2]}}
                                              for i in range(0, len(tokens), 2)])
    messages = [{'role': 'user', 'content': 't4 t5'}]
    make = lambda **extra: GenerationRequest(messages=messages, temperature=1.0, max_tokens=16, rl_capture=True,
                                             rl_top_k=2, native_recording=True, request_id='r', seed=1, **extra)
    whole = front.generate(make())
    deltas = []
    streamed = front.generate(make(on_delta=deltas.append))
    assert ''.join(d['text'] for d in deltas) == front.tokenizer.decode(words, skip_special_tokens=False)
    assert streamed['x_natlang_rl_capture'] == whole['x_natlang_rl_capture']
    assert front.sglang_requests == 2 and front.native_fallbacks == 0


def _span_fixture(tmp_path):
    import threading
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.serve import load_native_engine
    native = engine_fixture(tmp_path, 'lfm')
    front = load_native_engine(str(tmp_path / 'native.pt'), device='cpu', dtype=native.backbone.embedding_weight.dtype,
                               engine_class=SglangEngine)
    front.sglang_banned = []
    front._sglang_lock, front.sglang_requests, front.sglang_tokens, front.native_fallbacks = threading.Lock(), 0, 0, 0
    front.native_fallback_reasons, front.sglang_state_writes, front.sglang_capture_dir = {}, 0, None
    roles = front.role_tokens
    word = lambda text: list(front.tokenizer(text, add_special_tokens=False)['input_ids'])

    def capture(ids, start, end, adapter=None):
        # What the fork exports for ids[start:end]: the native forward of the whole sequence with its producing mask.
        import torch
        from natlang_neuralese.model.shared_state import native_step
        inside, mask = False, []
        for token in ids:
            mask.append(inside and token not in (roles.produce_open, roles.produce_close))
            inside = (inside or token == roles.produce_open) and token != roles.produce_close
        with torch.no_grad():
            out = native_step(native.shared_model, torch.tensor([ids]), producing=torch.tensor([mask]))
        rows = slice(start, end)
        return {'token_ids': out['token_ids'][rows], 'embeddings': out['embeddings'][rows],
                'positions': out['positions'][rows],
                'layers': tuple({key: (value if key == 'kind' else value[rows]) for key, value in layer.items()}
                                for layer in out['layers'])}
    front._sglang_capture = capture
    stored = {'native': [], 'front': []}
    for name, engine in (('native', native), ('front', front)):
        def store(*args, _engine=engine, _name=name, _original=type(engine)._native_store_capture, **kw):
            block, record_ = _original(_engine, *args, **kw)
            stored[_name].append(block)
            return block, record_
        engine._native_store_capture = store
    return native, front, roles, word, stored


def _scripted(native, front, tokens, make):
    """The native engine drawing exactly `tokens`, and the front given the same tokens by SGLang."""
    script = iter(tokens)
    native._native_token = lambda seq: (next(script), -0.5)
    records = [record(-0.5, [t, t + 1], [-0.5, -1.5]) for t in tokens]
    front._sglang_generate = lambda body: {'output_ids': tokens, 'meta_info': {'natlang_top_k': records}}
    front._sglang_stream = lambda body: iter([{'output_ids': tokens[i:i + 3], 'meta_info': {'natlang_top_k': records[i:i + 3]}}
                                              for i in range(0, len(tokens), 3)])
    return native.generate(make()), front.generate(make())


def _protocol_view(capture):
    samples = [{key: value for key, value in sample.items() if key not in ('logp', 'top_k')} for sample in capture['samples']]
    writes = [{key: value for key, value in write.items() if key not in ('state_id',)} for write in capture['writes']]
    return samples, capture['generated_inputs'], writes, capture['finish_reason']


def test_produce_span_follows_the_native_protocol_and_stores_the_same_block(tmp_path):
    import torch
    native, front, roles, word, stored = _span_fixture(tmp_path)
    tokens = (word('t4 t5') + [roles.produce_open] + word('t6 t7 t8') + [roles.produce_close] + word('t4')
              + [roles.produce_open, roles.produce_close] + [sorted(front.stop_ids)[0]])
    make = lambda **extra: GenerationRequest(messages=[{'role': 'user', 'content': 't4 t5'}], temperature=1.0,
                                             max_tokens=32, rl_capture=True, rl_top_k=2, native_recording=True,
                                             request_id='r', seed=1, **extra)
    expected, actual = _scripted(native, front, tokens, make)
    assert _protocol_view(actual['x_natlang_rl_capture']) == _protocol_view(expected['x_natlang_rl_capture'])
    assert len(stored['front']) == len(stored['native']) == 2 and front.sglang_state_writes == 2
    for mine, theirs in zip(stored['front'], stored['native']):
        assert torch.equal(mine.token_ids, theirs.token_ids) and torch.equal(mine.positions, theirs.positions)
        assert mine.embeddings.dtype == theirs.embeddings.dtype
        assert torch.allclose(mine.embeddings.float(), theirs.embeddings.float())
        for a, b in zip(mine.layers, theirs.layers):
            assert a['kind'] == b['kind'] and a.keys() == b.keys()
            for key in a:
                if key != 'kind':
                    assert a[key].dtype == b[key].dtype and torch.allclose(a[key], b[key], atol=1e-4, rtol=1e-3)
    strip = lambda content: [{k: v for k, v in part.items() if k != 'id'} if isinstance(part, dict) else part
                             for part in (content if isinstance(content, list) else [content])]
    assert strip(actual['choices'][0]['message']['content']) == strip(expected['choices'][0]['message']['content'])
    # Streamed: text and native-state deltas in order, the same capture.
    deltas = []
    stored['front'].clear()
    streamed = front.generate(make(on_delta=deltas.append))
    kinds = [next(iter(d)) for d in deltas]
    assert kinds.count('native_state') == 2 and kinds.index('native_state') > 0
    assert _protocol_view(streamed['x_natlang_rl_capture']) == _protocol_view(actual['x_natlang_rl_capture'])


@pytest.mark.parametrize('case', ['stop-inside', 'close-without-open', 'nested', 'consume-marker', 'length-inside'])
def test_protocol_failures_match_the_native_engine(tmp_path, case):
    native, front, roles, word, _ = _span_fixture(tmp_path)
    stop = sorted(front.stop_ids)[0]
    tokens, max_tokens = {
        'stop-inside': (word('t4') + [roles.produce_open] + word('t6') + [stop], 32),
        'close-without-open': (word('t4') + [roles.produce_close] + word('t6') + [stop], 32),
        'nested': (word('t4') + [roles.produce_open, roles.produce_open] + word('t6') + [stop], 32),
        'consume-marker': (word('t4') + [roles.consume_open] + word('t6') + [stop], 32),
        'length-inside': (word('t4') + [roles.produce_open] + word('t6 t7'), 4),
    }[case]
    make = lambda: GenerationRequest(messages=[{'role': 'user', 'content': 't4 t5'}], temperature=1.0,
                                     max_tokens=max_tokens, rl_capture=True, rl_top_k=2, native_recording=True,
                                     native_protocol_recovery=True, request_id='r', seed=1)
    expected, actual = _scripted(native, front, tokens, make)
    mine, theirs = actual['x_natlang_rl_capture'], expected['x_natlang_rl_capture']
    assert mine['protocol_failure'] == theirs['protocol_failure'] and mine['protocol_failure']['code'].startswith('neuralese')
    assert _protocol_view(mine) == _protocol_view(theirs)


def _import_fixture(tmp_path):
    native, front, roles, word, stored = _span_fixture(tmp_path)
    front.sglang_export = {'architecture': front.model_config.get('architecture'),
                           'layer_kinds': ['attention' if native.backbone.is_attention(i) else 'finite_convolution'
                                           for i in range(native.backbone.num_layers)],
                           'producer_embedding': {'input_id_offset': native.backbone.embedding_weight.shape[0]}}
    tokens = word('t4') + [roles.produce_open] + word('t6 t7') + [roles.produce_close] + [sorted(front.stop_ids)[0]]
    make = lambda: GenerationRequest(messages=[{'role': 'user', 'content': 't4 t5'}], temperature=1.0, max_tokens=32,
                                     rl_capture=True, rl_top_k=2, native_recording=True, request_id='r', seed=1)
    _scripted(native, front, tokens, make)
    block = stored['native'][0]
    block_id = native.store.put_native_state(block).id
    assert front.store.put_native_state(block).id == block_id
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 't4 '}, {'type': 'native_state', 'id': block_id},
                                             {'type': 'text', 'text': ' t5'}]}]
    return native, front, roles, word, block, messages


def test_imported_native_state_is_recorded_like_the_native_engine(tmp_path):
    """A turn that imports a produced block: same prompt receipt and imports on both engines; SGLang gets the block's
    positions as placeholders plus the import spec, whose file holds the block's exact tensors."""
    import torch
    from safetensors.torch import load_file
    native, front, roles, word, block, messages = _import_fixture(tmp_path)
    later = lambda: GenerationRequest(messages=messages, temperature=1.0, max_tokens=8, rl_capture=True, rl_top_k=2,
                                      native_recording=True, request_id='r2', seed=2)
    reply = word('t8') + [sorted(front.stop_ids)[0]]
    script = iter(reply)
    native._native_token = lambda seq: (next(script), -0.5)
    expected = native.generate(later())
    records = [record(-0.5, [t, t + 1], [-0.5, -1.5]) for t in reply]
    sent = []
    front._sglang_generate = lambda body: (sent.append(body), {'output_ids': reply, 'meta_info': {'natlang_top_k': records}})[1]
    actual = front.generate(later())
    mine, theirs = actual['x_natlang_rl_capture'], expected['x_natlang_rl_capture']
    assert mine['prompt'] == theirs['prompt'] and actual['native_state']['imports'] == expected['native_state']['imports']
    assert _protocol_view(mine) == _protocol_view(theirs)
    spec, = sent[0]['sampling_params']['custom_params']['natlang_import']
    ids, count = sent[0]['input_ids'], len(block.token_ids)
    assert spec['count'] == count and ids[spec['start'] - 1] == roles.consume_open
    assert ids[spec['start'] + count] == roles.consume_close
    assert all(i >= 2 * native.backbone.embedding_weight.shape[0] for i in ids[spec['start']:spec['start'] + count])
    assert len(ids) == mine['samples'][0]['query_own_position'] + 1 + count  # imported positions are not own positions
    saved = load_file(spec['path'])
    for index, layer in enumerate(block.layers):
        for name, value in layer.items():
            if name != 'kind':
                assert torch.equal(saved[f'layers.{index}.{name}'], value.cpu())


def test_decide_scores_on_sglang_with_the_native_decision_capture(tmp_path):
    from natlang_neuralese.serve.native_learning import decide_native
    native, front, roles, word, block, messages = _import_fixture(tmp_path)
    import concurrent.futures
    front._pool = concurrent.futures.ThreadPoolExecutor(4)
    body = {'messages': messages, 'options': ['t6', 't7 t8'], 'native_recording': True}
    expected = decide_native(native, body)
    sent = []

    def generate(request):
        sent.append(request)
        cont = request['input_ids'][request['logprob_start_len'] + 1:]
        return {'meta_info': {'input_token_logprobs': [[None, request['input_ids'][request['logprob_start_len']], None]]
                              + [[-0.25, t, None] for t in cont]}}
    front._sglang_generate = generate
    actual = decide_native(front, body)
    assert actual['native_decision_capture'] == expected['native_decision_capture']
    assert actual['tokens'] == expected['tokens'] and actual['native_state_imports'] == expected['native_state_imports']
    assert actual['log_probs'] == [-0.25 * n for n in expected['tokens']]
    assert all(r['sampling_params']['custom_params']['natlang_import'] for r in sent) and front.native_fallbacks == 0


def _adapters(engine, names):
    """An xs adapter and a free-lora adapter with nonzero factors on `names`."""
    from dataclasses import replace
    from natlang_neuralese.model.native_adapters import create_adapter
    made = []
    for kind, seed in (('xs', 1), ('free-lora', 2)):
        artifact = create_adapter(engine, {'kind': kind, 'rank': 2, 'seed': seed, 'parameter_names': names}).artifact
        generator = __import__('torch').Generator().manual_seed(seed)
        values = {name: __import__('torch').randn(value.shape, generator=generator) * .05 for name, value in artifact.tensors.items()}
        made.append(engine.store.put_native_tensor_artifact(replace(artifact, tensors=values)).id)
    return made


def test_lowrank_factors_reproduce_the_native_effective_deltas(tmp_path):
    import torch
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.model.native_adapters import (_effective_replacements, lowrank_factors, resolve_adapters,
                                                          target_parameters)
    engine = engine_fixture(tmp_path, 'lfm')
    available = target_parameters(engine.shared_model)
    names = [name for name in available if name.endswith(('q_proj.weight', 'v_proj.weight'))][:3]
    xs, free = _adapters(engine, names)
    entries = [{'id': xs, 'scale': .75}, {'id': free, 'scale': 1.5}]
    resolved = resolve_adapters(engine, entries)
    factors = lowrank_factors(resolved)
    float_params = {name: value.float() for name, value in available.items()}
    replacement, _ = _effective_replacements(float_params, resolved, None)
    for name in names:
        factor_b, factor_a = factors[name]
        assert factor_a.shape[0] == 4  # the two rank-2 adapters' ranks concatenated
        torch.testing.assert_close(float_params[name] + factor_b @ factor_a, replacement['model.' + name], atol=1e-6, rtol=1e-5)


def test_adapted_turns_bind_like_the_native_engine_and_pack_module_factors(tmp_path):
    """On SGLang an adapted turn records the native policy identity and adapter binding, and the slot file packs each
    native matrix's factors at its row offset inside the fork module (q|k|v)."""
    import threading
    import torch
    from safetensors.torch import load_file
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.serve import load_native_engine
    from natlang_neuralese.model.native_adapters import lowrank_factors, resolve_adapters, target_parameters
    native = engine_fixture(tmp_path, 'lfm')
    front = load_native_engine(str(tmp_path / 'native.pt'), device='cpu', dtype=native.backbone.embedding_weight.dtype,
                               engine_class=SglangEngine, store=native.store)
    available = target_parameters(native.shared_model)
    names = [name for name in available if name.endswith(('q_proj.weight', 'k_proj.weight'))][:2]
    layer = names[0].split('.layers.')[1].split('.')[0]
    q, k = (f'port.hf.model.layers.{layer}.self_attn.{part}_proj.weight' for part in ('q', 'k'))
    targets = {q: {'module': f'layers.{layer}.qkv_proj', 'offset': 0, 'shape': list(available[q].shape)},
               k: {'module': f'layers.{layer}.qkv_proj', 'offset': available[q].shape[0], 'shape': list(available[k].shape)}}
    xs, free = _adapters(native, [q, k])
    entries = [{'id': xs, 'scale': .5}, free]
    front.sglang_banned, front.sglang_capture_dir = [], None
    front._sglang_lock, front.sglang_requests, front.sglang_tokens, front.native_fallbacks = threading.Lock(), 0, 0, 0
    front.native_fallback_reasons, front.sglang_state_writes = {}, 0
    front.sglang_adapter_slots, front.sglang_adapter_rank, front.sglang_adapter_targets = 2, 8, targets
    front.native_target_shapes = {name: tuple(value.shape) for name, value in available.items()}
    front.sglang_module_rows = {f'layers.{layer}.qkv_proj': available[q].shape[0] + 2 * available[k].shape[0]}
    front._adapter_condition = threading.Condition()
    front._adapter_slot_of, front._adapter_slot_users, front._adapter_slot_key = {}, {}, {}
    front._adapter_files, front._adapter_clock = {}, 0
    word = lambda text: list(front.tokenizer(text, add_special_tokens=False)['input_ids'])
    tokens = word('t6 t7') + [sorted(front.stop_ids)[0]]
    make = lambda: GenerationRequest(messages=[{'role': 'user', 'content': 't4 t5'}], temperature=1.0, max_tokens=8,
                                     rl_capture=True, rl_top_k=2, native_recording=True, request_id='r', seed=1,
                                     adapters=entries)
    assert front._sglang_reason(make(), [('tokens', [1])]) is None
    expected, sent = None, []
    script = iter(tokens)
    native._native_token = lambda seq: (next(script), -0.5)
    expected = native.generate(make())
    records = [record(-0.5, [t, t + 1], [-0.5, -1.5]) for t in tokens]
    front._sglang_generate = lambda body: (sent.append(body), {'output_ids': tokens, 'meta_info': {'natlang_top_k': records}})[1]
    actual = front.generate(make())
    mine, theirs = actual['x_natlang_rl_capture'], expected['x_natlang_rl_capture']
    assert mine['schema'] == theirs['schema'] == 'natlang.native-rl-sampling/3'
    assert mine['policy'] == theirs['policy'] and mine['adapter_binding'] == theirs['adapter_binding']
    spec = sent[0]['sampling_params']['custom_params']['natlang_adapter']
    assert spec['slot'] == 1 and front._adapter_slot_users[1] == 0
    packed = load_file(spec['path'])
    factor_b, factor_a = packed[f'layers.{layer}.qkv_proj.B'], packed[f'layers.{layer}.qkv_proj.A']
    delta = factor_b @ factor_a
    expected_factors = lowrank_factors(resolve_adapters(native, entries))
    for name, offset in ((q, 0), (k, available[q].shape[0])):
        b, a = expected_factors[name]
        torch.testing.assert_close(delta[offset:offset + b.shape[0]], b @ a)
    torch.testing.assert_close(delta[available[q].shape[0] + available[k].shape[0]:], torch.zeros_like(delta[:available[k].shape[0]]))
    front.sglang_adapter_rank = 4  # summed rank 8 in qkv_proj no longer fits: the native engine serves it
    front._adapter_files.clear()
    assert front._sglang_reason(make(), [('tokens', [1])]) == 'adapter rank'


def test_privileged_state_guidance_imports_on_sglang_like_the_native_engine(tmp_path):
    """JOLT+ guidance in import mode (the writer's native state inserted into the teacher's prompt) runs on SGLang:
    same prompt receipt (privileged marks included), guidance receipt and imports as the native engine."""
    from natlang_neuralese.rl.privileged_context import REQUEST_SCHEMA, WRITER_FUNCTION
    native, front, roles, word, block, _ = _import_fixture(tmp_path)
    block_id = native.store.put_native_state(block).id
    request = {'schema': REQUEST_SCHEMA, 'invocation': {'name': 'run-1', 'role': 'root'}, 'mode': 'import',
               'custody': {'schema': 'natlang.jolt-privileged-custody/1', 'case_id': 'c', 'split': 'train',
                           'world_sha256': 'd' * 64, 'protected_sha256': 'c' * 64},
               'selection': {'implementation': 'nl', 'sha256': 'b' * 64},
               'writer': {'function': WRITER_FUNCTION, 'sha256': 'a' * 64}, 'state': {'id': block_id},
               'facts': [{'id': 'expected_result:0', 'kind': 'expected_result', 'text': 'The verified result is t6.'}]}
    messages = [{'role': 'system', 'content': 't7'}, {'role': 'user', 'content': 't4 t5'}]
    make = lambda: GenerationRequest(messages=messages, temperature=.9, seed=5, max_tokens=4, rl_capture=True, rl_top_k=3,
                                     privileged_context=request, native_recording=True, request_id='p')
    reply = word('t8') + [sorted(front.stop_ids)[0]]
    script = iter(reply)
    native._native_token = lambda seq: (next(script), -0.5)
    expected = native.generate(make())
    records = [record(-0.5, [t, t + 1, t + 2], [-0.5, -1.5, -2.5]) for t in reply]
    sent = []
    front._sglang_generate = lambda body: (sent.append(body), {'output_ids': reply, 'meta_info': {'natlang_top_k': records}})[1]
    assert front._sglang_reason(make(), front._prompt_plan(messages, None)) is None
    actual = front.generate(make())
    mine, theirs = actual['x_natlang_rl_capture'], expected['x_natlang_rl_capture']
    assert mine['prompt'] == theirs['prompt'] and mine['privileged'] == theirs['privileged']
    assert any(segment.get('privileged') and segment['kind'] == 'native_state' for segment in mine['prompt'])
    assert actual['native_state']['imports'] == expected['native_state']['imports'] and front.native_fallbacks == 0
    assert sent[0]['sampling_params']['custom_params']['natlang_import']
