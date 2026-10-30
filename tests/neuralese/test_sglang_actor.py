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
    assert 'native_state' in SglangEngine.sglang_unsupported(plain, [('tokens', [1]), ('native_state', 'b0')])
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
