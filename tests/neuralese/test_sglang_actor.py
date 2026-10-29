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
