"""Speculative decoding in the Python native engine (Engine._native_feed): a reply that repeats its own text drafts from
it, verifies chains in one forward and keeps the plain decode's tokens, capture and state (up to the floating-point
floor of a chunk forward against single-token forwards)."""
import copy
from concurrent.futures import Future

import pytest
import torch

from natlang_neuralese.model.shared_state import adopt_step_cache, native_step, truncate_step_cache
from test_native_decode_batch import model


@pytest.mark.parametrize('family', ['lfm', 'lfm3'])
def test_truncated_chunk_cache_is_the_cache_of_its_prefix(family):
    m = model(family)
    with torch.no_grad():
        base = native_step(m, torch.tensor([[2, 3, 4]]))['cache']
        for keep in range(1, 5):
            # a fresh chunk each time: the kept cache appends in place into the chunk's shared storage
            chunk = native_step(m, torch.tensor([[5, 6, 7, 8]]), base)
            cut = adopt_step_cache(truncate_step_cache(base, chunk, keep))
            direct = native_step(m, torch.tensor([[5, 6, 7, 8][:keep]]), base)['cache']
            assert cut.lengths == direct.lengths
            nxt = native_step(m, torch.tensor([[9]]), cut)
            ref = native_step(m, torch.tensor([[9]]), direct)
            torch.testing.assert_close(nxt['h_final'], ref['h_final'], atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('family', ['lfm'])
def test_speculative_reply_equals_plain_decode(tmp_path, family, monkeypatch):
    from natlang_neuralese.serve.engine import Sequence, GenerationRequest
    from test_native_serving_adapters import engine_fixture
    engine = engine_fixture(tmp_path, family)
    (tmp_path / 'reference').mkdir()
    reference = engine_fixture(tmp_path / 'reference', family)
    reference.shared_model = copy.deepcopy(engine.shared_model); reference.backbone = reference.shared_model.port
    engine.native_spec_draft, reference.native_spec_draft = 4, 0
    for e in (engine, reference):
        e.native_spec_layers_ok = True
    stop = next(iter(engine.stop_ids))
    pattern = [11, 12, 13, 14, 15]
    replies = {}
    try:
        for e in (engine, reference):
            seq = Sequence(GenerationRequest(messages=[{'role': 'user', 'content': 'repeat'}], max_tokens=40,
                                             temperature=.8, seed=3, rl_capture=True, request_id='r'), Future())
            seq.prompt_plan = e._prompt_plan(seq.request.messages, seq.request.tools)
            script = iter(pattern * 4 + [stop])

            def token(seq, script=script):
                value = next(script)
                return value, float(seq.logits.float().div(.8).log_softmax(-1)[0, value])
            e._native_token = token
            forwards = [0]
            forward = e._native_forward

            def counted(*args, forward=forward, forwards=forwards, **kwargs):
                forwards[0] += 1
                return forward(*args, **kwargs)
            e._native_forward = counted
            with torch.no_grad():
                for _ in range(40):
                    if seq.phase == 'done':
                        break
                    e._native_round([seq])
            seq.future.result()
            replies[e is engine] = (seq, forwards[0])
        (spec, spec_forwards), (plain, plain_forwards) = replies[True], replies[False]
        assert spec.items == plain.items and spec.finish_reason == plain.finish_reason == 'stop'
        assert spec.native_rl_inputs == plain.native_rl_inputs
        for x, y in zip(spec.rl_samples, plain.rl_samples):
            assert {k: v for k, v in x.items() if k != 'logp'} == {k: v for k, v in y.items() if k != 'logp'}
            assert x['logp'] == pytest.approx(y['logp'], abs=2e-5)
        assert engine._native_spec_drafted > 0 and spec_forwards < plain_forwards
    finally:
        engine.store.close(); reference.store.close()
