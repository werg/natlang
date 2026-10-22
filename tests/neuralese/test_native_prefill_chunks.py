"""Native prefill obeys the shared workspace setting with exact hybrid/KV continuation."""
from concurrent.futures import Future

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from natlang_neuralese.serve.engine import GenerationRequest, Sequence


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('chunk', [1, 3, 8])
def test_chunked_prefill_preserves_imports_hybrid_state_and_next_token(tmp_path, monkeypatch, family, chunk):
    engine = engine_fixture(tmp_path, family)
    try:
        block = engine.native_write_text('t5 t6')['id']
        plan = [('tokens', [4, 5, 6, 7, 8, 9, 10]), ('native_state', block),
                ('tokens', [11, 12, 13, 14, 15])]
        def sequence():
            return Sequence(GenerationRequest(messages=[], request_id='prefill-parity', native_recording=True),
                            Future(), prompt_plan=plan)
        with torch.inference_mode():
            engine.prefill_tokens = 8192
            reference = sequence()
            engine._native_prefill(reference)
            observed = []
            original = engine._native_forward
            def forward(ids, cache, **kwargs):
                observed.append(len(ids))
                return original(ids, cache, **kwargs)
            monkeypatch.setattr(engine, '_native_forward', forward)
            engine.prefill_tokens = chunk
            actual = sequence()
            engine._native_prefill(actual)
        assert max(observed) <= chunk
        assert actual.prompt_positions == reference.prompt_positions
        assert actual.native_own_position == reference.native_own_position
        assert actual.native_rl_prompt == reference.native_rl_prompt
        assert actual.native_imports == reference.native_imports
        assert actual.cache.lengths == reference.cache.lengths
        torch.testing.assert_close(actual.logits, reference.logits, atol=3e-6, rtol=2e-5)
        assert actual.logits.argmax().item() == reference.logits.argmax().item()
        for actual_state, reference_state in zip(actual.cache.states, reference.cache.states):
            if hasattr(actual_state, 'k'):
                count = actual.prompt_positions
                torch.testing.assert_close(actual_state.k[..., :count, :], reference_state.k[..., :count, :], atol=3e-6, rtol=2e-5)
                torch.testing.assert_close(actual_state.v[..., :count, :], reference_state.v[..., :count, :], atol=3e-6, rtol=2e-5)
            else:
                torch.testing.assert_close(actual_state.window, reference_state.window, atol=3e-6, rtol=2e-5)
    finally:
        engine.store.close()
