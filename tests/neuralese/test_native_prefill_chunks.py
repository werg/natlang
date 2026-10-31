"""Native prefill obeys the shared workspace setting with exact hybrid/KV continuation."""
from concurrent.futures import Future
import torch

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from natlang_neuralese.serve import native_prefill_configuration
from natlang_neuralese.serve.engine import GenerationRequest, Sequence


def test_small_gpu_prefill_policy_is_explicit_and_preserves_full_context():
    small = native_prefill_configuration('cuda:0', total_memory_bytes=12 * 1024**3)
    large = native_prefill_configuration('cuda:0', total_memory_bytes=12 * 1024**3 + 1)
    explicit = native_prefill_configuration('cpu', requested=4096)
    assert small['prefill_tokens'] == 2048
    assert large['prefill_tokens'] == 8192
    assert explicit['prefill_tokens'] == 4096
    assert small['full_prompt_preserved'] is large['full_prompt_preserved'] is True
    assert small['device_total_memory_bytes'] == 12 * 1024**3
    with pytest.raises(ValueError, match='positive integer'):
        native_prefill_configuration('cpu', requested=0)


def test_finished_request_releases_sequence_owned_kv_after_response(tmp_path, monkeypatch):
    engine = engine_fixture(tmp_path, 'lfm')
    try:
        record = engine.native_write_text('store-owned state')
        seq = Sequence(GenerationRequest(messages=[], request_id='release-kv'), Future())
        seq.cache = object()
        seq.logits = torch.ones(1)
        seq.native_capture_chunks.append(object())
        seq.native_capture_ids.append(7)
        monkeypatch.setattr(engine, '_native_response', lambda _seq: {'ok': True})
        engine._finish(seq, 'stop')
        assert seq.future.result() == {'ok': True}
        assert seq.cache is None and seq.logits is None
        assert seq.native_capture_chunks == [] and seq.native_capture_ids == []
        assert engine.store.get_native_state(record['id']) is not None
    finally:
        engine.store.close()


def test_failed_request_drops_gpu_traceback_frames_and_keeps_formatted_diagnostic(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    try:
        seq = Sequence(GenerationRequest(messages=[], request_id='release-failed-kv'), Future())
        seq.cache = object()
        seq.logits = torch.ones(1)
        seq.native_capture_chunks.append(object())
        try:
            raise RuntimeError('synthetic inference failure')
        except RuntimeError as caught:
            failure = caught
        engine._fail(seq, failure)
        stored = seq.future.exception()
        assert stored is failure
        assert stored.__traceback__ is None
        assert 'synthetic inference failure' in stored.natlang_traceback
        assert seq.cache is None and seq.logits is None
        assert seq.native_capture_chunks == []
    finally:
        engine.store.close()


@pytest.mark.parametrize('family', ['lfm'])
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


def test_chunked_prefill_forced_replay_preserves_exact_capture(tmp_path, monkeypatch):
    engine = engine_fixture(tmp_path, 'lfm')
    try:
        forced_ids = [7, next(iter(engine.stop_ids))]
        monkeypatch.setattr(engine, '_forced_message_plan',
            lambda _forced, _tools, opening: list(forced_ids))
        def replay(chunk, request_id):
            engine.prefill_tokens = chunk
            return engine.generate(GenerationRequest(messages=[{'role': 'user', 'content': 'tiny prompt'}],
                max_tokens=8, temperature=0.0, request_id=request_id,
                forced_message={'source_receipt': {'schema': 'test-only-forced-source'}},
                native_recording=True, rl_capture=True))
        reference = replay(8192, 'forced-reference')
        chunked = replay(2, 'forced-chunked')
        assert reference['choices'] == chunked['choices']
        ref_capture = reference['x_natlang_rl_capture']
        actual_capture = chunked['x_natlang_rl_capture']
        assert [row['token_id'] for row in ref_capture['samples']] == forced_ids
        assert [row['token_id'] for row in actual_capture['samples']] == forced_ids
        assert ref_capture['prompt'] == actual_capture['prompt']
        assert ref_capture['generated_inputs'] == actual_capture['generated_inputs']
    finally:
        engine.store.close()
