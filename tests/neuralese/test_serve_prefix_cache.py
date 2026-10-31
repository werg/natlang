"""Prompt-prefix reuse in the native server is exact: a resumed prefill equals a fresh one on the same grid."""
import torch

from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model, model_state
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.prefix_cache import PrefixCache

from test_native_serve import _tiny_source


def _engine(tmp_path, block):
    base = tmp_path / 'base'
    if not base.exists():
        base.mkdir()
        _tiny_source(base)
        model, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
        torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, tmp_path / 'native.pt')
    engine = load_native_engine(str(tmp_path / 'native.pt'), device='cpu', dtype=torch.float32)
    if block:
        engine.prefix_cache = PrefixCache(2**26, block)
    return engine


def _prefill(engine, ids):
    with torch.no_grad():
        output = engine._native_run(lambda: engine._native_prefill_forward(list(ids), None))
        return engine._native_logits(output['h_final']), output['cache']


def test_resumed_prefill_is_bitwise_a_fresh_prefill_on_the_grid(tmp_path):
    shared = [5 + i % 40 for i in range(21)]
    first, second = shared + [50, 51, 52], shared + [60, 61]
    warm = _engine(tmp_path, 4)
    _prefill(warm, first)
    assert warm.prefix_cache.hits == 0
    logits, cache = _prefill(warm, second)
    assert warm.prefix_cache.hits == 1 and warm.prefix_cache.reused_tokens == 20  # the last grid boundary in the shared prefix
    cold_logits, cold_cache = _prefill(_engine(tmp_path, 4), second)
    assert torch.equal(logits, cold_logits)
    assert cache.lengths == cold_cache.lengths
    for state, cold in zip(cache.states, cold_cache.states):
        for name in state.__dataclass_fields__:
            a, b = getattr(state, name), getattr(cold, name)
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b), name
    # A repeated whole prompt resumes from its end and gives the same next-token logits.
    again, _ = _prefill(warm, second)
    assert torch.equal(again, logits) and warm.prefix_cache.reused_tokens == 20 + len(second)
    # Off-grid default prefill computes the same function (numerically close, not bound to be bitwise).
    default_logits, _ = _prefill(_engine(tmp_path, 0), second)
    assert torch.allclose(default_logits, logits, atol=1e-4, rtol=1e-4)


def test_continuing_a_reused_state_leaves_the_stored_snapshot_unchanged(tmp_path):
    engine = _engine(tmp_path, 4)
    prompt = [7 + i % 30 for i in range(12)]
    base_logits, _ = _prefill(engine, prompt)
    for tail in ([40], [41, 42], [43]):
        _, cache = _prefill(engine, prompt)
        with torch.no_grad():
            engine._native_run(lambda: engine._native_forward(tail, cache))
    logits, _ = _prefill(engine, prompt)
    assert torch.equal(logits, base_logits)


def test_budget_evicts_least_recent_and_counts_shared_storage_once(tmp_path):
    cache = PrefixCache(1000, 4)
    tensor = torch.zeros(100)          # 400 bytes
    view = tensor[:50]
    cache.store('a', (tensor,), view)  # one storage
    assert cache.bytes == 400
    cache.store('b', (view,), torch.zeros(10))
    assert cache.bytes == 440
    cache.store('c', (torch.zeros(200),), torch.zeros(1))
    assert 'a' not in cache.entries and cache.bytes <= 1000
    digests = cache.digests({'k': 1}, list(range(10)))
    assert sorted(digests) == [4, 8, 10]
    assert cache.digests({'k': 2}, list(range(10)))[4] != digests[4]


def test_oversized_snapshot_does_not_exceed_budget_or_leave_storage_refs():
    cache = PrefixCache(100, 4)
    cache.store('small', (torch.zeros(10),), None)
    cache.store('oversized', (torch.zeros(100),), None)
    assert cache.bytes <= cache.budget
    assert 'oversized' not in cache.entries
    assert list(cache.entries) == ['small'] and cache.bytes == 40
    assert len(cache.refs) == 1 and next(iter(cache.refs.values())) == [40, 1]
    cache.store('next', (torch.zeros(10),), None)
    assert cache.bytes == 80 and list(cache.entries) == ['small', 'next']


def test_native_prefix_snapshot_retains_small_state_not_full_prefill_storage():
    from natlang_neuralese.model.lfm2_port import PortCache, ConvState, DeltaState, AttentionState, KVBuffer
    projected = torch.arange(8192, dtype=torch.float32).reshape(1, 8, 1024)
    hidden = torch.arange(32768, dtype=torch.float32).reshape(1, 1024, 32)
    keys, values = torch.zeros(1, 2, 16, 4), torch.ones(1, 2, 16, 4)
    buffer = KVBuffer(keys, values, 8)
    attention = AttentionState(keys[:, :, :8], values[:, :, :8], buffer)
    recurrent = torch.ones(1, 2, 4, 4)
    original = PortCache((ConvState(projected[..., -6:]),
                          DeltaState(projected[..., -4:], recurrent), attention), (1024,1024,8))
    cache = PrefixCache(4096, 256)
    cache.store('snapshot', original, hidden[:, -1:])
    # The large source tensors cannot fit this budget; their exact logical tails can.
    assert list(cache.entries) == ['snapshot'] and cache.bytes <= cache.budget
    stored, last, _ = cache.entries['snapshot']
    assert torch.equal(stored.states[0].window, original.states[0].window)
    assert torch.equal(stored.states[1].conv, original.states[1].conv)
    assert torch.equal(last, hidden[:, -1:])
    assert stored.states[0].window.untyped_storage().nbytes() == 8*6*4
    assert last.untyped_storage().nbytes() == 32*4
    assert stored.states[1].recurrent is recurrent and stored.states[2] is attention
    assert stored.states[2].buffer is buffer and stored.lengths == original.lengths


def test_native_snapshot_compaction_keeps_input_gradients():
    from natlang_neuralese.model.lfm2_port import PortCache, ConvState
    source = torch.randn(1, 4, 32, requires_grad=True)
    hidden = torch.randn(1, 32, 8, requires_grad=True)
    cache = PrefixCache(4096, 16)
    cache.store('differentiable', PortCache((ConvState(source[..., -3:]),),(32,)), hidden[:, -1:])
    stored, last, _ = cache.entries['differentiable']
    (stored.states[0].window.sum()+last.sum()).backward()
    assert torch.equal(source.grad[..., -3:], torch.ones_like(source[..., -3:]))
    assert torch.count_nonzero(source.grad[..., :-3]) == 0
    assert torch.equal(hidden.grad[:, -1:], torch.ones_like(hidden[:, -1:]))
    assert torch.count_nonzero(hidden.grad[:, :-1]) == 0


def test_prefix_configuration_is_bound_to_policy_and_metrics(tmp_path):
    from natlang_neuralese.serve import configure_native_prefix_cache
    engine = _engine(tmp_path, 0)
    initial = engine.serving_config['identity_sha256']
    config = configure_native_prefix_cache(engine, 0.01, 4)
    assert config['identity_sha256'] != initial
    assert engine.rl_policy_binding['serving_config_sha256'] == config['identity_sha256']
    assert config['prefix_cache']['execution'] == 'fixed-block-grid'
    _prefill(engine, list(range(5, 17)))
    _prefill(engine, list(range(5, 15)) + [40, 41])
    metrics = engine.progress_metrics()
    assert metrics['natlang_native_prefill_tokens_processed_total'] == 16
    assert metrics['natlang_prefix_cache_reused_tokens'] == 8
    disabled = configure_native_prefix_cache(engine, 0, 4)
    assert disabled['identity_sha256'] != config['identity_sha256']
    assert engine.prefix_cache is None
    assert disabled['prefix_cache']['execution'] == 'native-prefill-chunks'


def test_prefix_configuration_rejects_invalid_or_live_changes(tmp_path):
    import pytest
    from natlang_neuralese.serve import configure_native_prefix_cache
    engine = _engine(tmp_path, 0)
    for budget, block in ((-1, 4), (float('nan'), 4), (1, 0), (1, True)):
        with pytest.raises(ValueError):
            configure_native_prefix_cache(engine, budget, block)
    with pytest.raises(ValueError, match='exceeds the bound prefill'):
        configure_native_prefix_cache(engine, 1, engine.prefill_tokens + 1)
    engine._thread = object()
    with pytest.raises(ValueError, match='after engine start'):
        configure_native_prefix_cache(engine, 1, 4)


def test_default_prefix_grid_preserves_existing_prefill_chunks(tmp_path):
    from natlang_neuralese.serve import configure_native_prefix_cache
    engine = _engine(tmp_path, 0)
    engine.prefill_tokens = 8
    ids = list(range(5, 21))
    old_logits, old_cache = _prefill(engine, ids)
    config = configure_native_prefix_cache(engine, .01)
    assert config['prefix_cache']['block_tokens'] == 8
    cold_logits, cold_cache = _prefill(engine, ids)
    warm_logits, warm_cache = _prefill(engine, ids)
    assert torch.equal(old_logits, cold_logits) and torch.equal(cold_logits, warm_logits)
    assert engine.prefix_cache.reused_tokens == len(ids)
    for original, cached in zip(old_cache.states, cold_cache.states):
        for name in original.__dataclass_fields__:
            a, b = getattr(original, name), getattr(cached, name)
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
