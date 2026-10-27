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
