"""Forward-only adapted inference reuses a stack's effective weights: bitwise the uncached actor, far less work."""
import pytest
import torch

from test_native_invocation_actors import adapters
from test_native_serving_adapters import engine_fixture
from natlang_neuralese.model import native_adapters
from natlang_neuralese.model.native_adapters import InferenceReplacementCache, run_native


def _decode(engine, stack, steps=4):
    """Prefill then greedy decode under one adapter stack; returns every step's logits."""
    with torch.no_grad():
        out = run_native(engine, lambda: engine._native_forward(list(range(5, 17)), None), stack)
        cache, logits = out['cache'], []
        token = 20
        for _ in range(steps):
            out = run_native(engine, lambda: engine._native_forward([token], cache), stack)
            cache = out['cache']
            step = run_native(engine, lambda: engine.shared_model.logits(out['h_final'][:, -1]), stack)
            logits.append(step)
            token = int(step.argmax(-1)[0]) % 40 + 5
        return logits


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_cached_inference_is_bitwise_the_uncached_actor_and_resolves_once(tmp_path, family, monkeypatch):
    engine = engine_fixture(tmp_path, family)
    stack = [record.id for record in adapters(engine)]
    engine.native_adapter_cache_gb = 0
    uncached = _decode(engine, stack)
    del engine.native_adapter_cache_gb
    calls = []
    original = native_adapters.resolve_adapters
    monkeypatch.setattr(native_adapters, 'resolve_adapters', lambda *a, **k: calls.append(1) or original(*a, **k))
    cached = _decode(engine, stack)
    assert all(torch.equal(a, b) for a, b in zip(uncached, cached))
    assert len(calls) == 1, 'one stack resolves once across prefill, decode and readout'
    assert engine.native_adapter_cache.hits >= 8
    # Another stack is its own entry; plain execution never touches the cache.
    other = _decode(engine, stack[:1], steps=1)
    assert len(engine.native_adapter_cache.entries) == 2 and not torch.equal(other[0], cached[0])
    with torch.no_grad():
        run_native(engine, lambda: engine._native_forward([5, 6], None), None)
    assert len(engine.native_adapter_cache.entries) == 2


def test_gradient_enabled_execution_bypasses_the_cache(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    stack = [record.id for record in adapters(engine)]
    run_native(engine, lambda: engine._native_forward([5, 6, 7], None), stack)
    assert getattr(engine, 'native_adapter_cache', None) is None or not engine.native_adapter_cache.entries


def test_budget_bounds_entries_and_parameter_changes_miss():
    cache = InferenceReplacementCache(100)
    parameters = {'w': torch.zeros(2)}
    key = InferenceReplacementCache.key(['a'], parameters)
    cache.put(key, 'value', 60)
    assert cache.get(key) == 'value'
    cache.put(InferenceReplacementCache.key(['b'], parameters), 'other', 60)
    assert key not in cache.entries and cache.bytes == 60
    cache.put(('big',), 'x', 101)
    assert ('big',) not in cache.entries
    parameters['w'].add_(1)
    assert InferenceReplacementCache.key(['a'], parameters) != key
