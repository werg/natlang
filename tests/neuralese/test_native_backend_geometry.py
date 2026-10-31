from types import SimpleNamespace
from natlang_neuralese.eval import native_backend


def test_complete_runtime_forwards_long_source_and_consumer_geometry(monkeypatch):
    seen = []
    monkeypatch.setattr(native_backend, 'capture_training_rng_state', lambda _: 'rng')
    monkeypatch.setattr(native_backend, 'restore_training_rng_state', lambda *_: None)
    monkeypatch.setattr(native_backend, 'qualify_existing', lambda *a, **kw: seen.append(kw) or {'passed': False})
    opt = SimpleNamespace(_native_named_parameters={'weight': object()})
    native_backend.qualify_complete_runtime(None, None, None, {}, optimizer=opt, backend=None,
        fp8_profile={'kind': 'off'}, checkpoint_sha256='a'*64, forward_atol=1e-4,
        gradient_atol=1e-5, device='cpu', minimum_source_tokens=512, minimum_target_tokens=256)
    assert seen[0]['minimum_source_tokens'] == 512
    assert seen[0]['minimum_target_tokens'] == 256
