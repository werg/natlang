"""Declared vendor projection arithmetic for plain SFT updates (owner, 2026-11-04)."""
import pytest
import torch
import torch.nn.functional as F
from natlang_neuralese.maple import fused_moe
from natlang_neuralese.model import native_weight_carrier as carrier


@pytest.fixture
def projection():
    previous = carrier.training_projection_arithmetic()
    yield carrier.set_training_projection_arithmetic
    carrier.set_training_projection_arithmetic(previous)


def test_vendor_policy_replaces_forced_invariant_only_inside_plain_updates(projection, monkeypatch):
    x = torch.randn(5, 4); weight = torch.randn(3, 4)
    marker = object()
    monkeypatch.setattr(fused_moe._InvariantLinear, 'apply', lambda *args: marker)
    projection('invariant')
    with carrier.plain_training_arithmetic():
        assert fused_moe.invariant_linear(x, weight, allow_plain_fastpath=False) is marker
    projection('vendor')
    with carrier.plain_training_arithmetic():
        assert torch.equal(fused_moe.invariant_linear(x, weight, allow_plain_fastpath=False), F.linear(x, weight))
        # Carrier anchors (RL policy/VJP scopes) and non-IEEE arithmetic keep the invariant kernel.
        assert fused_moe.invariant_linear(x, weight, allow_plain_fastpath=False,
                                          cotangent_anchors=(torch.zeros(()),)) is marker
        assert fused_moe.invariant_linear(x, weight, precision='bf16', allow_plain_fastpath=False) is marker
        with torch.no_grad():
            assert fused_moe.invariant_linear(x, weight, allow_plain_fastpath=False) is marker
    assert fused_moe.invariant_linear(x, weight, allow_plain_fastpath=False) is marker
    with pytest.raises(ValueError, match='invariant or vendor'):
        projection('fast')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_vendor_linear_matches_autograd_linear_and_reuses_aliased_primals(device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(70, 48, generator=generator).to(device, dtype)
    x[40:50] = x[0:10]  # aliased prefix rows
    w = torch.randn(24, 48, generator=generator).to(device, dtype)
    seed = torch.randn(70, 24, generator=generator).to(device, dtype)
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    expected = F.linear(xa, wa); expected.backward(seed)
    xb, wb = x.clone().requires_grad_(), w.clone().requires_grad_()
    actual = fused_moe._VendorLinear.apply(xb, wb); actual.backward(seed)
    assert torch.equal(actual, expected) and torch.equal(xb.grad, xa.grad) and torch.equal(wb.grad, wa.grad)
    calls = []
    from natlang_neuralese.model import native_gradient_partitions as partitions
    def reused(value, active, operation):
        unique = torch.cat([torch.arange(0, 40), torch.arange(50, 70)]).to(value.device)
        inverse = torch.cat([torch.arange(0, 40), torch.arange(0, 10), torch.arange(40, 60)]).to(value.device)
        calls.append(len(unique))
        return operation(value.index_select(0, unique)).index_select(0, inverse)
    monkeypatch.setattr(partitions, 'reused_linear_primal', reused)
    xc, wc = x.clone().requires_grad_(), w.clone().requires_grad_()
    shared = fused_moe._VendorLinear.apply(xc, wc); shared.backward(seed)
    assert calls == [60]
    # Every original row occurrence keeps its own input gradient; dW sums all occurrences.
    assert torch.equal(xc.grad, xa.grad) and torch.equal(wc.grad, wa.grad)
    if device == 'cpu':
        assert torch.equal(shared, expected)


def test_precision_binding_records_vendor_only(projection, monkeypatch):
    from natlang_neuralese.model.native_model import configure_native_precision
    monkeypatch.delenv('NATLANG_PROJECTION_ARITHMETIC', raising=False)
    assert 'training_update_projection' not in configure_native_precision()
    assert carrier.training_projection_arithmetic() == 'invariant'
    monkeypatch.setenv('NATLANG_PROJECTION_ARITHMETIC', 'vendor')
    assert configure_native_precision()['training_update_projection'] == 'vendor-cublas-bf16-fp32-accumulate'
    assert carrier.training_projection_arithmetic() == 'vendor'
    monkeypatch.setenv('NATLANG_PROJECTION_ARITHMETIC', 'other')
    with pytest.raises(ValueError, match='invariant or vendor'):
        configure_native_precision()


def test_vendor_self_check_passes_on_device_and_rejects_wrong_rounding(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    weights = [torch.randn(96, 2048, device='cuda').to(torch.bfloat16), torch.randn(96, 2048, device='cuda').to(torch.bfloat16)]
    receipt = fused_moe.vendor_projection_self_check(weights, rows=(1, 64, 300), samples=16)
    assert receipt['passed'] and len(receipt['checks']) == 3  # one distinct shape, three row counts
    monkeypatch.setattr(fused_moe, '_vendor_adjoints',
                        lambda x, w, grad, a, b: ((grad.float() @ w.float()).bfloat16() * 1.01, grad.t().mm(x)))
    with pytest.raises(RuntimeError, match='FP32-accumulation bound'):
        fused_moe.vendor_projection_self_check(weights, rows=(64,), samples=16)
