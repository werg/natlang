from types import SimpleNamespace

import pytest
import torch


def test_adjoint_specialization_requires_explicit_rounding(monkeypatch):
    from natlang_neuralese.maple import fused_moe as fused
    calls = []
    def adjoint(grad, weight, *, precision):
        calls.append((precision, weight.clone()))
        return torch.zeros_like(grad)
    monkeypatch.setattr(fused, 'invariant_input_gradient', adjoint)
    weight = torch.tensor([[1.001]], dtype=torch.float32)
    def run(mode, w, grad):
        ctx = SimpleNamespace(saved_tensors=(grad, w), precision=mode,
            needs_input_grad=(True, False), anchors=(), carrier=False, scope=None)
        fused._InvariantLinear.backward(ctx, grad)
    run('bf16', weight, torch.ones(1, 1))
    assert calls[-1][0] == 'bf16x3_exact_weight'
    assert torch.equal(calls[-1][1], weight.bfloat16().float())
    run('ieee', weight, torch.ones(1, 1))
    assert calls[-1][0] == 'ieee' and torch.equal(calls[-1][1], weight)
    run('bf16', weight.bfloat16(), torch.ones(1, 1, dtype=torch.bfloat16))
    assert calls[-1][0] == 'ieee'
    run('bf16', weight, torch.ones(1, 1, dtype=torch.bfloat16))
    assert calls[-1][0] == 'ieee'


def test_reference_arithmetic_remains_higher_order_differentiable(monkeypatch):
    from natlang_neuralese.maple import fused_moe as fused
    from natlang_neuralese.common.native_higher_order import reference_arithmetic
    monkeypatch.setattr(fused._InvariantLinear, 'apply', lambda *a: pytest.fail('custom kernel used'))
    x = torch.randn(3, 4, requires_grad=True)
    w = torch.randn(5, 4, requires_grad=True)
    with reference_arithmetic():
        y = fused.invariant_linear(x, w, precision='bf16')
        dx, = torch.autograd.grad(y.square().sum(), x, create_graph=True)
        second, = torch.autograd.grad(dx.square().sum(), w)
    assert torch.isfinite(second).all() and second.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA invariant arithmetic')
def test_bf16_weight_adjoint_geometry_and_complete_graph(monkeypatch):
    from natlang_neuralese.maple import fused_moe as fused
    from natlang_neuralese.model.native_gradient_partitions import GradientPartitions, gradient_partitions
    torch.manual_seed(912)
    weight = (torch.randn(193, 129, device='cuda') * .05).bfloat16().float()
    grad = torch.randn(137, 193, device='cuda') * .02
    actual = fused.invariant_input_gradient(grad, weight, precision='bf16x3_exact_weight')
    split = torch.cat([fused.invariant_input_gradient(part, weight, precision='bf16x3_exact_weight')
                       for part in (grad[:79], grad[79:])])
    assert torch.equal(actual, split)
    ieee = fused.invariant_input_gradient(grad, weight, precision='ieee')
    torch.testing.assert_close(actual, ieee, atol=1e-5, rtol=0)

    x = torch.randn(137, 64, device='cuda') * .1
    w1 = torch.randn(96, 64, device='cuda') * .1
    w2 = torch.randn(48, 96, device='cuda') * .1
    def graph():
        a, b, c = (v.detach().clone().requires_grad_() for v in (x, w1, w2))
        with gradient_partitions(GradientPartitions((79, 58))):
            y = fused.invariant_linear(torch.nn.functional.silu(fused.invariant_linear(a, b, precision='bf16')),
                                       c, precision='bf16')
        grads = torch.autograd.grad(y.square().mean(), (a, b, c))
        return y.detach(), grads
    candidate, candidate_grads = graph()
    original = fused.invariant_input_gradient
    def baseline(g, w, *, precision='ieee'):
        return original(g, w, precision='ieee' if precision == 'bf16x3_exact_weight' else precision)
    monkeypatch.setattr(fused, 'invariant_input_gradient', baseline)
    reference, reference_grads = graph()
    torch.testing.assert_close(candidate, reference, atol=1e-4, rtol=0)
    for got, want in zip(candidate_grads, reference_grads):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0)
