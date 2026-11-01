import torch
import pytest
from torch import nn

from natlang_neuralese.model.native_parameter_precision import configure_norm_execution, rowwise_norm_forward


class Lfm3RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, dtype=torch.bfloat16))
        self.variance_epsilon = 1e-5

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(x.dtype)


@pytest.mark.parametrize("rows", [14, 128, 129, 270, 511])
def test_rowwise_norm_uses_single_adjoint_and_preserves_values_and_gradients(monkeypatch, rows):
    # Exercise the CUDA dispatch geometry on CPU without allocating a second GPU model.
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda _: True))
    torch.manual_seed(10)
    layer = Lfm3RMSNorm()
    original = layer.forward
    class Port(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'model_type': 'lfm3'})()
            self.norm = layer
    model = type('Model', (), {})()
    model.port = Port()
    configure_norm_execution(model)
    x = torch.randn(1, rows, 8, dtype=torch.bfloat16, requires_grad=True)
    incoming = torch.randn_like(x)
    actual = layer(x)
    assert type(actual.grad_fn).__name__ == '_RowwiseNormAdjointBackward'
    dx, dw = torch.autograd.grad(actual, (x, layer.weight), incoming)
    y = x.detach().clone().requires_grad_(True)
    expected = rowwise_norm_forward(original, y)
    ref_dx, ref_dw = torch.autograd.grad(expected, (y, layer.weight), incoming)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(dx.float(), ref_dx.float(), rtol=0, atol=0)
    torch.testing.assert_close(dw.float(), ref_dw.float(), rtol=0, atol=0)


def test_rowwise_norm_adjoint_preserves_higher_order(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda _: True))
    layer = Lfm3RMSNorm()
    original = layer.forward
    from natlang_neuralese.model.native_parameter_precision import _RowwiseNormAdjoint
    x = torch.randn(1, 9, 8, dtype=torch.bfloat16, requires_grad=True)
    incoming = torch.randn_like(x)
    y = _RowwiseNormAdjoint.apply(x, layer.weight, original)
    dx, = torch.autograd.grad(y, x, incoming, create_graph=True)
    actual, = torch.autograd.grad(dx.float().sum(), x)
    ref = x.detach().clone().requires_grad_(True)
    out = rowwise_norm_forward(original, ref)
    grad, = torch.autograd.grad(out, ref, incoming, create_graph=True)
    expected, = torch.autograd.grad(grad.float().sum(), ref)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("train_input,train_weight", [(True, False), (False, True)])
def test_rowwise_norm_higher_order_with_frozen_input_or_weight(monkeypatch, train_input, train_weight):
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda _: True))
    from natlang_neuralese.model.native_parameter_precision import _RowwiseNormAdjoint
    layer = Lfm3RMSNorm()
    layer.weight.requires_grad_(train_weight)
    x = torch.randn(1, 9, 8, dtype=torch.bfloat16, requires_grad=train_input)
    incoming = torch.randn_like(x)
    y = _RowwiseNormAdjoint.apply(x, layer.weight, layer.forward)
    target = x if train_input else layer.weight
    actual, = torch.autograd.grad(y, target, incoming, create_graph=True)
    out = rowwise_norm_forward(layer.forward, x)
    expected, = torch.autograd.grad(out, target, incoming, create_graph=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if train_input:
        actual2, = torch.autograd.grad(actual.float().sum(), x)
        expected2, = torch.autograd.grad(expected.float().sum(), x)
        torch.testing.assert_close(actual2, expected2, rtol=0, atol=0)
