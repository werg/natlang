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


def concatenated_norm_reference(original, x):
    import torch.nn.functional as F
    rows = x.reshape(-1, x.shape[-1])
    pieces = []
    for start in range(0, len(rows), 128):
        part = rows[start:start + 128]
        count = len(part)
        if count < 128:
            part = F.pad(part, (0, 0, 0, 128 - count))
        pieces.append(original(part)[:count])
    return torch.cat(pieces).reshape(x.shape)


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


@pytest.mark.parametrize('rows,selected', [(128, False), (129, True), (511, True), (1024, True), (1025, False)])
def test_batched_lfm3_adjoint_dispatch_and_exact_gradient(monkeypatch, rows, selected):
    from natlang_neuralese.model import native_parameter_precision as precision
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda _: True))
    torch.manual_seed(rows)
    layer = Lfm3RMSNorm()
    x = torch.randn(rows, 8, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(x) + torch.randn_like(x)
    calls = []
    original = precision._batched_lfm3_norm_adjoint
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(precision, '_batched_lfm3_norm_adjoint', counted)
    output = precision._RowwiseNormAdjoint.apply(x, layer.weight, layer.forward)
    dx, dw = torch.autograd.grad(output, (x, layer.weight), g)
    assert bool(calls) is selected
    reference_input = x.detach().clone().requires_grad_(True)
    reference_output = precision.rowwise_norm_forward(layer.forward, reference_input)
    ref_dx, ref_dw = torch.autograd.grad(reference_output, (reference_input, layer.weight), g)
    torch.testing.assert_close(dx, ref_dx, rtol=0, atol=0)
    torch.testing.assert_close(dw, ref_dw, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA rowwise normalization')
@pytest.mark.parametrize('weight_dtype', [torch.bfloat16, torch.float32])
def test_cuda_rowwise_norm_bounds_output_peak_and_preserves_exact_adjoints(weight_dtype):
    from natlang_neuralese.model.native_parameter_precision import _RowwiseNormAdjoint
    torch.manual_seed(17)
    layer = Lfm3RMSNorm().to(device='cuda', dtype=weight_dtype)
    x = torch.randn(1, 8193, 8, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    incoming = torch.randn(x.shape, device='cuda', dtype=weight_dtype)
    actual = _RowwiseNormAdjoint.apply(x, layer.weight, layer.forward)
    dx, dw = torch.autograd.grad(actual, (x, layer.weight), incoming)
    reference = concatenated_norm_reference(layer.forward, x)
    ref_dx, ref_dw = torch.autograd.grad(reference, (x, layer.weight), incoming)
    for a, b in ((actual, reference), (dx, ref_dx), (dw, ref_dw)):
        assert torch.equal(a, b)

    # A large output with small norm chunks isolates the removed cat copy.
    large = torch.randn(8192, 256, device='cuda', dtype=torch.bfloat16)
    layer.weight = nn.Parameter(torch.randn(256, device='cuda', dtype=weight_dtype))
    with torch.no_grad():
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        expected = concatenated_norm_reference(layer.forward, large)
        old_peak = torch.cuda.max_memory_allocated() - before
        expected_cpu = expected.cpu()
        output_bytes = expected.numel() * expected.element_size()
        del expected
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = rowwise_norm_forward(layer.forward, large)
        new_peak = torch.cuda.max_memory_allocated() - before
        assert torch.equal(result.cpu(), expected_cpu)
        assert new_peak <= old_peak - output_bytes // 2


def test_preallocated_norm_input_adjoint_retains_concatenated_layout(monkeypatch):
    from natlang_neuralese.model.native_parameter_precision import _RowwiseNormAdjoint
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda _: True))
    layer = Lfm3RMSNorm()
    x = torch.randn(8, 1025, dtype=torch.bfloat16).t().detach().requires_grad_(True)
    assert not x.is_contiguous()
    incoming = torch.randn_like(x)
    output = _RowwiseNormAdjoint.apply(x, layer.weight, layer.forward)
    dx, dw = torch.autograd.grad(output, (x, layer.weight), incoming)
    expected = concatenated_norm_reference(layer.forward, x)
    ref_dx, ref_dw = torch.autograd.grad(expected, (x, layer.weight), incoming)
    assert dx.is_contiguous()
    for a, b in ((output, expected), (dx, ref_dx), (dw, ref_dw)):
        assert torch.equal(a, b)
