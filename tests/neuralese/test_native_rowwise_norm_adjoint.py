import torch
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


def test_rowwise_norm_uses_single_adjoint_and_preserves_values_and_gradients(monkeypatch):
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
    x = torch.randn(1, 270, 8, dtype=torch.bfloat16, requires_grad=True)
    incoming = torch.randn_like(x)
    actual = layer(x)
    assert type(actual.grad_fn).__name__ == '_TorchStoredRMSNormBackward'
    dx, dw = torch.autograd.grad(actual, (x, layer.weight), incoming)
    y = x.detach().clone().requires_grad_(True)
    expected = rowwise_norm_forward(original, y)
    ref_dx, ref_dw = torch.autograd.grad(expected, (y, layer.weight), incoming)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(dx.float(), ref_dx.float(), rtol=0.02, atol=0.02)
    torch.testing.assert_close(dw.float(), ref_dw.float(), rtol=0.02, atol=0.1)
