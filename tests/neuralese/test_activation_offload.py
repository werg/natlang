import pytest
import torch

from natlang_neuralese.train.memory import offload_attention_tensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_saved_activation_offload_preserves_gradients_under_allocator_reuse():
    torch.manual_seed(0)
    x = torch.randn(1024, 256, device='cuda', requires_grad=True)
    def loss():
        value = x.square().mean()
        for i in range(24):
            # Temporary tensors are released during the forward; CUDA allocator
            # addresses can recur even though the saved values are different.
            part = torch.sin(x * (i + 1))
            value = value + part.square().mean()
        return value
    expected = loss()
    grad = torch.autograd.grad(expected, x)[0]
    with offload_attention_tensors(16*2**20, activations=True) as stats:
        actual = loss()
    got = torch.autograd.grad(actual, x)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(got, grad, atol=1e-7, rtol=1e-6)
    assert 0 < stats['offloaded_bytes'] <= 16*2**20


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_persistent_weights_are_not_offloaded():
    x = torch.randn(1024, 256, device='cuda', requires_grad=True)
    with offload_attention_tensors(4*2**20, activations=True, persistent_tensors=[x]) as stats:
        loss = x.square().mean()
    loss.backward()
    assert stats['offloaded_bytes'] == 0
