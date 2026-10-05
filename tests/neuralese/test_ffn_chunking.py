from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.model.lfm2_port import feed_forward_residual


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('chunk', [1, 5, 32])
def test_ffn_chunks_preserve_output_input_and_weight_gradients(device, chunk):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(0)
    layer = SimpleNamespace(
        ffn_norm=torch.nn.LayerNorm(8).to(device=device, dtype=torch.float64),
        feed_forward=torch.nn.Sequential(torch.nn.Linear(8, 24), torch.nn.SiLU(),
                                        torch.nn.Linear(24, 8)).to(device=device, dtype=torch.float64))
    h = torch.randn(2, 13, 8, device=device, dtype=torch.float64, requires_grad=True)
    params = [h, *layer.ffn_norm.parameters(), *layer.feed_forward.parameters()]
    reference = feed_forward_residual(layer, h)
    actual = feed_forward_residual(layer, h, chunk)
    torch.testing.assert_close(actual, reference)
    expected_grads = torch.autograd.grad(reference.square().sum(), params)
    actual_grads = torch.autograd.grad(actual.square().sum(), params)
    for actual_grad, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected)
