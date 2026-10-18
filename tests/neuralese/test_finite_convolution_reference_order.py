"""Minimal actual carrier witness for cached per-token finite-convolution fan-in."""
import torch
from torch import nn

from natlang_neuralese.model.native_gradient_partitions import GradientPartitions, gradient_partitions
from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope, stored_finite_convolution
from natlang_neuralese.model import ordered_finite_convolution


def _run(tmp_path, *, cached, prior=False, backend='cpu_file', device='cpu'):
    tmp_path.mkdir(parents=True)
    weight = nn.Parameter(torch.tensor([[[1.0]]], dtype=torch.bfloat16, device=device))
    histories = torch.ones((1, 3, 1, 1), dtype=torch.bfloat16, device=device)
    gates = torch.ones((1, 3, 1), dtype=torch.bfloat16, device=device)
    upstream = torch.tensor([1.704692840576172e-5, -175.0, 656.0], dtype=torch.bfloat16,
                            device=device).reshape(1, 3, 1)
    with ParameterCotangentScope([('conv.weight', weight)], directory=tmp_path,
                                 accumulator_backend=backend) as scope:
        if prior:
            prior_history = torch.ones((1, 1, 1, 1), dtype=torch.bfloat16, device=device)
            prior_gate = torch.ones((1, 1, 1), dtype=torch.bfloat16, device=device)
            prior_output = stored_finite_convolution(prior_history, weight, None, prior_gate)
            prior_output.backward(torch.tensor([[[-2.0e-5]]], dtype=torch.bfloat16, device=device))
            scope.new_backward()
        if cached:
            pieces = [stored_finite_convolution(histories[:, i:i + 1], weight, None,
                                                gates[:, i:i + 1]) for i in range(3)]
            output = torch.cat(pieces, dim=1)
        else:
            # This is the flat layer-major operation; its single custom backward
            # reduces all physical tokens in one call.
            with gradient_partitions(GradientPartitions((3,))):
                output = stored_finite_convolution(histories, weight, None, gates)
        output.backward(upstream)
        scope.finish()
    return output.detach().cpu(), weight.grad.detach().clone().cpu()


def test_flat_convolution_uses_cached_reverse_token_fanin_order(tmp_path):
    flat_output, flat_grad = _run(tmp_path / 'flat', cached=False)
    cached_output, cached_grad = _run(tmp_path / 'cached', cached=True)
    assert torch.equal(flat_output, cached_output)
    assert torch.equal(flat_grad, cached_grad)
    assert flat_grad.item() == 482.0


def test_cached_finite_convolution_folds_microbatch_prior_before_token_rows(tmp_path):
    flat_output, flat_grad = _run(tmp_path / 'flat-prior', cached=False, prior=True)
    cached_output, cached_grad = _run(tmp_path / 'cached-prior', cached=True, prior=True)
    assert torch.equal(flat_output, cached_output)
    assert torch.equal(flat_grad, cached_grad)
    assert flat_grad.item() == 482.0


def _run_bias(tmp_path, *, cached, backend='cpu_file', device='cpu'):
    tmp_path.mkdir()
    weight = nn.Parameter(torch.tensor([[[1.0]]], dtype=torch.bfloat16, device=device))
    bias = nn.Parameter(torch.tensor([0.0], dtype=torch.bfloat16, device=device))
    histories = torch.ones((1, 3, 1, 1), dtype=torch.bfloat16, device=device)
    gates = torch.ones((1, 3, 1), dtype=torch.bfloat16, device=device)
    upstream = torch.tensor([1.704692840576172e-5, -175.0, 656.0], dtype=torch.bfloat16,
                            device=device).reshape(1, 3, 1)
    with ParameterCotangentScope([('conv.weight', weight), ('conv.bias', bias)], directory=tmp_path,
                                 accumulator_backend=backend) as scope:
        if cached:
            output = torch.cat([stored_finite_convolution(histories[:, i:i + 1], weight, bias,
                                                           gates[:, i:i + 1]) for i in range(3)], 1)
        else:
            with gradient_partitions(GradientPartitions((3,))):
                output = stored_finite_convolution(histories, weight, bias, gates)
        output.backward(upstream)
        scope.finish()
    return (output.detach().cpu(), weight.grad.detach().clone().cpu(),
            bias.grad.detach().clone().cpu())


def _run_shaped(tmp_path, *, cached, backend, device):
    import pytest
    pytest.importorskip('triton')
    tmp_path.mkdir()
    generator = torch.Generator(device='cpu').manual_seed(104)
    weight = nn.Parameter(torch.randn((7, 1, 5), generator=generator, dtype=torch.float32)
                          .to(torch.bfloat16).to(device))
    bias = nn.Parameter(torch.randn((7,), generator=generator, dtype=torch.float32)
                        .to(torch.bfloat16).to(device))
    histories = torch.randn((1, 11, 5, 7), generator=generator, dtype=torch.float32)
    histories = histories.to(torch.bfloat16).to(device)
    gates = torch.randn((1, 11, 7), generator=generator, dtype=torch.float32)
    gates = gates.to(torch.bfloat16).to(device)
    upstream = torch.randn((1, 11, 7), generator=generator, dtype=torch.float32)
    upstream = upstream.to(torch.bfloat16).to(device)
    with ParameterCotangentScope([('conv.weight', weight), ('conv.bias', bias)], directory=tmp_path,
                                 accumulator_backend=backend) as scope:
        if cached:
            output = torch.cat([stored_finite_convolution(histories[:, i:i + 1], weight, bias,
                                                           gates[:, i:i + 1]) for i in range(11)], 1)
        else:
            with gradient_partitions(GradientPartitions((4, 7))):
                output = stored_finite_convolution(histories, weight, bias, gates)
        output.backward(upstream)
        scope.finish()
    return (output.detach().cpu(), weight.grad.detach().cpu().clone(),
            bias.grad.detach().cpu().clone())


def test_flat_and_cached_bias_follow_the_same_reverse_token_order(tmp_path):
    flat = _run_bias(tmp_path / 'bias-flat', cached=False)
    cached = _run_bias(tmp_path / 'bias-cached', cached=True)
    assert torch.equal(flat[0], cached[0])
    assert torch.equal(flat[1], cached[1])
    assert torch.equal(flat[2], cached[2])


def test_device_fp32_backend_matches_cached_reverse_order_and_prior(tmp_path):
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('CUDA is unavailable')
    for prior in (False, True):
        flat_output, flat_grad = _run(tmp_path / f'gpu-flat-{prior}', cached=False, prior=prior,
                                      backend='device_fp32', device='cuda')
        cached_output, cached_grad = _run(tmp_path / f'gpu-cached-{prior}', cached=True, prior=prior,
                                          backend='device_fp32', device='cuda')
        assert torch.equal(flat_output, cached_output)
        assert torch.equal(flat_grad, cached_grad)
    flat = _run_bias(tmp_path / 'gpu-bias-flat', cached=False,
                     backend='device_fp32', device='cuda')
    cached = _run_bias(tmp_path / 'gpu-bias-cached', cached=True,
                       backend='device_fp32', device='cuda')
    assert torch.equal(flat[0], cached[0])
    assert torch.equal(flat[1], cached[1])
    assert torch.equal(flat[2], cached[2])
    flat = _run_shaped(tmp_path / 'gpu-shape-flat', cached=False,
                       backend='device_fp32', device='cuda')
    cached = _run_shaped(tmp_path / 'gpu-shape-cached', cached=True,
                         backend='device_fp32', device='cuda')
    assert torch.equal(flat[0], cached[0])
    assert torch.equal(flat[1], cached[1])
    assert torch.equal(flat[2], cached[2])


def test_cpu_file_backend_matches_cached_reverse_order_for_multiple_channels(tmp_path):
    flat = _run_shaped(tmp_path / 'cpu-shape-flat', cached=False,
                       backend='cpu_file', device='cpu')
    cached = _run_shaped(tmp_path / 'cpu-shape-cached', cached=True,
                         backend='cpu_file', device='cpu')
    assert torch.equal(flat[0], cached[0])
    assert torch.equal(flat[1], cached[1])
    assert torch.equal(flat[2], cached[2])


def test_cpu_file_batches_keep_existing_accumulator_in_reverse_order(monkeypatch):
    generator = torch.Generator().manual_seed(991)
    q, channels, width = 19, 7, 5
    histories = torch.randn((1, q, width, channels), generator=generator, dtype=torch.float32)
    gradients = torch.randn((1, q, channels), generator=generator, dtype=torch.float32)
    initial = torch.randn((channels, 1, width), generator=generator, dtype=torch.float32)
    expected = initial.clone()
    for row in range(q - 1, -1, -1):
        contribution = (gradients[:, row].unsqueeze(1) * histories[:, row]).squeeze(0)
        expected.add_(contribution.permute(1, 0).unsqueeze(1))
    actual = initial.clone()
    monkeypatch.setattr(ordered_finite_convolution, '_CPU_PRODUCT_BATCH_BYTES', 700)
    ordered_finite_convolution.ordered_finite_convolution_accumulate(
        actual, histories, gradients, backend='cpu_file')
    assert torch.equal(actual, expected)


def test_ordered_accumulation_preserves_contiguous_and_strided_inputs():
    generator = torch.Generator().manual_seed(992)
    q, channels, width = 9, 6, 3
    histories_storage = torch.randn((1, q, width, channels * 2), generator=generator)
    histories = histories_storage[..., ::2]
    gradient_storage = torch.randn((1, q, channels * 2), generator=generator)
    gradients = gradient_storage[..., ::2]
    history_before, gradients_before = histories.clone(), gradients.clone()
    weight_target = torch.randn((channels, 1, width), generator=generator)
    weight_initial = weight_target.clone()
    weight_expected = weight_initial.clone()
    for row in range(q - 1, -1, -1):
        contribution = (gradients[:, row].unsqueeze(1) * histories[:, row]).squeeze(0)
        weight_expected.add_(contribution.permute(1, 0).unsqueeze(1))
    ordered_finite_convolution.ordered_finite_convolution_accumulate(
        weight_target, histories, gradients, backend='cpu_file')
    assert torch.equal(weight_target, weight_expected)

    bias_target = torch.randn((channels,), generator=generator)
    bias_initial = bias_target.clone()
    bias_expected = bias_initial.clone()
    for row in range(q - 1, -1, -1):
        bias_expected.add_(gradients[0, row])
    ordered_finite_convolution.ordered_finite_convolution_accumulate(
        bias_target, None, gradients, backend='cpu_file', bias=True)
    assert torch.equal(bias_target, bias_expected)
    assert torch.equal(histories, history_before)
    assert torch.equal(gradients, gradients_before)
