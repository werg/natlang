import pytest
import torch

from natlang_neuralese.eval.native_gradients import _backward_gradients_to_cpu


def test_streamed_leaf_and_input_gradients_match_autograd_grad_and_preserve_existing_grads():
    torch.manual_seed(31)
    first = torch.nn.Parameter(torch.randn(5, 3, dtype=torch.float64))
    second = torch.nn.Parameter(torch.randn(5, dtype=torch.float64))
    unused = torch.nn.Parameter(torch.randn(2, dtype=torch.float64))
    input_embeddings = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    old_grad = torch.randn_like(first)
    first.grad = old_grad
    old_value = old_grad.clone()
    old_version = old_grad._version
    old_input_grad = torch.randn_like(input_embeddings)
    input_embeddings.grad = old_input_grad
    old_input_value = old_input_grad.clone()
    old_input_version = old_input_grad._version

    reference = (input_embeddings @ first.t() + second).square().sum()
    expected = torch.autograd.grad(reference, (input_embeddings, first, second, unused), allow_unused=True)

    actual_loss = (input_embeddings @ first.t() + second).square().sum()
    streamed = _backward_gradients_to_cpu(actual_loss, ('first', 'second', 'unused'),
                                           (first, second, unused), input_embeddings)
    assert torch.equal(streamed['input_embeddings'], expected[0])
    assert torch.equal(streamed['first'], expected[1])
    assert torch.equal(streamed['second'], expected[2])
    assert streamed['unused'] is None
    assert first.grad is old_grad and first.grad._version == old_version
    assert torch.equal(first.grad, old_value)
    assert input_embeddings.grad is old_input_grad
    assert input_embeddings.grad._version == old_input_version
    assert torch.equal(input_embeddings.grad, old_input_value)
    assert second.grad is None and unused.grad is None


def test_streamed_gradients_fail_closed_for_nonleaf_parameter_targets():
    leaf = torch.nn.Parameter(torch.randn(4))
    nonleaf = leaf * 2
    with pytest.raises(ValueError, match='trainable leaves'):
        _backward_gradients_to_cpu(nonleaf.sum(), ('nonleaf',), (nonleaf,))


class _BackwardFailure(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise RuntimeError('injected backward failure')


def test_streamed_backward_exception_restores_all_prior_grad_objects():
    parameter = torch.nn.Parameter(torch.randn(4))
    old = torch.randn_like(parameter)
    parameter.grad = old
    with pytest.raises(RuntimeError, match='injected backward failure'):
        _backward_gradients_to_cpu(_BackwardFailure.apply(parameter).sum(), ('p',), (parameter,))
    assert parameter.grad is old
    assert torch.equal(parameter.grad, old)


def test_streamed_nonleaf_input_preserves_existing_grad_object_and_value():
    torch.manual_seed(77)
    parameter = torch.nn.Parameter(torch.randn(4, dtype=torch.float64))
    leaf = torch.randn(4, dtype=torch.float64, requires_grad=True)
    inputs = (leaf * 2).requires_grad_()
    inputs.retain_grad()
    old = torch.randn_like(inputs)
    expected_loss = (inputs * parameter).square().sum()
    expected = torch.autograd.grad(expected_loss, (inputs, parameter))
    inputs.grad = old
    old_value, old_version = old.clone(), old._version
    loss = (inputs * parameter).square().sum()
    actual = _backward_gradients_to_cpu(loss, ('weight',), (parameter,), inputs)
    assert torch.equal(actual['input_embeddings'], expected[0])
    assert torch.equal(actual['weight'], expected[1])
    assert inputs.grad is old and inputs.grad._version == old_version
    assert torch.equal(inputs.grad, old_value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA gradient stream check')
def test_cuda_streamed_gradient_values_are_cpu_resident_and_match_autograd_grad():
    torch.manual_seed(47)
    first = torch.nn.Parameter(torch.randn(7, 4, device='cuda', dtype=torch.bfloat16))
    second = torch.nn.Parameter(torch.randn(7, device='cuda', dtype=torch.bfloat16))
    inputs = torch.randn(5, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    loss = (inputs @ first.t() + second).float().square().sum()
    expected = torch.autograd.grad(loss, (inputs, first, second))
    loss = (inputs @ first.t() + second).float().square().sum()
    actual = _backward_gradients_to_cpu(loss, ('first', 'second'), (first, second), inputs)
    for name, reference in zip(('input_embeddings', 'first', 'second'), expected):
        assert actual[name].device.type == 'cpu'
        assert torch.equal(actual[name], reference.cpu())
    assert first.grad is None and second.grad is None
    assert inputs.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA gradient residency measurement')
def test_cuda_streaming_reduces_complete_parameter_gradient_residency():
    torch.manual_seed(59)
    parameters = tuple(torch.nn.Parameter(torch.randn(512, 512, device='cuda')) for _ in range(24))
    names = tuple(f'p{index}' for index in range(len(parameters)))
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    reference_loss = sum(parameter.square().sum() for parameter in parameters)
    reference = torch.autograd.grad(reference_loss, parameters)
    torch.cuda.synchronize()
    reference_peak = torch.cuda.max_memory_allocated() - base
    assert len(reference) == len(parameters) and all(value.device.type == 'cuda' for value in reference)
    del reference, reference_loss
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    streamed_loss = sum(parameter.square().sum() for parameter in parameters)
    streamed = _backward_gradients_to_cpu(streamed_loss, names, parameters)
    torch.cuda.synchronize()
    streamed_peak = torch.cuda.max_memory_allocated() - base
    assert all(value.device.type == 'cpu' for value in streamed.values())
    assert all(parameter.grad is None for parameter in parameters)
    assert streamed_peak < reference_peak
