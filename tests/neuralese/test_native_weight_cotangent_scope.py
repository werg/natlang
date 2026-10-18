"""Real low-precision Linear paths keep shared parameter cotangents FP32."""
import pytest
import torch
from torch import nn
from pathlib import Path

from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope


def _case(dtype=torch.bfloat16):
    torch.manual_seed(41)
    layer = nn.Linear(7, 5, bias=False).to(dtype=dtype)
    first = torch.randn(4, 7, dtype=dtype, requires_grad=True)
    second = torch.randn(3, 7, dtype=dtype, requires_grad=True)
    g1 = torch.randn(4, 5, dtype=dtype)
    g2 = torch.randn(3, 5, dtype=dtype)
    return layer, first, second, g1, g2


def test_split_backward_uses_one_fp32_cotangent_sum_and_preserves_forward_dx(tmp_path):
    layer, x1, x2, g1, g2 = _case()
    weight = layer.weight
    stored1, stored2 = weight.detach().clone(), weight.detach().clone()
    # Oracle uses the same stored BF16 operands but combines dW before casting.
    y1 = torch.nn.functional.linear(x1, stored1)
    y2 = torch.nn.functional.linear(x2, stored2)
    expected = (g1.float().T @ x1.detach().float()) + (g2.float().T @ x2.detach().float())
    expected_dx1 = g1 @ stored1
    expected_dx2 = g2 @ stored2
    with ParameterCotangentScope.for_model(layer, directory=tmp_path) as scope:
        scratch = Path(scope.directory)
        actual1 = layer(x1)
        assert torch.equal(actual1, y1)
        actual1.backward(g1)
        scope.new_backward()
        actual2 = layer(x2)
        assert torch.equal(actual2, y2)
        actual2.backward(g2)
        scope.finish()
    assert not scratch.exists()
    assert not scope.entries
    torch.testing.assert_close(weight.grad.float(), expected.to(weight.dtype).float(), rtol=0, atol=0)
    torch.testing.assert_close(x1.grad, expected_dx1, rtol=0, atol=0)
    torch.testing.assert_close(x2.grad, expected_dx2, rtol=0, atol=0)


def test_vjp_returns_fp32_fan_in_through_original_parameter_without_mutating_grad(tmp_path):
    layer, x1, x2, g1, g2 = _case()
    weight = layer.weight
    weight.grad = torch.full_like(weight, 3)
    before = weight.grad
    with ParameterCotangentScope.for_model(layer, mode='vjp', directory=tmp_path) as scope:
        loss = (layer(x1) * g1).sum() + (layer(x2) * g2).sum()
        actual, = torch.autograd.grad(loss, (weight,))
        scope.finish()
    expected = g1.float().T @ x1.detach().float() + g2.float().T @ x2.detach().float()
    torch.testing.assert_close(actual.float(), expected.to(weight.dtype).float(), rtol=0, atol=0)
    assert weight.grad is before and torch.equal(weight.grad, torch.full_like(weight, 3))


def test_unwrapped_low_precision_parameter_use_fails_closed(tmp_path):
    layer, x1, _, _, _ = _case()
    with ParameterCotangentScope.for_model(layer, directory=tmp_path) as scope:
        scratch = Path(scope.directory)
        # Deliberately bypass the wrapped module path.
        loss = (layer.weight * 2).sum() + layer(x1).float().sum()
        loss.backward()
        with pytest.raises(RuntimeError, match='bypassed cotangent carrier'):
            scope.finish()
    assert not scratch.exists()
    assert not scope.entries and not scope._auxiliary_entries and not scope._effective_routes


def test_vjp_rejects_direct_low_precision_gradient_without_any_carrier(tmp_path):
    from natlang_neuralese.model.native_weight_carrier import active_cotangent_scope

    layer, _, _, _, _ = _case()
    weight = layer.weight
    original_forward = layer.forward.__func__
    original_grad = torch.full_like(weight, 3)
    weight.grad = original_grad
    scratch = None
    with ParameterCotangentScope.for_model(layer, mode='vjp', directory=tmp_path) as scope:
        scratch = Path(scope.directory)
        with pytest.raises(RuntimeError, match='gradient arrived without a cotangent carrier'):
            torch.autograd.grad((weight * 2).sum(), (weight,))
    assert active_cotangent_scope() is None
    assert layer.forward.__func__ is original_forward
    assert weight.grad is original_grad and torch.equal(weight.grad, torch.full_like(weight, 3))
    assert scratch is not None and not scratch.exists()


def test_vjp_unused_low_precision_parameter_without_gradient_remains_allowed(tmp_path):
    class PartlyUsed(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(7, 5, bias=False).to(dtype=torch.bfloat16)
            self.unused = nn.Linear(7, 5, bias=False).to(dtype=torch.bfloat16)

        def forward(self, x):
            return self.used(x)

    model = PartlyUsed()
    x = torch.randn(2, 7, dtype=torch.bfloat16)
    with ParameterCotangentScope.for_model(model, mode='vjp', directory=tmp_path) as scope:
        used, unused = torch.autograd.grad(model(x).float().sum(),
                                           (model.used.weight, model.unused.weight),
                                           allow_unused=True)
        scope.finish()
    assert used is not None and unused is None


def test_embedding_split_lookups_accumulate_selected_fp32_rows(tmp_path):
    model = nn.Sequential(nn.Embedding(11, 7).to(dtype=torch.bfloat16))
    model[0].padding_idx = 0
    first = torch.tensor([[1, 4, 1, 0], [7, 4, 3, 1]])
    second = torch.tensor([[4, 4, 2]])
    g1 = torch.randn(2, 4, 7, dtype=torch.bfloat16)
    g2 = torch.randn(1, 3, 7, dtype=torch.bfloat16)
    weight = model[0].weight
    before = weight.detach().clone()
    expected = torch.zeros_like(weight, dtype=torch.float32)
    expected.index_add_(0, first.flatten(), g1.float().reshape(-1, 7))
    expected.index_add_(0, second.flatten(), g2.float().reshape(-1, 7))
    expected[0].zero_()
    with ParameterCotangentScope.for_model(model, directory=tmp_path) as scope:
        output = model(first)
        assert torch.equal(output, torch.nn.functional.embedding(first, before, padding_idx=0))
        output.backward(g1)
        scope.new_backward()
        model(second).backward(g2)
        scope.finish()
    torch.testing.assert_close(weight.grad.float(), expected.to(weight.dtype).float(), rtol=0, atol=0)
    previous = weight.grad
    with ParameterCotangentScope.for_model(model, mode='vjp', directory=tmp_path) as scope:
        loss = (model(first) * g1).sum()
        actual, = torch.autograd.grad(loss, (weight,))
        scope.finish()
    expected_vjp = torch.zeros_like(weight, dtype=torch.float32)
    expected_vjp.index_add_(0, first.flatten(), g1.float().reshape(-1, 7))
    expected_vjp[0].zero_()
    torch.testing.assert_close(actual.float(), expected_vjp.to(weight.dtype).float(), rtol=0, atol=0)
    assert weight.grad is previous


def test_embedding_frequency_scaling_and_padding_match_native_backward(tmp_path):
    module = nn.Embedding(9, 5, padding_idx=0, scale_grad_by_freq=True).to(dtype=torch.bfloat16)
    ids = torch.tensor([[2, 2, 0, 4], [2, 7, 4, 2]])
    upstream = torch.randn(2, 4, 5, dtype=torch.bfloat16)
    expected = torch.zeros_like(module.weight, dtype=torch.float32)
    expected.index_add_(0, ids.flatten(), upstream.float().reshape(-1, 5))
    counts = torch.bincount(ids.flatten(), minlength=module.num_embeddings)
    expected = expected / counts.clamp_min(1).float()[:, None]
    expected[0].zero_()
    with ParameterCotangentScope.for_model(module, directory=tmp_path) as scope:
        module(ids).backward(upstream)
        scope.finish()
    torch.testing.assert_close(module.weight.grad.float(), expected.to(module.weight.dtype).float(), rtol=0, atol=0)


def test_conv1d_split_invocations_keep_fp32_weight_fanin(tmp_path):
    torch.manual_seed(61)
    layer = nn.Conv1d(4, 5, 3, padding=1, bias=False).to(dtype=torch.bfloat16)
    left, right = (torch.randn(2, 4, 9, dtype=torch.bfloat16, requires_grad=True),
                   torch.randn(1, 4, 7, dtype=torch.bfloat16, requires_grad=True))
    gl, gr = torch.randn(2, 5, 9, dtype=torch.bfloat16), torch.randn(1, 5, 7, dtype=torch.bfloat16)
    weight = layer.weight.detach().clone()
    expected = (torch.nn.grad.conv1d_weight(left.float(), weight.shape, gl.float(), layer.stride,
                                            layer.padding, layer.dilation, layer.groups)
                + torch.nn.grad.conv1d_weight(right.float(), weight.shape, gr.float(), layer.stride,
                                              layer.padding, layer.dilation, layer.groups))
    left_ref = left.detach().requires_grad_(True)
    right_ref = right.detach().requires_grad_(True)
    expected_dx = (torch.autograd.grad(layer(left_ref), left_ref, gl)[0],
                   torch.autograd.grad(layer(right_ref), right_ref, gr)[0])
    with ParameterCotangentScope.for_model(layer, directory=tmp_path) as scope:
        out_left = layer(left)
        out_left.backward(gl)
        torch.testing.assert_close(left.grad, expected_dx[0], rtol=0, atol=0)
        scope.new_backward()
        out_right = layer(right)
        out_right.backward(gr)
        torch.testing.assert_close(right.grad, expected_dx[1], rtol=0, atol=0)
        scope.finish()
    torch.testing.assert_close(layer.weight.grad.float(), expected.to(weight.dtype).float(), rtol=0, atol=0)


def test_rmsnorm_split_invocations_keep_fp32_weight_fanin(tmp_path):
    from natlang_neuralese.maple.model import RMSNorm

    torch.manual_seed(73)
    layer = RMSNorm(7, 1e-5).to(dtype=torch.bfloat16)
    left, right = (torch.randn(3, 4, 7, dtype=torch.bfloat16, requires_grad=True),
                   torch.randn(2, 5, 7, dtype=torch.bfloat16, requires_grad=True))
    gl, gr = torch.randn_like(left), torch.randn_like(right)
    def normalized(x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + layer.eps)).to(x.dtype)
    expected = (gl.float() * normalized(left).float()).sum((0, 1)) + (gr.float() * normalized(right).float()).sum((0, 1))
    left_ref = left.detach().requires_grad_(True)
    right_ref = right.detach().requires_grad_(True)
    expected_dx = (torch.autograd.grad(layer(left_ref), left_ref, gl)[0],
                   torch.autograd.grad(layer(right_ref), right_ref, gr)[0])
    with ParameterCotangentScope.for_model(layer, directory=tmp_path) as scope:
        layer(left).backward(gl)
        torch.testing.assert_close(left.grad, expected_dx[0], rtol=0, atol=0)
        scope.new_backward()
        layer(right).backward(gr)
        torch.testing.assert_close(right.grad, expected_dx[1], rtol=0, atol=0)
        scope.finish()
    torch.testing.assert_close(layer.weight.grad.float(), expected.to(layer.weight.dtype).float(), rtol=0, atol=0)


def test_lfm_finite_history_split_invocations_keep_fp32_weight_fanin(tmp_path):
    """Exercise the actual layer-major LFM convolution reduction used by packed graphs."""
    from natlang_neuralese.model.native_weight_carrier import stored_finite_convolution

    torch.manual_seed(79)
    layer = nn.Conv1d(4, 4, 3, groups=4, bias=True).to(dtype=torch.bfloat16)
    weight, bias = layer.weight.detach().clone(), layer.bias.detach().clone()
    histories = [torch.randn(2, 5, 3, 4, dtype=torch.bfloat16, requires_grad=True),
                 torch.randn(1, 7, 3, 4, dtype=torch.bfloat16, requires_grad=True)]
    gates = [torch.randn(2, 5, 4, dtype=torch.bfloat16, requires_grad=True),
             torch.randn(1, 7, 4, dtype=torch.bfloat16, requires_grad=True)]
    output_gradients = [torch.randn_like(gates[0]), torch.randn_like(gates[1])]
    expected_dw = torch.zeros_like(weight, dtype=torch.float32)
    expected_db = torch.zeros_like(bias, dtype=torch.float32)
    expected_dx, expected_dc, expected_outputs = [], [], []
    for history, gate, grad in zip(histories, gates, output_gradients):
        history_ref = history.detach().requires_grad_(True)
        gate_ref = gate.detach().requires_grad_(True)
        out = (history_ref.float() * weight[:, 0].T.float()[None, None]).sum(2) + bias.float()
        result = gate_ref * out.to(gate_ref.dtype)
        dh, dc = torch.autograd.grad(result, (history_ref, gate_ref), grad)
        expected_dx.append(dh); expected_dc.append(dc); expected_outputs.append(result.detach())
        effective = (grad * gate).float()
        expected_dw += (effective.unsqueeze(2) * history.float()).sum((0, 1)).T[:, None, :]
        expected_db += effective.sum((0, 1))

    with ParameterCotangentScope.for_model(layer, directory=tmp_path) as scope:
        for index, (history, gate, grad) in enumerate(zip(histories, gates, output_gradients)):
            actual = stored_finite_convolution(history, layer.weight, layer.bias, gate)
            assert torch.equal(actual, expected_outputs[index])
            actual.backward(grad)
            torch.testing.assert_close(history.grad, expected_dx[index], rtol=0, atol=0)
            torch.testing.assert_close(gate.grad, expected_dc[index], rtol=0, atol=0)
            if index == 0:
                scope.new_backward()
        scope.finish()
    torch.testing.assert_close(layer.weight.grad.float(), expected_dw.to(weight.dtype).float(), rtol=0, atol=0)
    torch.testing.assert_close(layer.bias.grad.float(), expected_db.to(bias.dtype).float(), rtol=0, atol=0)


def test_distinct_parameters_sharing_storage_are_rejected(tmp_path):
    base = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
    view = torch.nn.Parameter(base.detach().view(2, 8))
    with pytest.raises(RuntimeError, match='overlapping distinct'):
        ParameterCotangentScope([('base', base), ('view', view)], directory=tmp_path)


def test_expert_bank_view_routes_fp32_cotangent_to_parent_parameter(tmp_path):
    from natlang_neuralese.maple.model import DenseExperts
    from natlang_neuralese.model.native_weight_carrier import stored_linear

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = DenseExperts(2, 7, 5)
            self.experts.make_latent(quantize=False)

    torch.manual_seed(83)
    model = Model()
    bank = model.experts.gate_up
    view = bank[1]
    x = torch.randn(4, 7, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(4, 10, dtype=torch.bfloat16)
    expected = torch.zeros_like(bank, dtype=torch.float32)
    expected[1] = grad.float().T @ x.detach().float()
    with ParameterCotangentScope.for_model(model, mode='vjp', directory=tmp_path) as scope:
        output = stored_linear(x, view, storage_dtype=torch.bfloat16)
        actual, = torch.autograd.grad((output * grad).sum(), (bank,))
        scope.finish()
    torch.testing.assert_close(actual.float(), expected.to(bank.dtype).float(), rtol=0, atol=0)
    torch.testing.assert_close(x.grad if x.grad is not None else torch.zeros_like(x), torch.zeros_like(x))


def test_nonreentrant_checkpoint_recompute_uses_scope_anchor(tmp_path):
    from torch.utils.checkpoint import checkpoint

    layer, x1, _, g1, _ = _case()
    weight = layer.weight
    expected = g1.float().T @ x1.detach().float()
    with ParameterCotangentScope.for_model(layer, mode='vjp', directory=tmp_path) as scope:
        result = checkpoint(layer, x1, use_reentrant=False)
        actual, = torch.autograd.grad((result * g1).sum(), (weight,))
        scope.finish()
    torch.testing.assert_close(actual.float(), expected.to(weight.dtype).float(), rtol=0, atol=0)


def test_one_shared_linear_module_alias_uses_one_anchor(tmp_path):
    class Aliased(nn.Module):
        def __init__(self):
            super().__init__()
            shared = nn.Linear(7, 5, bias=False).to(dtype=torch.bfloat16)
            self.left = shared
            self.right = shared

        def forward(self, left, right):
            return self.left(left), self.right(right)

    model = Aliased()
    left, right = torch.randn(2, 7, dtype=torch.bfloat16), torch.randn(3, 7, dtype=torch.bfloat16)
    gl, gr = torch.randn(2, 5, dtype=torch.bfloat16), torch.randn(3, 5, dtype=torch.bfloat16)
    weight = model.left.weight
    expected = gl.float().T @ left.float() + gr.float().T @ right.float()
    with ParameterCotangentScope.for_model(model, mode='vjp', directory=tmp_path) as scope:
        out_left, out_right = model(left, right)
        actual, = torch.autograd.grad((out_left * gl).sum() + (out_right * gr).sum(), (weight,))
        scope.finish()
    torch.testing.assert_close(actual.float(), expected.to(weight.dtype).float(), rtol=0, atol=0)
    assert model.left is model.right


def test_actor_replacements_share_base_fanin_and_keep_delta_vjps(tmp_path):
    from natlang_neuralese.model.native_weight_carrier import stored_linear

    torch.manual_seed(97)
    base = nn.Parameter(torch.randn(5, 7, dtype=torch.bfloat16))
    delta_a = torch.randn(5, 7, dtype=torch.float32, requires_grad=True)
    delta_b = torch.randn(5, 7, dtype=torch.float32, requires_grad=True)
    x_a = torch.randn(3, 7, dtype=torch.bfloat16)
    x_b = torch.randn(4, 7, dtype=torch.bfloat16)
    g_a = torch.randn(3, 5, dtype=torch.bfloat16)
    g_b = torch.randn(4, 5, dtype=torch.bfloat16)
    weight_a = base + delta_a.to(base.dtype)
    weight_b = base + delta_b.to(base.dtype)
    expected_a = g_a.float().T @ x_a.float()
    expected_b = g_b.float().T @ x_b.float()
    with ParameterCotangentScope([('shared.base', base)], mode='vjp', directory=tmp_path) as scope:
        scope.register_effective_weight(weight_a, base, [delta_a])
        scope.register_effective_weight(weight_b, base, [delta_b])
        output_a = stored_linear(x_a, weight_a, storage_dtype=base.dtype)
        output_b = stored_linear(x_b, weight_b, storage_dtype=base.dtype)
        actual_base, actual_a, actual_b = torch.autograd.grad(
            (output_a * g_a).sum() + (output_b * g_b).sum(), (base, delta_a, delta_b))
        scope.finish()
    torch.testing.assert_close(actual_base.float(), (expected_a + expected_b).to(base.dtype).float(), rtol=0, atol=0)
    torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=0)
    torch.testing.assert_close(actual_b, expected_b, rtol=0, atol=0)

