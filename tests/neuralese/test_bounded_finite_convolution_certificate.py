"""Regression certificate for full graph finite-convolution replacements."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from natlang_neuralese.model.native_graph_plan import ImportSpan, Invocation, compile_graph
from natlang_neuralese.model.shared_state import _convolve, _finite_convolution


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_complete_graph_convolution_forward_and_adjoint(device, dtype, checkpointed):
    torch.manual_seed(311)
    plan = compile_graph([
        Invocation('source', tuple(range(67)), producing=((1, 55),)),
        Invocation('stopped', tuple(range(40)), imports=(ImportSpan(3, 'source', 1, 55, stop_gradient=True),)),
        Invocation('live', tuple(range(37)), imports=(ImportSpan(2, 'source', 1, 55),)),
    ])
    width, channels = 7, 16
    source = torch.randn(1, plan.token_count, channels, device=device, dtype=dtype)
    gate = torch.randn_like(source)
    noise = torch.randn(1, plan.token_count, width, channels, device=device, dtype=dtype)[0]
    weights = torch.randn(channels, 1, width, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype)
    upstream = torch.randn_like(source)

    def run(reference):
        x, c, w, b, n = (v.detach().clone().requires_grad_() for v in
                          (source, gate, weights, bias, noise))
        kernel = SimpleNamespace(weight=w, bias=b)
        layer = SimpleNamespace(conv=SimpleNamespace(conv=kernel))
        def execute(x, c, w, b, n):
            if not reference:
                return _convolve(layer, x, c, plan, range(plan.token_count), noise=n)
            indices, stops = [], []
            for row in range(plan.token_count):
                history = plan.history(row, width)
                flags = plan.history_stops(row, width)
                pad = width - len(history)
                indices.append([-1] * pad + list(history))
                stops.append([False] * pad + list(flags))
            index = torch.tensor(indices, device=device)
            stop = torch.tensor(stops, device=device)
            padded = torch.cat((x, x.new_zeros(1, 1, channels)), 1)
            lengths = [len(node.token_ids) for node in plan.invocations if node.token_ids]
            data = torch.cat([padded[:, part] for part in index.split(lengths)], 1)
            data = torch.where(stop[None, :, :, None], data.detach(), data) + n[None]
            return _finite_convolution(kernel, data, c)
        out = checkpoint(execute, x, c, w, b, n, use_reentrant=False) if checkpointed else execute(x, c, w, b, n)
        grad = torch.autograd.grad(out, (x, c, w, b, n), upstream)
        return out.detach(), tuple(item.detach() for item in grad)

    actual, actual_grads = run(False)
    expected, expected_grads = run(True)
    assert torch.equal(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=0)
