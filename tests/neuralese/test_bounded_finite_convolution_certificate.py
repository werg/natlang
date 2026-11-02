"""Regression certificate for full graph finite-convolution replacements."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from natlang_neuralese.model.native_graph_plan import ImportSpan, Invocation, compile_graph
from natlang_neuralese.model.shared_state import _convolve, _finite_convolution


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_complete_graph_convolution_forward_and_adjoint(device, dtype, checkpointed, monkeypatch):
    from natlang_neuralese.model import shared_state
    monkeypatch.setattr(shared_state, '_BOUNDED_CONVOLUTION_MIN_ROWS', 0)
    torch.manual_seed(311)
    plan = compile_graph([
        Invocation('source', tuple(range(67)), producing=((1, 55),)),
        Invocation('stopped', tuple(range(40)), imports=(ImportSpan(3, 'source', 1, 55, stop_gradient=True),)),
        Invocation('live', tuple(range(37)), imports=(ImportSpan(2, 'source', 1, 55),)),
    ])
    width, channels = 7, 256
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
            data = (padded[:, index] if dtype == torch.float16 else
                    torch.cat([padded[:, part] for part in index.split(lengths)], 1))
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


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('source_dtype', [torch.float32, torch.float64])
def test_long_channel_partition_matches_complete_reduction(device, dtype, source_dtype, monkeypatch):
    from natlang_neuralese.model import shared_state
    torch.manual_seed(928)
    count, channels, width = 4096, 512, 7
    plan = compile_graph([Invocation('long', tuple(range(count)))])
    # FP32 shared state with BF16 parameters exercises mixed-precision history.
    source = torch.randn(1, count, channels, device=device, dtype=source_dtype)
    gate = torch.randn(1, count, channels, device=device, dtype=dtype)
    weight = torch.randn(channels, 1, width, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype)
    upstream = torch.randn_like(gate)

    def run(bounded):
        monkeypatch.setattr(shared_state, '_BOUNDED_CONVOLUTION_MIN_ROWS', 0 if bounded else count + 1)
        x, c, w, b = [v.detach().clone().requires_grad_() for v in (source, gate, weight, bias)]
        layer = SimpleNamespace(conv=SimpleNamespace(conv=SimpleNamespace(weight=w, bias=b)))
        result = shared_state._convolve(layer, x, c, plan, range(count))
        gradients = torch.autograd.grad(result, (x, c, w, b), upstream)
        return result.detach(), tuple(g.detach() for g in gradients)

    actual, actual_grad = run(True)
    expected, expected_grad = run(False)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


@pytest.mark.parametrize('reference', [False, True])
@pytest.mark.parametrize('source_dtype', [torch.float32, torch.float64])
def test_higher_order_reference_replay(monkeypatch, reference, source_dtype):
    from contextlib import nullcontext
    from natlang_neuralese.model import shared_state
    from natlang_neuralese.common.native_higher_order import reference_arithmetic
    torch.manual_seed(71)
    plan = compile_graph([Invocation('source', tuple(range(9)))])
    x0 = torch.randn(1, 9, 8, dtype=source_dtype) * .2
    c0 = torch.randn(1, 9, 8) * .2
    w0 = torch.randn(8, 1, 3) * .2
    b0 = torch.randn(8) * .2
    def run(bounded):
        monkeypatch.setattr(shared_state, '_BOUNDED_CONVOLUTION_MIN_ROWS', 0 if bounded else 100)
        x, c, w, b = (t.clone().requires_grad_() for t in (x0, c0, w0, b0))
        layer = SimpleNamespace(conv=SimpleNamespace(conv=SimpleNamespace(weight=w, bias=b)))
        with (reference_arithmetic() if reference else nullcontext()):
            out = shared_state._convolve(layer, x, c, plan, range(9))
            first = torch.autograd.grad(out.square().sum(), (x, c, w, b), create_graph=True)
            second = torch.autograd.grad(sum(v.square().sum() for v in first), (x, c, w, b))
        return first, second
    actual = run(True)
    expected = run(False)
    for actual_order, expected_order in zip(actual, expected):
        for a, b in zip(actual_order, expected_order):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


def test_reduction_sections_preserve_iterator_splits():
    from natlang_neuralese.model.bounded_finite_convolution import _reduction_sections
    assert list(_reduction_sections(38000, 7 * 2048 * 4)) == [(0, 19000), (19000, 38000)]
    sections = list(_reduction_sections(290623, 7 * 2048 * 4))
    assert sections[0][0] == 0 and sections[-1][1] == 290623
    assert all(end == next_begin for (_, end), (next_begin, _) in zip(sections, sections[1:]))
    assert all((end-begin)*7*2048*4 < 2**31 for begin, end in sections)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('partitioned', [False, True])
def test_bounded_scope_accumulates_existing_parameter_cotangents(device, partitioned, tmp_path, monkeypatch):
    from contextlib import nullcontext
    from natlang_neuralese.model import shared_state
    from natlang_neuralese.model.native_gradient_partitions import GradientPartitions, gradient_partitions
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope
    from natlang_neuralese.model import bounded_finite_convolution
    monkeypatch.setattr(bounded_finite_convolution, '_ORDERED_ROWS_PER_CHUNK', 5)
    torch.manual_seed(527)
    count, channels, width = 32, 256, 7
    plan = compile_graph([Invocation('a', tuple(range(13))), Invocation('b', tuple(range(19)))])
    x0 = torch.randn(1, count, channels, dtype=torch.bfloat16, device=device)
    c0 = torch.randn_like(x0)
    weight = torch.randn(channels, 1, width, dtype=torch.bfloat16, device=device)
    bias = torch.randn(channels, dtype=torch.bfloat16, device=device)
    upstream = torch.randn_like(x0)
    def run(bounded):
        monkeypatch.setattr(shared_state, '_BOUNDED_CONVOLUTION_MIN_ROWS', 0 if bounded else 100)
        conv = torch.nn.Conv1d(channels, channels, width, groups=channels, bias=True, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            conv.weight.copy_(weight); conv.bias.copy_(bias)
        layer = SimpleNamespace(conv=SimpleNamespace(conv=conv))
        x, c = x0.clone().requires_grad_(), c0.clone().requires_grad_()
        backend = 'device_fp32' if device == 'cuda' else 'cpu_file'
        (tmp_path / str(bounded)).mkdir()
        with ParameterCotangentScope.for_model(conv, directory=tmp_path / str(bounded), accumulator_backend=backend) as scope:
            context = gradient_partitions(GradientPartitions((13, 19))) if partitioned else nullcontext()
            with context:
                out = shared_state._convolve(layer, x, c, plan, range(count))
            out.backward(upstream)
            scope.new_backward()
            context = gradient_partitions(GradientPartitions((13, 19))) if partitioned else nullcontext()
            with context:
                out = shared_state._convolve(layer, x, c, plan, range(count))
            out.backward(-upstream / 2)
            scope.finish()
        return out.detach(), x.grad, c.grad, conv.weight.grad, conv.bias.grad
    actual, expected = run(True), run(False)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


def test_large_cuda_parameter_reduction_matches_iterator(monkeypatch):
    import os
    if os.environ.get('NATLANG_TEST_LARGE_CONVOLUTION') != '1' or not torch.cuda.is_available():
        pytest.skip('set NATLANG_TEST_LARGE_CONVOLUTION=1 for the 2.6 GiB CUDA reduction witness')
    from natlang_neuralese.model.bounded_finite_convolution import _reduction_sections
    torch.manual_seed(9)
    count, width, channels, chunk = 38001, 7, 2048, 128
    product = torch.randn(count, width, channels, device='cuda', dtype=torch.float32)
    expected = product.sum(0)
    parts = []
    for at in range(0, channels, chunk):
        total = None
        for lo, hi in _reduction_sections(count, width * channels * 4):
            piece = product[lo:hi, :, at:at+chunk].contiguous()
            value = piece.reshape(hi-lo, width, 1, chunk).expand(hi-lo, width, channels//chunk, chunk).sum(0)[:, 0]
            total = value if total is None else total + value
        parts.append(total)
    actual = torch.cat(parts, 1)
    assert torch.equal(actual, expected)


def test_large_cuda_peak_is_channel_bounded(monkeypatch):
    import os
    if os.environ.get('NATLANG_TEST_LARGE_CONVOLUTION') != '1' or not torch.cuda.is_available():
        pytest.skip('set NATLANG_TEST_LARGE_CONVOLUTION=1 for the CUDA peak-memory witness')
    from natlang_neuralese.model import shared_state
    torch.manual_seed(408)
    count, channels, width = 4096, 2048, 7
    plan = compile_graph([Invocation('long', tuple(range(count)))])
    x0 = torch.randn(1, count, channels, device='cuda', dtype=torch.bfloat16)
    c0 = torch.randn_like(x0)
    w0 = torch.randn(channels, 1, width, device='cuda', dtype=torch.bfloat16)
    upstream = torch.randn_like(x0)
    def measure(bounded):
        monkeypatch.setattr(shared_state, '_BOUNDED_CONVOLUTION_MIN_ROWS', 0 if bounded else count + 1)
        x, c, w = [v.clone().requires_grad_() for v in (x0, c0, w0)]
        layer = SimpleNamespace(conv=SimpleNamespace(conv=SimpleNamespace(weight=w, bias=None)))
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = shared_state._convolve(layer, x, c, plan, range(count))
        torch.autograd.grad(out, (x, c, w), upstream)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - start
    eager_peak = measure(False)
    bounded_peak = measure(True)
    assert bounded_peak < eager_peak * 0.85, (eager_peak, bounded_peak)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_bounded_subset_keeps_invocation_and_stopped_imports(device, monkeypatch):
    from natlang_neuralese.model import shared_state
    torch.manual_seed(118)
    plan = compile_graph([
        Invocation('a', tuple(range(20)), producing=((2, 15),)),
        Invocation('b', tuple(range(13)), imports=(ImportSpan(1, 'a', 2, 15, stop_gradient=True),)),
        Invocation('c', tuple(range(11)), imports=(ImportSpan(2, 'a', 2, 15),)),
    ])
    rows = tuple(range(0, plan.token_count, 2))
    channels = 256
    x0 = torch.randn(1, plan.token_count, channels, device=device, dtype=torch.bfloat16)
    c0 = torch.randn(1, len(rows), channels, device=device, dtype=torch.bfloat16)
    w0 = torch.randn(channels, 1, 7, device=device, dtype=torch.bfloat16)
    b0 = torch.randn(channels, device=device, dtype=torch.bfloat16)
    upstream = torch.randn_like(c0)
    def run(bounded):
        monkeypatch.setattr(shared_state, '_BOUNDED_CONVOLUTION_MIN_ROWS', 0 if bounded else 100)
        x, c, w, b = [v.clone().requires_grad_() for v in (x0, c0, w0, b0)]
        layer = SimpleNamespace(conv=SimpleNamespace(conv=SimpleNamespace(weight=w, bias=b)))
        out = shared_state._convolve(layer, x, c, plan, rows)
        return out.detach(), torch.autograd.grad(out, (x, c, w, b), upstream)
    actual, expected = run(True), run(False)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-4, rtol=0)
    for a, b in zip(actual[1], expected[1]):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
