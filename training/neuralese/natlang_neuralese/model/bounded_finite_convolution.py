"""Channel-bounded finite convolution for long graph replays.

Each channel is independent in the forward and source adjoint. Parameter
adjoints reduce all physical rows before casting to their storage dtype.
"""
from __future__ import annotations

from math import gcd
from types import SimpleNamespace

import torch

_ORDERED_ROWS_PER_CHUNK = 4096


def _kernel(weight, bias, canonical):
    return SimpleNamespace(weight=weight, bias=bias, _native_canonical_pointwise=canonical)


def _layer(weight, bias, canonical):
    return SimpleNamespace(conv=SimpleNamespace(conv=_kernel(weight, bias, canonical)))


def _history(source, index, stop, noise, begin, end):
    """Gather a bounded interval of flattened batch/query rows."""
    padded = torch.cat((source, source.new_zeros(source.shape[0], 1, source.shape[-1])), 1)
    q = index.shape[0]
    rows = torch.arange(begin, end, device=index.device)
    batches, queries = torch.div(rows, q, rounding_mode='floor'), rows.remainder(q)
    data = padded[batches[:, None], index[queries]]
    if source.dtype == torch.float64:
        data = data.float()
    if stop is not None:
        data = torch.where(stop[queries, :, None], data.detach(), data)
    if noise is not None:
        data = data + noise[queries].to(data.dtype)
    return data


def _reduction_sections(count, row_bytes):
    """Match TensorIterator's recursive 32-bit indexing splits for rows."""
    pending = [(0, count)]
    while pending:
        begin, end = pending.pop()
        if (end - begin) * row_bytes >= 2**31:
            middle = begin + (end - begin) // 2
            if middle == begin:
                raise ValueError('one convolution reduction row exceeds 32-bit indexing')
            pending.extend(((middle, end), (begin, middle)))
        else:
            yield begin, end


class _BoundedFiniteConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, gate, weight, bias, noise, plan, rows, index, stop, lengths, canonical,
                weight_anchor_count, *anchors):
        from .shared_state import _convolve_eager
        from .native_weight_carrier import active_cotangent_scope
        from .native_gradient_partitions import active_gradient_partitions
        ctx.save_for_backward(source, gate, weight, bias, noise, index, stop)
        ctx.plan, ctx.rows, ctx.lengths, ctx.canonical = plan, rows, lengths, canonical
        ctx.scope, ctx.partitions = active_cotangent_scope(), active_gradient_partitions()
        ctx.weight_anchors = anchors[:weight_anchor_count]
        ctx.bias_anchors = anchors[weight_anchor_count:]
        channels = source.shape[-1]
        ctx.chunk = gcd(channels, 128)
        pieces = []
        for begin in range(0, channels, ctx.chunk):
            end = begin + ctx.chunk
            layer = _layer(weight[begin:end], None if bias is None else bias[begin:end], canonical)
            pieces.append(_convolve_eager(layer, source[..., begin:end], gate[..., begin:end],
                plan, rows, index=(index, stop) if stop is not None else index,
                noise=None if noise is None else noise[..., begin:end]))
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def backward(ctx, gradient):
        from .shared_state import _convolve_eager
        source, gate, weight, bias, noise, index, stop = ctx.saved_tensors
        # Higher-order VJPs replay the original differentiable operation. This
        # path materializes the full history and is not memory-qualified for
        # long graphs. Its surrounding custom-autograd node can change FP32
        # accumulation order in later derivatives; the explicit reference
        # arithmetic scope bypasses this node altogether.
        if torch.is_grad_enabled():
            with torch.enable_grad():
                result = _convolve_eager(_layer(weight, bias, ctx.canonical), source, gate,
                    ctx.plan, ctx.rows, index=(index, stop) if stop is not None else index, noise=noise)
                tensors = (source, gate, weight, bias, noise)
                selected = [t for t in tensors if t is not None and t.requires_grad]
                values = iter(torch.autograd.grad(result, selected, gradient, create_graph=True, allow_unused=True))
                grads = [next(values) if t is not None and t.requires_grad else None for t in tensors]
            return (*grads, None, None, None, None, None, None, None, *([None] * (len(ctx.weight_anchors) + len(ctx.bias_anchors))))
        channels = source.shape[-1]
        dsource = torch.empty_like(source) if ctx.needs_input_grad[0] else None
        dgate = torch.empty_like(gate) if ctx.needs_input_grad[1] else None
        dweight = torch.empty_like(weight) if ctx.needs_input_grad[2] and not ctx.weight_anchors else None
        dbias = torch.empty_like(bias) if bias is not None and ctx.needs_input_grad[3] and not ctx.bias_anchors else None
        dnoise = torch.empty_like(noise) if noise is not None and ctx.needs_input_grad[4] else None
        for begin in range(0, channels, ctx.chunk):
            end = begin + ctx.chunk
            # Replay each channel independently; the invocation-split gather and
            # its scatter remain PyTorch's own operations and retain their order.
            from .native_weight_carrier import _COTANGENT_SCOPE
            scope_token = _COTANGENT_SCOPE.set(None)
            try:
                with torch.enable_grad():
                    x = source[..., begin:end].detach().requires_grad_(dsource is not None)
                    c = gate[..., begin:end].detach().requires_grad_(dgate is not None)
                    n = None if noise is None else noise[..., begin:end].detach().requires_grad_(dnoise is not None)
                    layer = _layer(weight[begin:end].detach(), None if bias is None else bias[begin:end].detach(), ctx.canonical)
                    result = _convolve_eager(layer, x, c, ctx.plan, ctx.rows,
                        index=(index, stop) if stop is not None else index, noise=n)
                    selected = [t for t in (x, c, n) if t is not None and t.requires_grad]
                    values = iter(torch.autograd.grad(result, selected, gradient[..., begin:end], allow_unused=True)) if selected else iter(())
                    for tensor, target in ((x, dsource), (c, dgate), (n, dnoise)):
                        if tensor is not None and tensor.requires_grad:
                            value = next(values)
                            if value is not None:
                                target[..., begin:end] = value
            finally:
                _COTANGENT_SCOPE.reset(scope_token)
            if dweight is not None or dbias is not None or ctx.weight_anchors or ctx.bias_anchors:
                effective = (gradient[..., begin:end] * gate[..., begin:end]).float()
                groups = channels // ctx.chunk
                ordered = bool((ctx.weight_anchors or ctx.bias_anchors) and ctx.partitions is not None)
                if ordered:
                    from .native_weight_carrier import _scope_add_ordered_finite
                    from .ordered_finite_convolution import ordered_finite_convolution_accumulate
                    if effective.shape[0] != 1 or ctx.partitions.token_count != effective.shape[1]:
                        raise RuntimeError('finite-convolution graph cotangent lost its token-row provenance')
                    local_w = torch.zeros_like(weight[begin:end], dtype=torch.float32) if dweight is not None else None
                    local_b = torch.zeros_like(bias[begin:end], dtype=torch.float32) if dbias is not None else None
                    for hi in range(effective.shape[1], 0, -_ORDERED_ROWS_PER_CHUNK):
                        lo = max(0, hi - _ORDERED_ROWS_PER_CHUNK)
                        history = _history(source[..., begin:end], index, stop,
                            None if noise is None else noise[..., begin:end], lo, hi)[None].detach()
                        value = effective[:, lo:hi]
                        if ctx.weight_anchors:
                            _scope_add_ordered_finite(weight[begin:end], history, value, ctx.weight_anchors, ctx.scope)
                        elif local_w is not None:
                            ordered_finite_convolution_accumulate(local_w, history, value,
                                backend='device_fp32' if weight.is_cuda else 'cpu_file')
                        if ctx.bias_anchors:
                            _scope_add_ordered_finite(bias[begin:end], None, value, ctx.bias_anchors, ctx.scope, bias=True)
                        elif local_b is not None:
                            ordered_finite_convolution_accumulate(local_b, None, value,
                                backend='device_fp32' if weight.is_cuda else 'cpu_file', bias=True)
                    if dweight is not None: dweight[begin:end] = local_w.to(weight.dtype)
                    if dbias is not None: dbias[begin:end] = local_b.to(bias.dtype)
                    continue
                if dweight is not None or ctx.weight_anchors:
                    b, q, chunk = effective.shape
                    width = index.shape[1]
                    flat = effective.reshape(b * q, chunk)
                    part = None
                    for lo, hi in _reduction_sections(b * q, width * channels * 4):
                        history = _history(source[..., begin:end], index, stop,
                                           None if noise is None else noise[..., begin:end], lo, hi)
                        product = history.contiguous().float() * flat[lo:hi, None, :]
                        expanded = product.reshape(hi-lo, width, 1, chunk).expand(hi-lo, width, groups, chunk)
                        value = expanded.sum(0)[:, 0, :]
                        part = value if part is None else part + value
                    part = part.T[:, None, :]
                    if ctx.weight_anchors:
                        from .native_weight_carrier import _scope_add
                        _scope_add(weight[begin:end], part, ctx.weight_anchors, ctx.scope)
                    else:
                        dweight[begin:end] = part.to(weight.dtype)
                if dbias is not None or ctx.bias_anchors:
                    b, q, chunk = effective.shape
                    flat = effective.reshape(b * q, chunk)
                    part = None
                    for lo, hi in _reduction_sections(b * q, channels * 4):
                        expanded = flat[lo:hi].reshape(hi-lo, 1, chunk).expand(hi-lo, groups, chunk)
                        value = expanded.sum(0)[0]
                        part = value if part is None else part + value
                    if ctx.bias_anchors:
                        from .native_weight_carrier import _scope_add
                        _scope_add(bias[begin:end], part, ctx.bias_anchors, ctx.scope)
                    else:
                        dbias[begin:end] = part.to(bias.dtype)
        anchor_grads = tuple(gradient.new_ones((), dtype=torch.float32)
                             for _ in (*ctx.weight_anchors, *ctx.bias_anchors))
        return dsource, dgate, dweight, dbias, dnoise, None, None, None, None, None, None, None, *anchor_grads


def bounded_finite_convolution(layer, source, gate, plan, rows, index, stop, lengths, noise):
    kernel = layer.conv.conv
    from .native_weight_carrier import _scope_anchors
    weight_anchors = _scope_anchors(kernel.weight)
    bias_anchors = _scope_anchors(kernel.bias) if kernel.bias is not None else ()
    return _BoundedFiniteConvolution.apply(source, gate, kernel.weight, kernel.bias, noise,
        plan, rows, index, stop, tuple(lengths), bool(getattr(kernel, '_native_canonical_pointwise', False)),
        len(weight_anchors), *weight_anchors, *bias_anchors)
