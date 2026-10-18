"""Grouped expert projections for the dense (BF16-latent) Mellum experts with weight gradients in the parameter
layout, and the shared per-row E4M3 quantizer of the FP8 expert path (maple/expert_gemm.py).

``GroupedLinear.apply(rows, weight, offsets, False)``: ``rows`` [P, K] are routed rows sorted by expert, ``offsets``
the int32 group ends (as ``torch._grouped_mm`` takes them), ``weight`` [E, N, K]; returns [P, N] = rows_e @ weight[e]ᵀ.
The weight gradient is one grouped GEMM over the row dimension straight into [E, N, K] (autograd through
``weight.transpose(-2, -1)`` produced it transposed and copied it into the parameter layout).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
BLOCK = 1024


@triton.jit
def _rowwise_quant(x_ptr, q_ptr, scale_ptr, C, FMAX: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK,), dtype=tl.float32)
    for c0 in range(0, C, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * C + offs, mask=offs < C, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    scale = tl.maximum(tl.max(amax, axis=0), 1e-30) / FMAX
    for c0 in range(0, C, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * C + offs, mask=offs < C, other=0.0).to(tl.float32)
        tl.store(q_ptr + row * C + offs, (x / scale).to(q_ptr.dtype.element_ty), mask=offs < C)
    tl.store(scale_ptr + row, scale)


def rowwise_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 values and FP32 scales (amax / 448) per row of a contiguous 2-D tensor."""
    x = x.contiguous()
    rows, columns = x.shape
    q = torch.empty_like(x, dtype=FP8)
    scale = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows:
        _rowwise_quant[(rows,)](x, q, scale, columns, FMAX=FP8_MAX, BLOCK=min(BLOCK, triton.next_power_of_2(columns)))
    return q, scale


class GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, weight, offsets, fp8, *anchors):
        if fp8:
            raise ValueError('FP8 expert GEMMs run through maple/expert_gemm.py (--fp8-training triton-experts)')
        from ..model.native_adapters import native_forward_storage_dtype
        dtype = native_forward_storage_dtype(weight)
        ctx.carrier = dtype is not None
        ctx.anchors = anchors
        ctx.scope_weight = weight if anchors else None
        from ..model.native_weight_carrier import active_cotangent_scope
        ctx.scope = active_cotangent_scope() if anchors else None
        weight = weight.to(dtype) if ctx.carrier else weight
        ctx.save_for_backward(rows, weight, offsets)
        return torch._grouped_mm(rows, weight.transpose(-2, -1), offs=offsets)

    @staticmethod
    def backward(ctx, grad):
        rows, weight, offsets = ctx.saved_tensors
        grad = grad.contiguous()
        grad_rows = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_rows = torch._grouped_mm(grad, weight, offs=offsets)
        if ctx.needs_input_grad[1]:
            grad_weight = (grouped_weight_gradient(grad, rows, offsets, weight.shape) if ctx.carrier else
                           torch._grouped_mm(grad.t(), rows, offs=offsets))
        anchor_grads = (None,) * len(ctx.anchors)
        if ctx.anchors and grad_weight is not None:
            from ..model.native_weight_carrier import _scope_add
            _scope_add(ctx.scope_weight, grad_weight, ctx.anchors, ctx.scope)
            grad_weight = None
            anchor_grads = tuple(grad.new_ones((), dtype=torch.float32) if anchor is not None else None
                                 for anchor in ctx.anchors)
        return grad_rows, grad_weight, None, None, *anchor_grads


@triton.jit
def _grouped_weight_gradient(g_ptr, a_ptr, ends_ptr, out_ptr, N: tl.constexpr, K: tl.constexpr,
                             SG0: tl.constexpr, SG1: tl.constexpr, SA0: tl.constexpr, SA1: tl.constexpr,
                             FP32_INPUTS: tl.constexpr, B: tl.constexpr):
    expert = tl.program_id(0)
    n = tl.program_id(1) * B + tl.arange(0, B)
    k = tl.program_id(2) * B + tl.arange(0, B)
    start = tl.load(ends_ptr + expert - 1, mask=expert > 0, other=0)
    stop = tl.load(ends_ptr + expert)
    acc = tl.zeros((B, B), tl.float32)
    for p0 in range(start, stop, B):
        p = p0 + tl.arange(0, B)
        grad = tl.load(g_ptr + p[None, :] * SG0 + n[:, None] * SG1,
                       mask=(p[None, :] < stop) & (n[:, None] < N), other=0)
        value = tl.load(a_ptr + p[:, None] * SA0 + k[None, :] * SA1,
                        mask=(p[:, None] < stop) & (k[None, :] < K), other=0)
        if FP32_INPUTS:
            grad, value = grad.to(tl.float32), value.to(tl.float32)
        acc += tl.dot(grad, value, input_precision='ieee')
    tl.store(out_ptr + expert * N * K + n[:, None] * K + k[None, :], acc,
             mask=(n[:, None] < N) & (k[None, :] < K))


def grouped_weight_gradient(grad, rows, offsets, shape):
    """Expert dW with FP32 accumulation/output and no host synchronization."""
    experts, n, k = shape
    result = torch.empty(shape, device=grad.device, dtype=torch.float32)
    if not grad.is_cuda:
        start = 0
        for expert, stop in enumerate(offsets.tolist()):
            result[expert] = grad[start:stop].float().T @ rows[start:stop].float()
            start = stop
        return result
    _grouped_weight_gradient[(experts, triton.cdiv(n, 32), triton.cdiv(k, 32))](
        grad, rows, offsets, result, n, k, *grad.stride(), *rows.stride(),
        FP32_INPUTS=grad.dtype == torch.float32 or rows.dtype == torch.float32, B=32,
        num_warps=4, num_stages=2)
    return result
