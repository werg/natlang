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
    def forward(ctx, rows, weight, offsets, fp8):
        if fp8:
            raise ValueError('FP8 expert GEMMs run through maple/expert_gemm.py (--fp8-training triton-experts)')
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
            grad_weight = torch._grouped_mm(grad.t(), rows, offs=offsets)  # [E, N, K]: the parameter's layout
        return grad_rows, grad_weight, None, None
