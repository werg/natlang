"""Fused RMSNorm (maple/model.py ``RMSNorm``) for BF16/FP16 activations on CUDA: one Triton kernel forward, one backward.

The eager rule ``(weight * (x.float() * rsqrt(mean(x.float()^2) + eps)).to(dtype)).to(dtype)`` runs about eight
full-size passes forward (FP32 copies of the activations) and as many again backward, and autograd keeps the FP32
intermediates. Here the forward computes the same FP32 normalization per row, rounds the normalized value to the
activation dtype, multiplies by the weight in FP32 and rounds once (the eager product's semantics for both an FP32 and
a BF16 weight); it saves only ``x`` and the per-row FP32 ``rsqrt``. The backward is the eager graph's chain rule in
FP32: ``grad_yb = round(g * w)`` (the cast's backward), then the normalization's gradient, rounded once to ``x``'s
dtype; the weight gradient is ``sum(g * yb)`` in the eager product's dtype. Only the FP32 row reduction order differs
from eager (tests/neuralese/test_fused_rms_norm.py measures the parity).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(x_ptr, w_ptr, out_ptr, rstd_ptr, columns, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < columns
    x = tl.load(x_ptr + row * columns + offsets, mask=valid, other=0.0).to(tl.float32)
    mean = tl.sum(x * x, axis=0) * (1.0 / columns)  # torch's mean: sum times the reciprocal count
    rstd = tl.math.rsqrt(mean + eps)
    normalized = (x * rstd).to(out_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * columns + offsets, (w * normalized).to(out_ptr.dtype.element_ty), mask=valid)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _backward_kernel(g_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, yb_ptr, columns, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < columns
    x = tl.load(x_ptr + row * columns + offsets, mask=valid, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * columns + offsets, mask=valid, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    # The rounded normalized value (for the weight gradient) and the gradient through the cast to it.
    tl.store(yb_ptr + row * columns + offsets, (x * rstd).to(yb_ptr.dtype.element_ty), mask=valid)
    grad_y = (g * w).to(dx_ptr.dtype.element_ty).to(tl.float32)
    # y = x * rstd, rstd = (mean(x^2) + eps)^-1/2: dx = grad_y * rstd + x * (2 / n) * (-1/2) rstd^3 * sum(grad_y * x)
    grad_rstd = tl.sum(grad_y * x, axis=0)
    grad_mean = grad_rstd * (-0.5 * rstd * rstd * rstd)
    dx = grad_y * rstd + (grad_mean / columns) * 2.0 * x
    tl.store(dx_ptr + row * columns + offsets, dx.to(dx_ptr.dtype.element_ty), mask=valid)


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        columns = x.shape[-1]
        rows_x = x.reshape(-1, columns)
        if not rows_x.is_contiguous():
            rows_x = rows_x.contiguous()
        w = weight.contiguous()
        out = torch.empty_like(rows_x)
        rstd = torch.empty(rows_x.shape[0], device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(columns)
        if rows_x.shape[0]:
            _forward_kernel[(rows_x.shape[0],)](rows_x, w, out, rstd, columns, float(eps), BLOCK=block,
                                                num_warps=min(16, max(1, block // 256)))
        ctx.save_for_backward(rows_x, w, rstd)
        ctx.shape = x.shape
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad):
        rows_x, w, rstd = ctx.saved_tensors
        columns = rows_x.shape[-1]
        g = grad.reshape(-1, columns)
        if not g.is_contiguous():
            g = g.contiguous()
        dx = torch.empty_like(rows_x)
        yb = torch.empty_like(rows_x)
        block = triton.next_power_of_2(columns)
        if rows_x.shape[0]:
            _backward_kernel[(rows_x.shape[0],)](g, rows_x, w, rstd, dx, yb, columns, BLOCK=block,
                                                 num_warps=min(16, max(1, block // 256)))
        grad_weight = None
        if ctx.needs_input_grad[1]:
            product = torch.promote_types(w.dtype, rows_x.dtype)
            grad_weight = (g.to(product) * yb.to(product)).sum(0).to(w.dtype)
        return dx.view(ctx.shape), grad_weight, None


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return FusedRMSNorm.apply(x, weight, eps)
