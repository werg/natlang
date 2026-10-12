"""Fused per-head RMSNorm + full rotary embedding + head-major layout for the q/k projections (Mellum attention).

Eager, per projection: the RMSNorm kernel, then ``apply_partial_rope`` (rotate_half's cat and negation, two products,
a sum and a cat copy, each a full pass over q), then ``transpose(1, 2)`` (SDPA copies it to contiguous), and as many
passes again in backward. Here one Triton program per token reads the projection's [heads * head_dim] row once and
writes the rotated, normalized heads in [B, H, T, D] order; the backward reads the gradient once and writes the
projection's input gradient and the norm gain's gradient.

The forward replays the eager roundings one by one (normalized value rounded to the activation dtype, times the gain,
rounded; ``rot * cos`` and ``rotate_half(rot) * sin`` each rounded, their sum rounded; no FMA contraction), so it
equals the eager path up to the RMS reduction order (tests/neuralese/test_rope_kernels.py). The backward follows autograd's BF16 roundings of
the same graph.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr, T, H: tl.constexpr, D: tl.constexpr, eps):
    token = tl.program_id(0).to(tl.int64)  # b * T + t
    b = token // T
    t = token % T
    d = tl.arange(0, D)
    half = D // 2
    rotated = (d + half) % D
    sign = tl.where(d < half, -1.0, 1.0)
    cos = tl.load(cos_ptr + token * D + d).to(tl.float32)
    sin = tl.load(sin_ptr + token * D + d).to(tl.float32)
    w = tl.load(w_ptr + d).to(tl.float32)
    w_rot = tl.load(w_ptr + rotated).to(tl.float32)
    for h in tl.static_range(H):
        base = token * H * D + h * D
        x = tl.load(x_ptr + base + d).to(tl.float32)
        x_rot = tl.load(x_ptr + base + rotated).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
        n = (w * (x * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        n_rot = (w_rot * (x_rot * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        a = (n * cos).to(tl.bfloat16).to(tl.float32)
        c = ((sign * n_rot) * sin).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + ((b * H + h) * T + t) * D + d, (a + c).to(out_ptr.dtype.element_ty))


@triton.jit
def _backward(g_ptr, x_ptr, w_ptr, cos_ptr, sin_ptr, dx_ptr, dw_ptr, T, H: tl.constexpr, D: tl.constexpr, eps):
    token = tl.program_id(0).to(tl.int64)
    b = token // T
    t = token % T
    d = tl.arange(0, D)
    half = D // 2
    rotated = (d + half) % D
    sign_t = tl.where(d < half, 1.0, -1.0)  # rotate_half's transpose: z[d + half] for d < half, -z[d - half] after
    cos = tl.load(cos_ptr + token * D + d).to(tl.float32)
    sin_rot = tl.load(sin_ptr + token * D + rotated).to(tl.float32)
    w = tl.load(w_ptr + d).to(tl.float32)
    dw = tl.zeros((D,), dtype=tl.float32)
    for h in tl.static_range(H):
        g_base = ((b * H + h) * T + t) * D
        g = tl.load(g_ptr + g_base + d).to(tl.float32)
        g_rot = tl.load(g_ptr + g_base + rotated).to(tl.float32)
        from_cos = (g * cos).to(tl.bfloat16).to(tl.float32)
        from_sin = (sign_t * (g_rot * sin_rot).to(tl.bfloat16).to(tl.float32))
        dn = (from_cos + from_sin).to(tl.bfloat16).to(tl.float32)
        base = token * H * D + h * D
        x = tl.load(x_ptr + base + d).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
        y = (x * rstd).to(tl.bfloat16).to(tl.float32)
        dw += dn * y
        grad_y = (dn * w).to(tl.bfloat16).to(tl.float32)
        grad_mean = tl.sum(grad_y * x, axis=0) * (-0.5 * rstd * rstd * rstd)
        dx = grad_y * rstd + (grad_mean / D) * 2.0 * x
        tl.store(dx_ptr + base + d, dx.to(dx_ptr.dtype.element_ty))
    tl.atomic_add(dw_ptr + d, dw)


class QKNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, cos, sin, heads, eps):
        B, T, width = x.shape
        D = width // heads
        x = x.contiguous()
        cos, sin = cos.expand(B, T, D).contiguous(), sin.expand(B, T, D).contiguous()
        out = torch.empty(B, heads, T, D, device=x.device, dtype=x.dtype)
        if B * T:
            _forward[(B * T,)](x, weight, cos, sin, out, T, H=heads, D=D, eps=float(eps), num_warps=1,
                                 enable_fp_fusion=False)
        ctx.save_for_backward(x, weight, cos, sin)
        ctx.heads, ctx.eps = heads, eps
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight, cos, sin = ctx.saved_tensors
        B, T, width = x.shape
        D = width // ctx.heads
        grad = grad.contiguous()
        dx = torch.empty_like(x)
        dw = torch.zeros(D, device=x.device, dtype=torch.float32)
        if B * T:
            _backward[(B * T,)](grad, x, weight, cos, sin, dx, dw, T, H=ctx.heads, D=D, eps=float(ctx.eps),
                                num_warps=1, enable_fp_fusion=False)
        return dx, (dw.to(weight.dtype) if ctx.needs_input_grad[1] else None), None, None, None, None


def qk_norm_rope(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, heads: int,
                 eps: float) -> torch.Tensor:
    """[B, T, heads * D] projection -> [B, heads, T, D]: per-head RMSNorm (gain ``weight``), full rotary (``cos``,
    ``sin`` [B, T, D]), head-major."""
    return QKNormRope.apply(x, weight, cos, sin, heads, eps)
