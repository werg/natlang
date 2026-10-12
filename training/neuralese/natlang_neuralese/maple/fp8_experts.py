"""Grouped expert projections for the dense (BF16-latent) Mellum experts, with weight gradients in parameter layout and
an optional FP8 path (owner 2026-10-12: real FP8 expert GEMMs; numerics may change).

``GroupedLinear.apply(rows, weight, offsets, fp8)``: ``rows`` [P, K] are routed rows sorted by expert, ``offsets`` the
int32 group ends (as ``torch._grouped_mm`` takes them), ``weight`` [E, N, K]; returns [P, N] = rows_e @ weight[e]ᵀ.

- The weight gradient is computed as one grouped GEMM over the row dimension straight into [E, N, K]
  (``_grouped_mm(gradᵀ, rows, offs)``). Autograd through ``weight.transpose(-2, -1)`` produced it transposed, and
  AccumulateGrad copied every expert latent's gradient into parameter layout (~1.25 s per Mellum update).
- ``fp8``: the forward and the input gradient run on FP8 tensor cores (cuBLASLt ``torch._scaled_mm`` per expert:
  ``_scaled_grouped_mm`` is not built for sm_121): E4M3 operands with per-row scales (rows of the activations and of
  the incoming gradient; per output row of the weight in the forward, per input column in the input gradient), FP32
  accumulation, BF16 out. The weight gradient stays BF16. 1.66x the BF16 GEMM rate on the GB10 (64 vs 38 TFLOP/s).
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


@triton.jit
def _columnwise_quant_t(w_ptr, q_ptr, scale_ptr, N, K, FMAX: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """W [E, N, K] -> Q [E, K, N] (transposed) with one scale per (e, k) over n."""
    e = tl.program_id(0).to(tl.int64)
    offs_k = tl.program_id(1) * BK + tl.arange(0, BK)
    k_mask = offs_k < K
    amax = tl.zeros((BK,), dtype=tl.float32)
    for n0 in range(0, N, BN):
        offs_n = n0 + tl.arange(0, BN)
        w = tl.load(w_ptr + e * N * K + offs_n[:, None] * K + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(w), axis=0))
    scale = tl.maximum(amax, 1e-30) / FMAX
    for n0 in range(0, N, BN):
        offs_n = n0 + tl.arange(0, BN)
        mask = (offs_n[:, None] < N) & k_mask[None, :]
        w = tl.load(w_ptr + e * N * K + offs_n[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
        tl.store(q_ptr + e * K * N + offs_k[None, :] * N + offs_n[:, None], (w / scale[None, :]).to(q_ptr.dtype.element_ty),
                 mask=mask)
    tl.store(scale_ptr + e * K + offs_k, scale, mask=k_mask)


def columnwise_quant_t(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """W [E, N, K] BF16 -> (E4M3 Wᵀ [E, K, N], FP32 scales [E, K]): the input-gradient operand W_e as [K, N] rows."""
    w = w.contiguous()
    E, N, K = w.shape
    q = torch.empty(E, K, N, device=w.device, dtype=FP8)
    scale = torch.empty(E, K, device=w.device, dtype=torch.float32)
    _columnwise_quant_t[(E, triton.cdiv(K, 64))](w, q, scale, N, K, FMAX=FP8_MAX, BN=64, BK=64)
    return q, scale


def _fp8_grouped(aq, sa, bq, sb, ends: list[int]) -> torch.Tensor:
    """out[s:e] = (aq[s:e] * sa) @ (bq[e] * sb[e])ᵀ per group: ``aq`` [P, K] E4M3 with row scales ``sa`` [P], ``bq``
    [E, N, K] E4M3 with row scales ``sb`` [E, N]; cuBLASLt FP8 per expert, written in place, BF16 out."""
    E, N, K = bq.shape
    out = torch.empty(aq.shape[0], N, device=aq.device, dtype=torch.bfloat16)
    start = 0
    for expert, end in enumerate(ends):
        if end > start:
            torch._scaled_mm(aq[start:end], bq[expert].t(), scale_a=sa[start:end, None], scale_b=sb[expert][None],
                             out_dtype=torch.bfloat16, out=out[start:end])
        start = end
    if start < aq.shape[0]:
        out[start:] = 0
    return out


class GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, weight, offsets, fp8):
        ctx.save_for_backward(rows, weight, offsets)
        ctx.fp8 = fp8
        if fp8:
            ctx.ends = offsets.tolist()
            E, N, K = weight.shape
            bq, sb = rowwise_quant(weight.reshape(E * N, K))
            return _fp8_grouped(*rowwise_quant(rows), bq.view(E, N, K), sb.view(E, N), ctx.ends)
        return torch._grouped_mm(rows, weight.transpose(-2, -1), offs=offsets)

    @staticmethod
    def backward(ctx, grad):
        rows, weight, offsets = ctx.saved_tensors
        grad = grad.contiguous()
        grad_rows = grad_weight = None
        if ctx.needs_input_grad[0]:
            if ctx.fp8:  # grad @ W_e: W_e as [K, N] rows (scaled per input column of the forward)
                grad_rows = _fp8_grouped(*rowwise_quant(grad), *columnwise_quant_t(weight), ctx.ends)
            else:
                grad_rows = torch._grouped_mm(grad, weight, offs=offsets)
        if ctx.needs_input_grad[1]:
            grad_weight = torch._grouped_mm(grad.t(), rows, offs=offsets)  # [E, N, K]: the parameter's layout
        return grad_rows, grad_weight, None, None
