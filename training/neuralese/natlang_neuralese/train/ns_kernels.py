"""Newton-Schulz GEMMs of MuonSR's cubic5 step (train/optim.py ``newton_schulz``) as two Triton kernels.

The folded eager step is ``M = a I + b X Xᵀ`` (``baddbmm``), then ``X <- M X`` (``bmm``), with X oriented wide
(rows <= cols); a tall latent (Mellum's expert ``down``, ``q_proj``) is transposed into that orientation and back,
two full copies of the stack per call. Here:

- ``gram``: ``a I + b X Xᵀ`` computes only the upper-triangle tiles (the product is symmetric) and writes each one
  and its mirror, so the Gram half of the step does ~57% of the GEMM work on Mellum's 896-row experts; FP32
  accumulation, one rounding to BF16, as ``baddbmm``.
- ``matmul``: a strided batched GEMM. A tall latent O (X = Oᵀ) never moves: its Gram is ``Oᵀ O`` read through
  swapped strides, and the step is ``O <- O M`` (M symmetric).

Same mathematics and roundings as the eager folded step up to FP32 accumulation order
(tests/neuralese/test_ns_kernels.py). ``NATLANG_MUON_NS_TRITON=0`` restores the eager path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

GRAM_TILE = 128


def _gram_configs():
    return [triton.Config({'BK': bk}, num_warps=w, num_stages=s) for bk in (32, 64) for w in (4, 8) for s in (3, 4)]


@triton.autotune(configs=_gram_configs(), key=['M', 'K'])
@triton.jit
def _gram_kernel(x_ptr, out_ptr, M, K, sxb, sxm, sxk, a, b, T: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    i = t // T
    j = t % T
    if i > j:
        return
    batch = tl.program_id(1).to(tl.int64)
    base = x_ptr + batch * sxb
    offs_i = i * BM + tl.arange(0, BM)
    offs_j = j * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    mi = offs_i < M
    mj = offs_j < M
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + offs_k
        mk = ks < K
        left = tl.load(base + offs_i[:, None] * sxm + ks[None, :] * sxk, mask=mi[:, None] & mk[None, :], other=0.0)
        right = tl.load(base + offs_j[None, :] * sxm + ks[:, None] * sxk, mask=mk[:, None] & mj[None, :], other=0.0)
        acc = tl.dot(left, right, acc)
    acc = acc * b
    if i == j:
        acc += tl.where(offs_i[:, None] == offs_j[None, :], a, 0.0)
    out = out_ptr + batch * M * M
    value = acc.to(tl.bfloat16)
    tl.store(out + offs_i[:, None] * M + offs_j[None, :], value, mask=mi[:, None] & mj[None, :])
    if i != j:
        tl.store(out + offs_j[:, None] * M + offs_i[None, :], tl.trans(value), mask=mj[:, None] & mi[None, :])


def _mm_configs():
    return [triton.Config({'BP': bp, 'BQ': bq, 'BK': bk, 'GROUP': 8}, num_warps=w, num_stages=s)
            for bp, bq in ((128, 128), (128, 256), (256, 128)) for bk in (32, 64) for w in (4, 8) for s in (3, 4)
            if not (bp * bq >= 32768 and w == 4)]


@triton.autotune(configs=_mm_configs(), key=['P', 'Q', 'K', 'A_ROWMAJOR', 'B_ROWMAJOR'])
@triton.jit
def _mm_kernel(a_ptr, b_ptr, c_ptr, P, Q, K, sab, sap, sak, sbb, sbk, sbq, A_ROWMAJOR: tl.constexpr,
               B_ROWMAJOR: tl.constexpr, BP: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    tiles_p = tl.cdiv(P, BP)
    tiles_q = tl.cdiv(Q, BQ)
    per_group = GROUP * tiles_q
    group = pid // per_group
    first = group * GROUP
    size = tl.minimum(tiles_p - first, GROUP)
    pid_p = first + (pid % per_group) % size
    pid_q = (pid % per_group) // size
    batch = tl.program_id(1).to(tl.int64)
    offs_p = pid_p * BP + tl.arange(0, BP)
    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = tl.arange(0, BK)
    mp = offs_p < P
    mq = offs_q < Q
    a_base = a_ptr + batch * sab
    b_base = b_ptr + batch * sbb
    acc = tl.zeros((BP, BQ), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + offs_k
        mk = ks < K
        left = tl.load(a_base + offs_p[:, None] * sap + ks[None, :] * sak, mask=mp[:, None] & mk[None, :], other=0.0)
        right = tl.load(b_base + ks[:, None] * sbk + offs_q[None, :] * sbq, mask=mk[:, None] & mq[None, :], other=0.0)
        acc = tl.dot(left, right, acc)
    tl.store(c_ptr + batch * P * Q + offs_p[:, None] * Q + offs_q[None, :], acc.to(tl.bfloat16),
             mask=mp[:, None] & mq[None, :])


def gram(x: torch.Tensor, a: float, b: float, *, transposed: bool = False) -> torch.Tensor:
    """``a I + b X Xᵀ`` (BF16, [B, m, m]) for a BF16 stack ``x`` [B, m, n]; ``transposed``: ``x`` is [B, n, m] and
    X is its transpose (the Gram is ``xᵀ x``)."""
    B = x.shape[0]
    if transposed:
        K, M = x.shape[1], x.shape[2]
        sxm, sxk = x.stride(2), x.stride(1)
    else:
        M, K = x.shape[1], x.shape[2]
        sxm, sxk = x.stride(1), x.stride(2)
    out = torch.empty(B, M, M, device=x.device, dtype=torch.bfloat16)
    T = triton.cdiv(M, GRAM_TILE)
    _gram_kernel[(T * T, B)](x, out, M, K, x.stride(0), sxm, sxk, float(a), float(b), T=T, BM=GRAM_TILE)
    return out


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched ``a @ b`` for BF16 stacks of any strides (a [B, P, K], b [B, K, Q]); a contiguous BF16 result."""
    B, P, K = a.shape
    Q = b.shape[2]
    c = torch.empty(B, P, Q, device=a.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(P, meta['BP']) * triton.cdiv(Q, meta['BQ']), B)
    _mm_kernel[grid](a, b, c, P, Q, K, a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2),
                     A_ROWMAJOR=a.stride(2) == 1, B_ROWMAJOR=b.stride(2) == 1)
    return c


def cubic5_steps(x: torch.Tensor, coefficients) -> torch.Tensor:
    """The folded cubic steps on a normalized BF16 stack ``x`` [B, rows, cols] in its stored orientation: wide
    (rows <= cols) ``X <- (a I + b X Xᵀ) X``; tall ``O <- O (a I + b Oᵀ O)`` (the same step on X = Oᵀ)."""
    tall = x.shape[-2] > x.shape[-1]
    for a, b in coefficients:
        m = gram(x, a, b, transposed=tall)
        x = matmul(x, m) if tall else matmul(m, x)
    return x
