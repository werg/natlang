"""Fused ternary MoE for Maple's frozen experts: one grouped GEMM per projection instead of a Python loop over experts.

``SparseMoE``'s reference path loops over the experts (a GPU→CPU sync for the counts in every layer) and rebuilds
each routed expert's dense BF16 weight from its int8 codes on every call. For a 1024-token chunk that is every
expert of the layer; for a decode step it is 8 experts plus ~10 launches each. Here the routed (token, expert)
pairs are grouped on the GPU into blocks of one expert each, and a Triton kernel multiplies a block of rows by the
expert's codes, converting int8 codes to BF16 and applying the row scale in registers (the dense weight never
exists in memory). The weight values are exactly ``codes * scale`` as in the reference; only the accumulation order
differs.

Gradients: to the input (``dx = (dy) @ (codes * scale)``, the same kernel on the transposed view). Codes and row
scales are frozen buffers. Learned block scales (nested QAT, ``TernaryExperts.learn_scales``) keep the reference
path, which also gives the scale gradients.
"""
from __future__ import annotations

import os
import sysconfig
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _python_headers():
    """Triton compiles its launcher against Python.h. Where the system lacks the dev headers (the DGX: no root to
    install python3.12-dev), use the exact-version package unpacked under NATLANG_PYTHON_HEADERS (default
    ~/.local/pydev/root: `apt-get download libpython3.12-dev python3.12-dev` and `dpkg -x` each)."""
    if (Path(sysconfig.get_paths()["include"]) / "Python.h").exists():
        return
    root = Path(os.environ.get("NATLANG_PYTHON_HEADERS", Path.home() / ".local/pydev/root"))
    version = f"python{sysconfig.get_python_version()}"
    # pyconfig.h includes <ARCH-linux-gnu/pythonX.Y/pyconfig.h>, so the include root itself is on the path too.
    found = [str(p) for p in (root / "usr/include" / version, root / "usr/include") if p.exists()]
    if found:
        os.environ["CPATH"] = os.pathsep.join(found + [x for x in [os.environ.get("CPATH")] if x])


_python_headers()



@triton.jit
def _grouped_ternary_mm(a_ptr, rows_ptr, codes_ptr, scale_ptr, out_ptr, block_expert_ptr,
                        P, N, K, stride_am, stride_ak, stride_ce, stride_inner, stride_outer,
                        stride_se, stride_srow, stride_scol, scale_block,
                        stride_om, stride_on,
                        HAS_ROWS: tl.constexpr, TRANS: tl.constexpr, SCALE_HALF: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out[m, n] = sum_k A[row(m), k] * W_e[k, n] for the block's expert e. W_e is the expert matrix at (inner=k,
    outer=n): codes[e] times its scale, where the scale is indexed by the weight's own (row, column // scale_block):
    row = outer and column = inner in the forward pass, the reverse for the input gradient (TRANS)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.load(block_expert_ptr + pid_m)
    if expert < 0:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    if HAS_ROWS:
        rows = tl.load(rows_ptr + offs_m, mask=offs_m < P, other=0)
    else:
        rows = offs_m
    n_mask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k_idx = k0 + offs_k
        mask_w = (k_idx[:, None] < K) & n_mask[None, :]
        a = tl.load(a_ptr + rows[:, None] * stride_am + k_idx[None, :] * stride_ak,
                    mask=(offs_m[:, None] < P) & (k_idx[None, :] < K), other=0.0)
        c = tl.load(codes_ptr + expert * stride_ce + k_idx[:, None] * stride_inner + offs_n[None, :] * stride_outer,
                    mask=mask_w, other=0)
        if TRANS:
            s = tl.load(scale_ptr + expert * stride_se + k_idx[:, None] * stride_srow
                        + (offs_n[None, :] // scale_block) * stride_scol, mask=mask_w, other=0.0)
        else:
            s = tl.load(scale_ptr + expert * stride_se + offs_n[None, :] * stride_srow
                        + (k_idx[:, None] // scale_block) * stride_scol, mask=mask_w, other=0.0)
        if SCALE_HALF:
            s = s.to(tl.float16)
        w = c.to(a.dtype) * s.to(a.dtype)  # the reference's weight values: codes * scale in the activation dtype
        acc += tl.dot(a, w)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc.to(out_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < P) & n_mask[None, :])


class Plan:
    """Routed pairs grouped into BM-row blocks of one expert, computed on the GPU without host syncs.

    ``rows[p]``: the token feeding padded position p (``tokens`` = the zero row for padding); ``block_expert[b]``: the
    expert of block b, or -1; ``position[f]``: the padded position of flat routed pair f = token * top_k + slot."""

    def __init__(self, index: torch.Tensor, experts: int, tokens: int, block: int):
        flat = index.reshape(-1)
        pairs = flat.numel()
        top_k = index.shape[-1]
        device = flat.device
        counts = torch.bincount(flat, minlength=experts)
        padded = (counts + block - 1) // block * block
        padded_end = torch.cumsum(padded, 0)
        padded_start = padded_end - padded
        order = torch.argsort(flat, stable=True)
        sorted_expert = flat[order]
        sorted_start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(pairs, device=device) - sorted_start[sorted_expert]
        position = padded_start[sorted_expert] + rank
        self.blocks = -(-pairs // block) + min(experts, pairs)  # upper bound on non-empty blocks
        self.size = self.blocks * block
        self.rows = torch.full((self.size,), tokens, dtype=torch.long, device=device)
        self.rows[position] = order // top_k
        block_start = torch.arange(self.blocks, device=device) * block
        expert_of = torch.searchsorted(padded_end, block_start, right=True)
        valid = expert_of < experts
        expert_clamped = expert_of.clamp(max=experts - 1)
        valid &= block_start >= padded_start[expert_clamped]
        self.block_expert = torch.where(valid, expert_of, torch.full_like(expert_of, -1)).to(torch.int32)
        self.position = torch.empty_like(position)
        self.position[order] = position
        self.block = block


class Weights:
    """One projection of every expert: int8 codes [E, rows, cols] and the scale the reference multiplies them by,
    per row ([E, rows, 1], BF16) or per row and ``block`` columns ([E, rows, cols / block], FP32 rounded to FP16)."""

    def __init__(self, codes, scale, block=None):
        self.codes, self.scale = codes, scale
        self.block = block or codes.shape[-1]
        self.half = block is not None


def _mm(a, rows, weights: Weights, plan: Plan, transpose: bool) -> torch.Tensor:
    """Grouped A @ W_eᵀ (forward; W_e = codes_e * scale_e is [N, K]) or A @ W_e (transpose: the input gradient)."""
    codes, scale = weights.codes, weights.scale
    E, R, C = codes.shape
    N, K = (R, C) if not transpose else (C, R)
    out = torch.empty(plan.size, N, device=a.device, dtype=a.dtype)
    # W[inner, outer]: forward inner = column (k), outer = row (n); transposed the other way round.
    inner, outer = (codes.stride(2), codes.stride(1)) if not transpose else (codes.stride(1), codes.stride(2))
    BM, BN, BK = plan.block, 64, 64
    grid = (plan.blocks, triton.cdiv(N, BN))
    _grouped_ternary_mm[grid](
        a, rows if rows is not None else a, codes, scale, out, plan.block_expert,
        plan.size, N, K, a.stride(0), a.stride(1), codes.stride(0), inner, outer,
        scale.stride(0), scale.stride(1), scale.stride(2), weights.block, out.stride(0), out.stride(1),
        HAS_ROWS=rows is not None, TRANS=transpose, SCALE_HALF=weights.half, BM=BM, BN=BN, BK=BK)
    return out


class _ExpertProjection(torch.autograd.Function):
    """y[p] = A[rows[p]] @ W_eᵀ; gradient to A only (gathered rows are scattered back). Codes and scales are frozen."""

    @staticmethod
    def forward(ctx, a, rows, weights, plan):
        ctx.save_for_backward(rows)
        ctx.weights, ctx.plan, ctx.a_shape = weights, plan, a.shape
        return _mm(a, rows, weights, plan, transpose=False)

    @staticmethod
    def backward(ctx, grad):
        (rows,) = ctx.saved_tensors
        grad_rows = _mm(grad.contiguous(), None, ctx.weights, ctx.plan, transpose=True)
        if rows is None:
            return grad_rows, None, None, None
        grad_a = torch.zeros(ctx.a_shape, device=grad.device, dtype=torch.float32)
        grad_a.index_add_(0, rows, grad_rows.float())
        return grad_a.to(grad.dtype), None, None, None


def expert_weights(experts) -> tuple[Weights, Weights] | None:
    """The weights the reference path would use (``TernaryExperts._scaled``), or None when they are being trained
    (learned block scales with autograd on: the reference path gives their gradients)."""
    from .ternary import STATE

    blocks = getattr(experts, "gate_up_blocks", None)
    if blocks is not None and STATE["enabled"]:
        if torch.is_grad_enabled() and (blocks.requires_grad or experts.down_blocks.requires_grad):
            return None
        return (Weights(experts.gate_up_codes, experts.gate_up_blocks, experts.block),
                Weights(experts.down_codes, experts.down_blocks, experts.block))
    return Weights(experts.gate_up_codes, experts.gate_up_scale), Weights(experts.down_codes, experts.down_scale)


def fused_experts(experts, x: torch.Tensor, index: torch.Tensor, weights: torch.Tensor, clamp: float,
                  projections: tuple[Weights, Weights]) -> torch.Tensor:
    """sum_slot weight * expert(x) for routed (token, expert) pairs; ``experts`` is a ``TernaryExperts``."""
    tokens, top_k = index.shape
    pairs = tokens * top_k
    gate_up_w, down_w = projections
    plan = Plan(index, experts.gate_up_codes.shape[0], tokens, 16 if pairs <= 512 else 64)
    padded_x = torch.cat([x, x.new_zeros(1, x.shape[-1])], 0)  # the padding rows read zeros
    gate_up = _ExpertProjection.apply(padded_x, plan.rows, gate_up_w, plan)
    gate, up = gate_up[:, :experts.ff], gate_up[:, experts.ff:]
    h = F.silu(gate.clamp(max=clamp)) * up.clamp(-clamp, clamp)
    out = _ExpertProjection.apply(h.contiguous(), None, down_w, plan)
    picked = out[plan.position].view(tokens, top_k, -1)
    return (picked.float() * weights[..., None]).sum(1)
