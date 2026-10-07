"""Fused ternary MoE for Maple's frozen experts: one grouped GEMM per projection instead of a Python loop over experts.

``SparseMoE``'s reference path loops over the experts (a GPU→CPU sync for the counts in every layer) and rebuilds
each routed expert's dense BF16 weight from its int8 codes on every call. For a 1024-token chunk that is every
expert of the layer; for a decode step it is 8 experts plus ~10 launches each. Here the routed (token, expert)
pairs are grouped on the GPU into blocks of one expert each, and a Triton kernel multiplies a block of rows by the
expert's codes, converting int8 codes to BF16 and applying the row scale in registers (the dense weight never
exists in memory). The weight values are exactly ``codes * scale`` as in the reference; only the accumulation order
differs.

Gradients: to the input (``dx = (dy) @ (codes * scale)``, the same kernel on the transposed view), and to learned
block scales (QAT, ``TernaryExperts.learn_scales``): ``ds[e, n, b] = sum_p dy[p, n] * sum_{k in b} x[p, k] *
codes[e, n, k]``, a grouped weight-gradient GEMM over each expert's rows whose tile is multiplied by the codes and
reduced over the scale block in registers (the dense weight gradient never exists). Codes stay frozen.
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
        if TRANS:
            c = tl.load(codes_ptr + expert * stride_ce + k_idx[:, None] * stride_inner + offs_n[None, :] * stride_outer,
                        mask=mask_w, other=0)
            s = tl.load(scale_ptr + expert * stride_se + k_idx[:, None] * stride_srow
                        + (offs_n[None, :] // scale_block) * stride_scol, mask=mask_w, other=0.0)
        else:  # tiles loaded [BN, BK] along the codes' contiguous columns (k), used transposed
            mask_t = n_mask[:, None] & (k_idx[None, :] < K)
            c = tl.load(codes_ptr + expert * stride_ce + offs_n[:, None] * stride_outer + k_idx[None, :] * stride_inner,
                        mask=mask_t, other=0)
            s = tl.load(scale_ptr + expert * stride_se + offs_n[:, None] * stride_srow
                        + (k_idx[None, :] // scale_block) * stride_scol, mask=mask_t, other=0.0)
        if SCALE_HALF:
            s = s.to(tl.float16)
        w = c.to(a.dtype) * s.to(a.dtype)  # the reference's weight values: codes * scale in the activation dtype
        acc += tl.dot(a, w if TRANS else tl.trans(w))
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc.to(out_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < P) & n_mask[None, :])


@triton.jit
def _grouped_scale_grad(a_ptr, rows_ptr, dy_ptr, codes_ptr, out_ptr, start_ptr, count_ptr,
                        N, K, stride_am, stride_ak, stride_dm, stride_dn, stride_ce, stride_cn, stride_ck,
                        stride_oe, stride_on, stride_ob,
                        HAS_ROWS: tl.constexpr, SB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                        BK: tl.constexpr):
    """out[e, n, b] = sum over expert e's routed rows p of dy[p, n] * sum_{k in block b} A[row(p), k] * codes[e, n, k].
    Expert e's rows are the padded positions start[e] .. start[e] + count[e] (``Plan``)."""
    expert = tl.program_id(0)
    pid_n = tl.program_id(1)
    block = tl.program_id(2)
    count = tl.load(count_ptr + expert)
    if count == 0:
        return
    start = tl.load(start_ptr + expert)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    total = tl.zeros((BN,), dtype=tl.float32)
    for kk in range(0, SB, BK):
        k_idx = block * SB + kk + tl.arange(0, BK)
        k_mask = k_idx < K
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for p0 in range(0, count, BM):
            local = p0 + tl.arange(0, BM)
            p_mask = local < count
            p = start + local
            if HAS_ROWS:
                rows = tl.load(rows_ptr + p, mask=p_mask, other=0)
            else:
                rows = p
            dy = tl.load(dy_ptr + p[:, None] * stride_dm + offs_n[None, :] * stride_dn,
                         mask=p_mask[:, None] & n_mask[None, :], other=0.0)
            a = tl.load(a_ptr + rows[:, None] * stride_am + k_idx[None, :] * stride_ak,
                        mask=p_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(tl.trans(dy), a)
        c = tl.load(codes_ptr + expert * stride_ce + offs_n[:, None] * stride_cn + k_idx[None, :] * stride_ck,
                    mask=n_mask[:, None] & k_mask[None, :], other=0)
        total += tl.sum(acc * c.to(tl.float32), axis=1)
    tl.store(out_ptr + expert * stride_oe + offs_n * stride_on + block * stride_ob, total, mask=n_mask)


class Plan:
    """Routed pairs grouped into BM-row blocks of one expert, computed on the GPU without host syncs.

    ``rows[p]``: the token feeding padded position p (``tokens`` = the zero row for padding); ``block_expert[b]``: the
    expert of block b, or -1; ``position[f]``: the padded position of flat routed pair f = token * top_k + slot."""

    def __init__(self, index: torch.Tensor, experts: int, tokens: int, block: int):
        flat = index.reshape(-1)
        pairs = flat.numel()
        top_k = index.shape[-1]
        device = flat.device
        # scatter_add, not bincount: CUDA bincount reads the max index back to the host (a sync per MoE layer).
        counts = torch.zeros(experts, dtype=torch.long, device=device).scatter_add_(0, flat, torch.ones_like(flat))
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
        self.top_k = top_k
        self.start, self.count = padded_start, counts  # expert e's rows: start[e] .. start[e] + count[e]


class Weights:
    """One projection of every expert: int8 codes [E, rows, cols] and the scale the reference multiplies them by,
    per row ([E, rows, 1], BF16) or per row and ``block`` columns ([E, rows, cols / block], FP32 rounded to FP16)."""

    def __init__(self, codes, scale, block=None):
        self.codes, self.scale = codes, scale
        self.block = block or codes.shape[-1]
        self.half = block is not None
        # Learned block scales with autograd on: the projection also returns their gradient.
        self.trainable = self.half and scale.requires_grad and torch.is_grad_enabled()


def _mm(a, rows, weights: Weights, plan: Plan, transpose: bool) -> torch.Tensor:
    """Grouped A @ W_eᵀ (forward; W_e = codes_e * scale_e is [N, K]) or A @ W_e (transpose: the input gradient)."""
    codes, scale = weights.codes, weights.scale
    E, R, C = codes.shape
    N, K = (R, C) if not transpose else (C, R)
    out = torch.empty(plan.size, N, device=a.device, dtype=a.dtype)
    # W[inner, outer]: forward inner = column (k), outer = row (n); transposed the other way round.
    inner, outer = (codes.stride(2), codes.stride(1)) if not transpose else (codes.stride(1), codes.stride(2))
    # Training-sized plans (128-row blocks): the kernel is bound by streaming each expert's int8 codes once per row
    # block, so larger blocks win; tiles measured on the GB10 for Maple's projections (scripts/bench_maple_moe.py,
    # 6270 tokens, top-8 of 256): 1.4-2.4x the earlier 64-row tiling.
    if plan.block >= 128:
        BM, BN, BK, tuning = (plan.block, 64, 32, dict(num_warps=8, num_stages=2)) if not transpose and K <= 512 else \
            (plan.block, 128, 32, dict(num_warps=4, num_stages=3 if transpose else 2))
    elif plan.block >= 64:
        BM, BN, BK, tuning = plan.block, 128, 64, dict(num_warps=4, num_stages=2)
    else:
        BM, BN, BK, tuning = plan.block, 64, 64, {}
    grid = (plan.blocks, triton.cdiv(N, BN))
    _grouped_ternary_mm[grid](
        a, rows if rows is not None else a, codes, scale, out, plan.block_expert,
        plan.size, N, K, a.stride(0), a.stride(1), codes.stride(0), inner, outer,
        scale.stride(0), scale.stride(1), scale.stride(2), weights.block, out.stride(0), out.stride(1),
        HAS_ROWS=rows is not None, TRANS=transpose, SCALE_HALF=weights.half, BM=BM, BN=BN, BK=BK, **tuning)
    return out


def _scale_grad(a, rows, grad, weights: Weights, plan: Plan) -> torch.Tensor:
    """Gradient of the block scales [E, N, K / block] (FP32; straight through the FP16 rounding of the forward)."""
    codes, scale = weights.codes, weights.scale
    E, N, K = codes.shape
    out = torch.zeros(scale.shape, device=grad.device, dtype=torch.float32)
    # Tiles measured on the GB10 (101,376 B shared memory) for Maple's projections: 1.36x (gate_up, N < K) and 1.54x
    # (down, N > K) over 64/64/32. A k-chunk never straddles two scale blocks.
    BN, BK, BM, tuning = (128, 32, 128, dict(num_warps=4, num_stages=2)) if N > K else \
        (64, 128, 32, dict(num_warps=4, num_stages=3))
    BK = min(BK, weights.block)
    grid = (E, triton.cdiv(N, BN), triton.cdiv(K, weights.block))
    _grouped_scale_grad[grid](
        a, rows if rows is not None else a, grad, codes, out, plan.start, plan.count,
        N, K, a.stride(0), a.stride(1), grad.stride(0), grad.stride(1), codes.stride(0), codes.stride(1),
        codes.stride(2), out.stride(0), out.stride(1), out.stride(2),
        HAS_ROWS=rows is not None, SB=weights.block, BM=BM, BN=BN, BK=BK, **tuning)
    return out


class _ExpertProjection(torch.autograd.Function):
    """y[p] = A[rows[p]] @ W_eᵀ; gradient to A (gathered rows are scattered back) and, when they train, to the learned
    block scales. Codes are frozen."""

    @staticmethod
    def forward(ctx, a, rows, weights, plan, scale):
        ctx.save_for_backward(a, rows)
        ctx.weights, ctx.plan, ctx.a_shape = weights, plan, a.shape
        return _mm(a, rows, weights, plan, transpose=False)

    @staticmethod
    def backward(ctx, grad):
        a, rows = ctx.saved_tensors
        grad = grad.contiguous()
        grad_scale = _scale_grad(a, rows, grad, ctx.weights, ctx.plan) if ctx.needs_input_grad[4] else None
        grad_rows = _mm(grad, None, ctx.weights, ctx.plan, transpose=True) if ctx.needs_input_grad[0] else None
        if grad_rows is None or rows is None:
            return grad_rows, None, None, None, grad_scale
        plan = ctx.plan
        tokens = plan.position.numel() // plan.top_k
        if ctx.a_shape[0] == tokens + 1:  # the padded token rows of fused_experts: gather each token's top_k rows
            summed = _fused(_gather_sum)(grad_rows, plan.position, tokens, plan.top_k)
            grad_a = torch.cat([summed, summed.new_zeros(1, summed.shape[-1])], 0)
        else:
            grad_a = torch.zeros(ctx.a_shape, device=grad.device, dtype=torch.float32)
            grad_a.index_add_(0, rows, grad_rows.float())
        return grad_a.to(grad.dtype), None, None, None, grad_scale


def expert_weights(experts) -> tuple[Weights, Weights] | None:
    """The weights the reference path would use (``TernaryExperts._scaled``); learned block scales train through the
    fused kernels."""
    from .ternary import STATE

    blocks = getattr(experts, "gate_up_blocks", None)
    if blocks is not None and STATE["enabled"]:
        return (Weights(experts.gate_up_codes, experts.gate_up_blocks, experts.block),
                Weights(experts.down_codes, experts.down_blocks, experts.block))
    return Weights(experts.gate_up_codes, experts.gate_up_scale), Weights(experts.down_codes, experts.down_scale)


def _gather_sum(src: torch.Tensor, position: torch.Tensor, tokens: int, top_k: int,
                weights: torch.Tensor | None = None) -> torch.Tensor:
    """sum over each token's top_k routed rows of src (FP32), optionally weighted: the combine of the expert outputs
    and, in the backward pass, the input gradient of the gathered rows (a gather, not an atomic scatter)."""
    picked = src[position].view(tokens, top_k, -1).float()
    if weights is not None:
        picked = picked * weights[..., None]
    return picked.sum(1)


def _swiglu(gate_up: torch.Tensor, ff: int, clamp: float) -> torch.Tensor:
    gate, up = gate_up[:, :ff], gate_up[:, ff:]
    return (F.silu(gate.clamp(max=clamp)) * up.clamp(-clamp, clamp)).contiguous()


_compiled = {}


def _fused(fn):
    """torch.compile'd elementwise/reduction glue around the expert kernels (one fused kernel instead of several
    full-size passes over memory); NATLANG_MAPLE_COMPILE=0 runs it eagerly."""
    if os.environ.get("NATLANG_MAPLE_COMPILE", "1") == "0":
        return fn
    if fn not in _compiled:
        _compiled[fn] = torch.compile(fn, dynamic=True)
    return _compiled[fn]


# Performance work on a live run: creating this file (e.g. via `docker exec`) saves the next training-sized call's
# inputs, routing and expert weights to NATLANG_MAPLE_MOE_DUMP_DIR for scripts/bench_maple_moe.py --dump.
_DUMP_REQUEST = os.environ.get('NATLANG_MAPLE_MOE_DUMP_REQUEST', '/tmp/natlang-moe-dump-request')


def _dump_call(x, index, gate_up_w, down_w):
    try:
        os.unlink(_DUMP_REQUEST)
    except FileNotFoundError:
        return
    directory = os.environ.get('NATLANG_MAPLE_MOE_DUMP_DIR', os.path.expanduser('~/natlang-moe-dump'))
    os.makedirs(directory, exist_ok=True)
    weight = lambda w: {'codes': w.codes.detach().cpu(), 'scale': w.scale.detach().cpu(), 'block': w.block,
                        'half': w.half}
    torch.save({'x': x.detach().cpu(), 'index': index.detach().cpu(), 'gate_up': weight(gate_up_w),
                'down': weight(down_w)}, os.path.join(directory, f'call-{x.shape[0]}-{os.getpid()}.pt'))


def fused_experts(experts, x: torch.Tensor, index: torch.Tensor, weights: torch.Tensor, clamp: float,
                  projections: tuple[Weights, Weights]) -> torch.Tensor:
    """sum_slot weight * expert(x) for routed (token, expert) pairs; ``experts`` is a ``TernaryExperts``."""
    tokens, top_k = index.shape
    pairs = tokens * top_k
    gate_up_w, down_w = projections
    if pairs > 512 and os.path.exists(_DUMP_REQUEST):
        _dump_call(x, index, gate_up_w, down_w)
    # Row blocks: each block streams its expert's codes once. Large calls (a whole training window) have ~100+ rows
    # per expert, where 128-row blocks halve that traffic (1.5-2x on a dumped warm-up call); small calls would
    # mostly pad them.
    plan = Plan(index, experts.gate_up_codes.shape[0], tokens, 16 if pairs <= 512 else 128 if pairs >= 32768 else 64)
    padded_x = torch.cat([x, x.new_zeros(1, x.shape[-1])], 0)  # the padding rows read zeros
    gate_up = _ExpertProjection.apply(padded_x, plan.rows, gate_up_w, plan, gate_up_w.scale if gate_up_w.trainable else None)
    h = _fused(_swiglu)(gate_up, experts.ff, clamp)
    out = _ExpertProjection.apply(h, None, down_w, plan, down_w.scale if down_w.trainable else None)
    return _fused(_gather_sum)(out, plan.position, tokens, top_k, weights)
