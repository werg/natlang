"""FP8 grouped expert GEMMs for the dense (BF16-latent) Mellum experts (owner 2026-10-12: real FP8 expert GEMMs, numerics
may change; ``torch._scaled_grouped_mm`` is not built for sm_121, so our own Triton kernel).

One launch per projection over all experts: row blocks of ``BM`` rows of one expert (block table built on the GPU, no
host sync), E4M3 operands with per-row scales applied in the epilogue, FP32 accumulation, BF16 out.

- forward: activations quantized per row once (the gate/up projection quantizes the [tokens, d] input before the
  routing gather and gathers FP8 rows inside the kernel: 8x less quantization than the routed rows), weights per output
  row;
- input gradient: incoming gradient per row, weights per input column (``columnwise_quant_t``), the routed-row result
  summed back per token by the scatter-free glue (maple/moe_glue.py);
- weight gradient: BF16 grouped GEMM straight into the parameter layout.

Measured on the GB10 (16k x 2304 x 1792): this kernel's FP8 GEMM 173 TFLOP/s, Triton BF16 97, cuBLASLt FP8 111, cuBLAS
BF16 59 (plans/neuralese/TRAINING_SPEED_20261012.md).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fp8_experts import FP8, columnwise_quant_t, rowwise_quant

BM, BN, BK, WARPS, STAGES = 128, 128, 64, 4, 3


@triton.jit
def _grouped_fp8_mm(a_ptr, rows_ptr, sa_ptr, b_ptr, sb_ptr, c_ptr, block_expert_ptr, block_row_ptr, end_ptr,
                    N, R, stride_am, stride_be, stride_bn, stride_sbe, stride_cm,
                    HAS_ROWS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C[m, n] = sa[src(m)] * sb[e, n] * sum_r A[src(m), r] * B[e, n, r] for the rows m of block b (expert e)."""
    pid = tl.program_id(0)
    tiles_n = tl.cdiv(N, BN)
    block = pid // tiles_n
    pid_n = pid % tiles_n
    expert = tl.load(block_expert_ptr + block)
    if expert < 0:
        return
    row0 = tl.load(block_row_ptr + block)
    end = tl.load(end_ptr + expert)
    offs_m = row0 + tl.arange(0, BM)
    m_mask = offs_m < end
    if HAS_ROWS:
        src = tl.load(rows_ptr + offs_m, mask=m_mask, other=0).to(tl.int64)
    else:
        src = offs_m.to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BK)
    a_p = a_ptr + src[:, None] * stride_am + offs_k[None, :]
    b_p = b_ptr + expert.to(tl.int64) * stride_be + offs_n[None, :].to(tl.int64) * stride_bn + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, R, BK):
        k_mask = (k0 + offs_k) < R
        a = tl.load(a_p, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_p, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_p += BK
        b_p += BK
    sa = tl.load(sa_ptr + src, mask=m_mask, other=0.0)
    sb = tl.load(sb_ptr + expert * stride_sbe + offs_n, mask=n_mask, other=0.0)
    out = acc * sa[:, None] * sb[None, :]
    tl.store(c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :], out.to(c_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class BlockTable:
    """Row blocks of ``BM`` rows of one expert over the expert-sorted rows (``offsets``: int32 group ends)."""

    def __init__(self, offsets: torch.Tensor, rows: int, block: int = BM):
        experts = offsets.numel()
        ends = offsets.long()
        starts = ends - torch.diff(ends, prepend=ends.new_zeros(1))
        counts = ends - starts
        blocks = (counts + block - 1) // block
        cumulative = torch.cumsum(blocks, 0)
        self.size = rows // block + experts  # upper bound on non-empty blocks
        index = torch.arange(self.size, device=offsets.device)
        expert = torch.searchsorted(cumulative, index, right=True)
        valid = expert < experts
        expert_c = expert.clamp(max=experts - 1)
        first = cumulative[expert_c] - blocks[expert_c]
        self.block_expert = torch.where(valid, expert_c, torch.full_like(expert_c, -1)).to(torch.int32)
        self.block_row = (starts[expert_c] + (index - first) * block).to(torch.int32)
        self.ends = ends.to(torch.int32).contiguous()


def grouped_fp8(aq, sa, rows, bq, sb, table: BlockTable, out_rows: int) -> torch.Tensor:
    """[out_rows, N] BF16: routed row m = (aq[rows[m]] * sa[rows[m]]) @ (bq[e] * sb[e])ᵀ, ``bq`` [E, N, R] E4M3."""
    E, N, R = bq.shape
    out = torch.empty(out_rows, N, device=aq.device, dtype=torch.bfloat16)
    if out_rows:
        grid = (table.size * triton.cdiv(N, BN),)
        _grouped_fp8_mm[grid](aq, rows if rows is not None else aq, sa, bq, sb, out, table.block_expert,
                              table.block_row, table.ends, N, R, aq.stride(0), bq.stride(0), bq.stride(1),
                              sb.stride(0), out.stride(0), HAS_ROWS=rows is not None, BM=BM, BN=BN, BK=BK,
                              num_warps=WARPS, num_stages=STAGES)
    return out


class Fp8ExpertProjection(torch.autograd.Function):
    """Routed rows (``rows``: source row of each expert-sorted row, or None when ``a`` is already in that order)
    times each expert's ``weight`` [E, N, K]ᵀ. Backward: the input gradient through FP8 (summed back per source row
    via ``slot_of`` when ``rows`` is given), the weight gradient in BF16 in the parameter layout."""

    @staticmethod
    def forward(ctx, a, rows, slot_of, top_k, weight, table, offsets):
        E, N, K = weight.shape
        aq, sa = rowwise_quant(a)
        wq, sw = rowwise_quant(weight.reshape(E * N, K))
        out_rows = rows.numel() if rows is not None else a.shape[0]
        ctx.save_for_backward(a, rows, slot_of, weight, offsets)
        ctx.table, ctx.top_k = table, top_k
        return grouped_fp8(aq, sa, rows, wq.view(E, N, K), sw.view(E, N), table, out_rows)

    @staticmethod
    def backward(ctx, grad):
        a, rows, slot_of, weight, offsets = ctx.saved_tensors
        grad = grad.contiguous()
        grad_a = grad_weight = None
        if ctx.needs_input_grad[0]:
            gq, sg = rowwise_quant(grad)
            wtq, swt = columnwise_quant_t(weight)  # [E, K, N], scales [E, K]
            grad_rows = grouped_fp8(gq, sg, None, wtq, swt, ctx.table, grad.shape[0])
            if rows is None:
                grad_a = grad_rows
            else:
                from .moe_glue import BLOCK as GLUE_BLOCK, _gather_backward
                grad_a = torch.empty_like(a)
                if a.shape[0]:
                    _gather_backward[(a.shape[0],)](grad_rows, slot_of, grad_a, grad_rows.shape[-1],
                                                    TOP_K=ctx.top_k, BLOCK=GLUE_BLOCK)
        if ctx.needs_input_grad[4]:
            routed = a[rows] if rows is not None else a
            grad_weight = torch._grouped_mm(grad.t(), routed, offs=offsets)
        return grad_a, None, None, None, grad_weight, None, None


def fp8_dense_experts(x, index, weights, gate_up, down, clamp):
    """``dense_grouped_experts`` with both projections on FP8 tensor cores (NATLANG_MOE_FP8)."""
    import torch.nn.functional as F

    from .moe_glue import Combine, routing_permutation
    tokens, top_k = index.shape
    experts, ff = gate_up.shape[0], down.shape[-1]
    order, slot_of = routing_permutation(index)
    flat = index.reshape(-1)
    offsets = torch.searchsorted(flat[order], torch.arange(experts, device=flat.device), right=True).to(torch.int32)
    table = BlockTable(offsets, flat.numel())
    source = (order // top_k).contiguous()
    hidden = Fp8ExpertProjection.apply(x.contiguous(), source, slot_of, top_k, gate_up, table, offsets)
    gate, up = hidden[:, :ff], hidden[:, ff:]
    if clamp is not None:
        gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
    out = Fp8ExpertProjection.apply((F.silu(gate) * up).contiguous(), None, None, top_k, down, table, offsets)
    return Combine.apply(out, slot_of, weights)
