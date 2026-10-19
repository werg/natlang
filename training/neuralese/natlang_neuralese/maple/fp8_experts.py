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
        from ..model.native_gradient_partitions import active_gradient_partitions
        ctx.partitions = active_gradient_partitions()
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
        deferred = False
        if ctx.needs_input_grad[1]:
            if ctx.carrier and ctx.anchors and ctx.scope.defer_expert_chunk(
                    ctx.scope_weight, grad, rows, None,
                    torch.cat((offsets.new_zeros(1), offsets[:-1])),
                    offsets - torch.cat((offsets.new_zeros(1), offsets[:-1])),
                    rows.shape[0], ctx.partitions, ctx.anchors):
                deferred = True
                grad_weight = None
            else:
                grad_weight = (grouped_weight_gradient(grad, rows, offsets, weight.shape) if ctx.carrier else
                               torch._grouped_mm(grad.t(), rows, offs=offsets))
        anchor_grads = (None,) * len(ctx.anchors)
        if deferred:
            # A unit edge schedules the shared anchor after all chunk routes
            # have arrived; the actual dW is flushed there in invocation order.
            anchor_grads = tuple(grad.new_ones((), dtype=torch.float32) for anchor in ctx.anchors)
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


def pack_grouped_expert_chunks(records, shape):
    """Restore one invocation's unpadded expert-row order from token chunks.

    Each record is ``(grad, activation, row_indices, starts, counts, pair_count)``.
    ``starts``/``counts`` describe the expert-major padded plan for that chunk;
    padding is removed before records are joined. The output tensors are packed
    once in the original stable token order, so the unchanged grouped dW kernel
    sees the same expert-local row stream and 32-row tiles as an unchunked call.

    All count/index operations stay on the operand device. In particular this
    helper does not call ``.item()`` or copy routing metadata to the host.
    Scratch is one invocation's packed operands, never another parameter-sized
    FP32 accumulator.
    """
    if not records:
        raise ValueError('cannot pack an empty grouped-expert invocation')
    experts, n, k = shape
    device = records[0][0].device
    dtype_g, dtype_a = records[0][0].dtype, records[0][1].dtype
    if any(record[0].device != device or record[1].device != device or
           record[0].dtype != dtype_g or record[1].dtype != dtype_a for record in records):
        raise RuntimeError('grouped expert chunks changed device or operand dtype within one invocation')
    if any(record[0].ndim != 2 or record[0].shape[1] != n or
           record[1].ndim != 2 or record[1].shape[1] != k for record in records):
        raise RuntimeError('grouped expert chunk operand geometry changed within one invocation')
    pair_count = sum(int(record[5]) for record in records)
    if pair_count <= 0:
        return (records[0][0].new_empty((0, n)), records[0][1].new_empty((0, k)),
                torch.zeros(experts, device=device, dtype=torch.int32))
    counts = torch.stack([record[4].to(device=device, dtype=torch.long) for record in records])
    if counts.ndim != 2 or counts.shape[1] != experts:
        raise RuntimeError('grouped expert chunk count geometry differs from its weight bank')
    totals = counts.sum(dim=0)
    starts_out = torch.cumsum(totals, dim=0) - totals
    chunk_prefix = torch.cumsum(counts, dim=0) - counts
    packed_grad = torch.zeros((pair_count, n), device=device, dtype=dtype_g)
    packed_rows = torch.zeros((pair_count, k), device=device, dtype=dtype_a)
    for chunk_index, (grad, activation, row_indices, starts, _counts, _pairs) in enumerate(records):
        padded_rows = grad.shape[0]
        if padded_rows == 0:
            continue
        row = torch.arange(padded_rows, device=device, dtype=torch.long)
        for expert in range(experts):
            count = counts[chunk_index, expert]
            valid = row < count
            source = (starts[expert].to(torch.long) + row).clamp(max=padded_rows - 1)
            destination = (starts_out[expert] + chunk_prefix[chunk_index, expert] + row).clamp(
                max=pair_count - 1)
            gradient_values = torch.where(valid[:, None], grad[source], torch.zeros((), device=device, dtype=dtype_g))
            if row_indices is None:
                activation_values = activation[source]
            else:
                token_rows = row_indices[source].to(torch.long)
                activation_values = activation[token_rows]
            activation_values = torch.where(valid[:, None], activation_values,
                                            torch.zeros((), device=device, dtype=dtype_a))
            # Invalid padded rows add exact zeros at destination 0. Valid
            # expert rows have unique destinations, so these cannot alter the
            # packed sequence or its values.
            packed_grad.index_add_(0, destination, gradient_values)
            packed_rows.index_add_(0, destination, activation_values)
    ends = torch.cumsum(totals, dim=0).to(torch.int32)
    return packed_grad, packed_rows, ends


@triton.jit
def _grouped_weight_gradient_chunk_tiles(g_ptrs, a_ptrs, row_ptrs, starts_ptr, counts_ptr, out_ptr,
                                         N: tl.constexpr, K: tl.constexpr, E: tl.constexpr, C,
                                         SG0: tl.constexpr, SG1: tl.constexpr,
                                         SA0: tl.constexpr, SA1: tl.constexpr,
                                         SR: tl.constexpr,
                                         HAS_ROWS: tl.constexpr, FP32_INPUTS: tl.constexpr,
                                         GTYPE: tl.constexpr, ATYPE: tl.constexpr, RTYPE: tl.constexpr,
                                         B: tl.constexpr):
    """One grouped dW tile stream, transparently assembled across FFN chunks.

    The device-side count/start table removes per-chunk plan padding. For each
    expert, the kernel maps a global expert-local row tile into the chunk that
    owns each row, assembles the whole 32-row operands in registers, and runs
    the same single dot and ordered FP32 add as `_grouped_weight_gradient`.
    """
    expert = tl.program_id(0)
    n = tl.program_id(1) * B + tl.arange(0, B)
    k = tl.program_id(2) * B + tl.arange(0, B)
    n_mask, k_mask = n < N, k < K
    total = 0
    for chunk in range(C):
        total += tl.load(counts_ptr + chunk * E + expert)
    acc = tl.zeros((B, B), dtype=tl.float32)
    for tile in range(tl.cdiv(total, B)):
        p = tile * B + tl.arange(0, B)
        g_tile = tl.zeros((B, B), dtype=GTYPE)
        a_tile = tl.zeros((B, B), dtype=ATYPE)
        prefix = 0
        for chunk in range(C):
            count = tl.load(counts_ptr + chunk * E + expert)
            start = tl.load(starts_ptr + chunk * E + expert)
            local = p - prefix
            valid = (local >= 0) & (local < count)
            source = start + local
            g_raw = tl.load(g_ptrs + chunk).to(tl.pointer_type(GTYPE))
            a_raw = tl.load(a_ptrs + chunk).to(tl.pointer_type(ATYPE))
            g = tl.load(g_raw + source[None, :] * SG0 + n[:, None] * SG1,
                        mask=valid[None, :] & n_mask[:, None], other=0)
            if HAS_ROWS:
                rows_raw = tl.load(row_ptrs + chunk).to(tl.pointer_type(RTYPE))
                token = tl.load(rows_raw + source * SR, mask=valid, other=0)
                a = tl.load(a_raw + token[:, None] * SA0 + k[None, :] * SA1,
                            mask=valid[:, None] & k_mask[None, :], other=0)
            else:
                a = tl.load(a_raw + source[:, None] * SA0 + k[None, :] * SA1,
                            mask=valid[:, None] & k_mask[None, :], other=0)
            g_tile = tl.where(valid[None, :] & n_mask[:, None], g, g_tile)
            a_tile = tl.where(valid[:, None] & k_mask[None, :], a, a_tile)
            prefix += count
        if FP32_INPUTS:
            left, right = g_tile.to(tl.float32), a_tile.to(tl.float32)
        else:
            left, right = g_tile, a_tile
        acc += tl.dot(left, right, input_precision='ieee')
    tl.store(out_ptr + expert * N * K + n[:, None] * K + k[None, :], acc,
             mask=n_mask[:, None] & k_mask[None, :])


def grouped_weight_gradient_chunk_tiles(records, shape):
    """Run the original ordered expert-tile dW over chunk-backed operands.

    This is the GPU path used by the cotangent anchor. It stores only small
    pointer/offset tables and one local 32x32 operand tile per program; long
    invocations are never concatenated into full activation/gradient buffers.
    """
    if not records:
        raise ValueError('cannot reduce an empty grouped-expert invocation')
    experts, n, k = shape
    first_grad, first_activation = records[0]['gradient'], records[0]['activation']
    device = first_grad.device
    if device.type != 'cuda':
        packed_grad, packed_rows, ends = pack_grouped_expert_chunks([
            (item['gradient'], item['activation'], item['row_indices'], item['starts'],
             item['counts'], item['pair_count']) for item in records
        ], shape)
        return grouped_weight_gradient(packed_grad, packed_rows, ends, shape)
    has_rows = records[0]['row_indices'] is not None
    if any((item['gradient'].device != device or item['activation'].device != device or
            (item['row_indices'] is not None) != has_rows or
            item['gradient'].dtype != first_grad.dtype or item['activation'].dtype != first_activation.dtype or
            tuple(item['gradient'].stride()) != tuple(first_grad.stride()) or
            tuple(item['activation'].stride()) != tuple(first_activation.stride())) for item in records):
        raise RuntimeError('expert chunk operands changed device, route kind, dtype, or layout')
    starts = torch.stack([item['starts'].to(device=device, dtype=torch.int32) for item in records]).contiguous()
    counts = torch.stack([item['counts'].to(device=device, dtype=torch.int32) for item in records]).contiguous()
    grad_ptrs = torch.tensor([item['gradient'].data_ptr() for item in records], device=device, dtype=torch.int64)
    activation_ptrs = torch.tensor([item['activation'].data_ptr() for item in records], device=device, dtype=torch.int64)
    row_ptrs = torch.tensor([0 if item['row_indices'] is None else item['row_indices'].data_ptr()
                             for item in records], device=device, dtype=torch.int64)
    result = torch.empty(shape, device=device, dtype=torch.float32)
    gtype = tl.float32 if first_grad.dtype == torch.float32 else (
        tl.float16 if first_grad.dtype == torch.float16 else tl.bfloat16)
    atype = tl.float32 if first_activation.dtype == torch.float32 else (
        tl.float16 if first_activation.dtype == torch.float16 else tl.bfloat16)
    fp32_inputs = first_grad.dtype == torch.float32 or first_activation.dtype == torch.float32
    if gtype != atype and not fp32_inputs:
        raise RuntimeError('grouped dW dot operands must share dtype unless both are promoted to FP32')
    row_dtype = next((item['row_indices'].dtype for item in records if item['row_indices'] is not None), None)
    rtype = tl.int64 if row_dtype == torch.int64 else tl.int32
    _grouped_weight_gradient_chunk_tiles[(experts, triton.cdiv(n, 32), triton.cdiv(k, 32))](
        grad_ptrs, activation_ptrs, row_ptrs, starts, counts, result, n, k, experts, len(records),
        *first_grad.stride(), *first_activation.stride(),
        0 if not has_rows else records[0]['row_indices'].stride(0),
        HAS_ROWS=has_rows, FP32_INPUTS=fp32_inputs, GTYPE=gtype, ATYPE=atype, RTYPE=rtype, B=32,
        num_warps=4, num_stages=2)
    return result
