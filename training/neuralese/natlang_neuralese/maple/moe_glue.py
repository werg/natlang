"""Permutation glue of the grouped dense MoE (maple/fused_moe.py ``dense_grouped_experts``) without scatters.

The eager glue gathered every token's row once per routed slot (``x[order // top_k]``), whose backward is a sort-based
scatter-add (``indexing_backward_kernel``), un-sorted the expert outputs (``out[argsort(order)]``, again a scatter in
backward) and combined them through an FP32 copy of all [tokens, top_k, width] outputs. On Mellum at 32k tokens this
glue was about a third of the MoE time (plans/neuralese/TRAINING_SPEED_20261012.md). Here, with ``slot_of`` the
inverse permutation (``slot_of[t * top_k + k]`` = the sorted position of token t's k-th route):

- ``gather_rows``: forward the same gather; backward ``grad_x[t] = sum_k grad_rows[slot_of[t, k]]`` (FP32, slot order),
  one Triton program per token, no scatter;
- ``combine``: forward ``y[t] = sum_k w[t, k] * out[slot_of[t, k]]`` in FP32, slot order (the eager product and sum);
  backward ``grad_out[slot_of[t, k]] = round(g[t] * w[t, k])`` written to its (unique) sorted row and
  ``grad_w[t, k] = sum_d g[t, d] * out[slot_of[t, k], d]``, one program per token.

Same values as eager up to FP32 summation order (tests/neuralese/test_moe_glue.py).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 512


@triton.jit
def _combine_forward(out_ptr, slot_ptr, w_ptr, y_ptr, D, TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    for d0 in range(0, D, BLOCK):
        offs = d0 + tl.arange(0, BLOCK)
        mask = offs < D
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(TOP_K):
            slot = tl.load(slot_ptr + t * TOP_K + k).to(tl.int64)
            w = tl.load(w_ptr + t * TOP_K + k)
            row = tl.load(out_ptr + slot * D + offs, mask=mask, other=0.0).to(tl.float32)
            acc += row * w
        tl.store(y_ptr + t * D + offs, acc, mask=mask)


@triton.jit
def _combine_backward(g_ptr, out_ptr, slot_ptr, w_ptr, grad_out_ptr, grad_w_ptr, D, TOP_K: tl.constexpr,
                      BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, TOP_K)
    dots = tl.zeros((TOP_K,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK):
        offs = d0 + tl.arange(0, BLOCK)
        mask = offs < D
        g = tl.load(g_ptr + t * D + offs, mask=mask, other=0.0).to(tl.float32)
        for k in tl.static_range(TOP_K):
            slot = tl.load(slot_ptr + t * TOP_K + k).to(tl.int64)
            w = tl.load(w_ptr + t * TOP_K + k)
            row = tl.load(out_ptr + slot * D + offs, mask=mask, other=0.0).to(tl.float32)
            dots += tl.where(ks == k, tl.sum(g * row, axis=0), 0.0)
            tl.store(grad_out_ptr + slot * D + offs, (g * w).to(grad_out_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_w_ptr + t * TOP_K + ks, dots)


@triton.jit
def _expert_of(end_ptr, row, E):
    """The expert of expert-sorted row ``row`` from the int32 group ends (at most 128 experts)."""
    ends = tl.load(end_ptr + tl.arange(0, 128), mask=tl.arange(0, 128) < E, other=2 ** 31 - 1)
    return tl.sum((ends <= row).to(tl.int32), axis=0).to(tl.int64)


@triton.jit
def _combine_backward_quant(g_ptr, out_ptr, slot_ptr, w_ptr, grad_out_ptr, grad_w_ptr, cs_ptr, end_ptr, q_ptr, s_ptr,
                            D, E, TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    """``_combine_backward`` plus each written row's FP8 quantization for the down projection's input gradient
    (``expert_gemm.scaled_rowwise_quant`` of the row times its expert's column scales), without re-reading the rows."""
    t = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, TOP_K)
    dots = tl.zeros((TOP_K,), dtype=tl.float32)
    amax = tl.zeros((TOP_K,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK):
        offs = d0 + tl.arange(0, BLOCK)
        mask = offs < D
        g = tl.load(g_ptr + t * D + offs, mask=mask, other=0.0).to(tl.float32)
        for k in tl.static_range(TOP_K):
            slot = tl.load(slot_ptr + t * TOP_K + k).to(tl.int64)
            w = tl.load(w_ptr + t * TOP_K + k)
            row = tl.load(out_ptr + slot * D + offs, mask=mask, other=0.0).to(tl.float32)
            dots += tl.where(ks == k, tl.sum(g * row, axis=0), 0.0)
            value = (g * w).to(grad_out_ptr.dtype.element_ty)
            tl.store(grad_out_ptr + slot * D + offs, value, mask=mask)
            cs = tl.load(cs_ptr + _expert_of(end_ptr, slot, E) * D + offs, mask=mask, other=0.0)
            amax = tl.maximum(amax, tl.where(ks == k, tl.max(tl.abs(value.to(tl.float32) * cs), axis=0), 0.0))
    tl.store(grad_w_ptr + t * TOP_K + ks, dots)
    scales = tl.maximum(amax, 1e-30) / 448.0
    for k in tl.static_range(TOP_K):
        slot = tl.load(slot_ptr + t * TOP_K + k).to(tl.int64)
        w = tl.load(w_ptr + t * TOP_K + k)
        expert = _expert_of(end_ptr, slot, E)
        scale = tl.sum(tl.where(ks == k, scales, 0.0), axis=0)
        for d0 in range(0, D, BLOCK):
            offs = d0 + tl.arange(0, BLOCK)
            mask = offs < D
            g = tl.load(g_ptr + t * D + offs, mask=mask, other=0.0).to(tl.float32)
            value = (g * w).to(grad_out_ptr.dtype.element_ty).to(tl.float32)
            cs = tl.load(cs_ptr + expert * D + offs, mask=mask, other=0.0)
            tl.store(q_ptr + slot * D + offs, (value * cs / scale).to(q_ptr.dtype.element_ty), mask=mask)
        tl.store(s_ptr + slot, scale)


@triton.jit
def _gather_backward(grad_rows_ptr, slot_ptr, grad_x_ptr, D, TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    for d0 in range(0, D, BLOCK):
        offs = d0 + tl.arange(0, BLOCK)
        mask = offs < D
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(TOP_K):
            slot = tl.load(slot_ptr + t * TOP_K + k).to(tl.int64)
            acc += tl.load(grad_rows_ptr + slot * D + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_x_ptr + t * D + offs, acc.to(grad_x_ptr.dtype.element_ty), mask=mask)


class GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, source, slot_of, top_k):
        ctx.save_for_backward(slot_of)
        ctx.top_k, ctx.tokens = top_k, x.shape[0]
        return x[source]

    @staticmethod
    def backward(ctx, grad_rows):
        (slot_of,) = ctx.saved_tensors
        grad_rows = grad_rows.contiguous()
        D = grad_rows.shape[-1]
        grad_x = torch.empty(ctx.tokens, D, device=grad_rows.device, dtype=grad_rows.dtype)
        if ctx.tokens:
            _gather_backward[(ctx.tokens,)](grad_rows, slot_of, grad_x, D, TOP_K=ctx.top_k, BLOCK=BLOCK)
        return grad_x, None, None, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, slot_of, weights, hand=None):
        out = out.contiguous()
        tokens, top_k = weights.shape
        D = out.shape[-1]
        w = weights.float().contiguous()
        y = torch.empty(tokens, D, device=out.device, dtype=torch.float32)
        if tokens:
            _combine_forward[(tokens,)](out, slot_of, w, y, D, TOP_K=top_k, BLOCK=BLOCK,
                                         enable_fp_fusion=False)  # the eager product, then the sum
        ctx.save_for_backward(out, slot_of, w)
        ctx.weights_dtype = weights.dtype
        ctx.hand = hand
        return y

    @staticmethod
    def backward(ctx, grad):
        out, slot_of, w = ctx.saved_tensors
        tokens, top_k = w.shape
        D = out.shape[-1]
        g = grad.float().contiguous()
        grad_out = torch.empty_like(out)
        grad_w = torch.empty(tokens, top_k, device=out.device, dtype=torch.float32)
        hand = ctx.hand
        if hand is not None and hand.col_scale is not None:  # quantize the rows for the down projection's dgrad
            from .fp8_experts import FP8
            q = torch.empty_like(out, dtype=FP8)
            scale = torch.empty(out.shape[0], device=out.device, dtype=torch.float32)
            if tokens:
                _combine_backward_quant[(tokens,)](g, out, slot_of, w, grad_out, grad_w, hand.col_scale, hand.ends,
                                                   q, scale, D, hand.col_scale.shape[0], TOP_K=top_k, BLOCK=BLOCK)
            hand.give(grad_out, q, scale)
        elif tokens:
            _combine_backward[(tokens,)](g, out, slot_of, w, grad_out, grad_w, D, TOP_K=top_k, BLOCK=BLOCK)
        return grad_out, None, grad_w.to(ctx.weights_dtype), None


def routing_permutation(index: torch.Tensor):
    """(order, slot_of): the routed (token, slot) pairs sorted by expert, and each pair's sorted position."""
    flat = index.reshape(-1)
    order = flat.argsort()
    slot_of = torch.empty_like(order)
    slot_of[order] = torch.arange(order.numel(), device=order.device)
    return order, slot_of.to(torch.int32).contiguous()



@triton.jit
def _swiglu_forward(h_ptr, out_ptr, F, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    valid = offs < F
    g = tl.load(h_ptr + row * 2 * F + offs, mask=valid, other=0.0).to(tl.float32)
    u = tl.load(h_ptr + row * 2 * F + F + offs, mask=valid, other=0.0).to(tl.float32)
    silu = (g / (1.0 + libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * F + offs, (silu * u).to(tl.bfloat16), mask=valid)


@triton.jit
def _swiglu_backward(grad_ptr, h_ptr, out_ptr, cs_ptr, end_ptr, q_ptr, s_ptr, F, E, QUANT: tl.constexpr,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    valid = offs < F
    grad = tl.load(grad_ptr + row * F + offs, mask=valid, other=0.0).to(tl.float32)
    g = tl.load(h_ptr + row * 2 * F + offs, mask=valid, other=0.0).to(tl.float32)
    u = tl.load(h_ptr + row * 2 * F + F + offs, mask=valid, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + libdevice.exp(-g))
    silu = (g / (1.0 + libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)  # the forward's saved value
    grad_up = (grad * silu).to(tl.bfloat16)  # mul backward: one rounding each
    grad_silu = (grad * u).to(tl.bfloat16).to(tl.float32)
    grad_gate = grad_silu * (sig * (1.0 + g * (1.0 - sig)))  # silu_backward in FP32, one rounding
    grad_gate = grad_gate.to(tl.bfloat16)
    tl.store(out_ptr + row * 2 * F + offs, grad_gate, mask=valid)
    tl.store(out_ptr + row * 2 * F + F + offs, grad_up, mask=valid)
    if QUANT:  # the gate_up projection's dgrad operand: (row * its expert's column scales) in FP8, per-row scale
        expert = _expert_of(end_ptr, row, E)
        cs_gate = tl.load(cs_ptr + expert * 2 * F + offs, mask=valid, other=0.0)
        cs_up = tl.load(cs_ptr + expert * 2 * F + F + offs, mask=valid, other=0.0)
        x_gate = grad_gate.to(tl.float32) * cs_gate
        x_up = grad_up.to(tl.float32) * cs_up
        scale = tl.maximum(tl.maximum(tl.max(tl.abs(x_gate), axis=0), tl.max(tl.abs(x_up), axis=0)), 1e-30) / 448.0
        tl.store(q_ptr + row * 2 * F + offs, (x_gate / scale).to(q_ptr.dtype.element_ty), mask=valid)
        tl.store(q_ptr + row * 2 * F + F + offs, (x_up / scale).to(q_ptr.dtype.element_ty), mask=valid)
        tl.store(s_ptr + row, scale)


class SwiGLU(torch.autograd.Function):
    """``silu(gate) * up`` of a contiguous BF16 [rows, 2F] (gate | up) projection (unclamped, as Mellum), one pass
    each way. The eager slices' backward materialized two zero-padded [rows, 2F] gradients and added them; here the
    gradient's two halves are written in place. Same roundings as eager (the silu rounded, the product rounded; in
    backward each product rounded, silu_backward in FP32)."""

    @staticmethod
    def forward(ctx, hidden, ff, hand=None):
        hidden = hidden.contiguous()
        out = torch.empty(hidden.shape[0], ff, device=hidden.device, dtype=hidden.dtype)
        if hidden.shape[0]:
            _swiglu_forward[(hidden.shape[0],)](hidden, out, ff, BLOCK=triton.next_power_of_2(ff),
                                                enable_fp_fusion=False)
        ctx.save_for_backward(hidden)
        ctx.ff, ctx.hand = ff, hand
        return out

    @staticmethod
    def backward(ctx, grad):
        (hidden,) = ctx.saved_tensors
        grad = grad.contiguous()
        out = torch.empty_like(hidden)
        hand = ctx.hand
        quant = hand is not None and hand.col_scale is not None
        if quant:
            from .fp8_experts import FP8
            q = torch.empty_like(hidden, dtype=FP8)
            scale = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32)
            cs, ends, experts = hand.col_scale, hand.ends, hand.col_scale.shape[0]
        else:
            q = scale = cs = ends = out
            experts = 0
        if hidden.shape[0]:
            _swiglu_backward[(hidden.shape[0],)](grad, hidden, out, cs, ends, q, scale, ctx.ff, experts, QUANT=quant,
                                                 BLOCK=triton.next_power_of_2(ctx.ff), enable_fp_fusion=False)
        if quant:
            hand.give(out, q, scale)
        return out, None, None


def swiglu(hidden: torch.Tensor, ff: int, hand=None) -> torch.Tensor:
    """``hand``: an ``expert_gemm.GradQuant`` of the projection that produced ``hidden``; the backward then also
    quantizes the gradient it writes for that projection's input gradient."""
    return SwiGLU.apply(hidden, ff, hand)
