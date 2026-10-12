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
    def forward(ctx, out, slot_of, weights):
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
        return y

    @staticmethod
    def backward(ctx, grad):
        out, slot_of, w = ctx.saved_tensors
        tokens, top_k = w.shape
        D = out.shape[-1]
        g = grad.float().contiguous()
        grad_out = torch.empty_like(out)
        grad_w = torch.empty(tokens, top_k, device=out.device, dtype=torch.float32)
        if tokens:
            _combine_backward[(tokens,)](g, out, slot_of, w, grad_out, grad_w, D, TOP_K=top_k, BLOCK=BLOCK)
        return grad_out, None, grad_w.to(ctx.weights_dtype)


def routing_permutation(index: torch.Tensor):
    """(order, slot_of): the routed (token, slot) pairs sorted by expert, and each pair's sorted position."""
    flat = index.reshape(-1)
    order = flat.argsort()
    slot_of = torch.empty_like(order)
    slot_of[order] = torch.arange(order.numel(), device=order.device)
    return order, slot_of.to(torch.int32).contiguous()
