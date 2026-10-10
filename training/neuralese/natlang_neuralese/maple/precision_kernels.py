"""One-pass elementwise kernels of the precision ramp ``w + mix·(Q(w) − w)`` (maple/ternary.precision_value) on CUDA.

The eager rule makes ~15 full-size FP32 passes over a weight; on Mellum's experts that was ~72% of a q4 warm-up update
(plans/mellum-port.md "Warm-up step profile"). Here the per-block / per-row scales stay the eager reductions of the
rule (maple/ternary.py ``q4_scales`` / ``ternary_row_stats``, shared with export), and everything elementwise runs in one
kernel: BF16 latent in, BF16 value out.

Bit-exact with the eager rule by construction: the kernel performs the eager rule's FP32 operations one by one with
round-to-nearest (IEEE division ``div_rn``, no FMA contraction: ``enable_fp_fusion=False``) and the same casts.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 2048


@triton.jit
def _q4_ramp_kernel(w_ptr, d_ptr, out_ptr, n, mix, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n
    x = tl.load(w_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    d = tl.load(d_ptr + offsets // GROUP, mask=valid, other=0.0)
    inverse = tl.where(d != 0.0, tl.math.div_rn(tl.full(d.shape, 1.0, tl.float32), d), 0.0)
    q = tl.minimum(tl.floor(x * inverse + 8.5), 15.0) - 8.0
    quantized = q * d.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + offsets, (x + mix * (quantized - x)).to(out_ptr.dtype.element_ty), mask=valid)


@triton.jit
def _ternary_ramp_kernel(w_ptr, threshold_ptr, alpha_ptr, out_ptr, n, mix, COLUMNS: tl.constexpr,
                         BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n
    x = tl.load(w_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    row = offsets // COLUMNS
    threshold = tl.load(threshold_ptr + row, mask=valid, other=0.0)
    alpha = tl.load(alpha_ptr + row, mask=valid, other=0.0).to(tl.float32)
    keep = tl.abs(x) > threshold
    quantized = tl.where(keep, tl.where(x > 0.0, alpha, -alpha), 0.0)
    tl.store(out_ptr + offsets, (x + mix * (quantized - x)).to(out_ptr.dtype.element_ty), mask=valid)


def q4_ramp(weight: torch.Tensor, d: torch.Tensor, mix: float, group: int) -> torch.Tensor:
    """``weight + mix·(Q4_0(weight) − weight)`` in ``weight``'s dtype; ``d`` the FP32 per-block ``max / -8``."""
    w = weight.contiguous()
    out = torch.empty_like(w)
    n = w.numel()
    _q4_ramp_kernel[(triton.cdiv(n, BLOCK),)](w, d.contiguous(), out, n, float(mix), GROUP=group, BLOCK=BLOCK,
                                              enable_fp_fusion=False)
    return out


def ternary_ramp(weight: torch.Tensor, threshold: torch.Tensor, alpha: torch.Tensor, mix: float) -> torch.Tensor:
    """``weight + mix·(T(weight) − weight)`` in ``weight``'s dtype; per-row FP32 ``threshold`` and BF16 ``alpha``."""
    w = weight.contiguous()
    out = torch.empty_like(w)
    n = w.numel()
    _ternary_ramp_kernel[(triton.cdiv(n, BLOCK),)](w, threshold.contiguous(), alpha.contiguous(), out, n, float(mix),
                                                   COLUMNS=w.shape[-1], BLOCK=BLOCK, enable_fp_fusion=False)
    return out
