"""One-pass elementwise kernels of MuonSR's step (train/optim.py ``MuonSR._update``) for BF16 CUDA latents.

The eager step made ~25 full-size passes over each latent outside Newton-Schulz (FP32 copies of the gradient, the
momentum, the Nesterov direction and the weight, and the stochastic rounding's integer arithmetic): about 6 s of a
Mellum update. Here:

- ``momentum_nesterov``: ``m = round_bf16(m * beta + g)`` (the stored momentum) and the Nesterov direction
  ``round_bf16(g + beta * m)`` (Newton-Schulz's BF16 input), from the BF16 gradient and momentum, in one pass;
- ``apply_update``: ``w = SR(w * factor + alpha * d)`` with the per-matrix decay factor and the active mask folded into
  ``factor`` and ``d``, and the stochastic rounding's noise drawn by the caller with ``torch.randint_like`` exactly as
  the eager ``stochastic_round_`` draws it.

Same FP32 operations as the eager step with the same roundings: separate products and sums where eager runs separate
kernels (``mul_rn``/``add_rn``), a fused multiply-add where eager's ``add_(x, alpha=)`` computes one, so the result is
bit-identical (tests/neuralese/test_muon_kernels.py).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 2048


@triton.jit
def _momentum_kernel(g_ptr, m_ptr, out_ptr, n, beta, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n
    g = tl.load(g_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    m = libdevice.mul_rn(m, beta)  # the eager mul_ then add_: two roundings
    m = libdevice.add_rn(m, g)
    tl.store(m_ptr + offsets, m.to(tl.bfloat16), mask=valid)
    nesterov = tl.fma(m, beta, g)  # the eager add_(m, alpha=beta): one fused multiply-add
    tl.store(out_ptr + offsets, nesterov.to(tl.bfloat16), mask=valid)


@triton.jit
def _apply_kernel(w_ptr, d_ptr, factor_ptr, active_ptr, noise_ptr, n, matrix, alpha, HAS_FACTOR: tl.constexpr,
                  HAS_ACTIVE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n
    w = tl.load(w_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    d = tl.load(d_ptr + offsets, mask=valid, other=0.0)
    if HAS_ACTIVE:  # the eager BF16 product direction * active
        d = (d.to(tl.float32) * tl.load(active_ptr + offsets // matrix, mask=valid, other=0.0)).to(tl.bfloat16)
    d = d.to(tl.float32)
    if HAS_FACTOR:
        factor = tl.load(factor_ptr + offsets // matrix, mask=valid, other=1.0)
        w = libdevice.mul_rn(w, factor)
    w = tl.fma(d, alpha, w)  # the eager add_(d, alpha=alpha)
    noise = tl.load(noise_ptr + offsets, mask=valid, other=0)
    bits = (w.to(tl.int32, bitcast=True) + noise) & -65536
    tl.store(w_ptr + offsets, bits.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=valid)


def momentum_nesterov(grad: torch.Tensor, momentum: torch.Tensor, beta: float) -> torch.Tensor:
    """Updates ``momentum`` (BF16) in place; returns the BF16 Nesterov direction."""
    out = torch.empty_like(momentum)
    n = grad.numel()
    _momentum_kernel[(triton.cdiv(n, BLOCK),)](grad, momentum, out, n, float(beta), BLOCK=BLOCK,
                                               enable_fp_fusion=False)
    return out


def apply_update(weight: torch.Tensor, direction: torch.Tensor, factor: torch.Tensor | None,
                 active: torch.Tensor | None, alpha: float) -> None:
    """``weight`` (BF16) <- SR(weight * factor + alpha * (direction * active)); ``factor`` and ``active`` are FP32 per
    matrix of a stack (flattened), or None."""
    n = weight.numel()
    matrix = weight.shape[-2] * weight.shape[-1]
    noise = torch.randint_like(weight, 0, 1 << 16, dtype=torch.int32)  # as stochastic_round_ draws it
    _apply_kernel[(triton.cdiv(n, BLOCK),)](weight, direction, factor if factor is not None else weight,
                                            active if active is not None else weight, noise, n, matrix, float(alpha),
                                            HAS_FACTOR=factor is not None, HAS_ACTIVE=active is not None,
                                            BLOCK=BLOCK, enable_fp_fusion=False)
