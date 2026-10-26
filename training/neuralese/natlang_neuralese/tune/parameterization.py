"""The two delta forms for one target matrix, including batched expert banks.

Native subspace: ``U_r C V_r^T`` with an FP32 ``r x r`` core on a canonical basis
slice. Free LoRA: ``(alpha/r) B A`` with ``B = 0`` and seeded nonzero ``A`` so the
initial delta is zero while ``B`` still receives gradients. ``factors`` folds any
scaling exactly once and returns ``(B, A)`` with ``delta = B @ A``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import TuneMathError


@dataclass
class SubspaceDelta:
    basis_id: str
    rank: int
    core: torch.Tensor  # (..., r, r) FP32

    def factors(self, bases):
        basis = bases.get(self.basis_id)
        if basis is None:
            raise TuneMathError(f'subspace delta names unavailable basis {self.basis_id}')
        left, right = basis.slice(self.rank)
        if self.core.dtype != torch.float32 or tuple(self.core.shape[-2:]) != (self.rank, self.rank):
            raise TuneMathError('subspace core must be an FP32 rank x rank matrix')
        core = self.core.to(left.device)
        return left @ core, right.transpose(-2, -1)


@dataclass
class FreeLoRADelta:
    B: torch.Tensor  # (..., rows, r)
    A: torch.Tensor  # (..., r, cols)
    alpha: float

    @property
    def rank(self):
        return self.A.shape[-2]

    def factors(self, bases=None):
        if self.B.shape[-1] != self.A.shape[-2] or not math.isfinite(self.alpha):
            raise TuneMathError('free LoRA factors have inconsistent rank or invalid alpha')
        return self.B * (self.alpha/self.rank), self.A


def dense(delta, bases=None):
    left, right = delta.factors(bases)
    return left @ right


def init_subspace(basis, rank, leading_shape=()):
    basis.slice(rank)
    return SubspaceDelta(basis.basis_id, rank, torch.zeros((*leading_shape, rank, rank), dtype=torch.float32))


def init_free(shape, rank, *, alpha=None, seed=0):
    """``shape`` is the target weight's (..., rows, cols)."""
    *leading, rows, cols = shape
    if type(rank) is not int or not 1 <= rank <= min(rows, cols):
        raise TuneMathError('free LoRA rank must fit the target matrix')
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn((*leading, rank, cols), generator=generator, dtype=torch.float32) / math.sqrt(cols)
    return FreeLoRADelta(torch.zeros((*leading, rows, rank), dtype=torch.float32), A,
                         float(rank if alpha is None else alpha))
