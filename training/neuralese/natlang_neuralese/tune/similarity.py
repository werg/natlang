"""Gauge-invariant effective-delta similarity for family discovery (§5.5).

Cosine over raw factors is invalid: ``BA = (BQ)(Q^-1 A)``. Inner products of
effective deltas use Gram products, ``<B1 A1, B2 A2>_F = sum((B1^T B2) * (A1 A2^T))``,
without dense expansion.
"""
from __future__ import annotations

import torch

from . import TuneMathError
from .gates import factor_sq_norm


def factor_inner(B1, A1, B2, A2):
    return ((B1.transpose(-2, -1) @ B2) * (A1 @ A2.transpose(-2, -1))).sum()


def delta_cosine(first, second, q):
    """q-weighted cosine of two adapters given as ``{m: (B, A)}``; absent matrices contribute zero."""
    names = set(first) | set(second)
    if names - set(q):
        raise TuneMathError(f'similarity lacks q weights for {sorted(names - set(q))}')
    inner = sum((q[name] * factor_inner(*first[name], *second[name]) for name in set(first) & set(second)),
                torch.zeros((), dtype=torch.float64))
    norm1 = sum((q[name] * factor_sq_norm(*first[name]) for name in first), torch.zeros((), dtype=torch.float64))
    norm2 = sum((q[name] * factor_sq_norm(*second[name]) for name in second), torch.zeros((), dtype=torch.float64))
    if float(norm1) <= 0 or float(norm2) <= 0:
        raise TuneMathError('similarity of a zero delta is undefined')
    return inner / torch.sqrt(norm1 * norm2)
