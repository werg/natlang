"""Per-binding family gates and the frozen family amplitude contract (§5.4).

Gates live in ``[0, a_max]`` with an L1 penalty; zero is an exact removable
membership. A family's amplitude ``||D_f||_q`` is fixed at ``rho_f`` while gates
train, so shrinking a gate cannot be undone by inflating the family factors.
"""
from __future__ import annotations

import math

import torch

from . import TuneMathError


def project_gates(gates, a_max=1.0):
    """In-place projection of every gate tensor into ``[0, a_max]``."""
    if not math.isfinite(a_max) or a_max <= 0:
        raise TuneMathError('a_max must be positive and finite')
    with torch.no_grad():
        for value in gates.values():
            value.clamp_(0., a_max)
    return gates


def gate_l1(gates, tau):
    """``tau * mean_lambda sum_f a[lambda,f]`` over bindings with at least one family."""
    values = [value.sum() for value in gates.values() if value.numel()]
    return tau * torch.stack(values).mean() if values else torch.zeros(())


def q_weights(shapes):
    """Default matrix-size normalization: inverse element count per target."""
    return {name: 1. / math.prod(shape) for name, shape in shapes.items()}


def factor_sq_norm(B, A):
    """``||B A||_F^2`` summed over leading (expert) axes without expanding the dense delta."""
    return ((B.transpose(-2, -1) @ B) * (A @ A.transpose(-2, -1))).sum()


def family_norm(factors, q):
    """``sqrt(sum_m q_m ||D_fm||_F^2)`` from ``{m: (B, A)}`` factors."""
    missing = set(factors) - set(q)
    if missing:
        raise TuneMathError(f'family norm lacks q weights for {sorted(missing)}')
    return torch.sqrt(sum(q[name] * factor_sq_norm(B, A) for name, (B, A) in factors.items()))


def normalize_family(factors, rho, q, *, absent_below=1e-12):
    """Scale the family's left factors so its q-norm is exactly ``rho``; a near-zero family is absent."""
    if not math.isfinite(rho) or rho <= 0:
        raise TuneMathError('family amplitude rho must be positive and finite')
    norm = float(family_norm(factors, q))
    if norm < absent_below:
        raise TuneMathError('family has no useful nonzero fit; treat it as absent')
    scale = rho / norm
    return {name: (B * scale, A) for name, (B, A) in factors.items()}, scale
