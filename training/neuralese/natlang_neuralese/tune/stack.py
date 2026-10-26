"""Lexical stack composition and exact export factors (§3.2, §5.3).

``W = W0 + sum_f a[f] D_f + sum_n s[n] D_n``: family members carry their binding
gate and lexical nodes their fixed deployment scale; both are static per
binding. Subspace members slicing one canonical basis sum in that basis (rank at
most ``max r``); everything else concatenates (rank at most the sum).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import TuneMathError
from .bases import BasisSlice, verify_nested
from .parameterization import SubspaceDelta

LEVELS = ('family', 'tree', 'folder', 'lambda', 'experiment')


@dataclass
class StackMember:
    node: str
    level: str
    scale: float
    deltas: dict = field(default_factory=dict)  # parameter name -> SubspaceDelta | FreeLoRADelta


@dataclass(frozen=True)
class TargetFactors:
    B: torch.Tensor | None   # (..., rows, rank); None for an explicit zero contribution
    A: torch.Tensor | None   # (..., rank, cols)
    rank: int
    mode: str                # 'zero' | 'common-basis' | 'concatenated'
    contributors: tuple

    def dense(self, shape):
        return torch.zeros(shape) if self.B is None else self.B @ self.A


def _check(members):
    nodes = [member.node for member in members]
    if len(set(nodes)) != len(nodes):
        raise TuneMathError('a stack names a node twice')
    for member in members:
        if member.level not in LEVELS or type(member.scale) not in (int, float) or not math.isfinite(member.scale):
            raise TuneMathError(f'stack member {member.node} has an invalid level or scale')


def stack_factors(members, bases, targets):
    """``targets`` maps every target parameter name to its weight shape; unknown deltas are rejected."""
    _check(members)
    unknown = {name for member in members for name in member.deltas} - set(targets)
    if unknown:
        raise TuneMathError(f'stack deltas name unknown targets: {sorted(unknown)}')
    out = {}
    for name, shape in targets.items():
        parts = [(member, member.deltas[name]) for member in members if name in member.deltas and member.scale != 0]
        if not parts:
            out[name] = TargetFactors(None, None, 0, 'zero', ())
            continue
        groups, free = {}, []
        for member, delta in parts:
            (groups.setdefault(delta.basis_id, []) if isinstance(delta, SubspaceDelta) else free).append((member, delta))
        Bs, As = [], []
        for group in groups.values():
            basis = verify_nested([BasisSlice(delta.basis_id, delta.rank) for _, delta in group], bases)
            top = max(delta.rank for _, delta in group)
            left, right = basis.slice(top)
            core = None
            for member, delta in group:
                padded = torch.nn.functional.pad(delta.core.to(left.device, torch.float32),
                                                 (0, top-delta.rank, 0, top-delta.rank)) * member.scale
                core = padded if core is None else core + padded
            Bs.append(left @ core)
            As.append(right.transpose(-2, -1))
        for member, delta in free:
            B, A = delta.factors(bases)
            Bs.append(B * member.scale)
            As.append(A)
        B, A = torch.cat(Bs, -1), torch.cat(As, -2)
        if tuple(B.shape[:-1]) != tuple(shape[:-1]) or tuple(A.shape[:-2]) + A.shape[-1:] != tuple(shape[:-2]) + tuple(shape[-1:]):
            raise TuneMathError(f'stack factors do not match target geometry for {name}')
        mode = 'common-basis' if len(groups) == 1 and not free else 'concatenated'
        out[name] = TargetFactors(B, A, B.shape[-1], mode, tuple(member.node for member, _ in parts))
    return out


def stack_dense(members, bases, targets):
    return {name: factors.dense(targets[name]) for name, factors in stack_factors(members, bases, targets).items()}


def effective_weights(base_weights, members, bases):
    targets = {name: tuple(weight.shape) for name, weight in base_weights.items()}
    deltas = stack_dense(members, bases, targets)
    return {name: weight.float() + deltas[name].to(weight.device) for name, weight in base_weights.items()}
