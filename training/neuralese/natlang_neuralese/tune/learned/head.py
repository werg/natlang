"""`coefficient-head-v1`: a zero-initialized readout over each available slot's decision hidden state.

The decision position of slot ``k`` is the position, after the slot cue, whose logits choose its sign token. The head
reads that position's final hidden state, so its outputs do not depend on which sign is then sampled. Per slot:

    rho_k = tanh(u_k) / 2                              refinement in grid steps
    raw_kg = beta * tanh(v_kg); delta_kg = raw_kg - sum_g w_kg raw_kg     group modulation, weighted mean zero
    c_kg = step * sign(n_k) * (|n_k| + rho_k) * (1 + delta_kg),  c_kg = 0 exactly when n_k = 0

``w_k`` is each target group's share of component ``k``'s update norm (from the bank). Centering can double the
range of ``raw``, so ``beta <= 1/2`` keeps ``1 + delta > 0``: a nonzero coefficient keeps its visible sign, and its
weighted mean strength ``step*(|n|+rho)`` rounds to its visible integer. With the zero-initialized output layer the
executed coefficients equal the token-only grid values exactly. The declared head-off mode uses ``rho = delta = 0``.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError

HEAD_SCHEMA = 'coefficient-head-v1'


class CoefficientHead(nn.Module):
    def __init__(self, width, groups, *, beta=.5):
        super().__init__()
        if type(groups) is not int or groups < 1:
            raise LearnedOptimizerError('the coefficient head needs at least one target group')
        if type(beta) is not float or not 0 <= beta <= .5:
            raise LearnedOptimizerError('the modulation bound beta must lie in [0, 1/2] so centered modulation keeps signs')
        self.groups, self.beta = groups, beta
        # FP32 master; routed to AdamW with the other native ordinary parameters.
        self.readout = nn.Linear(width, 1 + groups, dtype=torch.float32)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def config(self):
        return {'schema': HEAD_SCHEMA, 'width': self.readout.in_features, 'groups': self.groups, 'beta_hex': self.beta.hex()}

    def identity(self):
        values = [value.detach().cpu().double().reshape(-1).tolist() for value in (self.readout.weight, self.readout.bias)]
        return 'chd1:' + canonical_json_sha256_hex({**self.config(), 'values': [[float(x).hex() for x in v] for v in values]})

    def forward(self, hidden):
        """``hidden`` (K, width) at the slots' decision positions -> ``(u, v)`` of shapes (K,), (K, G)."""
        out = self.readout(hidden.float())
        return out[..., 0], out[..., 1:]


def refinement(u):
    return torch.tanh(u) / 2


def modulation(v, shares, beta):
    """Centered bounded modulation; ``shares`` (K, G) are nonnegative and sum to one per slot."""
    if shares.shape != v.shape:
        raise LearnedOptimizerError('group shares and modulation outputs differ in shape')
    raw = beta * torch.tanh(v)
    return raw - (shares * raw).sum(-1, keepdim=True)


def executed(integers, u, v, shares, *, step, beta, head=True):
    """Executed coefficients (K, G) for realized integers. ``head=False`` is the declared head-off mode."""
    n = torch.as_tensor(integers, dtype=torch.float32, device=shares.device)
    if not head:
        return (step * n)[:, None].expand(-1, shares.shape[-1]).clone()
    rho, delta = refinement(u), modulation(v, shares, beta)
    magnitude = torch.where(n != 0, n.abs() + rho, torch.zeros_like(rho))
    return step * torch.sign(n)[:, None] * magnitude[:, None] * (1 + delta)


def expected(mean, signed, u, v, shares, *, step, beta, head=True):
    """`token-expectation-v1` coefficients: ``step * (E[n] + rho (p(n>0) - p(n<0))) * (1 + delta)``."""
    if not head:
        return (step * mean)[:, None].expand(-1, shares.shape[-1]).clone()
    rho, delta = refinement(u), modulation(v, shares, beta)
    return step * (mean + rho * signed)[:, None] * (1 + delta)


def receipt(head_module, integers, u, v, coefficients, *, mode):
    if mode not in ('head', 'head-off'):
        raise LearnedOptimizerError('coefficient-head mode is head or head-off')
    hexes = lambda tensor: [float(x).hex() for x in tensor.detach().double().reshape(-1).tolist()]
    return {'schema': HEAD_SCHEMA, 'mode': mode, 'head': head_module.identity() if head_module is not None else None,
            'integers': [int(x) for x in integers], 'u': hexes(u) if u is not None else None,
            'v': hexes(v) if v is not None else None, 'shape': list(coefficients.shape), 'executed': hexes(coefficients)}


def shares_from_norms(norms):
    """Per-component group shares from (K, G) nonnegative update norms; a zero component gets uniform shares."""
    norms = torch.as_tensor(norms, dtype=torch.float32)
    if bool((norms < 0).any()) or not bool(torch.isfinite(norms).all()):
        raise LearnedOptimizerError('group update norms must be finite and nonnegative')
    totals = norms.sum(-1, keepdim=True)
    uniform = torch.full_like(norms, 1. / norms.shape[-1])
    return torch.where(totals > 0, norms / totals.clamp_min(math.ulp(1.)), uniform)
