"""Joint program objective normalization (§5.1). The model forward stays with the native engine.

``ell_i = -sum_t b[i,t] log p(z|h) / sum_t b[i,t]``;
``L = mean_lambda mean_{i in lambda}(w_i ell_i) + sum_n lambda_depth(n) sum_m q_m ||D[n,m]||_F^2
      + tau mean_lambda sum_f a[lambda,f]``.
"""
from __future__ import annotations

import math

import torch

from . import TuneMathError
from .gates import factor_sq_norm, gate_l1


def trajectory_losses(token_logprobs, token_weights):
    """Per-row positive-weight-normalized NLL. Rows are (N, T); padding has weight zero."""
    if token_logprobs.shape != token_weights.shape or token_logprobs.ndim != 2:
        raise TuneMathError('token log-probabilities and weights must share an (N, T) shape')
    if bool((token_weights < 0).any()) or not bool(torch.isfinite(token_weights).all()):
        raise TuneMathError('token weights must be finite and nonnegative')
    totals = token_weights.sum(-1)
    if bool((totals <= 0).any()):
        raise TuneMathError('a trajectory row has no positive target weight; reject it before training')
    return -(token_weights * token_logprobs).sum(-1) / totals


def program_loss(row_losses, row_bindings, evidence_weights, occurrences=None):
    """Mean over bound lambdas of the mean evidence-weighted row loss.

    ``occurrences`` (optional) names each row's recorded turn/graph occurrence; a
    producer consumed several times must still be counted once.
    """
    if not (len(row_bindings) == len(evidence_weights) == row_losses.shape[0]):
        raise TuneMathError('row losses, bindings and evidence weights differ in length')
    if occurrences is not None:
        if len(occurrences) != len(row_bindings) or len(set(occurrences)) != len(occurrences):
            raise TuneMathError('each recorded occurrence contributes exactly one loss row')
    weights = torch.as_tensor(evidence_weights, dtype=row_losses.dtype, device=row_losses.device)
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise TuneMathError('evidence weights must be finite and nonnegative')
    groups = {}
    for index, binding in enumerate(row_bindings):
        groups.setdefault(binding, []).append(index)
    weighted = weights * row_losses
    return torch.stack([weighted[torch.tensor(rows, device=row_losses.device)].mean() for rows in groups.values()]).mean()


def depth_penalty(node_factors, node_lambdas, q):
    """``sum_n lambda_depth(n) sum_m q_m ||D[n,m]||_F^2`` over ``{node: {m: (B, A)}}``."""
    total = torch.zeros(())
    for node, factors in node_factors.items():
        coefficient = node_lambdas.get(node)
        if coefficient is None or not math.isfinite(coefficient) or coefficient < 0:
            raise TuneMathError(f'node {node} lacks a valid declared depth penalty')
        for name, (B, A) in factors.items():
            if name not in q:
                raise TuneMathError(f'no matrix-size normalization recorded for {name}')
            total = total + coefficient * q[name] * factor_sq_norm(B, A)
    return total


def total_loss(row_losses, row_bindings, evidence_weights, node_factors, node_lambdas, q, gates, tau, occurrences=None):
    task = program_loss(row_losses, row_bindings, evidence_weights, occurrences)
    depth = depth_penalty(node_factors, node_lambdas, q).to(task)
    gate = gate_l1(gates, tau).to(task)
    return task + depth + gate, {'task': task.detach(), 'depth': depth.detach(), 'gate': gate.detach()}
