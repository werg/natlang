"""Small numerical measures shared by training objectives and their reports."""

from __future__ import annotations

import torch


def relative_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-example squared error divided by target squared norm."""
    if predicted.shape != target.shape:
        raise ValueError(f"relative-error tensors differ in shape: {predicted.shape} vs {target.shape}")
    if target.ndim == 0:
        return (predicted - target).pow(2) / target.pow(2).clamp_min(1e-12)
    dims = tuple(range(1, target.ndim))
    if not dims:
        return ((predicted - target).pow(2) / target.pow(2).clamp_min(1e-12)).mean()
    return ((predicted - target).pow(2).sum(dims) / target.pow(2).sum(dims).clamp_min(1e-12)).mean()
