"""Shared gold-aligned embedding anchor for Neuralese output projections."""

import torch


def relative_mse_positions(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-position relative MSE, with detached raw embedding targets."""
    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError('projection and embedding target shapes must match')
    target = target.detach().float()
    return ((predicted.float() - target).square().mean(-1) /
            target.square().mean(-1).clamp_min(1e-6))


def gold_aligned_projection_errors(full_projection: torch.Tensor, shallow_projection: torch.Tensor,
                                   target_embeddings: torch.Tensor,
                                   shallow_target_embeddings: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-content and shallow-feedback per-position errors against gold embeddings.

    Callers provide outputs at causal predictor states aligned to actual target IDs.
    """
    shallow_target_embeddings = (target_embeddings if shallow_target_embeddings is None
                                 else shallow_target_embeddings)
    if (full_projection.shape != shallow_projection.shape or full_projection.shape != target_embeddings.shape or
            shallow_projection.shape != shallow_target_embeddings.shape):
        raise ValueError('gold-aligned full, shallow, and target tensors must have equal shapes')
    return (relative_mse_positions(full_projection, target_embeddings),
            relative_mse_positions(shallow_projection, shallow_target_embeddings))


def scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Forward-identical gradient scaling for gentle backbone anchoring."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError('backbone gradient scale must be between zero and one')
    return value.detach() + scale * (value - value.detach())
