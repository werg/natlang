"""Shared token-level policy objective used by native recorded-policy training."""
from __future__ import annotations

import torch


def clipped_token_loss(logps, old_logps, advantage, *, clip_low=0.2, clip_high=0.28):
    """DAPO-style asymmetric clipping, with stable float32 ratios and signed gradients."""
    if not 0 <= clip_low < 1 or not 0 <= clip_high:
        raise ValueError('invalid RL clipping range')
    if logps.shape != old_logps.shape or logps.numel() == 0:
        raise ValueError('behavior/current token positions differ or are empty')
    if not torch.isfinite(logps).all() or not torch.isfinite(old_logps).all():
        raise ValueError('nonfinite RL log probabilities')
    ratios = (logps.float() - old_logps.float()).clamp(-20, 20).exp()
    a = torch.as_tensor(advantage, dtype=torch.float32, device=logps.device)
    if not torch.isfinite(a).all():
        raise ValueError('nonfinite advantage')
    clipped = ratios.clamp(1 - clip_low, 1 + clip_high)
    return -torch.minimum(ratios * a, clipped * a)
