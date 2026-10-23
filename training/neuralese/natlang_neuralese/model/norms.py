"""Small normalization layers shared by current native model components."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """FP32 RMS normalization with a learned scale and input-dtype output."""

    def __init__(self, dim: int, eps: float = 1e-5, gain: float = 1.0):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.full((dim,), float(gain)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)
