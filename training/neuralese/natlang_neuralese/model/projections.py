"""Historical learned block-delta projection (`D` of LEARNING_CONTINUUM §5).

The current native path optimizes positional KV and hybrid-state tensors directly; adapter-code projections were
retired. `DeltaProjection` remains only for historical readers and conversions.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

class DeltaProjection(nn.Module):
    """`D` (§5): a written block → a residual for a base block. Each base position's query (from its own vector)
    attends over the written block's vectors; the result, through a zero-initialised bias-free output, is that
    position's residual. A zero written block gives a zero delta; an untrained D gives the identity update; the delta
    has the base's shape whatever the written block's length."""

    def __init__(self, source_dialect: str, dim: int, hidden: int = 256, eps: float = 1e-6):
        super().__init__()
        self.source_dialect, self.dim, self.hidden, self.eps = source_dialect, dim, hidden, eps
        self.query = nn.Linear(dim, hidden, bias=False)
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()

    def forward(self, written: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """written [W, d], base [L, d] → delta [L, d] (scaled to the base's RMS, so updates are relative)."""
        if written.shape[0] == 0 or base.shape[0] == 0:
            return base.new_zeros(base.shape).float()
        w, b = self._norm(written), self._norm(base)
        scores = self.query(b) @ self.key(w).t() / math.sqrt(self.hidden)
        pooled = scores.softmax(-1) @ self.value(w)
        scale = base.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.out(F.gelu(pooled)) * scale
