"""Projections from Neuralese into other machinery (LEARNING_CONTINUUM.md §13a; `P` of §6.4, `D` of §5).

The model's port stays ordinary Neuralese. A projection is an adapter outside it: it maps a written block (a normal
value in the model's dialect) to a fixed-shape control of some target, here the coefficients of a weight adapter.

Form: one learned query per output row attends over the block's vectors (so a block of any length decodes to the
target's shape and the writer's stop head still decides length); bias-free key, value and output maps after an RMS
norm, so a zero block decodes to exactly zero, and a zero-initialised output so an untrained projection decodes
every block to zero (the identity update).

A projection is identified by its source dialect, its target (an adapter spec's dialect) and a hash of its weights,
and is refused for blocks of another dialect, as adapters are refused on another base.
"""

from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F
from torch import nn

SCHEMA = "natlang.block-projection/1"


class BlockProjection(nn.Module):
    def __init__(self, dim: int, rows: int, width: int, hidden: int = 256, eps: float = 1e-6):
        super().__init__()
        self.dim, self.rows, self.width, self.hidden, self.eps = dim, rows, width, hidden, eps
        self.queries = nn.Parameter(torch.randn(rows, hidden) / math.sqrt(hidden))
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        """[L, d] or [B, L, d] → [rows, width] or [B, rows, width]."""
        batched = block.dim() == 3
        x = block if batched else block[None]
        if x.shape[1] == 0:
            out = x.new_zeros(x.shape[0], self.rows, self.width)
            return out if batched else out[0]
        x = x.float()
        x = x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        scores = torch.einsum("rh,blh->brl", self.queries, self.key(x)) / math.sqrt(self.hidden)
        pooled = torch.einsum("brl,blh->brh", scores.softmax(-1), self.value(x))
        out = self.out(F.gelu(pooled))
        return out if batched else out[0]


class AdapterProjection(nn.Module):
    """`P`: a block of the source dialect → the coefficients of an adapter with spec `target`."""

    def __init__(self, source_dialect: str, target: str, dim: int, rows: int, width: int, hidden: int = 256):
        super().__init__()
        self.source_dialect, self.target = source_dialect, target
        self.project = BlockProjection(dim, rows, width, hidden)

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        return self.project(block)

    def identity(self) -> str:
        digest = hashlib.sha256(f"{SCHEMA}\0{self.source_dialect}\0{self.target}".encode())
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode() + tensor.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()[:16]

    def save(self, path: str, **extra):
        torch.save({"schema": SCHEMA, "source_dialect": self.source_dialect, "target": self.target,
                    "shape": [self.project.dim, self.project.rows, self.project.width, self.project.hidden],
                    "state": self.state_dict(), "identity": self.identity(), **extra}, path)

    @staticmethod
    def load(path: str, map_location="cpu") -> "AdapterProjection":
        saved = torch.load(path, map_location=map_location, weights_only=False)
        if saved.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a block projection")
        dim, rows, width, hidden = saved["shape"]
        projection = AdapterProjection(saved["source_dialect"], saved["target"], dim, rows, width, hidden)
        projection.load_state_dict(saved["state"])
        if projection.identity() != saved["identity"]:
            raise ValueError(f"{path}: weights do not match the recorded identity")
        return projection
