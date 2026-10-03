"""Dialect version tag and store entries for written blocks (S0 §3.4, §12).

A dialect is a lightweight version tag naming the model space a block was written in.
Blocks are only read by models of the same dialect; a mismatch means regenerate or
convert, never silent reuse.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

import torch

DIALECT = "nd:natlang@1"


def block_id(dialect: str, payload: torch.Tensor) -> str:
    """Content address: `nz1_` + base32 SHA-256 over dialect, shape, dtype and payload bytes."""
    data = payload.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(dialect.encode())
    digest.update(repr(tuple(data.shape)).encode())
    digest.update(str(data.dtype).encode())
    digest.update(data.view(torch.uint8).numpy().tobytes() if data.dtype == torch.bfloat16
                  else data.numpy().tobytes())
    return "nz1_" + base64.b32encode(digest.digest()).decode().rstrip("=").lower()


@dataclass(frozen=True)
class StoredBlock:
    id: str
    dialect: str
    type: str
    payload: torch.Tensor  # [length, width]
    truncated: bool
    producer: str = ""

    @property
    def length(self) -> int:
        return int(self.payload.shape[0])

    @staticmethod
    def make(payload: torch.Tensor, type: str, truncated: bool, dialect: str = DIALECT,
             producer: str = "") -> "StoredBlock":
        payload = payload.detach()
        return StoredBlock(block_id(dialect, payload), dialect, type, payload, truncated, producer)

    def check_dialect(self, dialect: str) -> None:
        if self.dialect != dialect:
            raise ValueError(f"block {self.id} is in {self.dialect}, reader speaks {dialect}: regenerate or convert")
