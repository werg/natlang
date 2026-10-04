"""Reading `.nz` files (spec/NEURALESE_FILES.md) in Python: a safetensors container whose tensors are blocks keyed by
content ID, with the natlang header (exports, block metadata) in the `natlang` metadata entry."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from safetensors import safe_open


@dataclass
class NzExport:
    name: str
    type: str
    block_id: str | None
    dialect: str | None
    payload: torch.Tensor | None  # [length, width] float32


def read_nz(path: str) -> tuple[dict, dict[str, NzExport]]:
    """(header, exports) of a `.nz` file. Exports that are single Neuralese values carry their block."""
    with safe_open(path, framework="pt") as stream:
        header = json.loads(stream.metadata()["natlang"])
        tensors = {key: stream.get_tensor(key) for key in stream.keys()}
    exports = {}
    for name, entry in header["exports"].items():
        value = entry.get("value")
        block_id = value.get("$neuralese", {}).get("id") if isinstance(value, dict) else None
        block = header["blocks"].get(block_id) if block_id else None
        exports[name] = NzExport(name, entry["type"], block_id, block["dialect"] if block else None,
                                 tensors[block_id].float() if block_id in tensors else None)
    return header, exports
