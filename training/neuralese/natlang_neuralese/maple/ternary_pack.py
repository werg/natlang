"""Compact storage of deployed ternary checkpoints (plans/STORAGE_POLICY.md, "Compact ternary milestones").

A qualified Maple/Mellum milestone exported in deployed form (maple/qat_export.py: every converted latent replaced by
its ternary value, Maple's rule) needs no BF16 latents unless QAT resumes from it. Each ternary tensor row holds only
{0, +a, -a}: stored as 2-bit codes (4 per byte) plus one BF16 scale per row, ~8x smaller than BF16. Non-ternary
tensors (embedding, head, router, norms) are stored unchanged. Unpacking reproduces every tensor bit-exactly (the
pack verifies it before writing), so the unpacked directory loads with ``load_maple`` like the export.

    python -m natlang_neuralese.maple.ternary_pack pack   --src EXPORT_DIR --out PACKED_DIR
    python -m natlang_neuralese.maple.ternary_pack unpack --src PACKED_DIR --out EXPORT_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

SCHEMA = "natlang.ternary-pack/1"


def pack_tensor(weight: torch.Tensor) -> dict | None:
    """``{"codes", "scale", "shape", "dtype"}`` when every row of ``weight`` is ternary (values 0, +a, -a with one a
    per row) and the packing round-trips exactly; None otherwise."""
    if weight.dim() < 2 or not weight.is_floating_point():
        return None
    rows = weight.reshape(-1, weight.shape[-1])
    scale = rows.abs().amax(-1)
    if not torch.all((rows == 0) | (rows.abs() == scale[:, None])):
        return None
    codes = torch.zeros_like(rows, dtype=torch.uint8)
    codes[rows > 0] = 1
    codes[rows < 0] = 2
    flat = codes.reshape(-1)
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    quads = flat.reshape(-1, 4)
    packed = quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)
    out = {"codes": packed, "scale": scale.to(weight.dtype), "shape": list(weight.shape), "dtype": str(weight.dtype)}
    if not torch.equal(unpack_tensor(out), weight):
        return None
    return out


def unpack_tensor(packed: dict) -> torch.Tensor:
    shape = packed["shape"]
    numel = 1
    for size in shape:
        numel *= size
    codes = packed["codes"]
    quads = torch.stack([(codes >> shift) & 3 for shift in (0, 2, 4, 6)], -1).reshape(-1)[:numel]
    rows = quads.reshape(-1, shape[-1])
    scale = packed["scale"].reshape(-1, 1)
    values = torch.zeros(rows.shape, dtype=scale.dtype)
    values = torch.where(rows == 1, scale.expand_as(values), values)
    values = torch.where(rows == 2, -scale.expand_as(values), values)
    return values.reshape(shape)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def pack_dir(src: Path, out: Path) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=False)
    meta = {"schema": SCHEMA, "source": str(src), "tensors": {}, "shards": []}
    raw_bytes = packed_bytes = 0
    for shard in sorted(src.glob("*.safetensors")):
        tensors = {}
        with safe_open(shard, "pt") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                raw_bytes += tensor.numel() * tensor.element_size()
                packed = pack_tensor(tensor)
                if packed is None:
                    tensors[name] = tensor
                    meta["tensors"][name] = {"shard": shard.name, "packed": False, "sha256": _digest(tensor)}
                else:
                    tensors[name + "::codes"], tensors[name + "::scale"] = packed["codes"], packed["scale"]
                    meta["tensors"][name] = {"shard": shard.name, "packed": True, "shape": packed["shape"],
                                             "sha256": _digest(tensor)}
        for value in tensors.values():
            packed_bytes += value.numel() * value.element_size()
        save_file(tensors, str(out / shard.name))
        meta["shards"].append(shard.name)
    for extra in src.iterdir():  # config, tokenizer, index, receipts: copied unchanged
        if extra.is_file() and extra.suffix != ".safetensors":
            shutil.copy2(extra, out / extra.name)
    meta["raw_bytes"], meta["packed_bytes"] = raw_bytes, packed_bytes
    (out / "ternary-pack.json").write_text(json.dumps(meta, indent=1) + "\n")
    return {"raw_gb": raw_bytes / 2**30, "packed_gb": packed_bytes / 2**30,
            "packed_tensors": sum(t["packed"] for t in meta["tensors"].values()), "tensors": len(meta["tensors"])}


def unpack_dir(src: Path, out: Path) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    meta = json.loads((src / "ternary-pack.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    for shard in meta["shards"]:
        tensors = {}
        with safe_open(src / shard, "pt") as handle:
            for name, info in meta["tensors"].items():
                if info["shard"] != shard:
                    continue
                if info["packed"]:
                    tensor = unpack_tensor({"codes": handle.get_tensor(name + "::codes"),
                                            "scale": handle.get_tensor(name + "::scale"), "shape": info["shape"]})
                else:
                    tensor = handle.get_tensor(name)
                if _digest(tensor) != info["sha256"]:
                    raise ValueError(f"{name}: unpacked tensor differs from the packed source")
                tensors[name] = tensor
        save_file(tensors, str(out / shard))
    for extra in src.iterdir():
        if extra.is_file() and extra.suffix != ".safetensors" and extra.name != "ternary-pack.json":
            shutil.copy2(extra, out / extra.name)
    return {"tensors": len(meta["tensors"]), "verified": True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("pack", "unpack"))
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    result = (pack_dir if a.action == "pack" else unpack_dir)(Path(a.src), Path(a.out))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
