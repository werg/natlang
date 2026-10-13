#!/usr/bin/env python3
"""Write the deployed GGUF of one quantization precision point (train/quantization.py) from a BF16 GGUF, losslessly.

A precision point maps module groups to formats; its deployed value of a weight is ``quantized_value(w, fmt)``
(maple/ternary.py), the same function training uses at λ = 1. This script stores exactly those values in existing
ggml types and checks every tensor bit for bit after packing:

- ``int4`` (group 32): Q4_0. The rule is llama.cpp's ``quantize_row_q4_0_ref`` with an FP16 scale, so Q4_0 holds it.
- ``ternary`` (Maple's per-row rule: every nonzero of a row is ±alpha, alpha BF16):
  - rows that are a multiple of 256 (Mellum's expert gate/up, 2304 columns): TQ2_0 (2.06 bits/weight). Each
    256-block's absmax is alpha, which FP16 holds exactly in its normal range.
  - other rows (Mellum's expert down, 896 columns): Q4_0 with d = alpha and codes 8 + {-1, 0, 1} (4.5 bits/weight).
    A 2-bit type with 128-column blocks is the planned replacement.
- groups the point does not select stay as they are (BF16 or F32).

Group names follow the recipe: ``experts`` (blk.N.ffn_{gate,up,down}_exps) and ``attention`` (blk.N.attn_{q,k,v,output}).
The point's ``layers`` selection is honoured when it is a list of layer indices.

    python scripts/mellum_deploy_gguf.py --gguf BF16.gguf --recipe training/neuralese/recipes/neuralese-direct-mellum-v4.json \
        --point ternary-experts --out DEPLOY.gguf

Run it under the memory ledger (about 3 GB plus the output's packed tensors, which are held until written).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training" / "neuralese"))

GROUPS = {
    "experts": re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$"),
    "attention": re.compile(r"^blk\.(\d+)\.attn_(q|k|v|output)\.weight$"),
}


def fork_gguf():
    from natlang_neuralese.export import fork_root

    sys.path.insert(0, str(fork_root() / "gguf-py"))
    import gguf

    return gguf


def point_modules(recipe_path: Path, name: str) -> dict:
    recipe = json.loads(recipe_path.read_text())
    spec = recipe.get("quantization") or {}
    for point in spec.get("points", []):
        if point["name"] == name:
            return point
    raise SystemExit(f"{recipe_path} declares no quantization point {name!r}")


def bf16_tensor(data: np.ndarray, shape):
    import torch

    raw = np.ascontiguousarray(data).view(np.int16).reshape(shape)
    return torch.from_numpy(raw).view(torch.bfloat16)


def pack_ternary_q4_0(values: np.ndarray) -> np.ndarray:
    """Q4_0 blocks holding ternary rows exactly: d = the block's absmax (alpha), codes 8 + {-1, 0, 1}."""
    rows = values.reshape(-1, 32)
    alpha = np.abs(rows).max(axis=1, keepdims=True)
    codes = np.where(alpha > 0, np.rint(rows / np.where(alpha > 0, alpha, 1)), 0).astype(np.int8)
    if np.abs(codes).max(initial=0) > 1:
        raise ValueError("not a ternary weight")
    q = (codes + 8).astype(np.uint8).reshape(-1, 2, 16)
    qs = q[:, 0, :] | (q[:, 1, :] << np.uint8(4))
    d = alpha.astype(np.float16).view(np.uint8)
    return np.concatenate([d, qs], axis=1).reshape(*values.shape[:-1], values.shape[-1] // 32 * 18)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--point", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    gguf = fork_gguf()
    import torch
    from natlang_neuralese.maple.ternary import quantized_value

    point = point_modules(Path(a.recipe), a.point)
    layers = point.get("layers")
    if layers not in (None, "all") and not isinstance(layers, list):
        raise SystemExit("this script honours a list of layers or all layers; resolve a sensitivity selection first")
    modules = {g: c for g, c in point["modules"].items() if g in GROUPS}
    reader = gguf.GGUFReader(a.gguf)
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(a.out, arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), kind, sub_type=sub)
    writer.add_string("natlang.precision_point", a.point)
    writer.add_string("natlang.precision_point.modules", json.dumps(point["modules"], sort_keys=True))
    report = {"point": a.point, "tensors": {}, "bytes_in": 0, "bytes_out": 0}
    for t in reader.tensors:
        group = next((g for g, rx in GROUPS.items() if g in modules and rx.match(t.name)), None)
        layer = int(GROUPS[group].match(t.name).group(1)) if group else None
        report["bytes_in"] += int(t.n_bytes)
        if group is None or (isinstance(layers, list) and layer not in layers) or modules[group]["format"] == "bf16":
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
            report["bytes_out"] += int(t.n_bytes)
            continue
        if t.tensor_type != gguf.GGMLQuantizationType.BF16:
            raise SystemExit(f"{t.name}: expected BF16, got {t.tensor_type.name}")
        config = modules[group]
        fmt = config["format"]
        options = {k: v for k, v in config.items() if k != "format"}
        shape = tuple(int(n) for n in reversed(t.shape.tolist()))  # numpy order: (..., rows, columns)
        w = bf16_tensor(t.data, shape)
        values = quantized_value(w, fmt, options).numpy().astype(np.float32)
        columns = shape[-1]
        if fmt == "int4":
            if options.get("group", 32) != 32:
                raise SystemExit("int4 deploys as Q4_0 (group 32)")
            qtype = gguf.GGMLQuantizationType.Q4_0
            packed = gguf.quants.quantize(w.float().numpy(), qtype)
        elif fmt == "ternary" and columns % 256 == 0:
            qtype = gguf.GGMLQuantizationType.TQ2_0
            packed = gguf.quants.quantize(values, qtype)
        elif fmt == "ternary":
            qtype = gguf.GGMLQuantizationType.Q4_0
            packed = pack_ternary_q4_0(values)
        else:
            raise SystemExit(f"{t.name}: unsupported format {fmt}")
        back = gguf.quants.dequantize(packed, qtype).reshape(values.shape)
        mismatch = int((back != values).sum())
        if mismatch:
            raise SystemExit(f"{t.name}: {mismatch} values differ after {qtype.name} packing (max "
                             f"{float(np.abs(back - values).max()):.3g}); the format does not hold this point")
        writer.add_tensor(t.name, packed, raw_dtype=qtype)
        report["bytes_out"] += int(packed.nbytes)
        report["tensors"][t.name] = {"group": group, "format": fmt, "ggml_type": qtype.name,
                                     "zero_fraction": float((values == 0).mean())}
        print(json.dumps({"tensor": t.name, **report["tensors"][t.name]}), flush=True)
        del w, values, back, packed
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    types = {}
    for info in report["tensors"].values():
        types[info["ggml_type"]] = types.get(info["ggml_type"], 0) + 1
    print(json.dumps({"out": a.out, "point": a.point, "bytes_in": report["bytes_in"], "bytes_out": report["bytes_out"],
                      "quantized_tensors": len(report["tensors"]), "types": types, "lossless": True}))


if __name__ == "__main__":
    main()
