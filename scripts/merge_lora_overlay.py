"""Merge a PEFT LoRA adapter into a sharded Hugging Face checkpoint without copying it.

    python scripts/merge_lora_overlay.py BASE_DIR ADAPTER_DIR OUT_DIR

OUT_DIR links every file of BASE_DIR, except that the weights the adapter changes are written merged
(W + alpha/r * B @ A, computed in fp32 and stored in W's dtype) to one extra shard, model-zz-lora-merged.safetensors,
and model.safetensors.index.json maps them there. The original shards still hold the unmerged copies: a loader that
reads shards in name order (llama.cpp's convert_hf_to_gguf.py does, keeping the last copy of a name) sees the merged
ones. Use it for conversion, not as a checkpoint to load with Transformers.

For adapters llama.cpp cannot convert as LoRA: Ling's converter splits MLA's kv_b_proj into a per-head transposed
k_b, which a low-rank pair cannot follow. The adapter must be plain LoRA on linear layers (no rsLoRA, DoRA, or
stacked-parameter targets).
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

OVERLAY = "model-zz-lora-merged.safetensors"


def main(base: Path, adapter: Path, out: Path):
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("use_rslora") or config.get("use_dora") or config.get("target_parameters"):
        raise SystemExit("only plain LoRA on linear layers can be merged here")
    scale = config["lora_alpha"] / config["r"]
    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    pairs = {}
    with safe_open(adapter / "adapter_model.safetensors", "pt") as f:
        for key in f.keys():
            module, _, part = key.replace("base_model.model.", "", 1).partition(".lora_")
            pairs.setdefault(module + ".weight", {})[part.split(".")[0]] = f.get_tensor(key)
    by_shard = {}
    for name, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise SystemExit(f"{name}: incomplete LoRA pair {sorted(pair)}")
        if name not in weight_map:
            raise SystemExit(f"{name}: not in the base checkpoint")
        by_shard.setdefault(weight_map[name], []).append(name)
    merged = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(base / shard, "pt") as f:
            for name in names:
                weight = f.get_tensor(name)
                delta = pairs[name]["B"].float() @ pairs[name]["A"].float()
                if delta.shape != weight.shape:
                    raise SystemExit(f"{name}: LoRA delta {tuple(delta.shape)} does not fit {tuple(weight.shape)}")
                merged[name] = (weight.float() + scale * delta).to(weight.dtype).contiguous()
    out.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        target = out / item.name
        if item.name == "model.safetensors.index.json" or target.exists() or target.is_symlink():
            continue
        os.symlink(os.path.relpath(item.resolve(), out.resolve()), target)
    save_file(merged, str(out / OVERLAY), metadata={"format": "pt"})
    index = {**index, "weight_map": {**weight_map, **{name: OVERLAY for name in merged}}}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"merged {len(merged)} weights (scale {scale:g}) into {out / OVERLAY}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", type=Path)
    ap.add_argument("adapter", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    main(a.base, a.adapter, a.out)
