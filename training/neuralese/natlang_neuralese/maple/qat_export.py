"""Export a full-latent QAT conversion (maple/qat_convert.py) as a deployed ternary checkpoint (plans/mellum-port.md).

The output is the source checkpoint's HF layout (same config, tokenizer and shard files) with every converted latent
(attention q/k/v/o, each expert's gate/up/down) replaced by its deployed ternary value under Maple's rule (per output
row: threshold 0.7 x row absmean, scale = mean kept magnitude, BF16). Maple's rule is idempotent on such weights, so
``load_maple`` reproduces the conversion's λ=1 forward exactly and loads the experts as ternary codes (config flag
``natlang_deployed_ternary``). From there the Mellum line runs the shared Maple-family pipeline unchanged: QAT install
(learned block scales, attention dense latents), foundation heads, N0 expert order, fused ternary MoE kernel.
Weights the conversion froze (embedding, head, router, norms) are copied unchanged.

    python -m natlang_neuralese.maple.qat_export --model DIR --checkpoint CONVERT/checkpoint.pt --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from .ternary import ternarize

EXPERT_PARTS = ("gate_proj", "up_proj", "down_proj")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def deployed_tensors(latents: dict, names: list[str], device="cpu") -> dict[str, torch.Tensor]:
    """The deployed values for checkpoint tensor ``names`` that have a latent; others are absent from the result."""
    out: dict[str, torch.Tensor] = {}
    experts: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in names:
        if name in latents:  # attention projections keep their checkpoint names
            out[name] = ternarize(latents[name].to(device)).to(torch.bfloat16).cpu()
            continue
        parts = name.split(".")
        if len(parts) >= 7 and parts[0] == "model" and parts[4] == "experts" and parts[6] in EXPERT_PARTS:
            layer, expert, part = int(parts[2]), int(parts[5]), parts[6]
            prefix = f"model.layers.{layer}.mlp.experts"
            if f"{prefix}.gate_up" not in latents:
                continue
            if layer not in experts:  # one ternarization per layer tensor (rows are independent)
                experts[layer] = (ternarize(latents[f"{prefix}.gate_up"].to(device)).to(torch.bfloat16).cpu(),
                                  ternarize(latents[f"{prefix}.down"].to(device)).to(torch.bfloat16).cpu())
            gate_up, down = experts[layer]
            ff = gate_up.shape[1] // 2
            value = {"gate_proj": gate_up[expert, :ff], "up_proj": gate_up[expert, ff:], "down_proj": down[expert]}[part]
            out[name] = value.contiguous()
    return out


def _ordered(index: dict, orders) -> dict[str, str]:
    """{exported tensor name: source tensor name} under the N0 expert order (maple/routing.py): exported expert j is
    source expert ``orders[layer][j]``; router rows are permuted the same way (``MapleForCausalLM.order_experts``)."""
    rename = {}
    for name in index:
        parts = name.split(".")
        if orders is not None and len(parts) >= 7 and parts[0] == "model" and parts[4] == "experts":
            layer, expert = int(parts[2]), int(parts[5])
            parts[5] = str(int(orders[layer][expert]))
        rename[name] = ".".join(parts)
    return rename


def export(model: Path, checkpoint: Path, out: Path, device: str = "cpu", order: Path | None = None) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    state = torch.load(checkpoint, map_location="cpu", mmap=True)
    latents = state["latents"]
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    by_file: dict[str, list[str]] = {}
    for name, file in index["weight_map"].items():
        by_file.setdefault(file, []).append(name)
    orders = torch.load(order)["orders"] if order else None
    rename = _ordered(index["weight_map"], orders)  # exported name -> source name (identity without an order)
    replaced = 0
    for file, names in sorted(by_file.items()):
        sources = [rename[name] for name in names]
        files = {index["weight_map"][src] for src in sources}
        tensors, metadata = {}, None
        for source_file in sorted(files):
            with safe_open(model / source_file, framework="pt", device="cpu") as source:
                for name, src in zip(names, sources):
                    if index["weight_map"][src] == source_file:
                        tensors[name] = source.get_tensor(src)
        if metadata is None:
            with safe_open(model / file, framework="pt", device="cpu") as source:
                metadata = source.metadata()
        deployed = deployed_tensors(latents, sources, device)
        for name, src in zip(names, sources):
            if src not in deployed:
                continue
            value = deployed[src]
            if value.shape != tensors[name].shape:
                raise ValueError(f"{name}: latent shape {tuple(value.shape)} != checkpoint {tuple(tensors[name].shape)}")
            tensors[name] = value.to(tensors[name].dtype)
            replaced += 1
        if orders is not None:
            for name in names:
                parts = name.split(".")
                if len(parts) == 6 and parts[4] == "gate" and parts[3] == "mlp" and parts[5] == "weight":
                    tensors[name] = tensors[name][orders[int(parts[2])].to(torch.long)].contiguous()
        save_file(tensors, out / file, metadata=metadata)
        del tensors, deployed
    expected = sum(1 for name in index["weight_map"] if name.split(".")[-2] in ("q_proj", "k_proj", "v_proj", "o_proj")
                   or (".mlp.experts." in name and name.split(".")[-2] in EXPERT_PARTS))
    if replaced != expected:
        raise ValueError(f"replaced {replaced} tensors, the checkpoint has {expected} attention/expert matrices")
    for item in model.iterdir():
        if item.is_file() and item.suffix != ".safetensors" and item.name != "config.json":
            shutil.copy2(item, out / item.name)
    config = json.loads((model / "config.json").read_text())
    provenance = {"source_model": str(model), "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                  "step": int(state.get("step", 0)), "rule": "maple-row-ternary-0.7", "replaced_tensors": replaced,
                  **({"expert_order": str(order), "expert_order_sha256": _sha256(Path(order))} if order else {})}
    config["natlang_deployed_ternary"] = provenance
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "qat-export.json").write_text(json.dumps(provenance, indent=1) + "\n")
    return provenance


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="the BF16 checkpoint the conversion started from")
    p.add_argument("--checkpoint", type=Path, required=True, help="qat_convert train --out DIR/checkpoint.pt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--order", type=Path, help="N0 expert-order.pt (maple/routing.py): export experts in that order, "
                                              "so nested members are prefixes of the exported checkpoint")
    a = p.parse_args(argv)
    print(json.dumps(export(a.model, a.checkpoint, a.out, a.device, a.order)))


if __name__ == "__main__":
    main()
