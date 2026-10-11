"""Read-only parameter drift summary for paired plain-text behavior reports.

The comparison is deliberately a descriptive parameter-distance statistic, not
a quality or capability gate. For adapted linear weights it compares effective
weights (base layer plus the declared LoRA update), while reporting the source
LoRA values through the checkpoint fingerprints and metadata.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from ..common.hashing import sha256_file_hex


def _base_tensor_path(snapshot: Path, key: str, index: dict | None):
    if index is None:
        path = snapshot / "model.safetensors"
    else:
        shard = index.get("weight_map", {}).get(key)
        if shard is None:
            return None
        path = snapshot / shard
    if not path.is_file():
        return None
    return path


def _lora_delta(state: dict, module_key: str, *, alpha: int, rank: int):
    prefix = module_key.removesuffix(".base_layer.weight")
    a_key = f"{prefix}.lora_A.neuralese.weight"
    b_key = f"{prefix}.lora_B.neuralese.weight"
    if a_key not in state or b_key not in state:
        return None
    a = state[a_key].detach().float()
    b = state[b_key].detach().float()
    return (b @ a) * (float(alpha) / rank)


def compare_checkpoint_weights(lineage_path: Path, candidate_path: Path, base_snapshot: Path) -> dict:
    """Compare original lineage and candidate layer tensors without changing them."""
    base_snapshot = base_snapshot.resolve()
    safe_index = base_snapshot / "model.safetensors.index.json"
    index = None
    if safe_index.is_file():
        import json
        index = json.loads(safe_index.read_text())
        index_hash = sha256_file_hex(safe_index)
    else:
        index_hash = None
    if not (base_snapshot / "model.safetensors").is_file() and index is None:
        raise FileNotFoundError(f"no model.safetensors or shard index under pinned base snapshot {base_snapshot}")

    lineage = torch.load(lineage_path, map_location="cpu", weights_only=False, mmap=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False, mmap=True)
    old_lora = lineage.get("lora") or {}
    new_lora = candidate.get("lora") or {}
    trainables = candidate.get("backbone_trainables") or {}
    old_rank = int((lineage.get("identity", {}).get("options") or {}).get("rank"))
    new_rank = int(candidate.get("lora_rank"))
    old_alpha = int((lineage.get("port_config") or {}).get("lora_alpha") or 2 * old_rank)
    new_alpha = int((candidate.get("port_config") or {}).get("lora_alpha") or 2 * new_rank)
    if old_rank < 1 or new_rank < 1 or not trainables:
        raise ValueError("checkpoint structures do not support exact effective-weight comparison")

    model_file_handles: dict[Path, object] = {}
    report_rows = []
    omitted = []
    family_sums = defaultdict(lambda: [0.0, 0.0, 0])
    total_diff_sq = 0.0
    total_reference_sq = 0.0
    try:
        for name, trained_value in trainables.items():
            if ".lora_A." in name or ".lora_B." in name:
                continue
            if name.startswith("model.layers.") is False:
                omitted.append({"name": name, "reason": "outside transformer-layer training scope"})
                continue
            base_key = name.replace(".base_layer.weight", ".weight")
            base_path = _base_tensor_path(base_snapshot, base_key, index)
            if base_path is None:
                omitted.append({"name": name, "base_key": base_key, "reason": "no exact pinned-base tensor key"})
                continue
            handle = model_file_handles.get(base_path)
            if handle is None:
                handle = safe_open(str(base_path), framework="pt", device="cpu")
                handle.__enter__()
                model_file_handles[base_path] = handle
            if base_key not in handle.keys():
                omitted.append({"name": name, "base_key": base_key, "reason": "base shard lacks exact key"})
                continue
            base_value = handle.get_tensor(base_key).float()
            candidate_value = trained_value.detach().float()
            if tuple(base_value.shape) != tuple(candidate_value.shape):
                omitted.append({"name": name, "base_key": base_key, "reason": "base/candidate tensor shape mismatch",
                                "base_shape": list(base_value.shape), "candidate_shape": list(candidate_value.shape)})
                continue
            original_value = base_value
            candidate_effective = candidate_value
            module_key = name.removesuffix(".base_layer.weight")
            old_delta = _lora_delta(old_lora, module_key + ".base_layer.weight", alpha=old_alpha, rank=old_rank)
            new_delta = _lora_delta(new_lora, module_key + ".base_layer.weight", alpha=new_alpha, rank=new_rank)
            if old_delta is not None:
                if tuple(old_delta.shape) != tuple(base_value.shape):
                    omitted.append({"name": name, "reason": "lineage LoRA effective delta shape mismatch"})
                    continue
                original_value = base_value + old_delta
            if new_delta is not None:
                if tuple(new_delta.shape) != tuple(candidate_value.shape):
                    omitted.append({"name": name, "reason": "candidate LoRA effective delta shape mismatch"})
                    continue
                candidate_effective = candidate_value + new_delta
            difference = candidate_effective - original_value
            diff_norm = float(torch.linalg.vector_norm(difference))
            reference_norm = float(torch.linalg.vector_norm(original_value))
            relative = diff_norm / max(reference_norm, 1e-30)
            report_rows.append({"name": name, "base_key": base_key, "numel": difference.numel(),
                                "lineage_lora_included": old_delta is not None,
                                "candidate_lora_included": new_delta is not None,
                                "difference_l2": diff_norm, "lineage_effective_l2": reference_norm,
                                "relative_l2": relative,
                                "max_abs_difference": float(difference.abs().max())})
            total_diff_sq += diff_norm * diff_norm
            total_reference_sq += reference_norm * reference_norm
            family = "attention" if ".self_attn." in name else (
                     "convolution" if ".conv." in name else (
                     "feed_forward" if ".feed_forward." in name else "other_layer"))
            acc = family_sums[family]
            acc[0] += diff_norm * diff_norm
            acc[1] += reference_norm * reference_norm
            acc[2] += difference.numel()
    finally:
        for handle in model_file_handles.values():
            handle.__exit__(None, None, None)

    return {"schema": "natlang.plain-checkpoint-weight-drift/1",
            "interpretation": "read-only descriptive parameter drift; not a behavior or qualification gate",
            "lineage_checkpoint_sha256": sha256_file_hex(lineage_path),
            "candidate_checkpoint_sha256": sha256_file_hex(candidate_path),
            "base_snapshot": str(base_snapshot),
            "base_weights_sha256": sha256_file_hex(base_snapshot / "model.safetensors") if (base_snapshot / "model.safetensors").is_file() else None,
            "base_shard_index_sha256": index_hash,
            "lineage_lora_rank": old_rank, "lineage_lora_alpha": old_alpha,
            "candidate_lora_rank": new_rank, "candidate_lora_alpha": new_alpha,
            "candidate_trainable_tensor_count": len(trainables),
            "compared_tensor_count": len(report_rows), "compared_numel": sum(r["numel"] for r in report_rows),
            "omitted_tensor_count": len(omitted), "omitted": omitted,
            "aggregate_difference_l2": total_diff_sq ** 0.5,
            "aggregate_lineage_effective_l2": total_reference_sq ** 0.5,
            "aggregate_relative_l2": (total_diff_sq / max(total_reference_sq, 1e-60)) ** 0.5,
            "families": {name: {"relative_l2": (values[0] / max(values[1], 1e-60)) ** 0.5,
                                 "compared_numel": values[2]}
                         for name, values in family_sums.items()},
            "per_tensor": report_rows,
            "caveats": ["LoRA contributions are reconstructed as B @ A scaled by alpha/rank.",
                        "Only exact matched transformer layer tensors in candidate backbone_trainables are compared.",
                        "Relative tensor drift does not establish output or task-quality impact."]}
