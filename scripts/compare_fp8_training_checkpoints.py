#!/usr/bin/env python3
"""Compare final BF16/TE training states from the FP8 checkpoint-resume exercise.

Run inside the pinned training image with CPU device visibility. It compares the
actual student/projection parameters, optimizer state, RNG state and TE scaling
extra state. Wall-clock fields and output paths are intentionally excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _tensor_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"same_shape": left.shape == right.shape, "same_dtype": left.dtype == right.dtype,
                "left_shape": list(left.shape), "right_shape": list(right.shape),
                "left_dtype": str(left.dtype), "right_dtype": str(right.dtype), "exact": False}
    a, b = left.detach().to(device="cpu"), right.detach().to(device="cpu")
    if a.is_floating_point() or a.is_complex():
        delta = (a.to(torch.float64) - b.to(torch.float64)).abs()
        denom = a.to(torch.float64).abs().max().item() if a.numel() else 0.0
        return {"same_shape": True, "same_dtype": True, "exact": bool(torch.equal(a, b)),
                "numel": a.numel(), "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
                "relative_to_left_max": float(delta.max().item() / denom) if delta.numel() and denom else None,
                "finite_left": bool(torch.isfinite(a).all()), "finite_right": bool(torch.isfinite(b).all())}
    return {"same_shape": True, "same_dtype": True, "exact": bool(torch.equal(a, b)),
            "numel": a.numel()}


def _walk(left: Any, right: Any, path: str, result: dict[str, Any]) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        result[path] = _tensor_stats(left, right)
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            result[path] = {"exact": False, "same_shape": left.shape == right.shape,
                            "same_dtype": left.dtype == right.dtype,
                            "left_shape": list(left.shape), "right_shape": list(right.shape),
                            "left_dtype": str(left.dtype), "right_dtype": str(right.dtype)}
        elif left.dtype.kind in "fc":
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            denom = float(np.max(np.abs(left.astype(np.float64)))) if left.size else 0.0
            result[path] = {"exact": bool(np.array_equal(left, right)), "numel": int(left.size),
                            "max_abs": float(np.max(delta)) if delta.size else 0.0,
                            "relative_to_left_max": float(np.max(delta) / denom)
                            if delta.size and denom else None,
                            "finite_left": bool(np.isfinite(left).all()),
                            "finite_right": bool(np.isfinite(right).all())}
        else:
            result[path] = {"exact": bool(np.array_equal(left, right)), "numel": int(left.size),
                            "dtype": str(left.dtype)}
        return
    if isinstance(left, np.generic) and isinstance(right, np.generic):
        left, right = left.item(), right.item()
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys, right_keys = set(left), set(right)
        result[path + ".$keys"] = {"exact": left_keys == right_keys,
                                    "only_left": sorted(map(str, left_keys - right_keys)),
                                    "only_right": sorted(map(str, right_keys - left_keys))}
        for key in sorted(left_keys & right_keys, key=str):
            _walk(left[key], right[key], f"{path}.{key}", result)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        result[path + ".$length"] = {"exact": len(left) == len(right), "left": len(left), "right": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            _walk(a, b, f"{path}[{index}]", result)
        return
    if type(left) is type(right) and isinstance(left, (str, int, float, bool, type(None))):
        result[path] = {"exact": left == right, "left": left, "right": right}
        return
    if isinstance(left, bytes) and isinstance(right, bytes):
        result[path] = {"exact": left == right, "left_bytes": len(left), "right_bytes": len(right),
                        "left_sha256": hashlib.sha256(left).hexdigest(),
                        "right_sha256": hashlib.sha256(right).hexdigest()}
        return
    result[path] = {"exact": False, "left_type": type(left).__name__, "right_type": type(right).__name__}


def compare(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = torch.load(left_path, map_location="cpu", weights_only=False, mmap=True)
    right = torch.load(right_path, map_location="cpu", weights_only=False, mmap=True)
    fields = ("schema", "step", "student_parameters", "neuralese_projection", "token_choice_projection",
              "optimizer", "rng", "fp8_training")
    details: dict[str, Any] = {}
    left_identity = dict(left.get("identity") or {})
    right_identity = dict(right.get("identity") or {})
    left_options = dict(left_identity.get("options") or {})
    right_options = dict(right_identity.get("options") or {})
    left_options.pop("out", None)
    right_options.pop("out", None)
    identity_match = {
        "options_match_ignoring_output_path": left_options == right_options,
        "inputs_match": left_identity.get("inputs") == right_identity.get("inputs"),
    }
    for key in fields:
        if key in left or key in right:
            if key not in left or key not in right:
                details[key] = {"exact": False, "present_left": key in left, "present_right": key in right}
            else:
                _walk(left[key], right[key], key, details)
    by_family = {}
    for key, value in details.items():
        family = key.split(".", 1)[0]
        by_family.setdefault(family, []).append(value)
    family_summary = {}
    for family, values in by_family.items():
        tensors = [value for value in values if "numel" in value]
        numeric = [value for value in tensors if value.get("same_shape") and value.get("same_dtype")]
        family_summary[family] = {
            "entries": len(values), "tensor_entries": len(tensors),
            "exact_tensor_entries": sum(bool(value.get("exact")) for value in tensors),
            "comparable_tensor_entries": len(numeric),
            "all_entries_exact": all(bool(value.get("exact")) for value in values),
            "mismatch_entry_count": sum(not bool(value.get("exact")) for value in values),
            "max_abs_delta": max((value.get("max_abs", 0.0) for value in numeric), default=None),
            "max_relative_to_left": max((value["relative_to_left_max"] for value in numeric
                                          if value.get("relative_to_left_max") is not None), default=None),
            "nonfinite_tensor_entries": sum(not value.get("finite_left", True) or
                                             not value.get("finite_right", True) for value in numeric),
        }
    return {
        "schema": "natlang.fp8-checkpoint-state-comparison/1",
        "interrupted_resumed_checkpoint": str(left_path.resolve()),
        "uninterrupted_control_checkpoint": str(right_path.resolve()),
        "left_step": left.get("step"), "right_step": right.get("step"),
        "identity_match": identity_match,
        "compared_fields": list(fields), "families": family_summary, "details": details,
        "interpretation": "Exact equality is reported per tensor/scalar; nonexact floating tensors include max absolute and relative deltas. This compares one shared schedule, not task or runtime qualification.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumed", required=True, type=Path)
    parser.add_argument("--uninterrupted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare(args.resumed, args.uninterrupted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "steps": [result["left_step"], result["right_step"]],
                      "families": result["families"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
