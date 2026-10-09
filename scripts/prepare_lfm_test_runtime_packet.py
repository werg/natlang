#!/usr/bin/env python3
"""Build a gold-separated runtime packet from a ready corpus's protected test rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256, canonical_json_sha256_hex as canonical_sha  # noqa: E402


def line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def prepare(source: Path, out: Path, *, source_manifest: Path) -> dict:
    source, out, source_manifest = Path(source), Path(out), Path(source_manifest)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("sha256") != sha256(source):
        raise ValueError("source is not the exact corpus pinned by its manifest")
    if "audit" in manifest and manifest.get("audit", {}).get("ready") is not True:
        raise ValueError("source audit manifest is not ready")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty packet directory {out}")
    out.mkdir(parents=True, exist_ok=True)

    group_splits: dict[str, set[str]] = defaultdict(set)
    alias_splits: dict[str, dict[str, set[str]]] = {
        "source_ids": defaultdict(set), "source_program_ids": defaultdict(set),
        "program_ids": defaultdict(set),
    }
    test_cases = {}
    train_rows = test_rows = 0
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split")
            if split not in ("train", "test"):
                raise ValueError(f"row {line_number} lacks an explicit train/test assignment")
            if split == "train":
                train_rows += 1
            else:
                test_rows += 1
            for group in row.get("source_groups", []) or []:
                group_splits[str(group)].add(split)
            for alias_kind in alias_splits:
                values = row.get(alias_kind, []) or []
                if alias_kind == "program_ids":
                    values = [row.get("program_id")] if row.get("program_id") else []
                for value in values:
                    alias_splits[alias_kind][str(value)].add(split)
            if split != "test":
                continue
            task = row.get("task", {})
            ir = task.get("program_ir") if isinstance(task, dict) else None
            if not isinstance(ir, dict) or not isinstance(ir.get("id"), str):
                raise ValueError(f"test row {line_number} has no embedded program IR")
            ir_id = ir["id"]
            fingerprint = canonical_sha(ir)
            existing = test_cases.get(ir_id)
            if not existing:
                semantics = ir.get("semantics", {})
                if not isinstance(semantics, dict) or "expected" not in semantics:
                    raise ValueError(f"test program {ir_id} has no separately recordable expected value")
                expected = semantics["expected"]
                test_cases[ir_id] = {
                    "program_ir": ir,
                    "program_ir_sha256": fingerprint,
                    "source_ref": row.get("source_ref"),
                    "source_ids": sorted(set(row.get("source_ids", []) or [])),
                    "source_groups": sorted(set(row.get("source_groups", []) or [])),
                    "expected": expected,
                    "expected_sha256": canonical_sha(expected),
                    "program_ir_variants": [fingerprint],
                    "decision_rows": 0,
                }
            else:
                if fingerprint not in existing["program_ir_variants"]:
                    existing["program_ir_variants"].append(fingerprint)
                if canonical_sha(ir.get("semantics", {}).get("expected")) != existing["expected_sha256"]:
                    raise ValueError(f"test rows disagree on gold value for program IR {ir_id}")
                # A case can span multiple teacher decision rows; retain the union
                # so the later result joins are auditable at case level.
                existing["source_ids"] = sorted(set(existing["source_ids"]) | set(row.get("source_ids", []) or []))
                existing["source_groups"] = sorted(set(existing["source_groups"]) | set(row.get("source_groups", []) or []))
            test_cases[ir_id]["decision_rows"] += 1

    cross = sorted(group for group, labels in group_splits.items() if len(labels) != 1)
    if cross:
        raise ValueError(f"source groups cross train/test boundary: {cross[:10]}")
    alias_cross = {kind: sorted(alias for alias, labels in aliases.items() if len(labels) != 1)
                   for kind, aliases in alias_splits.items()}
    alias_cross = {kind: values for kind, values in alias_cross.items() if values}
    if alias_cross:
        raise ValueError(f"source identity aliases cross train/test boundary: {alias_cross}")
    if not test_cases:
        raise ValueError("no test program IRs found")

    ir_path = out / "cases.ir.jsonl"
    gold_path = out / "gold-reference.jsonl"
    selected_path = out / "selection.jsonl"
    variants_path = out / "ir-variants.jsonl"
    groups_path = out / "protected-group-closure.json"
    with ir_path.open("x", encoding="utf-8") as ir_out, gold_path.open("x", encoding="utf-8") as gold_out, selected_path.open("x", encoding="utf-8") as sel_out, variants_path.open("x", encoding="utf-8") as variant_out:
        for ir_id in sorted(test_cases):
            case = test_cases[ir_id]
            ir_out.write(json.dumps(case["program_ir"], ensure_ascii=False, separators=(",", ":")) + "\n")
            gold_out.write(json.dumps({"id": ir_id, "program_ir_sha256": case["program_ir_sha256"],
                                       "expected": case["expected"]}, ensure_ascii=False,
                                      separators=(",", ":")) + "\n")
            sel_out.write(json.dumps({"id": ir_id, "program_ir_sha256": case["program_ir_sha256"],
                                      "split": "test", "source_ids": case["source_ids"],
                                      "source_groups": case["source_groups"],
                                      "decision_rows": case["decision_rows"],
                                      "program_ir_variant_count": len(case["program_ir_variants"])}, ensure_ascii=False,
                             separators=(",", ":")) + "\n")
            if len(case["program_ir_variants"]) > 1:
                variant_out.write(json.dumps({"id": ir_id,
                                              "selected_program_ir_sha256": case["program_ir_sha256"],
                                              "program_ir_variant_sha256": case["program_ir_variants"]},
                                             ensure_ascii=False, separators=(",", ":")) + "\n")
    group_receipt = {
        "schema": "lfm-test-source-group-closure/1",
        "source_group_split": {key: sorted(labels)[0] for key, labels in sorted(group_splits.items())},
        "aliases": {kind: {key: sorted(labels)[0] for key, labels in sorted(values.items())}
                    for kind, values in alias_splits.items()},
    }
    groups_path.write_text(json.dumps(group_receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    receipt = {
        "schema": "lfm-test-runtime-packet/1",
        "status": "prepared_test_only_not_executed",
        "source": {"path": str(source.resolve()), "sha256": sha256(source),
                   "manifest": str(source_manifest.resolve()),
                   "manifest_sha256": sha256(source_manifest),
                   "source_manifest_version": manifest.get("version")},
        "protected_split": {"train_decision_rows": train_rows, "test_decision_rows": test_rows,
                            "test_programs": len(test_cases), "source_groups": len(group_splits),
                            "source_ids": len(alias_splits["source_ids"]),
                            "source_program_ids": len(alias_splits["source_program_ids"]),
                            "train_test_crossing_source_groups": 0,
                            "train_test_crossing_aliases": 0},
        "program_ir_variant_cases": sum(len(case["program_ir_variants"]) > 1 for case in test_cases.values()),
        "artifacts": {name: {"path": str(path.resolve()), "sha256": sha256(path), "rows": line_count(path)}
                      for name, path in (("cases_ir", ir_path), ("gold_reference", gold_path),
                                         ("selection", selected_path), ("ir_variants", variants_path))},
        "source_group_closure": {"path": str(groups_path.resolve()), "sha256": sha256(groups_path),
                                 "labels": len(group_splits) + sum(len(v) for v in alias_splits.values())},
        "gold_separation": "gold values are repeated in the runtime oracle metadata inside cases.ir.jsonl but are not part of the model-visible prompt; gold-reference.jsonl is a separate audit join",
        "source_component_check": "all source_groups observed in the source corpus have exactly one split label",
    }
    receipt_path = out / "packet-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.out, source_manifest=args.source_manifest),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
