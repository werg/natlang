#!/usr/bin/env python3
"""Make new, oracle-preserving inputs for frozen map and map/count programs.

Each variant stays in its parent's source group so related trajectories cannot
land on opposite sides of a program-level train/holdout split.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from program_ir import validate


def variant(record: dict, seed: int) -> dict | None:
    sem = record.get("semantics", {})
    if record.get("kind") != "lambda_source" or sem.get("operation") not in {"map", "map_count"}:
        return None
    items = sem.get("inputs", {}).get("tickets")
    oracles = sem.get("leaf_oracles")
    if not isinstance(items, list) or len(items) < 3 or not isinstance(oracles, list):
        return None
    rubric = sem["inputs"].get("rubric")
    gold = {}
    for oracle in oracles:
        supplied = oracle.get("input", {})
        if rubric is not None and supplied.get("rubric") != rubric:
            return None
        key = supplied.get("ticket")
        if not isinstance(key, str) or key in gold:
            return None
        gold[key] = oracle["output"]
    if len(set(items)) != len(items) or any(item not in gold for item in items):
        return None
    rng = random.Random(f"map-input-variant/1:{seed}:{record['id']}")
    order = list(range(len(items)))
    rng.shuffle(order)
    order = order[:max(2, len(items) - 1)]
    selected = [items[index] for index in order]
    if selected == items:
        return None
    updated = copy.deepcopy(record)
    updated["id"] = record["id"] + f":input-variant-{seed}"
    updated["semantics"]["inputs"]["tickets"] = selected
    updated["semantics"]["leaf_oracles"] = [oracle for oracle in oracles
        if oracle["input"]["ticket"] in selected]
    values = [gold[item] for item in selected]
    updated["semantics"]["expected"] = values if sem["operation"] == "map" else sum(values)
    updated["source_ids"] = record["source_ids"]
    updated["source_groups"] = record["source_groups"]
    updated["input_variant"] = {"parent_id": record["id"], "seed": seed,
                                  "selection": order}
    return validate(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    source_hash = hashlib.sha256()
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.source.open("rb") as source, args.output.open("x") as output:
        for raw in source:
            source_hash.update(raw)
            made = variant(json.loads(raw), args.seed)
            if made is not None:
                output.write(json.dumps(made, ensure_ascii=False) + "\n")
                count += 1
    manifest = {"schema": "natlang.map_input_variants/1", "source": str(args.source),
                "source_sha256": source_hash.hexdigest(), "seed": args.seed,
                "variants": count}
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"{count} variants -> {args.output}")


if __name__ == "__main__":
    main()
