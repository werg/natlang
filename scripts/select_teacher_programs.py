#!/usr/bin/env python3
"""Select a reproducible, family-balanced teacher batch from frozen program IR."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def select(source: Path, *, count: int, seed: int) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    source_hash = hashlib.sha256()
    with source.open("rb") as stream:
        for raw in stream:
            source_hash.update(raw)
            row = json.loads(raw)
            if row.get("split") != "train" or row.get("semantics", {}).get("contains_templates"):
                continue
            grouped[row.get("family", row["kind"])].append(row)
    if count > sum(map(len, grouped.values())):
        raise ValueError("requested more eligible teacher programs than available")
    for family, rows in grouped.items():
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest())
    positions = Counter()
    chosen = []
    families = sorted(grouped)
    while len(chosen) < count:
        for family in families:
            index = positions[family]
            if index < len(grouped[family]):
                chosen.append(grouped[family][index])
                positions[family] += 1
                if len(chosen) == count:
                    break
    if len({row["id"] for row in chosen}) != len(chosen):
        raise ValueError("source IR contains duplicate selected program IDs")
    return chosen, {"schema": "natlang.teacher_program_selection/1", "source": str(source),
                    "source_sha256": source_hash.hexdigest(), "seed": seed,
                    "programs": len(chosen),
                    "families": dict(Counter(row.get("family", row["kind"]) for row in chosen)),
                    "selected_ids": [row["id"] for row in chosen]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.output.exists():
        parser.error("count must be positive and output must be new")
    rows, manifest = select(args.source, count=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"{len(rows)} programs across {len(manifest['families'])} families -> {args.output}")


if __name__ == "__main__":
    main()
