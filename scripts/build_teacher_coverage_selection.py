#!/usr/bin/env python3
"""Atomically select a balanced teacher batch from every current reviewed corpus."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def build(paths: list[Path], *, per_family: int, seed: int) -> tuple[list[dict], dict]:
    grouped, identities, sources, added = defaultdict(list), {}, {}, set()
    for path in sorted(set(paths)):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train" or row.get("semantics", {}).get("contains_templates"):
                continue
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            revision = hashlib.sha256(encoded).hexdigest()
            previous = identities.setdefault(row["id"], revision)
            if previous != revision:
                raise ValueError(f"conflicting revisions for program id {row['id']}")
            if row["id"] in added:
                continue
            added.add(row["id"])
            grouped[row.get("family", row["kind"])].append(row)
    short = {family: len(rows) for family, rows in grouped.items() if len(rows) < per_family}
    if short:
        raise ValueError(f"families below {per_family} eligible programs: {short}")
    selected = []
    for family, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest())
        selected.extend(rows[:per_family])
    return selected, {"schema": "natlang.teacher_coverage_selection/1", "seed": seed,
                      "per_family": per_family, "families": len(grouped),
                      "programs": len(selected), "sources": sources,
                      "selected_ids": [row["id"] for row in selected]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--glob", action="append", dest="patterns",
                        default=["data/external_pilot/synthetic-*-reviewed-*.ir.jsonl"])
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=909)
    args = parser.parse_args()
    paths = [Path(path) for pattern in args.patterns for path in glob.glob(pattern)]
    if not paths or args.per_family < 1:
        parser.error("corpus glob found no files or per-family is not positive")
    rows, manifest = build(paths, per_family=args.per_family, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = args.out.with_suffix(args.out.suffix + ".building")
    staging.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    staging.replace(args.out)
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"{len(rows)} programs across {manifest['families']} families -> {args.out}")


if __name__ == "__main__":
    main()
