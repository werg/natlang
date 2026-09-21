#!/usr/bin/env python3
"""Build a deterministic teacher-polish set biased toward whole programs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ap.add_argument("--leaf-programs", type=int, default=128)
    args = ap.parse_args()
    if args.leaf_programs < 0:
        ap.error("--leaf-programs must be nonnegative")
    manifest_path = args.destination.with_suffix(args.destination.suffix + ".manifest.json")
    if args.destination.exists() or manifest_path.exists():
        ap.error("destination already exists")

    programs: dict[str, list[dict]] = defaultdict(list)
    with args.source.open() as stream:
        for line in stream:
            row = json.loads(line)
            programs[row.get("program_id") or row["id"]].append(row)

    whole = {pid for pid, rows in programs.items()
             if any(row.get("family") == "teacher_program" for row in rows)}
    leaf = [pid for pid, rows in programs.items()
            if pid not in whole and any(row.get("family") == "teacher_leaf" for row in rows)]
    leaf.sort(key=lambda pid: hashlib.sha256(pid.encode()).hexdigest())
    chosen = whole | set(leaf[:args.leaf_programs])
    rows = [row for pid in sorted(chosen) for row in programs[pid]]

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with args.destination.open("x") as out:
        for row in rows:
            text = json.dumps(row, ensure_ascii=False) + "\n"
            out.write(text)
            h.update(text.encode())
    source_manifest = args.source.with_suffix(args.source.suffix + ".manifest.json")
    prior = json.loads(source_manifest.read_text()) if source_manifest.exists() else {}
    result = {
        "schema": "natlang.teacher_polish_selection/1",
        "source": str(args.source),
        "source_sha256": digest(args.source),
        "renderer": prior.get("renderer"),
        "rows": len(rows),
        "programs": len(chosen),
        "whole_programs": len(whole),
        "leaf_programs": len(chosen - whole),
        "family_rows": dict(Counter(row.get("family") or "unknown" for row in rows)),
        "sha256": h.hexdigest(),
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
