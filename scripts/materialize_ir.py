#!/usr/bin/env python3
"""Verify program IR in the current harness and emit template-neutral turns."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import run_program
from audit_trajectory_admission import audit_turns
from natlang.native import _strip_private
from program_ir import read_jsonl, lower

SOURCE_FILES = ("scripts/materialize_ir.py", "scripts/program_ir.py", "scripts/generate.py",
                "scripts/generate_external.py", "scripts/materialize_ir_shards.py")


def harness_hashes():
    repo = Path(__file__).resolve().parent.parent
    paths = [repo / name for name in SOURCE_FILES]
    paths.extend((repo / "natlang").rglob("*.py"))
    paths.extend((repo / "natlang" / "prompts").rglob("*.md"))
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths)) if path.exists()}


def materialize_record(record, recovery_rate=0):
    samples, episodes = run_program(lower(record), recovery_seed=record["id"],
                                    recovery_rate=recovery_rate)
    if record["kind"] == "lambda_scenario":
        audit_turns(record, samples)
    provenance = {key: record[key] for key in ("source", "split", "source_ids",
                   "source_groups", "source_revisions", "license", "gold_sources")}
    lines = []
    for turn, sample in enumerate(samples):
        lines.append({"id": f"{record['id']}:{turn}", "program_id": record["id"],
                      "family": record["kind"], "ir_version": record["version"],
                      "provisional_gold": bool(record["semantics"].get("contains_templates")),
                      "ir_digest": hashlib.sha256(json.dumps(record, sort_keys=True,
                                       ensure_ascii=False).encode()).hexdigest(),
                      **provenance, **sample, "tools": _strip_private(sample["tools"])})
    return lines, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--max-programs", type=int)
    parser.add_argument("--recovery-rate", type=float, default=0)
    args = parser.parse_args()
    if args.max_programs is not None and args.max_programs < 1:
        parser.error("max-programs must be positive")
    source_hashes = harness_hashes()
    ir_hash = hashlib.sha256(args.src.read_bytes()).hexdigest()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    seen = set()
    opener = gzip.open if args.dst.suffix == ".gz" else open
    with opener(args.dst, "wt", encoding="utf-8") as out:
        for number, record in enumerate(read_jsonl(args.src)):
            if args.max_programs is not None and number >= args.max_programs:
                break
            if record["id"] in seen:
                raise ValueError(f"duplicate IR id: {record['id']}")
            seen.add(record["id"])
            lines, episodes = materialize_record(record, args.recovery_rate)
            for line in lines:
                out.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
                counts["turns"] += 1
                counts["provisional_turns" if line["provisional_gold"] else "eligible_turns"] += 1
                counts["skill:" + line["skill"]] += 1
            counts["programs"] += 1
            counts["provisional_programs" if record["semantics"].get("contains_templates") else "eligible_programs"] += 1
            counts["episodes"] += episodes
            if counts["programs"] % 1000 == 0:
                print(f"verified {counts['programs']} programs, {counts['turns']} turns", flush=True)
    manifest = {"materializer": "natlang.materializer/1", "ir_file": str(args.src),
                "ir_sha256": ir_hash, "harness_sha256": source_hashes,
                "recovery_rate": args.recovery_rate, "counts": dict(counts)}
    args.dst.with_suffix(args.dst.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.dst), "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
