#!/usr/bin/env python3
"""Parallel, resumable, atomically sharded build of IR verification traces."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from materialize_ir import harness_hashes, materialize_record
from program_ir import read_jsonl


def run_shard(job):
    first, records, destination, recovery_rate = job
    path = Path(destination)
    temp = path.with_suffix(path.suffix + ".tmp")
    counts = Counter()
    with gzip.open(temp, "wt", encoding="utf-8") as stream:
        for record in records:
            lines, episodes = materialize_record(record, recovery_rate)
            provisional = bool(record["semantics"].get("contains_templates"))
            for line in lines:
                stream.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
                counts["turns"] += 1
                counts["provisional_turns" if provisional else "eligible_turns"] += 1
                counts["skill:" + line["skill"]] += 1
            counts["programs"] += 1
            counts["provisional_programs" if provisional else "eligible_programs"] += 1
            counts["episodes"] += episodes
    # The part only appears after the whole shard has passed verification.
    temp.replace(path)
    return first, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path, help="output directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--max-programs", type=int)
    parser.add_argument("--recovery-rate", type=float, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.shard_size < 1:
        parser.error("workers and shard-size must be positive")
    if args.dst.suffix:
        parser.error("dst must be a directory, not a file")
    args.dst.mkdir(parents=True, exist_ok=True)
    identity = {"materializer": "natlang.materializer-shards/1",
                "ir_sha256": hashlib.sha256(args.src.read_bytes()).hexdigest(),
                "harness_sha256": harness_hashes(), "recovery_rate": args.recovery_rate,
                "shard_size": args.shard_size, "max_programs": args.max_programs}
    manifest = args.dst / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["identity"] != identity:
            parser.error("source or harness changed; choose a new output directory")
    elif any(args.dst.iterdir()):
        parser.error("output directory has files but no manifest")
    else:
        manifest.write_text(json.dumps({"identity": identity}, indent=2) + "\n")

    jobs = []
    batch = []
    first = 0
    total = 0
    for record in read_jsonl(args.src):
        if args.max_programs is not None and total >= args.max_programs:
            break
        if not batch:
            first = total
        batch.append(record)
        total += 1
        if len(batch) == args.shard_size:
            path = args.dst / f"part-{first:07d}.jsonl.gz"
            if not path.exists():
                jobs.append((first, batch, str(path), args.recovery_rate))
            batch = []
    if batch:
        path = args.dst / f"part-{first:07d}.jsonl.gz"
        if not path.exists():
            jobs.append((first, batch, str(path), args.recovery_rate))
    counts = Counter()
    with multiprocessing.Pool(args.workers) as pool:
        for first, result in pool.imap_unordered(run_shard, jobs):
            counts.update(result)
            print(f"verified shard {first}: {result['programs']} programs, {result['turns']} turns", flush=True)
    manifest.write_text(json.dumps({"identity": identity, "programs_in_ir": total,
                                    "new_shards": len(jobs), "new_counts": dict(counts)}, indent=2) + "\n")
    print(json.dumps({"output": str(args.dst), "programs_in_ir": total,
                      "new_shards": len(jobs), "new_counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
