#!/usr/bin/env python3
"""Fail-closed disk preflight for the bounded-memory training corpus preparer."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="filesystem whose free space will be checked")
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="each exact corpus input; its stored byte size is included")
    parser.add_argument("--gzip-expanded-input", type=Path, action="append", default=[],
                        help="gzip JSONL source whose expanded byte size is streamed and included")
    parser.add_argument("--extra-estimated-input-bytes", type=int, default=0,
                        help="explicit downstream materialization growth estimate; reported separately")
    parser.add_argument("--estimate-note", action="append", default=[],
                        help="explain the origin and limits of each growth estimate")
    parser.add_argument("--multiplier", type=int, default=5)
    parser.add_argument("--reserve-bytes", type=int, default=2 * 1024 ** 3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.multiplier < 1 or args.reserve_bytes < 0 or args.extra_estimated_input_bytes < 0:
        parser.error("multiplier must be positive; reserve and growth estimate must be nonnegative")
    root = args.path.resolve()
    inputs = []
    for raw in args.input:
        path = raw.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"training corpus input missing: {path}")
        inputs.append({"path": str(path), "bytes": path.stat().st_size})
    expanded_inputs = []
    for raw in args.gzip_expanded_input:
        path = raw.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"compressed input missing: {path}")
        expanded = 0
        with gzip.open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                expanded += len(block)
        expanded_inputs.append({"path": str(path), "compressed_bytes": path.stat().st_size,
                                "expanded_bytes": expanded})
    stored_total = sum(item["bytes"] for item in inputs)
    expanded_total = sum(item["expanded_bytes"] for item in expanded_inputs)
    total = stored_total + expanded_total + args.extra_estimated_input_bytes
    required = total * args.multiplier + args.reserve_bytes
    free = shutil.disk_usage(root).free
    report = {
        "schema": "natlang.training_resource_preflight/1",
        "filesystem_path": str(root),
        "inputs": inputs,
        "gzip_expanded_inputs": expanded_inputs,
        "estimated_materialization_growth_bytes": args.extra_estimated_input_bytes,
        "estimate_notes": args.estimate_note,
        "estimate_is_peak_guarantee": False,
        "peak_space_note": "This is a conservative preflight floor based on declared inputs and estimates, not a guarantee of peak free-space sufficiency; SQLite, rendered copies, caches, and filesystem overhead can increase peak use.",
        "stored_input_bytes": stored_total,
        "expanded_source_bytes": expanded_total,
        "input_bytes": total,
        "multiplier": args.multiplier,
        "reserve_bytes": args.reserve_bytes,
        "required_free_bytes": required,
        "available_free_bytes": free,
        "ready": free >= required,
        "policy": "same conservative 5x input plus 2 GiB reserve used by prepare_training_stages.py --streaming; includes streamed gzip expansion and separately declared materialization estimate",
        "training_started": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    if not report["ready"]:
        print(f"resource_wait: need {required} free bytes, have {free}", file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
