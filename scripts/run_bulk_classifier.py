#!/usr/bin/env python3
"""Run bounded, resumable Jev labeling now or across successive quota windows.

Run once from a daily scheduler, or pass --continuous to keep filling the
dataset. The service's free fast limit is currently 20,000 items/day; the
default 18,000-item tranche leaves room for retries and other API users.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def retry_after(batch_log: Path) -> float | None:
    if not batch_log.exists():
        return None
    with batch_log.open("rb") as stream:
        last = None
        for line in stream:
            if line.strip():
                last = line
    if last is None:
        return None
    row = json.loads(last)
    if row.get("http_status") != 429:
        return None
    value = row.get("headers", {}).get("retry-after")
    try:
        delay = float(value)
    except (TypeError, ValueError):
        return None
    return delay if 0 < delay <= 86400 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="shuffled normalized task JSONL")
    parser.add_argument("output", type=Path, help="append-only labeled JSONL")
    parser.add_argument("--daily-items", type=int, default=18000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--continuous", action="store_true",
                        help="wait across free-tier quota windows until all input tasks are labeled")
    args = parser.parse_args()
    if not 1 <= args.daily_items <= 20000 or not 1 <= args.batch_size <= 1000:
        parser.error("daily-items must be 1..20000 and batch-size 1..1000")
    labeler = Path(__file__).with_name("label_classifier.py")
    batch_log = args.output.with_suffix(args.output.suffix + ".batches.jsonl")
    while True:
        started = time.monotonic()
        before = lines(args.output)
        command = [sys.executable, str(labeler), str(args.input), str(args.output),
                   "--max-items", str(args.daily_items), "--max-requests", "1000",
                   "--batch-size", str(args.batch_size), "--compact-audit", "--retry-errors"]
        result = subprocess.run(command, check=False)
        written = lines(args.output) - before
        print(f"tranche wrote {written} item rows; labeler exit {result.returncode}", file=sys.stderr)
        if not args.continuous:
            return result.returncode
        if result.returncode != 0:
            delay = retry_after(batch_log)
            if delay is None:
                return result.returncode
            print(f"free quota reached; waiting {delay:.0f}s", file=sys.stderr)
            time.sleep(delay + 2)
            continue
        if written < args.daily_items:
            print("all available tasks processed", file=sys.stderr)
            return 0
        delay = max(0, 86400 - (time.monotonic() - started))
        print(f"daily tranche complete; waiting {delay:.0f}s", file=sys.stderr)
        time.sleep(delay)


if __name__ == "__main__":
    raise SystemExit(main())
