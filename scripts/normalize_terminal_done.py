#!/usr/bin/env python3
"""Convert legacy trajectory JSONL to the reply-only tool surface before SFT export.

Accepts one .jsonl/.jsonl.gz file or a directory of part-*.jsonl.gz shards.
The destination must not exist, so the original corpus stays intact.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.terminal import reply_only_sample


def _open(path, mode):
    return gzip.open(path, mode + "t") if path.suffix == ".gz" else path.open(mode)


def convert_file(src, dst):
    rows = converted = 0
    with _open(src, "r") as inp, _open(dst, "w") as out:
        for line in inp:
            sample = json.loads(line)
            normalized = reply_only_sample(sample)
            converted += sample["skill"] == "done"
            out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            rows += 1
    return rows, converted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    if not args.src.exists() or args.dst.exists():
        ap.error("source must exist and destination must be new")
    sources = sorted(args.src.glob("part-*.jsonl.gz")) if args.src.is_dir() else [args.src]
    if not sources:
        ap.error("no JSONL shards found")
    if args.src.is_dir():
        args.dst.mkdir(parents=True)
    else:
        args.dst.parent.mkdir(parents=True, exist_ok=True)
    rows = converted = 0
    for src in sources:
        dst = args.dst / src.name if args.src.is_dir() else args.dst
        n, c = convert_file(src, dst)
        rows += n
        converted += c
    print(f"{rows} rows; {converted} terminal targets converted to replies -> {args.dst}")


if __name__ == "__main__":
    main()
