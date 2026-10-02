#!/usr/bin/env python3
"""Concatenate JSONL inputs to an atomically published JSONL file, validating each row."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=args.output.name + ".", suffix=".pending", dir=args.output.parent)
    rows = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for raw in args.inputs:
                source = Path(raw)
                with source.open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise ValueError(f"{source}:{line_number}: expected JSON object")
                        output.write(line if line.endswith("\n") else line + "\n")
                        rows += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, args.output)
        dfd = os.open(args.output.parent, os.O_RDONLY)
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
    print(json.dumps({"output": str(args.output), "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
