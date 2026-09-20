#!/usr/bin/env python3
"""Show progress in a live natlang experiment without imposing a run limit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from applications.long_run import inspect_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if args.journal is None and args.trace_dir is None:
        parser.error("supply --journal or --trace-dir")
    print(json.dumps(inspect_run(journal=args.journal, trace_dir=args.trace_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
