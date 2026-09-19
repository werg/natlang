#!/usr/bin/env python3
"""Read a reduction trace without loading or executing its programme."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from natlang.trace import TraceReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--kind", help="show only records of this kind")
    parser.add_argument("--state", action="store_true", help="show final captured state")
    args = parser.parse_args()
    reader = TraceReader.open(args.trace)
    if args.state:
        print(json.dumps({"coverage": reader.coverage(), "state": reader.final_state()}, indent=2,
                         ensure_ascii=False))
    else:
        print(json.dumps(reader.of_kind(args.kind) if args.kind else reader.events, indent=2,
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
