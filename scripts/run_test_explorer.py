#!/usr/bin/env python3
"""Explore dependency-plan contracts with isolated natlang trials."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from applications.test_explorer import Explorer, GraphCase
from natlang.native import NativeCallDecoder
from natlang.tool_agent import ToolAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cases", type=Path, default=ROOT / "codebases/test_explorer/scenarios/cases.jsonl")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--analyst-seed", type=int, required=True)
    parser.add_argument("--target-seed", type=int, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    trace_dir = args.out.parent / (args.out.stem + "-traces")
    if args.out.exists() or trace_dir.exists():
        parser.error("output or trace directory already exists")
    cases = [GraphCase(row["id"], row["group"], row["description"],
                       tuple(row["tasks"]), row["split"])
             for row in (json.loads(line) for line in args.cases.read_text().splitlines() if line.strip())]
    decoder = NativeCallDecoder(base_url=args.server)
    factory = lambda lam: ToolAgent(decoder, validation_feedback="caller")
    result = Explorer(analyst_factory=factory, target_factory=factory, model_id=args.model_id,
                      analyst_seed=args.analyst_seed, target_seed=args.target_seed,
                      trace_dir=trace_dir).run("Find violations of the dependency-plan graph contract",
                                              cases, budget=args.budget, split=args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"{len(result['observations'])} cases; report: {args.out}")


if __name__ == "__main__":
    main()
