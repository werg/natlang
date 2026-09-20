#!/usr/bin/env python3
"""Analyse one checked or draft natlang function with a frozen evidence snapshot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from applications.type_studio import TypeStudio
from applications.teacher import teacher_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target")
    parser.add_argument("operation", choices=("infer", "check"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--evidence", type=Path, help="JSON with obligations, witnesses and required_effects")
    parser.add_argument("--draft", action="store_true", help="source has no complete signature; analyse as read-only text")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    evidence = json.loads(args.evidence.read_text()) if args.evidence else {}
    allowed = {"obligations", "witnesses", "required_effects"}
    if set(evidence) - allowed:
        parser.error("unknown evidence keys")
    studio = (TypeStudio.from_draft_text(args.target, args.source.read_text(), **evidence)
              if args.draft else TypeStudio.from_file(args.source, args.target, **evidence))
    trace = args.out.with_suffix(".trace.jsonl")
    if trace.exists():
        parser.error("trace already exists")
    result = studio.run(args.operation,
                        agent_factory=teacher_factory(args.server),
                        model_id=args.model_id, root_seed=args.seed, trace_path=trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as target:
        json.dump(result, target, indent=2, ensure_ascii=False)
        target.write("\n")
    print(f"{result['outcome']}: {args.out}; trace: {trace}")


if __name__ == "__main__":
    main()
