#!/usr/bin/env python3
"""Run a finite, trace-linked natlang experiment over one semantic-merge programme."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from applications.experiment_lab import (Candidate, ExperimentLab, MergeTrialBackend,
                                         load_merge_cases, merge_source_revision)
from applications.teacher import teacher_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--program", default="merge_history")
    parser.add_argument("--question", default="Which merge strategy is repeatable, and which results need review?")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--strategies", nargs="+", choices=("history", "incremental"),
                        default=("history", "incremental"))
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--world-seed", type=int, required=True)
    parser.add_argument("--analyst-seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args()
    trace_dir = args.out.parent / (args.out.stem + "-traces")
    journal_path = args.out.parent / (args.out.stem + ".journal.jsonl")
    if args.out.exists() or trace_dir.exists() or journal_path.exists():
        parser.error("the output, trace directory or journal already exists")
    cases = [case for case in load_merge_cases(args.cases)
             if case.payload["program"] == args.program]
    agent_factory = teacher_factory(args.server)
    candidates = [Candidate(strategy, merge_source_revision(args.program, strategy),
                            args.model_id, args.model_seed, strategy)
                  for strategy in args.strategies]
    lab = ExperimentLab(agent_factory=agent_factory,
                        backend=MergeTrialBackend(agent_factory=agent_factory, trace_dir=trace_dir),
                        world_seed=args.world_seed, analyst_seed=args.analyst_seed,
                        analyst_model_id=args.model_id, trace_dir=trace_dir)
    result = lab.run(args.question, cases, candidates, budget=args.budget,
                     repeats=args.repeats, split=args.split, journal_path=journal_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(f"{len(result['report']['trials'])} trials; report: {args.out}; traces: {trace_dir}")


if __name__ == "__main__":
    main()
