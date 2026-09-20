#!/usr/bin/env python3
"""Run a trace-linked, unbounded teacher evaluation of one natlang application."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from applications.evaluation_backends import (BACKENDS, application_revision,
                                              evaluation_harness_revision, load_application_cases)
from applications.experiment_lab import Candidate, ExperimentLab
from applications.teacher import teacher_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=sorted(BACKENDS))
    parser.add_argument("out", type=Path, help="new directory for journal, traces, raw turns and report")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--budget", type=int, default=1, help="number of cases to select; not a model run limit")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--world-seed", type=int, required=True)
    parser.add_argument("--analyst-seed", type=int, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists")
    cases = [case for case in load_application_cases(args.cases)
             if case.payload["suite"] == args.suite]
    if not any(case.split == args.split for case in cases):
        parser.error("no cases in the requested suite and split")
    args.out.mkdir(parents=True)
    traces = args.out / "traces"
    journal = args.out / "journal.jsonl"
    turns = args.out / "teacher_turns.jsonl"
    factory = teacher_factory(args.server, turns_path=turns, validation_feedback="local")
    candidate = Candidate(args.suite, application_revision(args.suite),
                          args.model_id, args.model_seed, args.suite)
    lab = ExperimentLab(agent_factory=factory,
                        backend=BACKENDS[args.suite](agent_factory=factory, trace_dir=traces),
                        world_seed=args.world_seed, analyst_seed=args.analyst_seed,
                        analyst_model_id=args.model_id, trace_dir=traces,
                        harness_revision=evaluation_harness_revision(),
                        harness_config={"typed_chat": True,
                                        "validation_feedback": "local"})
    print(f"Running {args.suite}; journal: {journal}; traces: {traces}", flush=True)
    try:
        result = lab.run(f"Which {args.suite} cases complete and meet their independent rubric?",
                         cases, [candidate], budget=args.budget, repeats=args.repeats,
                         split=args.split, journal_path=journal)
    except (Exception, KeyboardInterrupt) as exc:
        with (args.out / "failure.json").open("x") as stream:
            json.dump({"type": type(exc).__name__, "message": str(exc),
                       "traceback": traceback.format_exc()}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        raise
    report = args.out / "result.json"
    with report.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Completed: {report}; raw teacher turns: {turns}", flush=True)


if __name__ == "__main__":
    main()
