#!/usr/bin/env python3
"""Run semantic-merge cases with a pinned teacher and capture reviewable trajectories.

Mechanical admission checks provenance, typed execution and expected conflict shape.
Semantic correctness still requires review against each case's rubric before SFT export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from natlang.codebase import load_function
from natlang.host import load
from natlang.invocation import ModelSettings, RunOptions, SeedPolicy
from natlang.native import NativeCallDecoder
from natlang.runtime import Runtime
from natlang.tool_agent import TOOLS_PROMPT, ToolAgent
from natlang.trace import TraceReader, TraceRecorder
from natlang.values import dump


ROOT = Path(__file__).resolve().parent.parent


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def mechanical_checks(case: dict, outcome: str, value: dict | None, reader: TraceReader) -> dict:
    source_ids = {update["id"] for update in case["inputs"]["updates"]}
    claims = ([*value["applied"], *(id for alt in value["alternatives"] for id in alt["update_ids"])]
              if value else [])
    complete = len(claims) == len(source_ids) and len(set(claims)) == len(source_ids) and set(claims) == source_ids
    conflict_shape = ("unresolved" if value and value["alternatives"] else "merged")
    if case["program"] == "merge_history" and value:
        conflict_shape = value["status"]
    try:
        reader.reconstruct()
        trace_ok = True
    except ValueError:
        trace_ok = False
    return {"done": outcome == "done", "complete_provenance": complete,
            "expected_conflict_shape": conflict_shape == case["expected"],
            "trace_reconstructable": trace_ok,
            "ready_for_semantic_review": (outcome == "done" and complete and
                                          conflict_shape == case["expected"] and trace_ok)}


def collect(case: dict, decoder, *, model_id: str, trace_path: Path,
            max_turns: int = 64, max_tokens: int = 4000, max_seconds: float = 900,
            temperature: float = 0) -> dict:
    if digest(case["inputs"]) != case["input_sha256"]:
        raise ValueError(f"input digest mismatch for {case['case_id']}")
    entry = ROOT / "codebases/semantic_merge" / f"{case['program']}.nl"
    definition = load_function(entry)
    source_hash = digest(definition.to_inline())
    options = RunOptions(seed=SeedPolicy("derived", case["root_seed"]),
                         model=ModelSettings(temperature=temperature, max_turns=max_turns,
                                             max_tokens=max_tokens, max_seconds=max_seconds))
    recorder = TraceRecorder({"run_id": options.run_id, "case_id": case["case_id"],
                              "source_sha256": source_hash, "input_sha256": case["input_sha256"],
                              "model": model_id, "seed_policy": vars(options.seed),
                              "model_settings": vars(options.model), "tool_schema": "tools-v2",
                              "capture": "semantic-merge-teacher"}, trace_path)
    turns = []
    rt = Runtime(lambda lam: ToolAgent(decoder, teacher_turns=turns,
                                      validation_feedback="caller"),
                 options=options, trace_sink=recorder)
    try:
        outcome, result = rt.run_root(load(entry, case["inputs"]))
        value = dump(result) if outcome.kind == "done" else None
        error = None
    except Exception as exc:
        outcome, value = None, None
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        recorder.close()
    reader = TraceReader(recorder.events)
    checks = mechanical_checks(case, outcome.kind if outcome else "exception", value, reader)
    return {"schema": "semantic-merge-teacher-run/v1", "case_id": case["case_id"],
            "group": case["group"], "model": model_id, "run_id": options.run_id,
            "source_sha256": source_hash, "input_sha256": case["input_sha256"],
            "prompt_sha256": hashlib.sha256(TOOLS_PROMPT.encode()).hexdigest(),
            "root_seed": case["root_seed"], "model_settings": vars(options.model),
            "trace": str(trace_path), "outcome": outcome.kind if outcome else "exception",
            "detail": outcome.detail if outcome else "", "error": error, "value": value,
            "rubric": case["rubric"], "checks": checks,
            "semantic_review": "pending", "teacher_turns": turns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="?", default=ROOT / "codebases/semantic_merge/scenarios/cases.jsonl")
    parser.add_argument("out", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--case-id")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-seconds", type=float, default=900)
    args = parser.parse_args()
    decoder = NativeCallDecoder(base_url=args.server)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = args.out.parent / (args.out.stem + "-traces")
    trace_dir.mkdir(exist_ok=False)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.case_id:
        cases = [case for case in cases if case["case_id"] == args.case_id]
    with args.out.open("x") as target:
        for case in cases[:args.limit]:
            trace_path = trace_dir / (case["case_id"].replace(":", "-") + ".jsonl")
            row = collect(case, decoder, model_id=args.model_id, trace_path=trace_path,
                          max_turns=args.max_turns, max_tokens=args.max_tokens,
                          max_seconds=args.max_seconds, temperature=args.temperature)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            target.flush()
            print(f"{case['case_id']}: {row['outcome']} reviewable={row['checks']['ready_for_semantic_review']}",
                  flush=True)


if __name__ == "__main__":
    main()
