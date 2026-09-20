#!/usr/bin/env python3
"""Small paired Bonsai probe; writes one audited result per invocation.

Run each case with --segment-turns off and a small integer. This is a probe,
not a corpus collector, and never writes training data or reference values.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from applications.teacher import PROMPT
from natlang.decoder import LlamaServerDecoder
from natlang.invocation import RunOptions, SeedPolicy
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.trace import TraceRecorder
from natlang.values import dump, load_program
from scripts.collect_scenario_teacher import _root
from scripts.program_ir import lower


def case(name):
    if name == "partial_record":
        return (load_program({"$lambda": {
            "type": "Lambda<{ values: Num[] }, { sum: Num, count: Num }>",
            "instructions": ("Use run_code to add the numbers in args/values. Write that result to "
                             "return/sum. Then use run_code to count the numbers and write that "
                             "result to return/count.")}, }),
            {"sum": 14, "count": 4})
    if name == "map_then_count":
        path = ROOT / "runs/teacher-program-simple-s73-remaining.ir.jsonl"
        for line in path.open():
            row = json.loads(line)
            record = row["task"]["program_ir"]
            if record["id"] == "73:map_then_count:13":
                program = lower(record)
                return _root(program), program.expected
    raise ValueError(f"unknown case: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("partial_record", "map_then_count"), required=True)
    parser.add_argument("--segment-turns", required=True, help="off or a positive integer")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    parser.add_argument("--seed", type=int, default=907)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".trace.jsonl").exists():
        parser.error("output or trace already exists")
    segment_turns = None if args.segment_turns == "off" else int(args.segment_turns)
    if segment_turns is not None and segment_turns < 1:
        parser.error("segment turns must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    root, expected = case(args.case)
    if args.case == "partial_record":
        root.in_["values"] = [2, 3, 4, 5]
    decoder = LlamaServerDecoder(args.server,
        chat_extra={"thinking_budget_tokens": 256, "top_p": 0.95, "top_k": 20,
                    "chat_template_kwargs": {"reasoning_effort": "low"}},
        tool_aliases={"call": "call_function"}, json_text_values=True)
    captured = []
    recorder = TraceRecorder({"run_id": f"continuation-{args.case}-{args.segment_turns}",
                              "capture": "continuation-comparison"},
                             args.out.with_suffix(".trace.jsonl"))
    options = RunOptions(seed=SeedPolicy("derived", args.seed))
    started = time.monotonic()
    try:
        outcome, value = Runtime(lambda lam: ToolAgent(
            decoder, system_prompt=PROMPT, temperature=0, validation_feedback="caller",
            segment_turns=segment_turns, teacher_turns=captured, max_seconds=360),
            options=options, trace_sink=recorder).run_root(root)
        actual = dump(value) if outcome.kind == "done" else None
        requests = [e for e in recorder.events if e["kind"] == "model_request" and e.get("phase") == "end"]
        checkpoints = [e for e in recorder.events if e["kind"] == "checkpoint"]
        actions = [e for e in recorder.events if e["kind"] == "action"]
        summary = {"case": args.case, "segment_turns": segment_turns, "seed": args.seed,
                   "probe_deadline_seconds": 360,
                   "status": outcome.kind, "detail": outcome.detail, "expected": expected,
                   "actual": actual, "correct": outcome.kind == "done" and actual == expected,
                   "elapsed_seconds": round(time.monotonic() - started, 2),
                   "model_requests": len(requests), "checkpoints": len(checkpoints),
                   "prompt_tokens": sum(e.get("prompt_tokens") or 0 for e in requests),
                   "completion_tokens": sum(e.get("completion_tokens") or 0 for e in requests),
                   "max_prompt_tokens": max((e.get("prompt_tokens") or 0 for e in requests), default=0),
                   "max_messages": max((len(t["messages_before"]) for t in captured), default=0),
                   "actions": [{"name": e["name"], "outcome": e["outcome"]} for e in actions],
                   "notes": [e.get("note", "") for e in checkpoints]}
        args.out.write_text(json.dumps({"summary": summary, "teacher_turns": captured}, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
