#!/usr/bin/env python3
"""Paired live-model probe for tools-v2 versus explicit-engine tools-v3."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.invocation import ModelSettings, RunOptions, SeedPolicy
from natlang.native import NativeCallDecoder
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.trace import TraceRecorder, TraceReader
from natlang.values import load_program

PROGRAM = {"$lambda": {"type": "Lambda<{ values: Num[] }, Num>",
                       "instructions": "Add all numbers in args/values exactly. Use run_code to compute the sum, "
                                       "then write that value to return.",
                       "args": {"values": [4, 7, 9]}}}
EXPECTED = 20


def run_profile(decoder, *, explicit_engine: bool, seed: int, model_id: str):
    profile = "tools-v3" if explicit_engine else "tools-v2"
    options = RunOptions(seed=SeedPolicy("derived", seed),
                         model=ModelSettings(temperature=0, max_turns=12,
                                             max_tokens=1200, turn_tokens=300))
    trace = TraceRecorder({"run_id": options.run_id,
                           "source_sha256": hashlib.sha256(json.dumps(PROGRAM, sort_keys=True).encode()).hexdigest(),
                           "model": model_id, "tool_schema": profile, "seed_policy": vars(options.seed),
                           "capture": "paired-student-probe"})
    log = []
    runtime = Runtime(lambda lam: ToolAgent(decoder, validation_feedback="local", log=log),
                      options=options, trace_sink=trace, engine_selection=explicit_engine)
    outcome, value = runtime.run_root(load_program(PROGRAM))
    actions = TraceReader(trace.events).of_kind("action")
    code_actions = [a for a in actions if a.get("name") == "run_code"]
    return {"profile": profile, "outcome": outcome.kind, "value": value if outcome.kind == "done" else None,
            "correct": outcome.kind == "done" and value == EXPECTED,
            "false_success": outcome.kind == "done" and value != EXPECTED,
            "wrong_engine": sum(a["outcome"] in ("rejected", "error") and
                                ("engine" not in a.get("arguments", {}) or
                                 a["arguments"].get("engine") != "quickjs-isolated")
                                for a in code_actions),
            "malformed_code": sum(a["outcome"] == "error" for a in code_actions),
            "wrong_bindings": sum(a.get("name") == "call" and a["outcome"] in ("rejected", "refused")
                                  for a in actions),
            "actions": len(actions), "turns": getattr(decoder, "usage", {}).get("turns"),
            "completion_tokens": getattr(decoder, "usage", {}).get("completion_tokens"),
            "trace": trace.events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing overwrite: {args.out}")
    results = [run_profile(NativeCallDecoder(base_url=args.server), explicit_engine=version,
                           seed=args.seed, model_id=args.model_id) for version in (False, True)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"version": "engine-surface-pilot/1", "seed": args.seed,
                                    "model": args.model_id, "results": results}, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "trace"} for row in results], indent=2))


if __name__ == "__main__":
    main()
