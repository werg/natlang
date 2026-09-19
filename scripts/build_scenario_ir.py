#!/usr/bin/env python3
"""Freeze current failure and support cases before their harness is rendered."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_failures import matched_cases
from generate_agent_support import support_cases
from natlang.corpus import file_digest
from program_ir import VERSION


def semantic_steps(steps):
    operations = []
    for action, args in steps:
        if action == "mark_done":
            continue
        if action == "call":
            op = {"op": "invoke", "function": args["function"], "target": args["to"]}
            if "done" in args:
                op["completion"] = args["done"]
            for old, new in (("inputs", "arguments"), ("over", "foreach"),
                             ("init", "initial"), ("until", "until"), ("max", "max_steps")):
                if old in args:
                    op[new] = args[old]
        elif action == "write":
            op = {"op": "assign", "target": args["path"], "value_type": args["type"]}
            if "done" in args:
                op["completion"] = args["done"]
            if "source" in args:
                op["from"] = args["source"]
            elif "value" in args:
                op["value"] = args["value"]
        elif action == "report_error":
            op = {"op": "fail", "message": args["message"]}
        elif action == "report_blocker":
            op = {"op": "block", "missing": args["missing"]}
        else:
            raise ValueError(f"unsupported scenario step: {action}")
        operations.append(op)
    return operations


def scenario_record(case, source, seed, group, generator_revisions):
    program_id = f"scenario:{source}:{seed}:{group}:{case['name']}"
    fault = case.get("fault")
    if fault:
        action, args, expected_result = fault
        fault = {"action": action, "arguments": args,
                 "expected_result": expected_result}
    operations = semantic_steps(case["steps"])
    required = []
    for op in operations:
        if op["op"] == "invoke":
            required.append({"tool": "call", "arguments": {"function": op["function"], "to": op["target"],
                **({"inputs": op["arguments"]} if "arguments" in op else {}),
                **({"over": op["foreach"]} if "foreach" in op else {})}})
        elif op["op"] == "assign":
            required.append({"tool": "write", "arguments": {"path": op["target"], "type": op["value_type"],
                "source" if "from" in op else "value": op.get("from", op.get("value"))}})
        elif op["op"] == "fail":
            required.append({"tool": "report_error", "arguments": {"message": op["message"]}})
        elif op["op"] == "block":
            required.append({"tool": "report_blocker", "arguments": {"missing": op["missing"]}})
    constrained = []
    mandatory_call = ((case.get("allowed_failure_call") or (fault and fault["arguments"]))
                      if case.get("failure") else None)
    if mandatory_call:
        constrained.append({"function": mandatory_call["function"], "to": mandatory_call["to"],
                            "inputs": mandatory_call.get("inputs", {})})
    terminal = operations[-1] if operations else {}
    outcome = case.get("failure") or "done"
    contract = {"kind": outcome, "value": case.get("expected"),
                "explanation": terminal.get("message", terminal.get("missing")),
                "effects": case.get("effects") or [], "required_actions": required,
                "constrained_calls": constrained}
    return {"version": VERSION, "id": program_id, "kind": "lambda_scenario",
            "source": "synthetic-" + source, "split": "train",
            "source_ids": [program_id],
            "source_groups": [f"scenario:{source}:{seed}:{group}"],
            "source_revisions": generator_revisions,
            "license": "project-generated", "gold_sources": ["synthetic-generator"],
            "semantics": {"root": case["root"], "operations": operations,
                          "contract": contract, "injected_fault": fault}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", choices=["failures", "support"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--groups", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.groups < 1:
        ap.error("groups must be positive")
    repo = Path(__file__).resolve().parent.parent
    generators = [repo / "scripts/generate_failures.py"]
    if args.source == "support":
        generators.append(repo / "scripts/generate_agent_support.py")
    revisions = [file_digest(path) for path in generators]
    cases = matched_cases if args.source == "failures" else support_cases
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = args.out.with_suffix(args.out.suffix + ".building")
    digest = hashlib.sha256()
    counts = Counter()
    with staging.open("w") as stream:
        for group in range(args.groups):
            for case in cases(args.seed, group):
                record = scenario_record(case, args.source, args.seed, group, revisions)
                line = json.dumps(record, ensure_ascii=False) + "\n"
                stream.write(line)
                digest.update(line.encode())
                counts[case["name"]] += 1
    staging.replace(args.out)
    manifest = {"version": VERSION, "source": args.source, "seed": args.seed,
                "groups": args.groups, "generators_sha256": {
                    str(path.relative_to(repo)): revision
                    for path, revision in zip(generators, revisions)},
                "programs": sum(counts.values()), "cases": dict(counts),
                "ir_sha256": digest.hexdigest()}
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
