#!/usr/bin/env python3
"""Check candidate actions against a scenario's semantic outcome contract.

This is a structural admission gate. Effect and final-value claims also require
execution in the runtime; materialize_ir runs that execution for references.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from program_ir import read_jsonl
from natlang.scenario import CALL_MODES, action_matches

SEMANTIC_TOOLS = {"call", "write", "write_value", "copy_value", *CALL_MODES,
                  "report_error", "report_blocker"}
SCOPE_TOOLS = {"eval", "read_value", "write_value", "return_value", "mark_lines",
               "report_blocker", "report_error"}


def target_actions(rows):
    actions = []
    for row in rows:
        for call in row["target"].get("tool_calls", []):
            name = call["function"]["name"]
            if name in SEMANTIC_TOOLS:
                actions.append({"tool": name, "arguments": json.loads(call["function"]["arguments"])})
    return actions


def audit_scope_turns(rows):
    """Validate the shape of scope-eval targets without applying legacy action contracts.

    Scope eval deliberately expresses work as ordinary declarations and line
    closure.  Its exact semantics are checked by the teacher replay and
    runtime admission; this pass only guards the shared IR/training boundary
    against silently dropping or inventing tool choices.
    """
    for row in rows:
        target = row.get("target") or {}
        for call in target.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if name not in SCOPE_TOOLS:
                raise ValueError(f"{row.get('id', '<row>')}: unknown scope-eval tool: {name}")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{row.get('id', '<row>')}: invalid scope-eval arguments") from exc
            if not isinstance(args, dict):
                raise ValueError(f"{row.get('id', '<row>')}: scope-eval arguments are not an object")
    return True


def audit_actions(record, actions, *, complete=True):
    """Raise on changed destinations, omitted actions, or a wrong terminal result.

    For proposal reviews, use complete=False to check a proposed prefix.
    Ancillary arguments such as line completion flags are ignored.
    """
    if record["kind"] != "lambda_scenario":
        raise ValueError("outcome admission requires a lambda_scenario record")
    contract = record["semantics"]["contract"]
    expected = contract["required_actions"]
    constraints = {item["function"]: item for item in contract["constrained_calls"]}
    position = 0
    for action in actions:
        name, args = action["tool"], action["arguments"]
        if name not in SEMANTIC_TOOLS:
            continue
        if name in ({"call"} | CALL_MODES) and args.get("function") in constraints:
            rule = constraints[args["function"]]
            if not action_matches({"name": name, "arguments": args}, {"name": "call", "arguments": rule}):
                raise ValueError(f"{record['id']}: changed required call destination or inputs: {args}")
            # An impossible direct call may be attempted and rejected before the error report.
            if position < len(expected) and expected[position]["tool"] != "call":
                continue
        if position >= len(expected):
            raise ValueError(f"{record['id']}: extra semantic action: {action}")
        required = expected[position]
        if not action_matches({"name": name, "arguments": args},
                              {"name": required["tool"], "arguments": required["arguments"]}):
            raise ValueError(f"{record['id']}: expected {required}, got {action}")
        position += 1
    if complete and position != len(expected):
        raise ValueError(f"{record['id']}: missing required actions: {expected[position:]}")
    if complete and contract["kind"] in {"error", "blocked"}:
        terminal = expected[-1]
        if contract["explanation"] != next(iter(terminal["arguments"].values())):
            raise ValueError(f"{record['id']}: terminal explanation differs from contract")
    return True


def audit_turns(record, rows, *, complete=True):
    offered = {tool["function"]["name"] for row in rows for tool in row.get("tools", [])
               if isinstance(tool, dict) and isinstance(tool.get("function"), dict)}
    if "eval" in offered or "return_value" in offered:
        return audit_scope_turns(rows)
    return audit_actions(record, target_actions(rows), complete=complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path)
    parser.add_argument("turns", type=Path, help="JSONL or JSONL.gz turns grouped by program_id")
    args = parser.parse_args()
    records = {record["id"]: record for record in read_jsonl(args.ir)}
    opener = gzip.open if args.turns.suffix == ".gz" else open
    grouped = {}
    with opener(args.turns, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            grouped.setdefault(row["program_id"], []).append(row)
    for program_id, rows in grouped.items():
        audit_turns(records[program_id], rows)
    print(json.dumps({"admitted_programs": len(grouped), "turns": sum(map(len, grouped.values()))}))


if __name__ == "__main__":
    main()
