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


def target_actions(rows):
    actions = []
    for row in rows:
        for call in row["target"].get("tool_calls", []):
            name = call["function"]["name"]
            if name in SEMANTIC_TOOLS:
                actions.append({"tool": name, "arguments": json.loads(call["function"]["arguments"])})
    return actions


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
