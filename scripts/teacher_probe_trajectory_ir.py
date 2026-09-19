#!/usr/bin/env python3
"""Preserve teacher behavior-probe decisions as template-independent trajectory IR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.teacher_trajectory_ir import VERSION, digest, legacy_action_sequence, legacy_turns, new_turns


def convert_probe(doc, row, *, path: Path, index: int):
    modern = "teacher_turns" in row
    trajectory, opening = (new_turns(row), []) if modern else legacy_turns(row)
    limits = [] if modern else ["raw_server_response_unavailable", "reasoning_not_recorded",
                                "rejected_action_turns_may_be_absent"]
    if not modern and row.get("status") == "done" and trajectory and trajectory[-1]["assistant"]["calls"]:
        limits.append("terminal_turn_missing")
    return {"version": VERSION,
            "id": "teacher-probe:" + digest([str(path), index, row["case"], row.get("group")])[:20],
            "task": {"kind": "behavior_probe", "case": row["case"], "group": row.get("group"),
                     "leaf_program": row["program"], "source_program_ids": []},
            "provenance": {"audit_file": str(path), "audit_row": index,
                           "audit_row_sha256": digest(row),
                           "model": doc.get("model"),
                           "teacher_system_prompt": doc.get("system_prompt"),
                           "probe_args": doc.get("args"),
                           "source_hashes": {k: v for k, v in doc.items() if k.endswith("_sha256")}},
            "outcome": {"status": row.get("status"), "detail": row.get("detail"),
                        "value": row.get("value"), "accepted": row.get("pass"),
                        "expected": row.get("expected"),
                        "expected_failure": row.get("expected_failure"),
                        "required_calls_exact": row.get("required_calls_exact"),
                        "effects_correct": row.get("effects_correct")},
            "trajectory": trajectory, "harness_opening_context": opening,
            "legacy_action_log": row.get("log") if not modern else None,
            "legacy_action_sequence": legacy_action_sequence(row) if not modern else None,
            "capture_limits": limits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    if args.dst.exists():
        parser.error(f"refusing overwrite: {args.dst}")
    staged = args.dst.with_suffix(args.dst.suffix + ".building")
    if staged.exists():
        parser.error(f"refusing overwrite: {staged}")
    doc = json.loads(args.src.read_text())
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with staged.open("x") as target:
        for index, row in enumerate(doc["rows"], 1):
            target.write(json.dumps(convert_probe(doc, row, path=args.src, index=index),
                                    ensure_ascii=False) + "\n")
    staged.replace(args.dst)
    print(f"{len(doc['rows'])} converted teacher probe trajectories -> {args.dst}")


if __name__ == "__main__":
    main()
