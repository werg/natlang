#!/usr/bin/env python3
"""Make a disposable training view of teacher trajectory IR without changing its source."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.teacher_trajectory_ir import VERSION


def project(row, *, tool_map, accepted_only=False, drop_reasoning=False,
            empty_success_reply=False):
    if row["version"] != VERSION:
        raise ValueError(f"unsupported teacher trajectory IR: {row['version']}")
    if accepted_only and not row["outcome"]["accepted"]:
        return None
    selected = copy.deepcopy(row)
    for turn in selected["trajectory"]:
        assistant = turn["assistant"]
        if drop_reasoning:
            assistant["reasoning"] = None
            for review in turn["reviews"]:
                review["reasoning"] = None
        indexes = {}
        calls = []
        for old_index, call in enumerate(assistant["calls"]):
            name = tool_map.get(call["tool"], call["tool"])
            if name is None:
                if call["tool"] == "end_turn":
                    turn["terminal_migrated"] = True
                continue
            indexes[old_index] = len(calls)
            calls.append({**call, "tool": name})
        assistant["calls"] = calls
        turn["executions"] = [{**execution, "call_index": indexes[execution["call_index"]],
                               "name": calls[indexes[execution["call_index"]]]["tool"]}
                              for execution in turn["executions"] if execution["call_index"] in indexes]
        turn["reviews"] = [{**review, "call_index": indexes[review["call_index"]]}
                           for review in turn["reviews"] if review["call_index"] in indexes]
    if empty_success_reply and selected["outcome"]["status"] == "done":
        if selected["trajectory"] and selected["trajectory"][-1]["assistant"]["calls"]:
            selected["trajectory"].append({"index": len(selected["trajectory"]),
                                            "function": selected["trajectory"][-1].get("function") or selected["task"].get("function"),
                                            "context": [], "tools_offered": None,
                                            "assistant": {"content": "", "reasoning": None,
                                                          "calls": []},
                                            "reviews": [], "executions": [],
                                            "raw_response_sha256": None,
                                            "synthesized_end_turn": True})
        elif selected["trajectory"]:
            selected["trajectory"][-1]["assistant"]["content"] = ""
    selected["projection"] = {"tool_map": tool_map, "accepted_only": accepted_only,
                              "drop_reasoning": drop_reasoning,
                              "empty_success_reply": empty_success_reply}
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--drop-reasoning", action="store_true")
    parser.add_argument("--empty-success-reply", action="store_true")
    parser.add_argument("--tool-map", type=Path,
                        help='JSON object of canonical tool renames; null drops a tool, e.g. {"end_turn":null}')
    args = parser.parse_args()
    if args.dst.exists():
        parser.error(f"refusing overwrite: {args.dst}")
    staged = args.dst.with_suffix(args.dst.suffix + ".building")
    if staged.exists():
        parser.error(f"refusing overwrite: {staged}")
    tool_map = json.loads(args.tool_map.read_text()) if args.tool_map else {}
    if not isinstance(tool_map, dict) or any(not isinstance(k, str) or
                                             (v is not None and not isinstance(v, str))
                                             for k, v in tool_map.items()):
        parser.error("--tool-map must be a JSON object from tool names to names or null")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.src.open() as source, staged.open("x") as target:
        for line in source:
            if not line.strip():
                continue
            row = project(json.loads(line), tool_map=tool_map,
                          accepted_only=args.accepted_only,
                          drop_reasoning=args.drop_reasoning,
                          empty_success_reply=args.empty_success_reply)
            if row is not None:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    staged.replace(args.dst)
    print(f"{count} projected teacher trajectories -> {args.dst}")


if __name__ == "__main__":
    main()
