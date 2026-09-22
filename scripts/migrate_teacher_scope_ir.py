#!/usr/bin/env python3
"""Conservatively project unambiguous legacy teacher leaves to scope-eval-v1.

The source trajectory remains embedded verbatim.  Only accepted leaf episodes
whose useful work is one successful return write (plus optional reads) are
projected.  Everything involving correction, nested writes, edits, calls, or
ambiguous completion is reported for regeneration instead of being rewritten.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from scripts.teacher_trajectory_ir import VERSION, digest


def scope_expression(path: str) -> str:
    parts = path.split("/")
    if not parts or parts[0] not in ("args", "let", "return"):
        raise ValueError("unsupported workspace path")
    expression = "result" if parts[0] == "return" else parts[0]
    if parts[0] == "let":
        if len(parts) < 2:
            raise ValueError("incomplete local path")
        expression, parts = parts[1], parts[2:]
    else:
        parts = parts[1:]
    for part in parts:
        if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", part):
            expression += "." + part
        elif part.isdigit():
            expression += "[" + part + "]"
        else:
            expression += "[" + json.dumps(part, ensure_ascii=False) + "]"
    return expression


def call(tool: str, arguments: dict, source=None) -> dict:
    return {"tool": tool, "source_tool": source or tool,
            "arguments": arguments, "call_id": None}


def turn(index: int, source: dict | None, calls: list[dict], *, synthesized: str | None = None) -> dict:
    original = source or {}
    assistant = original.get("assistant") or {}
    result = {"index": index, "function": original.get("function"), "phase": "action",
              "segment_turns": original.get("segment_turns"),
              "segment_messages": original.get("segment_messages"), "note": None,
              "context": [], "tools_offered": None,
              "assistant": {"content": assistant.get("content") or "",
                            "reasoning": assistant.get("reasoning"), "calls": calls},
              "reviews": copy.deepcopy(original.get("reviews") or []),
              "executions": [{"call_index": position, "name": item["tool"],
                              "args": copy.deepcopy(item["arguments"]), "kind": "ok",
                              "projected": True}
                             for position, item in enumerate(calls)],
              "raw_response_sha256": original.get("raw_response_sha256")}
    if synthesized:
        result["synthesized"] = synthesized
        result["assistant"]["content"] = ""
        result["assistant"]["reasoning"] = None
        result["reviews"] = []
    return result


def substantive_ranges(instructions: str) -> list[tuple[int, int]]:
    indexes = [i for i, line in enumerate(instructions.splitlines(), 1) if line.strip()]
    ranges = []
    for index in indexes:
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], index)
        else:
            ranges.append((index, index))
    return ranges


def migrate(row: dict) -> tuple[dict | None, str]:
    if row.get("version") != VERSION:
        return None, "unsupported-version"
    if not row.get("outcome", {}).get("accepted") or row["outcome"].get("status") != "done":
        return None, "not-accepted-done"
    if row.get("task", {}).get("kind") != "generative_leaf" or not row["task"].get("leaf_program"):
        return None, "not-linked-leaf"
    if row.get("provenance", {}).get("tool_schema") == "scope-eval-v1":
        return None, "already-scope-eval"

    useful = []
    for source_turn in row.get("trajectory") or []:
        executions = source_turn.get("executions") or []
        if any(item.get("kind") not in (None, "ok", "done", "completed") for item in executions):
            return None, "rejected-or-failed-action"
        for item in (source_turn.get("assistant") or {}).get("calls") or []:
            if item.get("tool") not in ("read", "write", "end_turn"):
                return None, "unsupported-action"
            if item.get("tool") != "end_turn":
                useful.append((source_turn, item))
    writes = [(source, item) for source, item in useful if item["tool"] == "write"]
    if len(writes) != 1:
        return None, "not-one-write"
    write_turn, write = writes[0]
    args = write.get("arguments") or {}
    if args.get("path") != "return" or set(args) - {"path", "type", "value", "source", "done"}:
        return None, "ambiguous-return-write"
    if ("value" in args) == ("source" in args):
        return None, "ambiguous-return-write"

    migrated_turns = []
    for source_turn, item in useful:
        legacy_args = item.get("arguments") or {}
        if item["tool"] == "read":
            try:
                expression = scope_expression(str(legacy_args.get("path") or ""))
            except ValueError:
                return None, "unsupported-read"
            migrated_turns.append(turn(len(migrated_turns), source_turn,
                                       [call("read_value", {"expression": expression}, "read")]))
        elif "value" in legacy_args:
            target = {"name": "result", "value": legacy_args["value"]}
            if legacy_args.get("type"):
                target["as_type"] = legacy_args["type"]
            migrated_turns.append(turn(len(migrated_turns), source_turn,
                                       [call("write_value", target, "write")]))
        else:
            try:
                expression = scope_expression(str(legacy_args["source"]))
            except ValueError:
                return None, "unsupported-copy"
            type_text = legacy_args.get("type")
            annotation = f": {type_text}" if type_text else ""
            code = f"const result{annotation} = {expression}; result"
            migrated_turns.append(turn(len(migrated_turns), source_turn,
                                       [call("eval", {"code": code}, "write")]))

    lam = row["task"]["leaf_program"]["$lambda"]
    ranges = substantive_ranges(lam.get("instructions", ""))
    if not ranges:
        return None, "no-substantive-lines"
    for start, end in ranges:
        args = {"start": start}
        if end != start:
            args["end"] = end
        migrated_turns.append(turn(len(migrated_turns), None, [call("mark_lines", args)],
                                   synthesized="line-closure"))
    migrated_turns.append(turn(len(migrated_turns), None,
                               [call("return_value", {"variable": "result"})],
                               synthesized="typed-return"))
    migrated_turns.append(turn(len(migrated_turns), None, [], synthesized="natural-end-turn"))

    selected = copy.deepcopy(row)
    selected["id"] = row["id"] + ":scope-eval-v1"
    selected["source_trajectory"] = copy.deepcopy(row.get("trajectory") or [])
    selected["trajectory"] = migrated_turns
    selected.setdefault("provenance", {})["source_tool_schema"] = selected["provenance"].get("tool_schema")
    selected["provenance"]["tool_schema"] = "scope-eval-v1"
    selected["projection"] = {"kind": "legacy-leaf-to-scope-eval-v1",
                              "source_row_sha256": digest(row),
                              "preserved_source_trajectory": True,
                              "synthesized": ["line-closure", "typed-return", "natural-end-turn"]}
    return selected, "migrated"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dst", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    if args.dst.exists():
        parser.error(f"refusing overwrite: {args.dst}")
    staged = args.dst.with_suffix(args.dst.suffix + ".building")
    if staged.exists():
        parser.error(f"refusing overwrite: {staged}")
    counts, selected, seen = Counter(), [], set()
    for source in args.sources:
        with source.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                migrated, reason = migrate(row)
                counts[reason] += 1
                if migrated is None:
                    continue
                key = (migrated["task"].get("reference_key"),
                       digest(migrated["outcome"].get("value")))
                if key in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(key)
                selected.append(migrated)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with staged.open("x") as out:
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    staged.replace(args.dst)
    print(json.dumps({"written": len(selected), "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
