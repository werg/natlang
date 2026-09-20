#!/usr/bin/env python3
"""Normalize teacher leaf audits into template-independent, linked trajectory IR.

The audit remains the immutable record of the exact server replies. This IR
keeps ordered decisions, results, exposed reasoning and provenance so later
renderers can migrate or select actions without parsing a model chat template.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.gen.codebases import ref_key

VERSION = "natlang.teacher_trajectory/1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def program_links(path: Path):
    links = defaultdict(set)
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            oracles = record["semantics"].get("leaf_oracles", {})
            if not isinstance(oracles, dict):
                continue
            for function, oracle in oracles.items():
                for case in oracle.get("cases", []):
                    links[ref_key(function, case["input"])].add(record["id"])
    return {key: sorted(ids) for key, ids in links.items()}


def leaf_programs(path: Path):
    """Recover exact executable leaf requests from frozen codebase definitions."""
    programs = {}

    def find_child(node, function):
        if isinstance(node, dict):
            lam = node.get("$lambda")
            if isinstance(lam, dict) and function in lam.get("codebase", {}):
                return lam["codebase"][function]
            for value in node.values():
                found = find_child(value, function)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = find_child(value, function)
                if found is not None:
                    return found
        return None

    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            oracles = record["semantics"].get("leaf_oracles", {})
            if not isinstance(oracles, dict):
                continue
            for function, oracle in oracles.items():
                child = find_child(record["semantics"].get("root"), function)
                if child is None or "args" not in child or "returns" not in child:
                    continue
                signature = "Lambda<{ " + ", ".join(f"{name}: {ty}" for name, ty in child["args"].items())
                signature += " }, " + child["returns"] + ">"
                for case in oracle.get("cases", []):
                    key = ref_key(function, case["input"])
                    doc = {"$lambda": {"type": signature, "function": function,
                                       "types": child.get("types", {}),
                                       "instructions": child["instructions"],
                                       "args": case["input"],
                                       "codebase": child.get("codebase", {})}}
                    prior = programs.setdefault(key, doc)
                    if prior != doc:
                        raise ValueError(f"conflicting frozen leaf definitions for {key}")
    return programs


def normalize_call(name, args, call_id=None):
    # Preserve the source spelling so a future migration can choose differently.
    canonical = {"call_function": "call", "done": "end_turn"}.get(name, name)
    return {"tool": canonical, "source_tool": name, "arguments": args,
            "call_id": call_id}


def raw_calls(message):
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        encoded = fn.get("arguments") or "{}"
        if isinstance(encoded, dict):
            args = encoded
        else:
            try:
                args = json.loads(encoded)
            except (TypeError, ValueError):
                args = {"__unparsed__": encoded}
        calls.append(normalize_call(fn.get("name", ""), args, call.get("id")))
    return calls


def new_turns(audit):
    result = []
    for index, source in enumerate(audit["teacher_turns"]):
        response = source.get("response")
        message = ((response or {}).get("choices") or [{}])[0].get("message") or {}
        raw = raw_calls(message)
        calls = []
        for position, pair in enumerate(source.get("calls") or []):
            name, args = pair
            calls.append(normalize_call(name, args, raw[position]["call_id"] if position < len(raw) else None))
        reviews = []
        for review in source.get("reviews") or []:
            review_response = review.get("response")
            review_message = ((review_response or {}).get("choices") or [{}])[0].get("message") or {}
            reviews.append({"call_index": review.get("call_index"),
                            "decision": review.get("decision"), "reason": review.get("reason"),
                            "content": review_message.get("content"),
                            "reasoning": (review_message.get("reasoning_content") or review_message.get("reasoning")
                                          or review_message.get("thinking")),
                            "raw_response_sha256": digest(review_response) if review_response is not None else None})
        result.append({"index": index, "function": source.get("function"),
                       "phase": source.get("phase", "action"),
                       "segment_turns": source.get("segment_turns"),
                       "context": source.get("messages_before") or [],
                       "tools_offered": source.get("tools_offered"),
                       "assistant": {"content": message.get("content") if response is not None else source.get("text", ""),
                                     "reasoning": (message.get("reasoning_content") or message.get("reasoning")
                                                   or message.get("thinking")),
                                     "calls": calls},
                       "reviews": reviews,
                       "executions": source.get("executions") or [],
                       "raw_response_sha256": digest(response) if response is not None else None})
    return result


def legacy_turns(audit):
    turns, by_call_id = [], {}
    transcript = audit.get("transcript") or []
    opening = []
    first_assistant = next((i for i, message in enumerate(transcript)
                            if message.get("role") == "assistant"), len(transcript))
    if (len(transcript) > first_assistant + 1 and
            transcript[first_assistant + 1].get("role") == "tool" and
            transcript[first_assistant + 1].get("tool_call_id") == "call_0" and
            any(call.get("call_id") == "call_0" and call["tool"] == "read"
                for call in raw_calls(transcript[first_assistant]))):
        opening, transcript = transcript[:first_assistant + 2], transcript[first_assistant + 2:]
    else:
        opening, transcript = transcript[:first_assistant], transcript[first_assistant:]
    for message in transcript:
        if message.get("role") == "assistant":
            calls = raw_calls(message)
            turn = {"index": len(turns), "function": audit.get("function", "root"),
                    "context": [], "tools_offered": None,
                    "assistant": {"content": message.get("content", ""),
                                                   "reasoning": None, "calls": calls},
                    "reviews": [], "executions": [], "raw_response_sha256": None}
            turns.append(turn)
            for position, call in enumerate(calls):
                if call["call_id"]:
                    by_call_id[call["call_id"]] = (turn, position)
        elif message.get("role") == "tool" and message.get("tool_call_id") in by_call_id:
            turn, position = by_call_id[message["tool_call_id"]]
            call = turn["assistant"]["calls"][position]
            turn["executions"].append({"call_index": position, "name": call["tool"],
                                       "args": call["arguments"], "kind": None,
                                       "text": message.get("content", "")})
    # Earlier audits omitted rejected actions from the transcript. Retain the
    # complete log separately; do not guess which assistant turn produced them.
    return turns, opening


def legacy_action_sequence(audit):
    actions = []
    for index, event in enumerate(audit.get("log") or []):
        name, _, encoded = event.get("action", "").partition(" ")
        try:
            args = json.loads(encoded) if encoded else {}
        except ValueError:
            args = {"__unparsed__": encoded}
        actions.append({"index": index, "call": normalize_call(name, args),
                        "result_kind": event.get("kind"),
                        "attempt": event.get("attempt")})
    return actions


def convert(audit, *, audit_path: Path, line_number: int, links: dict, program_ir_hash: str,
            leaf_program_by_key=None, allow_unlinked=False):
    key = ref_key(audit["function"], audit["args"])
    if audit["key"] != key:
        raise ValueError(f"{audit_path}:{line_number}: reference key mismatch")
    if key not in links and not allow_unlinked:
        raise ValueError(f"{audit_path}:{line_number}: key is absent from program IR: {key}")
    modern = "teacher_turns" in audit
    trajectory, opening = (new_turns(audit), []) if modern else legacy_turns(audit)
    metadata = audit.get("model_metadata") or {}
    models = metadata.get("data") or []
    limits = ([] if modern else ["raw_server_response_unavailable", "reasoning_not_recorded",
                                 "rejected_action_turns_may_be_absent"])
    if not modern and audit.get("status") == "done" and trajectory and trajectory[-1]["assistant"]["calls"]:
        limits.append("terminal_turn_missing")
    if key not in links:
        limits.append("unlinked_program")
    return {"version": VERSION,
            "id": "teacher-leaf:" + digest([key, str(audit_path), line_number])[:20],
            "task": {"kind": "generative_leaf", "reference_key": key,
                     "function": audit["function"], "arguments": audit["args"],
                     "leaf_program": audit.get("leaf_program") or (leaf_program_by_key or {}).get(key),
                     "source_program_ids": links.get(key, [])},
            "provenance": {"audit_file": str(audit_path), "audit_line": line_number,
                           "audit_row_sha256": digest(audit),
                           "program_ir_sha256": program_ir_hash,
                           "model": models[0].get("id") if models else None,
                           "teacher_system_prompt": audit.get("system_prompt"),
                           "temperature": audit.get("temperature"),
                           "thinking": audit.get("thinking"),
                           "reasoning_effort": audit.get("reasoning_effort"),
                           "json_text_values": audit.get("json_text_values")},
            "outcome": {"status": audit.get("status"), "detail": audit.get("detail"),
                        "value": audit.get("value"), "accepted": audit.get("accepted"),
                        "admitted": audit.get("admitted"), "checks": audit.get("checks")},
            "trajectory": trajectory,
            "harness_opening_context": opening,
            "legacy_action_log": audit.get("log") if not modern else None,
            "legacy_action_sequence": legacy_action_sequence(audit) if not modern else None,
            "capture_limits": limits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("program_ir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--allow-unlinked", action="store_true",
                        help="preserve audits whose leaf keys are absent from this frozen program IR")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing overwrite: {args.out}")
    staged = args.out.with_suffix(args.out.suffix + ".building")
    if staged.exists():
        parser.error(f"refusing overwrite: {staged}")
    links = program_links(args.program_ir)
    programs = leaf_programs(args.program_ir)
    program_ir_hash = hashlib.sha256(args.program_ir.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.audit.open() as source, staged.open("x") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = convert(json.loads(line), audit_path=args.audit, line_number=line_number,
                          links=links, program_ir_hash=program_ir_hash,
                          leaf_program_by_key=programs, allow_unlinked=args.allow_unlinked)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    staged.replace(args.out)
    print(f"{count} converted teacher trajectories -> {args.out}")


if __name__ == "__main__":
    main()
