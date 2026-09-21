#!/usr/bin/env python3
"""Collect a whole-program teacher trajectory against frozen program IR."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from natlang.gen.programs import BLOCKED
from natlang.invocation import RunOptions, SeedPolicy
from natlang.runtime import Runtime
from natlang.scenario import ScenarioContract, admit
from natlang.tool_agent import TOOLS_PROMPT, ToolAgent
from natlang.trace import TraceReader, TraceRecorder
from natlang.types import TypeEnv
from natlang.values import coerce, dump, load_program
from scripts.program_ir import digest, lower, validate

VERSION = "natlang.teacher_trajectory/1"


def _root(program):
    if program.loader is not None:
        return program.loader()
    root = load_program(program.root)
    env = root.env(TypeEnv())
    for name, value in program.inputs.items():
        root.in_[name] = coerce(value, root.type.params.get(name)[0], env, yaml=False, path=f"args/{name}")
    return root


def _turn(source):
    response = source.get("response") or {}
    message = ((response.get("choices") or [{}])[0].get("message") or {})
    reasoning = message.get("reasoning_content") or message.get("reasoning") or message.get("thinking")
    return {"function": source.get("function"), "call_id": source.get("call_id"),
            "phase": source.get("phase", "action"), "segment_turns": source.get("segment_turns"),
            "segment_messages": source.get("segment_messages"),
            "note": ({"text": source.get("text", ""), "author": "teacher"}
                     if source.get("phase") == "checkpoint" else None),
            "context": source.get("messages_before") or [],
            "tools_offered": source.get("tools_offered"),
            "assistant": {"content": source.get("text") or "", "reasoning": reasoning,
                          "calls": [{"tool": name, "source_tool": name,
                                     "arguments": args, "call_id": None}
                                    for name, args in source.get("calls") or []]},
            "executions": source.get("executions") or [], "reviews": source.get("reviews") or [],
            "raw_response_sha256": digest(response) if response else None}


def collect(record: dict, decoder, *, model_id: str, options: RunOptions | None = None,
            system_prompt: str = TOOLS_PROMPT, trace_path: Path | None = None,
            segment_turns: int | None = 6,
            segment_messages: int | None = 12) -> tuple[dict, TraceRecorder]:
    validate(record)
    program = lower(record)
    options = options or RunOptions(seed=SeedPolicy("derived", 0))
    recorder = TraceRecorder({"run_id": options.run_id, "source_sha256": digest(record),
                              "semantic_version": record["version"], "tool_schema": "tools-v2",
                              "model": model_id, "seed_policy": vars(options.seed),
                              "capture": "teacher-whole-program"}, trace_path)
    captured = []
    rt = Runtime(lambda lam: ToolAgent(decoder, system_prompt=system_prompt,
                                      temperature=0, validation_feedback="caller",
                                      teacher_turns=captured, segment_turns=segment_turns,
                                      segment_messages=segment_messages),
                 options=options, capabilities=program.capabilities, trace_sink=recorder)
    try:
        outcome, value = rt.run_root(_root(program))
    finally:
        recorder.close()
    actual = dump(value) if outcome.kind == "done" else None
    expected_kind = "quiesced" if program.outcome in ("blocked", "error") or program.expected == BLOCKED else "done"
    value_ok = (actual == program.expected if not callable(program.expected) else
                bool(program.expected(actual))) if expected_kind == "done" else True
    accepted = outcome.kind == expected_kind and value_ok and (
        program.expected_effects is None or rt.emitted == program.expected_effects)
    limits = []
    if any(turn.get("reviews") for turn in captured):
        limits.append("review_path_requires_separate_replay")
        accepted = False
    trajectory = [_turn(turn) for turn in captured]
    action_ledger = [{"name": event.get("name"), "arguments": event.get("arguments"),
                      "outcome": event.get("outcome")}
                     for event in TraceReader(recorder.events).of_kind("action")]
    admission = None
    if accepted:
        semantic = record["semantics"].get("contract") if record["kind"] == "lambda_scenario" else None
        required = tuple({"name": item["tool"], "arguments": item["arguments"]}
                         for item in (semantic or {}).get("required_actions", []))
        constraints = tuple((semantic or {}).get("constrained_calls", []))
        effects = (tuple(("out.emit", [payload]) for payload in program.expected_effects)
                   if program.expected_effects is not None else None)
        try:
            admission = admit(TraceReader(recorder.events), ScenarioContract(
                expected_kind, actual if expected_kind == "done" else None,
                effects=effects, required_actions=required, constrained_calls=constraints))
        except ValueError as exc:
            limits.append("semantic_admission_failed:" + str(exc))
            accepted = False
    row = {"version": VERSION, "id": "teacher-program:" + digest([record["id"], model_id, options.run_id])[:20],
           "task": {"kind": "whole_program", "program_ir": record,
                    "source_program_ids": [record["id"]]},
           "provenance": {"model": model_id, "program_ir_sha256": digest(record),
                          "tool_schema": "tools-v2", "seed_policy": vars(options.seed),
                          "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
                          "trace_sha256": digest(recorder.events)},
           "outcome": {"status": outcome.kind, "detail": outcome.detail, "value": actual,
                       "effects": rt.emitted, "accepted": accepted, "admission": admission,
                       "action_ledger": action_ledger},
           "trajectory": trajectory, "capture_limits": limits}
    return row, recorder


def _resume_count(output: Path, records: list[dict],
                  model_id: str, root_seed: int) -> int:
    if not output.exists():
        return 0
    count = 0
    with output.open() as stream:
        for line in stream:
            row = json.loads(line)
            if count >= len(records):
                raise ValueError("existing teacher output exceeds the requested source range")
            record = records[count]
            if (row["task"]["program_ir"]["id"] != record["id"] or
                    row["provenance"]["program_ir_sha256"] != digest(record) or
                    row["provenance"]["model"] != model_id or
                    row["provenance"]["seed_policy"] != vars(SeedPolicy("derived", root_seed))):
                raise ValueError(f"existing teacher row {count} does not match this run")
            count += 1
    return count


def _trace_path(output: Path, index: int) -> Path:
    base = output.parent / f"{output.stem}-{index}.trace.jsonl"
    base.unlink(missing_ok=True)
    for retry in output.parent.glob(f"{output.stem}-{index}.retry*.trace.jsonl"):
        retry.unlink()
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--start", type=int, default=0, help="first zero-based program row in a frozen batch")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="append after verifying completed rows; discard non-resumable interrupted traces")
    parser.add_argument("--segment-turns", type=int, default=6,
                        help="conversation work turns before a continuation checkpoint")
    parser.add_argument("--segment-messages", type=int, default=12,
                        help="conversation items before a safe continuation checkpoint")
    parser.add_argument("--system-file", type=Path,
                        default=Path(__file__).resolve().parent.parent / "natlang/prompts/tools_teacher_compact.md")
    args = parser.parse_args()
    if args.start < 0 or args.limit < 1 or args.segment_turns < 1 or args.segment_messages < 5:
        parser.error("start must be nonnegative; limit and segment turns positive, segment messages at least 5")
    from natlang.decoder import LlamaServerDecoder
    decoder = LlamaServerDecoder(
        args.server,
        chat_extra={"thinking_budget_tokens": 256, "top_p": 0.95, "top_k": 20,
                    "chat_template_kwargs": {"reasoning_effort": "low"}},
        tool_aliases={"call": "call_function"}, json_text_values=True)
    system_prompt = args.system_file.read_text()
    records = []
    with args.ir.open() as source:
        for index, line in enumerate(source):
            if index >= args.start + args.limit:
                break
            if index >= args.start:
                records.append(json.loads(line))
    if len(records) != args.limit:
        parser.error("requested source range exceeds the frozen batch")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = (_resume_count(args.out, records,
                               args.model_id, args.root_seed) if args.resume else 0)
    with args.out.open("a" if args.resume and args.out.exists() else "x") as target:
        for offset in range(completed, args.limit):
            index = args.start + offset
            record = records[offset]
            options = RunOptions(seed=SeedPolicy("derived", args.root_seed))
            row, _ = collect(record, decoder, model_id=args.model_id, options=options,
                             system_prompt=system_prompt,
                             segment_turns=args.segment_turns, segment_messages=args.segment_messages,
                             trace_path=_trace_path(args.out, index))
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            target.flush()
            print(f"{record['id']}: {row['outcome']['status']} accepted={row['outcome']['accepted']}", flush=True)


if __name__ == "__main__":
    main()
