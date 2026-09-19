#!/usr/bin/env python3
"""Replay accepted teacher decisions in the current harness and emit structured training turns."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.decoder import ChatTurn
from natlang.native import _strip_private
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.values import dump, load_program
from natlang.trace import TraceRecorder, TraceReader
from natlang.scenario import ScenarioContract, admit
from scripts.project_teacher_trajectory_ir import project
from scripts.teacher_trajectory_ir import VERSION, digest


class ReplayDecoder:
    def __init__(self, turns):
        self.turns = turns
        self.cursor = 0
        self.samples = []

    def chat(self, messages, tools, **_kwargs):
        if self.cursor >= len(self.turns):
            raise ValueError("teacher trajectory ended before the harness completed")
        turn = self.turns[self.cursor]
        assistant = turn["assistant"]
        calls = [(call["tool"], call["arguments"]) for call in assistant["calls"]]
        if any(name == "end_turn" for name, _ in calls):
            raise ValueError("migrate legacy end_turn before replay")
        target_calls = [{"id": f"teacher_{self.cursor}_{i}", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
                        for i, (name, args) in enumerate(calls)]
        self.samples.append({"messages": copy.deepcopy(messages), "tools": _strip_private(copy.deepcopy(tools)),
                             "target": {"role": "assistant", "content": assistant.get("content") or "",
                                        **({"tool_calls": target_calls} if target_calls else {})},
                             "skill": "+".join(name for name, _ in calls) if calls else "reply",
                             "teacher_reasoning": assistant.get("reasoning")})
        self.cursor += 1
        return ChatTurn(calls, assistant.get("content") or "", target_calls, completion_tokens=1)


def materialize(row, *, system_prompt: str):
    if row["version"] != VERSION:
        raise ValueError("unsupported teacher trajectory version")
    if row["task"]["kind"] != "generative_leaf":
        raise ValueError("this materializer accepts leaf trajectories only")
    if not row["outcome"]["accepted"] or row["outcome"]["status"] != "done":
        raise ValueError("teacher trajectory was not accepted as a completed leaf")
    if not row["task"].get("source_program_ids") or "unlinked_program" in row.get("capture_limits", []):
        raise ValueError("teacher trajectory is not linked to a frozen program")
    program = row["task"].get("leaf_program")
    if program is None:
        raise ValueError("teacher trajectory lacks a frozen leaf program")
    decoder = ReplayDecoder(row["trajectory"])
    log = []
    root = load_program(program)
    recorder = TraceRecorder({"run_id": row["id"], "source_sha256": digest(program),
                              "teacher_trajectory_sha256": digest(row),
                              "tool_schema": "tools-v2", "engine_bindings": ["quickjs-isolated"],
                              "capture": "recorded-teacher-replay"})
    outcome, value = Runtime(lambda lam: ToolAgent(decoder, system_prompt=system_prompt,
                                                  validation_feedback="caller", log=log),
                             max_episodes=4, trace_sink=recorder).run_root(root)
    if outcome.kind != "done" or dump(value) != row["outcome"]["value"]:
        raise ValueError(f"replay changed outcome: {outcome.kind}: {outcome.detail}")
    if decoder.cursor != len(decoder.turns):
        raise ValueError("teacher trajectory has unconsumed turns")
    actual = [(event["action"].split(" ", 1)[0], event["kind"]) for event in log]
    expected = [(execution["name"], execution["kind"])
                for turn in row["trajectory"] for execution in turn["executions"]
                if execution.get("kind") is not None]
    if expected and actual != expected:
        raise ValueError(f"replay changed tool outcomes: {actual!r} != {expected!r}")
    admission = admit(TraceReader(recorder.events), ScenarioContract("done", row["outcome"]["value"]))
    provenance = {"teacher_trajectory_id": row["id"],
                  "teacher_trajectory_digest": digest(row),
                  "source_program_ids": row["task"]["source_program_ids"],
                  "reference_key": row["task"]["reference_key"],
                  "teacher_model": row["provenance"]["model"],
                  "gold_sources": ["checked-teacher-trajectory"],
                  "license": "project-generated", "split": "train",
                  "source": "teacher-leaf"}
    return [{"id": f"{row['id']}:{index}", "program_id": row["id"],
             "family": "teacher_leaf", "ir_version": VERSION,
             "provisional_gold": False, "trace_admission": admission, **provenance, **sample}
            for index, sample in enumerate(decoder.samples)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--system-file", type=Path,
                        default=Path(__file__).resolve().parent.parent / "natlang/prompts/tools_small.md")
    parser.add_argument("--tool-map", type=Path,
                        help='JSON tool migration; defaults to dropping legacy end_turn')
    parser.add_argument("--drop-reasoning", action="store_true")
    args = parser.parse_args()
    if args.dst.exists():
        parser.error(f"refusing overwrite: {args.dst}")
    staged = args.dst.with_suffix(args.dst.suffix + ".building")
    if staged.exists():
        parser.error(f"refusing overwrite: {staged}")
    tool_map = json.loads(args.tool_map.read_text()) if args.tool_map else {"end_turn": None}
    system_prompt = args.system_file.read_text()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    trajectories = turns = 0
    with args.src.open() as source, staged.open("x") as target:
        for line in source:
            if not line.strip():
                continue
            row = project(json.loads(line), tool_map=tool_map, accepted_only=True,
                          drop_reasoning=args.drop_reasoning, empty_success_reply=True)
            if row is None:
                continue
            samples = materialize(row, system_prompt=system_prompt)
            for sample in samples:
                target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            trajectories += 1
            turns += len(samples)
    staged.replace(args.dst)
    print(f"{trajectories} replayed teacher trajectories, {turns} turns -> {args.dst}")


if __name__ == "__main__":
    main()
