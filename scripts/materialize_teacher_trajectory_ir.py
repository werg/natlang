#!/usr/bin/env python3
"""Replay accepted teacher decisions in the current harness and emit structured training turns."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from natlang.decoder import ChatTurn
from natlang.native import _strip_private
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.values import dump, load_program
from natlang.trace import TraceRecorder, TraceReader
from natlang.scenario import ScenarioContract, admit
from scripts.program_ir import lower
from scripts.collect_scenario_teacher import _root as whole_root
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
                             "call_id": turn.get("call_id"), "function": turn.get("function"),
                             "target": {"role": "assistant", "content": assistant.get("content") or "",
                                        **({"tool_calls": target_calls} if target_calls else {})},
                             "skill": "checkpoint" if turn.get("phase") == "checkpoint" else
                                      "+".join(name for name, _ in calls) if calls else "reply",
                             "teacher_reasoning": assistant.get("reasoning")})
        self.cursor += 1
        return ChatTurn(calls, assistant.get("content") or "", target_calls, completion_tokens=1)


def materialize(row, *, system_prompt: str):
    if row["version"] != VERSION:
        raise ValueError("unsupported teacher trajectory version")
    task_kind = row["task"]["kind"]
    if task_kind not in ("generative_leaf", "whole_program"):
        raise ValueError("unsupported teacher trajectory task")
    if not row["outcome"]["accepted"]:
        raise ValueError("teacher trajectory was not accepted")
    if task_kind == "generative_leaf" and row["outcome"]["status"] != "done":
        raise ValueError("teacher leaf was not completed")
    if not row["task"].get("source_program_ids") or "unlinked_program" in row.get("capture_limits", []):
        raise ValueError("teacher trajectory is not linked to a frozen program")
    program = row["task"].get("leaf_program") if task_kind == "generative_leaf" else row["task"].get("program_ir")
    if program is None:
        raise ValueError("teacher trajectory lacks a frozen program")
    decoder = ReplayDecoder(row["trajectory"])
    checkpoint_limits = {turn["segment_turns"] for turn in row["trajectory"]
                         if turn.get("phase") == "checkpoint"}
    if len(checkpoint_limits) > 1:
        raise ValueError("teacher trajectory has inconsistent checkpoint settings")
    segment_turns = next(iter(checkpoint_limits)) if checkpoint_limits else None
    log = []
    lowered = lower(program) if task_kind == "whole_program" else None
    root = whole_root(lowered) if lowered is not None else load_program(program)
    recorder = TraceRecorder({"run_id": row["id"], "source_sha256": digest(program),
                              "teacher_trajectory_sha256": digest(row),
                              "tool_schema": "tools-v2", "engine_bindings": ["quickjs-isolated"],
                              "capture": "recorded-teacher-replay"})
    outcome, value = Runtime(lambda lam: ToolAgent(decoder, system_prompt=system_prompt,
                                                  validation_feedback="caller", log=log,
                                                  segment_turns=segment_turns),
                             max_episodes=2000 if lowered is not None else 4,
                             capabilities=lowered.capabilities if lowered is not None else None,
                             trace_sink=recorder).run_root(root)
    if outcome.kind != row["outcome"]["status"] or (
            outcome.kind == "done" and dump(value) != row["outcome"]["value"]):
        raise ValueError(f"replay changed outcome: {outcome.kind}: {outcome.detail}")
    if decoder.cursor != len(decoder.turns):
        raise ValueError("teacher trajectory has unconsumed turns")
    if task_kind == "whole_program":
        actual = [{"name": event.get("name"), "arguments": event.get("arguments"),
                   "outcome": event.get("outcome")}
                  for event in TraceReader(recorder.events).of_kind("action")]
        expected = row["outcome"].get("action_ledger")
    else:
        actual = [(event["action"].split(" ", 1)[0], event["kind"]) for event in log]
        expected = [(execution["name"], execution["kind"])
                    for turn in row["trajectory"] for execution in turn["executions"]
                    if execution.get("kind") is not None]
        if not expected and "raw_server_response_unavailable" in row.get("capture_limits", []):
            expected = None  # old audits lack a per-turn execution ledger
    if expected is not None and actual != expected:
        raise ValueError(f"replay changed tool outcomes: {actual!r} != {expected!r}")
    effects = (tuple(("out.emit", [payload]) for payload in lowered.expected_effects)
               if lowered is not None and lowered.expected_effects is not None else None)
    semantic = (program["semantics"].get("contract") if task_kind == "whole_program" and
                program["kind"] == "lambda_scenario" else None)
    required = tuple({"name": item["tool"], "arguments": item["arguments"]}
                     for item in (semantic or {}).get("required_actions", []))
    constraints = tuple((semantic or {}).get("constrained_calls", []))
    admission = admit(TraceReader(recorder.events), ScenarioContract(
        row["outcome"]["status"], row["outcome"]["value"], effects=effects,
        required_actions=required, constrained_calls=constraints))
    provenance = {"teacher_trajectory_id": row["id"],
                  "teacher_trajectory_digest": digest(row),
                  "training_admission": row.get("training_admission"),
                  "source_program_ids": row["task"]["source_program_ids"],
                  "reference_key": row["task"].get("reference_key"),
                  "teacher_model": row["provenance"]["model"],
                  "gold_sources": ["checked-teacher-trajectory"],
                  "license": program.get("license", "project-generated") if task_kind == "whole_program" else "project-generated",
                  "split": program.get("split", "train") if task_kind == "whole_program" else "train",
                  "source": "teacher-program" if task_kind == "whole_program" else "teacher-leaf"}
    if task_kind == "whole_program":
        provenance["source_groups"] = program.get("source_groups") or [program["id"]]
    split_id = (provenance["source_groups"][0] if task_kind == "whole_program" else row["id"])
    return [{"id": f"{row['id']}:{index}", "program_id": split_id,
             "family": "teacher_program" if task_kind == "whole_program" else "teacher_leaf", "ir_version": VERSION,
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
