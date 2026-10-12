"""S7 task-success rollouts: GRPO groups over train-split task rows, run by the collector, scored by classify_case.

The reward is the task-success eval's own outcome class (`eval.task_packet.classify_case`): 1 for `task_success`,
0 for every model-caused failure, None (unscored) for `infrastructure_error` and `unresolved`, which must not teach
the policy. Rollouts run through the same frozen-collector path as the held-task eval (`scripts/run_bonsai_queue.py`
against an OpenAI-compatible `natlang_neuralese.serve`), so RL trains and is judged on the same execution path.

  prepare  derive an authored-root packet from train rows (refuses any non-train row) and a queue of `group`
           sampled single-case entries per row (distinct seeds, temperature > 0).
  score    read the queue journal and results, classify every rollout and report per-group rewards, advantages
           (grpo.group_advantages) and the spread curriculum: groups with mixed outcomes are GRPO-ready; groups
           with zero spread go to rejection sampling (all fail: teacher seeding; all succeed: retire).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ..eval.task_packet import classify_case, read_jsonl
from .grpo import group_advantages

UNSCORED_OUTCOMES = {"infrastructure_error", "unresolved"}


def reward_of(outcome: str) -> float | None:
    """Binary task-success reward; infrastructure outcomes are unscored, never 0."""
    if outcome in UNSCORED_OUTCOMES:
        return None
    return 1.0 if outcome == "task_success" else 0.0


def authored_root_row(row: dict[str, Any]) -> dict[str, Any]:
    """The held-task packet's derivation applied to a train row: the root eval is the row's authored reference root,
    so rollouts exercise the model inside the child calls exactly as the held-task eval does."""
    if row.get("split") != "train":
        raise ValueError(f"refusing non-train row for RL rollouts: {row.get('id')} (split={row.get('split')!r})")
    curriculum = dict(row.get("curriculum") or {})
    root = next((args for kind, args in ((curriculum.get("reference") or {}).get("root") or []) if kind == "eval"), None)
    if not isinstance(root, dict) or not isinstance(root.get("code"), str) or not root["code"].strip():
        raise ValueError(f"row has no authored reference root eval: {row.get('id')}")
    curriculum["trajectory_contract"] = {
        "version": "natlang.trajectory_contract/1", "authored_root_eval": True,
        "authored_root_sha256": hashlib.sha256(root["code"].encode()).hexdigest(),
        "min_child_invocations": 0, "min_state_edges_from_children": 0, "min_iteration_steps": 0,
    }
    return {**row, "curriculum": curriculum}


def rollout_turns(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The policy's own model turns of a collector result as `natlang.rollout/1` turns ({messages, tools, target}),
    replayable as `logLikelihood` terms. Host-authored actions (`action_provenance`, e.g. the authored root eval) are
    not policy samples and are skipped."""
    turns = []
    for step in (result or {}).get("trajectory") or []:
        assistant = step.get("assistant")
        if step.get("phase") != "action" or not isinstance(assistant, dict) or assistant.get("action_provenance") \
                or step.get("action_provenance") or not isinstance(step.get("context"), list):
            continue
        target: dict[str, Any] = {"role": "assistant", "content": assistant.get("content") or ""}
        calls = [{"id": f"call_{index}", "type": "function",
                  "function": {"name": call.get("source_tool") or call.get("tool"),
                               "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False)}}
                 for index, call in enumerate(assistant.get("calls") or [])]
        if calls:
            target["tool_calls"] = calls
        turns.append({"messages": step["context"], "tools": step.get("tools_offered") or None, "target": target})
    return turns


def build_queue(packet: Path, rows: int, group: int, out: Path, *, temperature: float, case_seconds: int,
                max_model_requests: int, max_turns: int, seed_base: int) -> list[dict[str, Any]]:
    if group < 2:
        raise ValueError("a GRPO group needs at least two rollouts")
    if temperature <= 0:
        raise ValueError("group rollouts need temperature > 0; greedy samples are identical")
    entries = []
    for index in range(rows):
        for sample in range(group):
            case = out / f"row-{index:03d}" / f"sample-{sample:02d}"
            entries.append({
                "key": f"s7-task-{index:03d}-{sample:02d}", "index": index, "count": 1, "source": str(packet),
                "seed": seed_base + 1000 * index + sample, "temperature": temperature,
                "case_seconds": case_seconds, "max_model_requests": max_model_requests, "max_turns": max_turns,
                "jobs": str(case / "jobs"), "log": str(case / "collector.log"), "output": str(case / "results.jsonl"),
                "group": index, "sample": sample,
            })
    return entries


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = read_jsonl(args.cases)
    selected = [authored_root_row(row) for row in source[args.offset:args.offset + args.rows]]
    packet = out / "packet.jsonl"
    packet.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    entries = build_queue(packet, len(selected), args.group, out, temperature=args.temperature,
                          case_seconds=args.case_seconds, max_model_requests=args.max_model_requests,
                          max_turns=args.max_turns, seed_base=args.seed_base)
    (out / "queue.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries), encoding="utf-8")
    return {"packet": str(packet), "rows": len(selected), "group": args.group, "queue_entries": len(entries),
            "source_cases": str(args.cases), "ids": [row["id"] for row in selected]}


def score(run_dir: Path, emit: Path | None = None, round_: int = 0) -> dict[str, Any]:
    """Classify every rollout; with `emit`, also write `natlang.rollout/1` records for `grpo.py round`."""
    run_dir = run_dir.resolve()
    emitted: list[dict[str, Any]] = []
    packet = read_jsonl(run_dir / "packet.jsonl")
    entries = read_jsonl(run_dir / "queue.jsonl")
    journal_path = run_dir / "journal.jsonl"
    journal = read_jsonl(journal_path) if journal_path.exists() else []
    finishes = {row["key"]: row for row in journal if row.get("event") == "finish"}
    rollouts = []
    for entry in entries:
        output = Path(entry["output"])
        results = read_jsonl(output) if output.is_file() else []
        result = results[0] if results else None
        outcome = classify_case(result, finishes.get(entry["key"]))
        task = packet[entry["index"]]
        detail = ((result or {}).get("outcome") or {}).get("detail") if result else None
        if emit is not None:
            emitted.append({"schema": "natlang.rollout/1", "round": round_, "task_id": task["id"],
                            "family": task.get("family"), "group": entry["group"], "sample": entry["sample"],
                            "seed": entry["seed"], "policy": {"temperature": entry.get("temperature")},
                            "turns": rollout_turns(result), "outcome": outcome,
                            "reward": None if reward_of(outcome) is None else {"quality": reward_of(outcome)},
                            "unscored": outcome if reward_of(outcome) is None else None})
        rollouts.append({"task_id": task["id"], "family": task.get("family"), "group": entry["group"],
                         "sample": entry["sample"], "seed": entry["seed"], "outcome": outcome,
                         "reward": reward_of(outcome), "detail": detail if isinstance(detail, str) else None,
                         "finish_status": (finishes.get(entry["key"]) or {}).get("status")})
    groups = []
    for index, task in enumerate(packet):
        members = [r for r in rollouts if r["group"] == index]
        advantages, stats = group_advantages([r["reward"] for r in members])
        for member, advantage in zip(members, advantages):
            member["advantage"] = advantage
        scored = [r["reward"] for r in members if r["reward"] is not None]
        if stats["spread"]:
            route = "grpo"
        elif not scored:
            route = "unscored"
        elif scored[0] == 1.0:
            route = "retire"
        else:
            route = "rejection-sampling"
        groups.append({"task_id": task["id"], "route": route, **stats,
                       "outcomes": sorted(r["outcome"] for r in members)})
    counts: dict[str, int] = {}
    for r in rollouts:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    report = {"schema": "natlang.s7-task-rollout-score/1", "rollouts": len(rollouts), "outcome_counts": counts,
              "scored": sum(r["reward"] is not None for r in rollouts),
              "successes": sum(r["reward"] == 1.0 for r in rollouts),
              "groups_with_spread": sum(g["spread"] for g in groups), "groups": groups}
    (run_dir / "rewards.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rollouts), encoding="utf-8")
    if emit is not None:
        emit.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in emitted), encoding="utf-8")
        report["emitted_rollouts"] = str(emit)
    (run_dir / "score.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cases", type=Path, required=True, help="train-split natlang.program/2 task rows")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--case-seconds", type=int, default=600)
    p.add_argument("--max-model-requests", type=int, default=32)
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=7000)
    s = sub.add_parser("score")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--emit-rollouts", type=Path, default=None, help="write natlang.rollout/1 records here")
    s.add_argument("--round", type=int, default=0)
    args = parser.parse_args(argv)
    result = prepare(args) if args.command == "prepare" else score(args.run_dir, args.emit_rollouts, args.round)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
