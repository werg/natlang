#!/usr/bin/env python3
"""Project accepted Studio teacher jobs into template-neutral training turns."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def strip_private(value):
    if isinstance(value, list):
        return [strip_private(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_private(item) for key, item in value.items()
                if not key.startswith("x-")}
    return value


def materialize(row: dict) -> list[dict]:
    if row.get("schema") != "natlang.studio_teacher_trajectory/1":
        raise ValueError("unsupported Studio teacher trajectory")
    if not row.get("outcome", {}).get("accepted"):
        return []
    case, provenance = row["case"], row["provenance"]
    if case["source_revision"] != provenance["source_revision"]:
        raise ValueError("Studio source revision mismatch")
    admission = {"admitted": True, "kind": "exact-studio-state-oracle",
                 "source_revision": case["source_revision"],
                 "expected_sha256": digest(case["expected"]),
                 "reducer_trace_sha256": digest(row["runs"]["reducer"]["trace"]),
                 "view_trace_sha256": digest(row["runs"]["view"]["trace"])}
    samples = []
    for index, exchange in enumerate(row["exchanges"]):
        assistant = exchange.get("assistant")
        if not assistant:
            raise ValueError("Studio exchange lacks normalized assistant choice")
        calls = assistant.get("calls") or []
        target_calls = [{"id": f"studio_{index}_{number}", "type": "function",
                         "function": {"name": call["tool"],
                                      "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}
                        for number, call in enumerate(calls)]
        target = {"role": "assistant", "content": assistant.get("content") or ""}
        if target_calls:
            target["tool_calls"] = target_calls
        samples.append({"id": f"{row['id']}:{index}", "program_id": case["id"],
            "family": "teacher_studio", "ir_version": row["schema"],
            "provisional_gold": False, "trace_admission": admission,
            "teacher_trajectory_id": row["id"], "teacher_trajectory_digest": digest(row),
            "training_admission": {"kind": "exact-studio-state-oracle", "approved": True},
            "source_program_ids": [case["id"]], "teacher_model": provenance["model"],
            "gold_sources": ["checked-teacher-trajectory", "exact-host-oracle"],
            "license": "project-generated", "split": case["split"],
            "source": "teacher-studio", "source_groups": [case["target"]],
            "messages": exchange["request"]["messages"],
            "tools": strip_private(exchange["request"]["tools"]), "target": target,
            "skill": "+".join(call["tool"] for call in calls) if calls else "reply",
            "teacher_reasoning": assistant.get("reasoning")})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cases", type=Path,
                        help="only admit results matching a case id and revision in this current snapshot")
    parser.add_argument("--include-eval", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.replace:
        parser.error("output already exists")
    rows, turns = 0, 0
    current = None
    if args.cases:
        current = {(case["id"], case["source_revision"]) for case in
                   map(json.loads, args.cases.read_text().splitlines())}
    staging = args.out.with_suffix(args.out.suffix + ".building")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        staging.unlink()
    with staging.open("x") as target:
        for path in sorted(args.jobs.glob("*.result.json")):
            row = json.loads(path.read_text())
            if current is not None and (row["case"]["id"], row["case"]["source_revision"]) not in current:
                continue
            if row["case"]["split"] != "train" and not args.include_eval:
                continue
            samples = materialize(row)
            for sample in samples:
                target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            rows += bool(samples)
            turns += len(samples)
    staging.replace(args.out)
    print(f"{rows} accepted Studio trajectories, {turns} turns -> {args.out}")


if __name__ == "__main__":
    main()
