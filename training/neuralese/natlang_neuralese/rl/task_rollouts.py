"""S7 task-success rollouts: GRPO groups over train-split task rows, run by the collector, scored by classify_case.

The reward is the task-success eval's own outcome class (`eval.task_packet.classify_case`): 1 for `task_success`,
0 for every model-caused failure, None (unscored) for `infrastructure_error` and `unresolved`, which must not teach
the policy. Rollouts run through the same frozen-collector path as the held-task eval (`scripts/run_bonsai_queue.py`
against an OpenAI-compatible `natlang_neuralese.serve`), so RL trains and is judged on the same execution path.

  prepare  derive a train-only packet and a queue of `group` sampled single-case entries per row
           (distinct seeds, temperature > 0). The root mode is declared: authored for
           child-policy rollouts, student for complete student-policy rollouts.
  score    read the queue journal and results, classify every rollout and report per-group rewards, advantages
           (grpo.group_advantages) and the spread curriculum: groups with mixed outcomes are GRPO-ready; groups
           with zero spread go to rejection sampling (all fail: teacher seeding; all succeed: retire).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..eval.task_packet import classify_case, read_jsonl
from .rollout_math import group_advantages
from ..common.task_roots import student_root_row, authored_root_row

UNSCORED_OUTCOMES = {"infrastructure_error", "unresolved"}


def reward_of(outcome: str, finish: dict[str, Any] | None = None) -> float | None:
    """Binary task-success reward; infrastructure outcomes are unscored, never 0. A host wall-clock stop
    (`case_wall_budget`) is a collection resource limit that depends on server load, not on the policy (the journal
    says so: "not a language deadline or model-negative label"), so it is unscored too; a model-request budget stays 0."""
    if outcome == "budget_exhaustion" and (finish or {}).get("resource_limit_reason") == "case_wall_budget":
        return None
    if outcome in UNSCORED_OUTCOMES:
        return None
    return 1.0 if outcome == "task_success" else 0.0


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
        reasoning = assistant.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            target["reasoning_content"] = reasoning
        response = step.get("model_response") or {}
        # Raw calls are the sampled action. Normalized calls can rename tools,
        # reserialize JSON, or even infer a tool call from plain output text.
        # Those are runtime interpretations, not the policy's sampled tokens.
        raw_calls = response.get("raw_calls")
        if isinstance(raw_calls, list) and raw_calls:
            target["tool_calls"] = copy.deepcopy(raw_calls)
        elif "raw_calls" not in response:
            # Legacy trajectories did not retain raw calls. Keep them usable for
            # RFT; the receipt explicitly excludes exact on-policy replay.
            calls = [{"id": call.get("call_id") or f"call_{index}", "type": "function",
                      "function": {"name": call.get("source_tool") or call.get("tool"),
                                   "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False)}}
                     for index, call in enumerate(assistant.get("calls") or [])]
            if calls:
                target["tool_calls"] = calls
        turns.append({"messages": copy.deepcopy(step["context"]),
                      "tools": copy.deepcopy(step.get("tools_offered") or None), "target": target,
                      "invocation_id": step.get("invocation_id"),
                      **({'sampling': copy.deepcopy(response['rl_sampling'])} if response.get('rl_sampling') else {}),
                      **({'native_state_imports': copy.deepcopy(response['native_state_imports'])}
                         if response.get('native_state_imports') else {}),
                      "replay": {"raw_calls_retained": "raw_calls" in response,
                                 "request_sha256": step.get("request_sha256"),
                                 "raw_response_sha256": step.get("raw_response_sha256")}})
    return turns


def _budget_terminated_prefix(entry: dict[str, Any], task: dict[str, Any], finish: dict[str, Any] | None,
                              *, packet_source_path: Path, packet_source_sha256: str | None
                              ) -> dict[str, Any] | None:
    """Recover only a completely captured, exact model-request-budget prefix.

    The collector has no result row for this terminal path. It does, however,
    durably append the interrupted runtime snapshot and action frames to the
    latest partial-evidence sidecar before writing the bound terminal error.
    An incomplete or ambiguous capture is deliberately not a training sample.
    """
    if not isinstance(finish, dict) or finish.get("resource_limit_reason") != "model_request_budget":
        return None
    if task.get("split") != "train":
        return None
    accounting = finish.get("output_accounting")
    if not isinstance(accounting, dict) or packet_source_sha256 is None:
        return None
    job_index = entry.get("index")
    program_id = task.get("id")
    states = accounting.get("job_states")
    matching_states = [row for row in states or [] if isinstance(row, dict) and
                       type(row.get("index")) is int and row.get("index") == job_index]
    state = matching_states[0] if len(matching_states) == 1 else None
    if (not isinstance(state, dict) or state.get("program_id") != program_id or
            state.get("state") != "failed" or state.get("resource_limit_reason") != "model_request_budget"):
        return None
    jobs = Path(entry["jobs"])
    source_path = Path(entry.get("source", ""))
    try:
        if (source_path.is_symlink() or not source_path.is_file() or
                source_path.resolve(strict=True) != packet_source_path.resolve(strict=True)):
            return None
    except OSError:
        return None
    error_path = jobs / f"{int(job_index):06d}.error.json"
    digest = state.get("digest")
    if not isinstance(digest, str) or len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
        return None
    partial_path = jobs / f"{int(job_index):06d}-{digest[:16]}.partial.json"
    try:
        error_bytes = error_path.read_bytes()
        error = json.loads(error_bytes)
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    source_sha = accounting.get("source_sha256")
    terminal_error = state.get("terminal_error") or {}
    if (terminal_error.get("sha256") != hashlib.sha256(error_bytes).hexdigest() or
            Path(str(terminal_error.get("path", ""))).name != error_path.name or
            terminal_error.get("error") != str(error.get("error", "")) or
            error.get("index") != job_index or error.get("program_id") != program_id or
            error.get("code") != "NATLANG_MODEL_REQUEST_BUDGET" or
            source_sha != packet_source_sha256 or error.get("source_sha256") != packet_source_sha256 or
            terminal_error.get("source_sha256") != packet_source_sha256 or
            terminal_error.get("record_binding_verified") is not True or
            error.get("program_ir_sha256") != state.get("digest") or
            (state.get("terminal_error") or {}).get("program_ir_sha256") != state.get("digest") or
            partial.get("version") != "natlang.teacher_partial.native/1" or partial.get("program_id") != program_id or
            (partial.get("provenance") or {}).get("program_ir_sha256") != state.get("digest") or
            not isinstance(partial.get("execution_attempt_id"), str)):
        return None

    snapshots = partial.get("evidence_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        return None
    request_attempt_ids = partial.get("request_attempt_ids")
    if (not isinstance(request_attempt_ids, list) or not request_attempt_ids or
            not all(isinstance(value, str) and value for value in request_attempt_ids)):
        return None
    manifest = snapshots[-1]
    if (not isinstance(manifest, dict) or manifest.get("schema") != "natlang.teacher_partial_evidence/1" or
            manifest.get("status") != "execution_interrupted" or
            manifest.get("attempt_id") != request_attempt_ids[-1] or
            not isinstance(manifest.get("path"), str) or Path(manifest["path"]).name != manifest["path"]):
        return None
    evidence_path = jobs / manifest["path"]
    try:
        if evidence_path.is_symlink() or not evidence_path.is_file():
            return None
        evidence_bytes = evidence_path.read_bytes()
    except OSError:
        return None
    if (len(evidence_bytes) != manifest.get("bytes") or
            hashlib.sha256(evidence_bytes).hexdigest() != manifest.get("sha256")):
        return None
    try:
        records = [json.loads(line) for line in evidence_bytes.splitlines() if line]
    except (ValueError, TypeError):
        return None
    if (len(records) != manifest.get("records") or
            any(row.get("version") != "natlang.teacher_partial_evidence_record/1" or
                row.get("attempt_id") != manifest.get("attempt_id") or row.get("sequence") != index
                for index, row in enumerate(records))):
        return None
    payloads = [row.get("payload") for row in records]
    snapshots = [row for row in payloads if isinstance(row, dict) and row.get("kind") == "execution_snapshot"]
    if len(snapshots) != 1:
        return None
    snapshot = snapshots[0]
    if (snapshot.get("schema") != "natlang.partial_execution_snapshot/1" or snapshot.get("status") != "interrupted" or
            snapshot.get("execution_attempt_id") != partial["execution_attempt_id"] or
            not isinstance(snapshot.get("logical_run_id"), str) or not snapshot["logical_run_id"] or
            snapshot.get("run_id") != f"{snapshot['logical_run_id']}/execution-{partial['execution_attempt_id']}" or
            snapshot.get("pending_children") is not False or
            not isinstance(snapshot.get("failure_reason"), str) or
            not re.fullmatch(r"whole-case model request budget exceeded \([1-9][0-9]*\)",
                             snapshot["failure_reason"])):
        return None
    if sum(row.get("kind") == "execution_snapshot" for row in payloads if isinstance(row, dict)) != 1:
        return None
    all_action_frames = [row for row in payloads if isinstance(row, dict) and
                         row.get("kind") == "teacher_action_frame"]
    policy_sources = {"live_provider", "journal_replay", "handoff_replay"}
    known_sources = policy_sources | {"authored_reference_root"}
    if any(row.get("source") not in known_sources for row in all_action_frames):
        return None
    action_frames = [row for row in all_action_frames if row.get("source") in policy_sources]
    frames = [row.get("response_frame") for row in action_frames]
    verified_action_frames = action_frames + [row for row in all_action_frames
                                              if row.get("source") == "authored_reference_root"]
    verified_frames = [row.get("response_frame") for row in verified_action_frames]
    if (not frames or any(not isinstance(frame, dict) or frame.get("phase") != "action" or
                          not isinstance(frame.get("invocation_id"), str) or
                          not isinstance(frame.get("context"), list) or
                          not isinstance(frame.get("model_response"), dict) or
                          "raw_calls" not in frame["model_response"] for frame in verified_frames)):
        return None
    saved_turns = partial.get("turns")
    if not isinstance(saved_turns, list):
        return None
    saved_indexes = []
    for evidence_row, frame in zip(verified_action_frames, verified_frames):
        index = evidence_row.get("saved_turn_index")
        if (type(index) is not int or index < 0 or index >= len(saved_turns) or
                index in saved_indexes or not isinstance(evidence_row.get("request_sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", evidence_row["request_sha256"])):
            return None
        saved_indexes.append(index)
        saved = saved_turns[index]
        response = saved.get("response") if isinstance(saved, dict) else None
        model_response = frame.get("model_response")
        if (not isinstance(response, dict) or saved.get("request_sha256") != evidence_row.get("request_sha256") or
                saved.get("invocation_id") != evidence_row.get("invocation_id") or
                response.get("calls") != model_response.get("calls") or
                # The native collector's durable turn envelope stores transport metadata under raw_response;
                # the exact raw call payload is retained in the separately hashed action frame above.
                ("raw_calls" in response and response.get("raw_calls") != model_response.get("raw_calls")) or
                not isinstance(model_response.get("raw_calls"), list) or
                response.get("text", "") != model_response.get("text", "")):
            return None
    if any(row.get("kind") == "record_omitted" for row in payloads if isinstance(row, dict)):
        return None

    trace_rows = [row for row in payloads if isinstance(row, dict) and row.get("kind") == "trace_event"]
    invocation_rows = [row for row in payloads if isinstance(row, dict) and row.get("kind") == "invocation_snapshot"]
    if (sum(row.get("trace_role") == "root" for row in trace_rows) != snapshot.get("root_event_count") or
            len(invocation_rows) != snapshot.get("invocation_count")):
        return None
    call_ids = [row.get("call_id") for row in invocation_rows]
    if (any(not isinstance(call_id, str) or not call_id for call_id in call_ids) or len(call_ids) != len(set(call_ids)) or
            any((row.get("trace_role") == "root" and row.get("invocation_id") != snapshot.get("run_id")) or
                (row.get("trace_role") == "child" and row.get("invocation_id") not in call_ids) or
                row.get("trace_role") not in {"root", "child"} for row in trace_rows)):
        return None
    for invocation in invocation_rows:
        events = [row for row in trace_rows if row.get("trace_role") == "child" and
                  row.get("invocation_id") == invocation.get("call_id")]
        if len(events) != invocation.get("event_count"):
            return None
    graph = [row["event"] for row in trace_rows if isinstance(row.get("event"), dict) and
             isinstance(row["event"].get("node"), str) and isinstance(row["event"].get("inputs"), list)]
    reference_roots = [row for row in verified_action_frames if row.get("source") == "authored_reference_root"]
    if reference_roots:
        # An authored root is harness reference material, not a sampled policy turn. Keep its
        # invocation node as the exact caller edge, but never detach its model/state dependencies
        # from the child's trainable graph.
        if len(reference_roots) != 1:
            return None
        reference = reference_roots[0]
        root_invocation = reference.get("invocation_id")
        if (root_invocation != snapshot.get("run_id") or
                reference.get("response_frame", {}).get("invocation_id") != root_invocation):
            return None
        root_events = [row["event"] for row in trace_rows if row.get("trace_role") == "root" and
                       isinstance(row.get("event"), dict) and isinstance(row["event"].get("node"), str) and
                       isinstance(row["event"].get("inputs"), list)]
        root_turns = [event for event in root_events if event.get("kind") == "model_turn"]
        root_call_node = f"call:{root_invocation}"
        root_calls = [event for event in root_events if event.get("kind") == "invocation" and
                      event.get("node") == root_call_node]
        if (len(root_turns) != 1 or root_turns[0].get("node") != f"{root_invocation}#turn1" or
                len(root_calls) != 1 or any(event.get("kind") in {"block_write", "block_read"}
                                           for event in root_events)):
            return None
        child_events = [row["event"] for row in trace_rows if row.get("trace_role") == "child" and
                        isinstance(row.get("event"), dict) and isinstance(row["event"].get("node"), str) and
                        isinstance(row["event"].get("inputs"), list)]
        for event in child_events:
            for edge in event["inputs"]:
                if isinstance(edge, dict) and edge.get("node") == root_turns[0]["node"]:
                    return None
                if isinstance(edge, dict) and edge.get("node") == root_call_node and not (
                        event.get("kind") == "invocation" and edge.get("port") == "caller"):
                    return None
        graph = child_events + root_calls
    trajectory = {"trajectory": frames}
    turns = rollout_turns(trajectory)
    if not turns:
        return None
    try:
        from .episode_graph import validate_rollout_graph
        validate_rollout_graph({"turns": turns, "execution_graph": graph})
    except (TypeError, ValueError):
        return None
    return {"turns": turns, "trajectory": frames, "execution_graph": graph,
            "evidence": {"path": str(evidence_path), "sha256": manifest["sha256"],
                         "attempt_id": manifest["attempt_id"], "failure_reason": snapshot["failure_reason"]}}


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
    root_mode = getattr(args, "root_mode", "authored")
    transform = authored_root_row if root_mode == "authored" else student_root_row
    selected = [transform(row) for row in source[args.offset:args.offset + args.rows]]
    packet = out / "packet.jsonl"
    packet.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    entries = build_queue(packet, len(selected), args.group, out, temperature=args.temperature,
                          case_seconds=args.case_seconds, max_model_requests=args.max_model_requests,
                          max_turns=args.max_turns, seed_base=args.seed_base)
    preparation = {"schema": "natlang.task-rollout-preparation/1", "root_mode": root_mode,
                   "source_cases": str(args.cases.resolve()),
                   "source_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                   "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest()}
    policy_id = getattr(args, 'policy_checkpoint_sha256', None)
    if policy_id:
        if len(policy_id) != 64 or any(c not in '0123456789abcdef' for c in policy_id):
            raise ValueError('policy checkpoint must be a SHA-256 digest')
        preparation['policy_checkpoint_sha256'] = policy_id
    seed_reference = getattr(args, "match_seeds_from", None)
    if seed_reference is not None:
        reference = seed_reference.resolve()
        reference_packet = read_jsonl(reference / "packet.jsonl")
        reference_queue = read_jsonl(reference / "queue.jsonl")
        by_task_sample = {}
        for entry in reference_queue:
            key = (reference_packet[int(entry["index"])]["id"], int(entry["sample"]))
            if key in by_task_sample:
                raise ValueError(f"reference seed is ambiguous for {key}")
            by_task_sample[key] = int(entry["seed"])
        for entry in entries:
            key = (selected[int(entry["index"])]["id"], int(entry["sample"]))
            if key not in by_task_sample:
                raise ValueError(f"reference has no matching task and sample: {key}")
            entry["seed"] = by_task_sample[key]
        preparation["matched_seed_queue"] = str(reference / "queue.jsonl")
        preparation["matched_seed_queue_sha256"] = hashlib.sha256((reference / "queue.jsonl").read_bytes()).hexdigest()
    (out / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
    (out / "queue.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries), encoding="utf-8")
    return {"packet": str(packet), "root_mode": root_mode, "rows": len(selected), "group": args.group, "queue_entries": len(entries),
            "source_cases": str(args.cases), "ids": [row["id"] for row in selected]}


def score(run_dir: Path, emit: Path | None = None, round_: int = 0) -> dict[str, Any]:
    """Classify every rollout; with `emit`, also write `natlang.rollout/1` records for `grpo.py round`."""
    run_dir = run_dir.resolve()
    emitted: list[dict[str, Any]] = []
    packet_path = run_dir / "packet.jsonl"
    packet = read_jsonl(packet_path)
    packet_sha256 = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    preparation_path = run_dir / "preparation.json"
    preparation = json.loads(preparation_path.read_text()) if preparation_path.exists() else {"root_mode": "authored"}
    root_mode = preparation["root_mode"]
    if root_mode == "student":
        for index, task in enumerate(packet):
            if ((task.get("curriculum") or {}).get("trajectory_contract") or {}).get("authored_root_eval"):
                raise ValueError(f"student-root packet row {index} still requests authored-root replay")
    entries = read_jsonl(run_dir / "queue.jsonl")
    journal_path = run_dir / "journal.jsonl"
    journal = read_jsonl(journal_path) if journal_path.exists() else []
    finishes = {row["key"]: row for row in journal if row.get("event") == "finish"}
    rollouts = []
    for entry in entries:
        output = Path(entry["output"])
        results = read_jsonl(output) if output.is_file() else []
        result = results[0] if results else None
        if root_mode == "student" and result:
            for turn in result.get("trajectory") or []:
                if (turn.get("action_provenance") or {}).get("kind") == "authored_reference_root_eval":
                    raise ValueError(f"student-root rollout {entry['key']} replayed an authored root")
        outcome = classify_case(result, finishes.get(entry["key"]))
        reward = reward_of(outcome, finishes.get(entry["key"]))
        expected_packet_sha256 = packet_sha256 if preparation.get("packet_sha256") == packet_sha256 else None
        budget_prefix = _budget_terminated_prefix(
            entry, packet[entry["index"]], finishes.get(entry["key"]),
            packet_source_path=packet_path, packet_source_sha256=expected_packet_sha256) \
            if result is None and outcome == "budget_exhaustion" and \
            (finishes.get(entry["key"]) or {}).get("resource_limit_reason") == "model_request_budget" else None
        task = packet[entry["index"]]
        detail = ((result or {}).get("outcome") or {}).get("detail") if result else None
        if emit is not None:
            training_reward = reward
            if result is None and reward is not None and budget_prefix is None:
                # Preserve the task-level failure score in this report, but never
                # offer an empty or unverified partial as a native policy sample.
                training_reward = None
            emitted.append({"schema": "natlang.rollout/1", "round": round_, "task_id": task["id"],
                            "family": task.get("family"), "group": entry["group"], "sample": entry["sample"],
                            "seed": entry["seed"], "split": task.get("split"),
                            **({"jolt_role": entry["jolt_role"]} if "jolt_role" in entry else {}),
                            "policy": {"temperature": entry.get("temperature"),
                                "checkpoint_sha256": preparation.get("policy_checkpoint_sha256"),
                                "endpoint": entry.get("rl_endpoint")},
                            "turns": budget_prefix["turns"] if budget_prefix else rollout_turns(result),
                            "execution_graph": budget_prefix["execution_graph"] if budget_prefix else
                                copy.deepcopy(((result or {}).get("outcome") or {}).get("execution_graph") or []),
                            "outcome": outcome,
                            "reward": None if training_reward is None else {"quality": training_reward},
                            "unscored": ("budget_prefix_capture_unavailable" if result is None and reward is not None and
                                         budget_prefix is None else outcome if reward is None else None),
                            **({"budget_termination_evidence": budget_prefix["evidence"]} if budget_prefix else {})})
        rollouts.append({"task_id": task["id"], "family": task.get("family"), "group": entry["group"],
                         "sample": entry["sample"], "seed": entry["seed"], "outcome": outcome,
                         "reward": reward, "detail": detail if isinstance(detail, str) else None,
                         "finish_status": (finishes.get(entry["key"]) or {}).get("status"),
                         **({"jolt_role": entry["jolt_role"]} if "jolt_role" in entry else {})})
    groups = []
    for index, task in enumerate(packet):
        # Group statistics and routes describe the student policy; privileged teacher attempts are reported apart.
        members = [r for r in rollouts if r["group"] == index and r.get("jolt_role", "student") == "student"]
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
    report = {"schema": "natlang.s7-task-rollout-score/1", "root_mode": root_mode,
              "rollouts": len(rollouts), "outcome_counts": counts,
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
    p.add_argument("--policy-checkpoint-sha256", help="full checkpoint actually served for this rollout queue")
    p.add_argument("--match-seeds-from", type=Path,
                   help="copy seeds by task ID and sample from an existing prepared run; fail on missing or ambiguous matches")
    p.add_argument("--root-mode", choices=("authored", "student"), default="authored",
                   help="authored replays the reference root and samples children; student samples root and children")
    s = sub.add_parser("score")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--emit-rollouts", type=Path, default=None, help="write natlang.rollout/1 records here")
    s.add_argument("--round", type=int, default=0)
    args = parser.parse_args(argv)
    result = prepare(args) if args.command == "prepare" else score(args.run_dir, args.emit_rollouts, args.round)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
