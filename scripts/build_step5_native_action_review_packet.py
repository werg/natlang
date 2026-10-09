#!/usr/bin/env python3
"""Build a trace-bound per-action review packet for materialized Step 5 rows.

This derives byte pins, canonical JSON target/context digests, and exact action
event joins. Human semantic dispositions are read from a separate annotation
file and are never inferred from collector success or materializer status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from assemble_admitted_neuralese_cohort import target_digest


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha(data)


def file_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve().relative_to(ROOT)), "bytes": len(raw), "sha256": sha(raw)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        rows.append(value)
    return rows


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("result", "trace", "native-rows", "materialization-manifest", "capture-snapshot",
                 "annotations", "source-cases", "launch-plan", "runtime-manifest", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--source-row-index", required=True, type=int)
    parser.add_argument("--controller-provenance", required=True,
                         help="Explicitly identify whether the controller/root was authored or sampled")
    args = parser.parse_args()
    result = load_json(args.result)
    native_rows = read_jsonl(args.native_rows)
    trace_rows = read_jsonl(args.trace)
    annotations = load_json(args.annotations)
    launch_plan = load_json(args.launch_plan)

    ledger = result.get("outcome", {}).get("action_ledger", [])
    trace_actions = [event for event in trace_rows if event.get("kind") == "action"]
    invocations = {entry.get("invocation_id"): entry for entry in result.get("outcome", {}).get("invocation_ledger", [])}
    annotation_rows = annotations.get("rows")
    if not isinstance(annotation_rows, dict):
        raise ValueError("annotations.rows must map native row IDs to explicit review assessments")
    source_lines = args.source_cases.read_bytes().splitlines(keepends=True)
    if args.source_row_index < 0 or args.source_row_index >= len(source_lines):
        raise ValueError("source row index is outside source-cases JSONL")
    selected_source_line = source_lines[args.source_row_index]
    source_record = json.loads(selected_source_line)
    source_binding = launch_plan.get("source", {})
    expected_source_path = (ROOT / source_binding.get("path", "")).resolve()
    if (not expected_source_path.is_relative_to(ROOT) or expected_source_path != args.source_cases.resolve()
            or not expected_source_path.is_file() or sha(expected_source_path.read_bytes()) != source_binding.get("sha256")):
        raise ValueError("source-cases file does not match the exact launch-plan source path/hash")
    if source_binding.get("index") != args.source_row_index:
        raise ValueError("source row index does not match the launch-plan source binding")
    if sha(selected_source_line.rstrip(b"\r\n")) != source_binding.get("row_sha256"):
        raise ValueError("selected source row does not match the launch-plan row hash")

    packet_rows = []
    seen = set()
    for row in native_rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in seen:
            raise ValueError(f"missing or duplicate native row id: {row_id}")
        seen.add(row_id)
        annotation = annotation_rows.get(row_id)
        if not isinstance(annotation, dict) or annotation.get("disposition") not in {"candidate", "hold"}:
            raise ValueError(f"missing explicit candidate/hold assessment for {row_id}")
        source_ref = row.get("source_ref", {})
        invocation_id = source_ref.get("invocation_id")
        decision = row.get("decision", {})
        calls = decision.get("assistant", {}).get("calls", [])
        exact_calls = []
        for call in calls:
            outcome = call.get("outcome", {})
            seq = outcome.get("trace_seq")
            name = outcome.get("name") or call.get("tool")
            matches = [event for event in ledger if event.get("call_id") == invocation_id and
                       event.get("seq") == seq and event.get("name") == name and
                       canonical_digest(event.get("arguments")) == canonical_digest(outcome.get("arguments"))]
            trace_matches = [event for event in trace_actions if event.get("call_id") == invocation_id and
                             event.get("seq") == seq and event.get("name") == name and
                             canonical_digest(event.get("arguments")) == canonical_digest(outcome.get("arguments"))]
            exact_calls.append({
                "name": name,
                "trace_seq": seq,
                "tool_call_id": outcome.get("tool_call_id"),
                "recorded_status": outcome.get("status"),
                "action_ledger_matches": len(matches),
                "trace_action_matches": len(trace_matches),
                "target_arguments_sha256_canonical_sorted_json": canonical_digest(outcome.get("arguments")),
                "ledger_outcome": matches[0].get("outcome") if len(matches) == 1 else None,
                "ledger_diagnostics": matches[0].get("diagnostics") if len(matches) == 1 else None,
            })
        invocation = invocations.get(invocation_id, {})
        target = row.get("target")
        packet_rows.append({
            "native_id": row_id,
            "decision_index": decision.get("index"),
            "split": row.get("split"),
            "source_groups": row.get("source_groups"),
            "invocation_id": invocation_id,
            "parent_invocation_id": source_ref.get("parent_invocation_id"),
            "materializer_source_row_sha256": source_ref.get("source_row_sha256"),
            "native_row_training_admission": row.get("training_admission"),
            "target_sha256_canonical_sorted_json": target_digest(row),
            "messages_sha256_canonical_sorted_json": canonical_digest(row.get("messages")),
            "tools_sha256_canonical_sorted_json": canonical_digest(row.get("tools")),
            "target_call_names": [call.get("function", {}).get("name") for call in target.get("tool_calls", [])],
            "exact_call_events": exact_calls,
            "typed_result_write_receipts": [receipt for call in calls for receipt in
                call.get("outcome", {}).get("typed_result_writes", [])],
            "invocation_host_result": invocation.get("host_result"),
            "invocation_failure": invocation.get("failure"),
            "source_validation": source_ref.get("inline_instruction_site", {}).get("validation"),
            "assessment": annotation,
        })

    missing_annotations = set(annotation_rows) - seen
    if missing_annotations:
        raise ValueError(f"annotations refer to absent native rows: {sorted(missing_annotations)}")
    root_completed = result.get("outcome", {}).get("status") == "done" and result.get("outcome", {}).get("accepted") is True
    case_provenance = (
        "Per-action sampled outputs from a successfully completed Step 5 root call."
        if root_completed else
        "Per-action sampled outputs from a Step 5 root call that did not complete successfully."
    )
    packet = {
        "schema": "natlang.step5-native-action-review-packet/1",
        "status": "review proposal; no training admission",
        "provenance": f"{case_provenance} Candidate labels below are human review recommendations, not admissions.",
        "controller_provenance": args.controller_provenance,
        "pins": {
            "result": file_pin(args.result),
            "trace": file_pin(args.trace),
            "native_rows": file_pin(args.native_rows),
            "materialization_manifest": file_pin(args.materialization_manifest),
            "capture_snapshot_manifest": file_pin(args.capture_snapshot),
            "annotations": file_pin(args.annotations),
            "source_cases": file_pin(args.source_cases),
            "launch_plan": file_pin(args.launch_plan),
            "runtime_manifest": file_pin(args.runtime_manifest),
        },
        "source_binding": {
            "physical_row_index": args.source_row_index,
            "row_bytes_including_line_feed": len(selected_source_line),
            "row_sha256_with_line_feed": sha(selected_source_line),
            "row_sha256_without_line_feed": sha(selected_source_line.rstrip(b"\r\n")),
            "materializer_source_row_sha256_basis": "nativeRowDigest(result JSON object); intentionally distinct from physical source-cases JSONL row hashes",
            "case_id": source_record.get("id") or source_record.get("case_id"),
            "source_groups": source_record.get("source_groups") or source_record.get("groups"),
            "split": source_record.get("split"),
        },
        "route_identity": {
            "model": result.get("provenance", {}).get("model"),
            "transport": result.get("provenance", {}).get("transport"),
            "physical_network_request_count_known": False,
            "request_telemetry_scope": result.get("request_telemetry"),
        },
        "collector_outcome": result.get("outcome", {}).get("status"),
        "collector_accepted": result.get("outcome", {}).get("accepted") is True,
        "trajectory_review": result.get("trajectory_review"),
        "rows": packet_rows,
        "counts": {
            "materialized_rows": len(packet_rows),
            "candidate_recommendations": sum(item["assessment"]["disposition"] == "candidate" for item in packet_rows),
            "held_recommendations": sum(item["assessment"]["disposition"] == "hold" for item in packet_rows),
            "rows_with_unrecorded_action": sum(any(call["recorded_status"] == "not_recorded" for call in item["exact_call_events"]) for item in packet_rows),
        },
        "limits": {
            "parent_root_completed": root_completed,
            "authored_controller_synthesis_claim": False,
            "whole_trajectory_admission": False,
            "physical_network_request_count_known": False,
            "training_admission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output.read_bytes()), "counts": packet["counts"]}))


if __name__ == "__main__":
    main()
