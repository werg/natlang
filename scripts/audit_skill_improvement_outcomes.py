#!/usr/bin/env python3
"""Summarize host-recorded skill-improvement results without treating model history as an oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def records(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def diagnostic_codes(artifact: dict[str, Any]) -> list[dict[str, str]]:
    result = []
    for item in records(artifact.get("skillDiagnostics")):
        code, severity, path = item.get("code"), item.get("severity", "error"), item.get("path", "")
        if isinstance(code, str):
            result.append({"code": code, "severity": str(severity), "path": str(path)})
    return result


def trace_kind(trace: dict[str, Any]) -> str:
    outcome = str(trace.get("outcome", "unknown")).lower()
    detail = str(trace.get("detail", "")).lower()
    if outcome in {"done", "returned"}:
        return "completed_tool_exchange"
    if outcome == "quiesced":
        if any(word in detail for word in ("not enough turns", "remaining turns", "need to create", "more turns", "remaining file")):
            return "turn_or_output_budget_exhaustion"
        return "quiesced_without_completed_work"
    if outcome == "error":
        if any(word in detail for word in ("malformed", "invalid json", "schema", "argument")):
            return "tool_protocol_or_schema_error"
        return "tool_execution_error"
    if outcome in {"cancelled", "canceled", "timeout"}:
        return "cancelled_or_timed_out"
    if outcome == "failed":
        return "tool_execution_error"
    return "other_nonterminal_exchange"


def host_failure_categories(artifact: dict[str, Any]) -> list[str]:
    search = mapping(artifact.get("search"))
    categories: set[str] = set()
    if artifact.get("disposition") in {"failed", "incomplete", "interrupted"}:
        categories.add("collection_incomplete_or_failed")
    error = str(search.get("error", ""))
    if "reported quality does not match independently executed selected source" in error.lower():
        categories.add("model_claimed_score_disagrees_with_host_measurement")
    elif error:
        categories.add("host_search_error")
    diagnostics = diagnostic_codes(artifact)
    if any(row["severity"] == "error" and row["code"].startswith("skill-") for row in diagnostics):
        categories.add("invalid_skill_metadata_or_contract")
    kinds = [trace_kind(trace) for trace in records(artifact.get("traces"))]
    categories.update(kind for kind in kinds if kind != "completed_tool_exchange")
    if artifact.get("disposition") == "not-promoted":
        baseline = mapping(search.get("baseline"))
        selected = mapping(search.get("validation"))
        if ((baseline.get("source") is not None and baseline.get("source") != artifact.get("baseline")) or
                (selected.get("source") is not None and selected.get("source") != artifact.get("selected"))):
            categories.add("host_source_digest_mismatch")
        elif artifact.get("baseline") == artifact.get("selected") and baseline.get("gatesPassed") is True and selected.get("gatesPassed") is True:
            if baseline.get("quality") == 1 and selected.get("quality") == 1:
                categories.add("no_observed_headroom_at_support_ceiling")
            elif baseline.get("quality") == selected.get("quality"):
                categories.add("no_measured_support_quality_gain")
            else:
                categories.add("baseline_retained_after_search")
        else:
            categories.add("not_promoted_without_verified_gain")
    if artifact.get("baseline") != artifact.get("selected") and artifact.get("selected") is not None:
        if any(row["severity"] == "error" and row["code"].startswith("skill-") for row in diagnostics):
            categories.add("edited_candidate_invalid_skill_metadata")
        elif not mapping(artifact.get("query")):
            categories.add("edited_candidate_without_final_query")
        elif mapping(search.get("validation")).get("gatesPassed") is not True:
            categories.add("edited_candidate_failed_support_validation")
    return sorted(categories)


def validation_summary(value: Any) -> dict[str, Any] | None:
    row = mapping(value)
    if not row:
        return None
    scores = records(row.get("scores"))
    return {"split": row.get("split"), "source": row.get("source"), "quality": row.get("quality"),
            "gatesPassed": row.get("gatesPassed"), "passed": row.get("passed"), "total": row.get("total"),
            "modelCalls": row.get("modelCalls"), "evidence_sha256": row.get("evidence"),
            "sourceBytes": row.get("sourceBytes"),
            "scores": [{"caseId": score.get("caseId"), "quality": score.get("quality")} for score in scores],
            "outcomes": [{"caseId": outcome.get("caseId"), "passed": outcome.get("passed"),
                          "quality": outcome.get("quality"), "failure_kind":
                            "host_failure" if outcome.get("error") else None}
                         for outcome in records(row.get("outcomes"))]}


def query_summary(value: Any) -> dict[str, Any] | None:
    row = mapping(value)
    if not row:
        return None
    output: dict[str, Any] = {"effect": row.get("effect"), "baseline": validation_summary(row.get("baseline")),
                              "selected": validation_summary(row.get("selected"))}
    return output


def support_references(definition: Any) -> list[dict[str, Any]]:
    result = []
    for case in records(mapping(definition).get("cases")):
        result.append({"case_id": case.get("id"), "group": case.get("group"), "split": case.get("split"),
                       "input_sha256": canonical_sha({"args": case.get("args"), "folder": case.get("folder")}),
                       "expected_sha256": canonical_sha(case.get("expected")),
                       "service_ids": sorted(mapping(case.get("services")))})
    return result


def authored_requests(exchanges: Any) -> list[dict[str, Any]]:
    result = []
    for index, exchange in enumerate(records(exchanges)):
        request = mapping(exchange.get("request"))
        turn = mapping(exchange.get("turn"))
        if not request:
            result.append({"ordinal": index, "request_fingerprint": None, "fingerprint_status": "missing_serialized_request"})
            continue
        result.append({"ordinal": index, "request_fingerprint": canonical_sha(request),
                       "recording_version": exchange.get("recording_version"),
                       "call_count": len(records(turn.get("calls"))),
                       "wire_attempts": len(records(exchange.get("wireExchanges"))),
                       "fingerprint_status": "full_serialized_request_including_messages_tools_and_limits"})
    return result


def artifact_summary(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    artifact = json.loads(raw)
    search = mapping(artifact.get("search"))
    state = mapping(search.get("state"))
    baseline = mapping(search.get("baseline"))
    selected_validation = mapping(search.get("validation"))
    selected_files = artifact.get("selectedFiles")
    file_hashes = []
    if isinstance(selected_files, dict):
        for path_name, content in sorted(selected_files.items()):
            if isinstance(path_name, str) and isinstance(content, str):
                file_hashes.append({"path": path_name, "sha256": sha(content.encode())})
    traces = records(artifact.get("traces"))
    outcomes = Counter(str(trace.get("outcome", "unknown")) for trace in traces)
    validation = validation_summary(selected_validation)
    attempt_parts = path.relative_to(root).parts
    task_attempt = "/".join(attempt_parts[:-1])
    history = records(state.get("history"))
    diagnostics = diagnostic_codes(artifact)
    changed = artifact.get("baseline") != artifact.get("selected") and artifact.get("selected") is not None
    baseline_validation = validation_summary(baseline)
    host_error = str(search.get("error", ""))
    error_type = re.match(r"([A-Za-z][A-Za-z0-9_.]*)", host_error)
    return {
        "artifact": path.relative_to(root).as_posix(), "attempt_key": task_attempt,
        "result_sha256": sha(raw), "collection_sha256": artifact.get("collection_sha256"),
        "trajectory_identity": artifact.get("identity"), "episode": artifact.get("episode"),
        "family": artifact.get("family"), "disposition": artifact.get("disposition"), "positive_claim": artifact.get("positive"),
        "source_groups": artifact.get("source_groups", []),
        "baseline_source_digest": artifact.get("baseline"), "selected_source_digest": artifact.get("selected"),
        "selected_files": file_hashes,
        "search": {"disposition": search.get("disposition"), "iteration_counter": state.get("iteration"),
                   "history_record_count_unverified": len(history), "stop_reason_category": state.get("stopReason") is not None,
                   "baseline_support_validation": baseline_validation,
                   "selected_support_validation": validation,
                   "validation_case_scores": validation.get("scores", []) if validation else [],
                   "skill_diagnostics": diagnostic_codes(artifact),
                   "host_error_type": error_type.group(1) if error_type else None,
                   "host_error_sha256": sha(host_error.encode()) if host_error else None,
                   "host_error_category": [category for category in host_failure_categories(artifact)
                                           if category in {"model_claimed_score_disagrees_with_host_measurement", "host_search_error"}],
                   "trace_outcomes": dict(sorted(outcomes.items())),
                   "trace_failure_categories": dict(sorted(Counter(trace_kind(t) for t in traces if trace_kind(t) != "completed_tool_exchange").items()))},
        "support_evidence_refs": support_references(artifact.get("searchDefinition")),
        "final_query": query_summary(artifact.get("query")), "final_transfer": query_summary(artifact.get("transfer")),
        "revision_review": ({"baseline_source_digest": artifact.get("baseline"), "candidate_source_digest": artifact.get("selected"),
                             "candidate_files": file_hashes, "candidate_skill_diagnostics": diagnostics,
                             "host_support_validation": validation, "final_query_present": bool(mapping(artifact.get("query"))),
                             "support_quality_delta": (validation.get("quality") - baseline_validation.get("quality")
                               if validation and baseline_validation and isinstance(validation.get("quality"), (int, float))
                               and isinstance(baseline_validation.get("quality"), (int, float)) else None),
                             "status": "invalid_skill_metadata" if any(row["severity"] == "error" and row["code"].startswith("skill-") for row in diagnostics)
                               else "support_only_no_final_query" if not mapping(artifact.get("query"))
                               else "requires_independent_query_and_replay_review"} if changed else None),
        "author_request_fingerprints": authored_requests(artifact.get("authorExchanges")),
        "executor_exchange_count": len(records(artifact.get("executorExchanges"))),
        "classification": host_failure_categories(artifact),
    }


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    mirror_manifest = root.parent / "source-manifest.json"
    mirror = json.loads(mirror_manifest.read_text(encoding="utf-8")) if mirror_manifest.is_file() else None
    paths = sorted(root.rglob("result.json"))
    artifacts = [artifact_summary(path, root) for path in paths]
    dispositions = Counter(str(row.get("disposition", "unknown")) for row in artifacts)
    categories = Counter(category for row in artifacts for category in row["classification"])
    fingerprint_occurrences: dict[str, list[dict[str, Any]]] = {}
    for row in artifacts:
        for request in row["author_request_fingerprints"]:
            fingerprint = request.get("request_fingerprint")
            if fingerprint:
                fingerprint_occurrences.setdefault(fingerprint, []).append({"attempt_key": row["attempt_key"], "ordinal": request["ordinal"]})
    repeated = {fingerprint: rows for fingerprint, rows in fingerprint_occurrences.items() if len(rows) > 1}
    return {"schema": "natlang.skill-improvement-outcomes-audit/1", "input_root": str(root),
            "source_mirror": ({"manifest": mirror_manifest.name, "manifest_sha256": sha(mirror_manifest.read_bytes()),
                               "remote_host": mirror.get("remote_host"), "remote_root": mirror.get("remote_root"),
                               "captured_utc": mirror.get("captured_utc"), "remote_inventory_sha256_verified": mirror.get("remote_inventory_sha256_verified")}
                              if mirror else None),
            "artifact_count": len(artifacts), "dispositions": dict(sorted(dispositions.items())),
            "host_failure_categories": dict(sorted(categories.items())), "artifacts": artifacts,
            "request_fingerprint_index": {"unique_full_request_hashes": len(fingerprint_occurrences),
                                           "exact_repeated_request_hashes": repeated,
                                           "dpo_pair_readiness": "none_without_same-context paired outcome and causal lineage",
                                           "dpo_pairs_admitted": 0},
            "interpretation_policy": {
                "history_is_oracle": False,
                "model_claimed_history_or_freeform_trace_detail_used_as_quality": False,
                "query_transfer_payloads_copied": False,
                "incomplete_or_invalid_contract_are_quality_negatives": False,
                "positive_training_admission": False,
                "note": "Saved validation summaries and source digests are host-recorded evidence; history reasons are retained only as an unverified count."}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing mirrored result.json artifacts")
    parser.add_argument("--out", type=Path, help="exclusive-create audit JSON path; stdout when omitted")
    args = parser.parse_args()
    value = audit(args.root)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
