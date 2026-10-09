#!/usr/bin/env python3
"""Report status for explicitly registered, reviewed generation pools.

Read-only: this script does not inspect or mutate generation authority, catalogs,
workers, or provider state. It reports only the explicit cohort named by the
registry file; later pools require a new registry version.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402


REPO = Path(__file__).resolve().parents[1]


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def pin(path: Path, inputs: dict[str, dict[str, Any]]) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    key = str(resolved)
    if key not in inputs:
        inputs[key] = {
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        }
    return inputs[key]["sha256"]


def read_json(path: Path, inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    digest = pin(path, inputs)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if sha256_file(path.resolve()) != digest:
        raise ValueError(f"input changed while being read: {path}")
    return value


def jsonl_rows(path: Path, inputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    digest = pin(path, inputs)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    if sha256_file(path.resolve()) != digest:
        raise ValueError(f"input changed while being read: {path}")
    return rows


def source_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source_ids: set[str] = set()
    source_groups: set[str] = set()
    target_ids: set[str] = set()
    component_ids: set[str] = set()
    row_count = 0
    for row in rows:
        row_count += 1
        target_id = row.get("id")
        if isinstance(target_id, str):
            target_ids.add(target_id)
        program = row.get("program_ir") if isinstance(row.get("program_ir"), dict) else row
        if not isinstance(program, dict):
            continue
        for field, output in (("source_ids", source_ids), ("source_groups", source_groups)):
            values = program.get(field, [])
            if isinstance(values, list):
                output.update(value for value in values if isinstance(value, str))
        generation = program.get("generation")
        if isinstance(generation, dict):
            values = generation.get("composite_component_ids", [])
            if isinstance(values, list):
                component_ids.update(value for value in values if isinstance(value, str))
    return {
        "target_rows": row_count,
        "unique_target_ids": len(target_ids),
        "unique_source_ids": len(source_ids),
        "unique_source_groups": len(source_groups),
        "unique_component_program_ids": len(component_ids),
    }


def terminal_counts(
    queue_path: Path,
    journal_path: Path,
    cases_path: Path,
    inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    queue = jsonl_rows(queue_path, inputs)
    cases = jsonl_rows(cases_path, inputs)
    journal = jsonl_rows(journal_path, inputs)

    queue_keys = [row.get("key") for row in queue]
    queue_key_counts = collections.Counter(key for key in queue_keys if isinstance(key, str))
    duplicate_queue_keys = sorted(key for key, count in queue_key_counts.items() if count > 1)
    queue_indices = [row.get("index") for row in queue]
    duplicate_indices = sorted(
        str(index)
        for index, count in collections.Counter(
            index for index in queue_indices if isinstance(index, int)
        ).items()
        if count > 1
    )
    expected_indices = set(range(len(cases)))
    actual_indices = {index for index in queue_indices if isinstance(index, int)}
    queue_ir_link_errors = []
    for entry in queue:
        index = entry.get("index")
        source = entry.get("source")
        if not isinstance(index, int) or index < 0 or index >= len(cases):
            queue_ir_link_errors.append({"key": entry.get("key"), "problem": "index_out_of_range"})
            continue
        if not isinstance(source, str) or repo_path(source).resolve() != cases_path.resolve():
            queue_ir_link_errors.append({"key": entry.get("key"), "problem": "source_path_mismatch"})
            continue
        if cases[index].get("id") is None:
            queue_ir_link_errors.append({"key": entry.get("key"), "problem": "case_ir_missing_id"})

    starts = [event for event in journal if event.get("event") == "start"]
    finishes = [event for event in journal if event.get("event") == "finish"]
    start_keys = [event.get("key") for event in starts if isinstance(event.get("key"), str)]
    finish_keys = [event.get("key") for event in finishes if isinstance(event.get("key"), str)]
    start_counts = collections.Counter(start_keys)
    finish_counts = collections.Counter(finish_keys)
    orphan_starts = sorted(set(start_keys) - set(queue_key_counts))
    orphan_finishes = sorted(set(finish_keys) - set(queue_key_counts))
    duplicate_start_keys = sorted(key for key, count in start_counts.items() if count > 1)
    duplicate_finish_keys = sorted(key for key, count in finish_counts.items() if count > 1)
    # A key can be retried. Determine current terminality from its latest
    # attempt boundary in append order rather than set-differencing all starts
    # and finishes (which hides a retry started after an earlier finish).
    latest_attempt_event: dict[str, dict[str, Any]] = {}
    for event in journal:
        key = event.get("key")
        if isinstance(key, str) and event.get("event") in {"start", "finish"}:
            latest_attempt_event[key] = event
    started = set(start_keys)
    finished = set(finish_keys)
    terminal_latest = {
        key for key, event in latest_attempt_event.items() if event.get("event") == "finish"
    }
    terminal_state_counts = collections.Counter(
        str(event.get("status", "unknown")) for event in finishes
    )
    journal_latest = max(
        (event for event in journal if isinstance(event.get("time"), (int, float))),
        key=lambda event: float(event["time"]),
        default=None,
    )
    exact_export_finishes = 0
    accounted_result_rows = 0
    verified_journal_ir_links = 0
    journal_ir_link_errors = []
    resource_limited_finishes = 0
    queue_by_key = {row.get("key"): row for row in queue if isinstance(row.get("key"), str)}
    for event in finishes:
        accounting = event.get("output_accounting")
        if isinstance(accounting, dict):
            if accounting.get("complete") is True and accounting.get("disposition") == "all_exact_results_exported":
                exact_export_finishes += 1
                rows = accounting.get("output_rows")
                if isinstance(rows, int):
                    accounted_result_rows += rows
            queue_row = queue_by_key.get(event.get("key"))
            if queue_row is not None and isinstance(accounting.get("job_states"), list):
                expected_index = queue_row.get("index")
                expected_id = cases[expected_index].get("id") if isinstance(expected_index, int) and 0 <= expected_index < len(cases) else None
                job_states = accounting["job_states"]
                if len(job_states) != queue_row.get("count"):
                    journal_ir_link_errors.append({"key": event.get("key"), "problem": "accounted_job_count_mismatch"})
                for job in job_states:
                    if not isinstance(job, dict) or job.get("index") != expected_index or job.get("program_id") != expected_id:
                        journal_ir_link_errors.append({"key": event.get("key"), "problem": "finished_job_ir_identity_mismatch"})
                    else:
                        verified_journal_ir_links += 1
            elif queue_row is not None:
                journal_ir_link_errors.append({"key": event.get("key"), "problem": "missing_finished_job_identity_accounting"})
        elif event.get("key") in queue_by_key:
            journal_ir_link_errors.append({"key": event.get("key"), "problem": "missing_output_accounting"})
        if event.get("resource_limit_reason") or event.get("status") in {"quiesced", "resource_limited"}:
            resource_limited_finishes += 1

    return {
        "assigned_case_count": len(queue),
        "source_case_count": len(cases),
        "queue_ir_case_count_matches": len(queue) == len(cases),
        "queue_indices_cover_ir": actual_indices == expected_indices,
        "queue_index_count_matches_ir": len(queue) == len(cases),
        "queue_unique_keys": len(queue_key_counts),
        "duplicate_queue_keys": duplicate_queue_keys,
        "duplicate_queue_indices": duplicate_indices,
        "queue_ir_link_errors": queue_ir_link_errors,
        "journal_events": len(journal),
        "journal_latest_event_at_utc": (
            datetime.fromtimestamp(float(journal_latest["time"]), tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            if journal_latest else None
        ),
        "journal_latest_event_type": journal_latest.get("event") if journal_latest else None,
        "queue_ir_link_errors": queue_ir_link_errors,
        "journal_start_events": len(starts),
        "journal_finish_events": len(finishes),
        "unique_started_queue_keys": len(started & set(queue_key_counts)),
        "unique_terminal_queue_keys": len(terminal_latest & set(queue_key_counts)),
        "queued_not_started_keys": len(set(queue_key_counts) - started),
        "started_not_terminal_keys": len(started - terminal_latest),
        "latest_attempt_nonterminal_keys": sorted(
            key for key, event in latest_attempt_event.items()
            if key in queue_key_counts and event.get("event") == "start"
        ),
        "finished_job_ir_identity_links_verified": verified_journal_ir_links,
        "finished_job_ir_identity_link_errors": journal_ir_link_errors,
        "terminal_status_counts": dict(sorted(terminal_state_counts.items())),
        "duplicate_start_keys": duplicate_start_keys,
        "duplicate_terminal_keys": duplicate_finish_keys,
        "orphan_start_keys": orphan_starts,
        "orphan_terminal_keys": orphan_finishes,
        "exact_export_finish_count_from_supervisor_accounting": exact_export_finishes,
        "accounted_result_rows_from_supervisor_accounting": accounted_result_rows,
        "resource_limited_or_quiesced_finish_count": resource_limited_finishes,
        "admission_ledger": "absent_for_this_local_pool; do not count outcomes as admitted/rejected",
        "admitted_result_count": None,
        "semantic_rejection_count": None,
    }


def qwen_pool(config: dict[str, Any], inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    root = repo_path(config["root"])
    preparation_path = root / "preparation.json"
    assignment_path = root / "assignment.json"
    ir_path = root / "bundle/cases.ir.jsonl"
    proof_path = root / "bundle/selection-proof.jsonl"
    import_status_path = root / "import-status.json"
    import_ledger_path = root / "import-ledger.jsonl"
    worker_status_path = root / "runtime-import-staging/worker-status.json"

    preparation = read_json(preparation_path, inputs)
    assignment = read_json(assignment_path, inputs)
    import_status = read_json(import_status_path, inputs)
    worker_status = read_json(worker_status_path, inputs)
    ir = jsonl_rows(ir_path, inputs)
    proof = jsonl_rows(proof_path, inputs)
    ledger = jsonl_rows(import_ledger_path, inputs)

    candidate_count = preparation.get("candidate_alternatives")
    rejected = preparation.get("rejections", {})
    excluded_count = sum(value for value in rejected.values() if isinstance(value, int))
    computed_eligible = candidate_count - excluded_count if isinstance(candidate_count, int) else None
    reported_eligible = preparation.get("eligible_remaining_after_all_prior_dedup")
    selected_count = len(ir)
    proof_payloads = [row.get("payload_signature") for row in proof]
    proof_target_ids = [row.get("id") for row in proof]
    ledger_sha = [row.get("sha256") for row in ledger]
    ledger_assignment_hashes = {row.get("assignment_sha256") for row in ledger}
    ledger_dispositions = collections.Counter(row.get("disposition", "unknown") for row in ledger)
    status_total = import_status.get("total_unique_artifacts")
    artifact_count_matches = len(ledger) == len(set(ledger_sha)) == status_total == selected_count
    source_ir_path = repo_path(str(preparation.get("source_ir_path")))
    source_ir_actual_sha = pin(source_ir_path, inputs)
    ir_ids = [row.get("id") for row in ir]
    proof_by_id = {row.get("id"): row for row in proof}
    ir_by_id = {row.get("id"): row for row in ir}
    selected_source_joins_match = (
        len(ir_by_id) == len(ir)
        and set(proof_by_id) == set(ir_by_id)
        and all(row.get("index") == index and row.get("id") == ir[index].get("id")
                for index, row in enumerate(proof))
    )
    if selected_source_joins_match:
        for case_id, case in ir_by_id.items():
            proof_row = proof_by_id[case_id]
            if proof_row.get("source_ids") != (case.get("program_ir", {}).get("source_ids") if isinstance(case.get("program_ir"), dict) else case.get("source_ids")):
                selected_source_joins_match = False
                break

    # The import ledger's digest is not sufficient by itself: reopen each
    # referenced immutable result, verify its bytes, and join its embedded
    # task program to the selected IR row.
    result_artifact_errors: list[dict[str, Any]] = []
    imported_program_ids: list[str] = []
    verified_result_bytes = 0
    verified_result_program_joins = 0
    allowed_artifact_root = (root / "imports/jobs").resolve()
    for ledger_index, entry in enumerate(ledger):
        result_path_value = entry.get("path")
        result_path = repo_path(result_path_value) if isinstance(result_path_value, str) else None
        if result_path is None:
            result_artifact_errors.append({"row": ledger_index, "problem": "missing_result_path"})
            continue
        resolved_result = result_path.resolve()
        try:
            resolved_result.relative_to(allowed_artifact_root)
        except ValueError:
            result_artifact_errors.append({"row": ledger_index, "problem": "result_path_outside_import_jobs"})
            continue
        try:
            actual_result_sha = pin(resolved_result, inputs)
            if actual_result_sha != entry.get("sha256"):
                result_artifact_errors.append({"row": ledger_index, "problem": "result_bytes_sha256_mismatch"})
                continue
            verified_result_bytes += 1
            result = read_json(resolved_result, inputs)
        except (OSError, ValueError, TypeError) as error:
            result_artifact_errors.append({"row": ledger_index, "problem": "result_unreadable", "detail": str(error)})
            continue
        program = result.get("task", {}).get("program_ir") if isinstance(result.get("task"), dict) else None
        program_id = program.get("id") if isinstance(program, dict) else None
        imported_program_ids.append(program_id if isinstance(program_id, str) else "")
        if result.get("id") != entry.get("id"):
            result_artifact_errors.append({"row": ledger_index, "problem": "result_artifact_id_mismatch"})
        if program_id != entry.get("program_id"):
            result_artifact_errors.append({"row": ledger_index, "problem": "ledger_program_id_mismatch"})
        if program_id not in ir_by_id:
            result_artifact_errors.append({"row": ledger_index, "problem": "result_program_absent_from_selected_ir"})
        elif program_id == entry.get("program_id") and result.get("id") == entry.get("id"):
            verified_result_program_joins += 1
        if entry.get("assignment_sha256") != inputs[str(assignment_path.resolve())]["sha256"]:
            result_artifact_errors.append({"row": ledger_index, "problem": "result_assignment_sha256_mismatch"})
    artifact_program_counts = collections.Counter(imported_program_ids)
    result_ir_coverage_matches = (
        not result_artifact_errors
        and len(ledger) == selected_count
        and set(artifact_program_counts) == set(ir_by_id)
        and all(count == 1 for count in artifact_program_counts.values())
    )

    checks = {
        "preparation_eligible_arithmetic_matches": computed_eligible == reported_eligible,
        "source_ir_matches_preparation_pin": source_ir_actual_sha == preparation.get("source_ir_sha256"),
        "selection_proof_rows_match_ir": len(proof) == selected_count,
        "selection_proof_payload_signatures_unique": len(proof_payloads) == len(set(proof_payloads)),
        "selection_proof_target_ids_unique": len(proof_target_ids) == len(set(proof_target_ids)),
        "selection_proof_ids_and_source_ids_join_selected_ir": selected_source_joins_match,
        "assignment_case_count_matches_ir": assignment.get("cases") == selected_count,
        "assignment_selected_ir_hash_matches": assignment.get("selected_ir_sha256") == inputs[str(ir_path.resolve())]["sha256"],
        "assignment_selection_proof_hash_matches": assignment.get("selection_proof_sha256") == inputs[str(proof_path.resolve())]["sha256"],
        "import_ledger_unique_artifact_count_matches_status_and_ir": artifact_count_matches,
        "import_ledger_bound_to_one_assignment": ledger_assignment_hashes == {inputs[str(assignment_path.resolve())]["sha256"]},
        "import_dispositions_sum_to_ledger": sum(ledger_dispositions.values()) == len(ledger),
        "imported_result_bytes_and_program_ids_match_selected_ir": result_ir_coverage_matches,
    }
    if not all(checks.values()):
        issues = [name for name, passed in checks.items() if not passed]
    else:
        issues = []

    metrics = source_metrics(ir)
    remaining = computed_eligible - selected_count if isinstance(computed_eligible, int) else None
    return {
        "pool_id": config["pool_id"],
        "provider": config["provider"],
        "cohort_status": config.get("cohort_status", "registered_snapshot"),
        "purpose": preparation.get("purpose"),
        "coverage_claim": "alternate composite task coverage over known source groups; not fresh-source coverage",
        "source_inventory": {
            "path": str(preparation.get("source_ir_path")),
            "sha256": source_ir_actual_sha,
            "source_rows": preparation.get("source_rows"),
            "source_groups_considered": preparation.get("source_groups_considered"),
            "excluded_nonprimitive_components": preparation.get("excluded_nonprimitive_components"),
        },
        "target_counts": {
            "candidate_alternatives_before_history_and_split_filters": candidate_count,
            "previously_seen_payloads": rejected.get("prior_payload"),
            "nontrain_or_protected_candidates": rejected.get("nontrain"),
            "eligible_alternate_compositions_before_v9v3_selection": computed_eligible,
            "selected_v9v3_target_cases": selected_count,
            "duplicate_selected_target_ids": len(ir_ids) - len(set(ir_ids)),
            "duplicate_selected_payload_signatures": len(proof_payloads) - len(set(proof_payloads)),
            "eligible_alternate_compositions_not_selected_by_this_selection": remaining,
            "residual_definition": "candidate_alternatives minus all preparation rejection counts, then minus this selection's validated IR rows; it is a historical selection remainder and may include targets selected by later cohorts",
        },
        "source_coverage_for_selected_cases": metrics,
        "attempts_and_results": {
            "import_checked_at": import_status.get("checked_at"),
            "worker_state": worker_status.get("state"),
            "worker_progress": worker_status.get("progress"),
            "import_status_total_unique_artifacts": status_total,
            "current_import_new_rows": import_status.get("new_rows"),
            "current_import_new_admitted": import_status.get("new_admitted"),
            "current_import_new_rejected": import_status.get("new_rejected"),
            "current_assignment_held": import_status.get("new_assignment_held"),
            "cumulative_import_ledger_dispositions": dict(sorted(ledger_dispositions.items())),
        "cumulative_import_ledger_rows": len(ledger),
            "import_ledger_unique_result_hashes": len(set(ledger_sha)),
            "verified_imported_result_bytes_count": verified_result_bytes,
            "verified_imported_result_program_joins": verified_result_program_joins,
            "imported_result_artifact_errors": result_artifact_errors,
            "admitted_and_rejected_counts_are_from_import_ledger": True,
        },
        "validation": {"checks": checks, "issues": issues},
    }


def local_pool(
    *,
    config: dict[str, Any],
    inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pool_id = config["pool_id"]
    provider = config["provider"]
    plan_path = repo_path(config["plan_path"])
    worker_entry = config.get("worker")
    plan = read_json(plan_path, inputs)
    if worker_entry is None:
        raise ValueError(f"{pool_id}: missing reviewed worker entry")
    queue_path = repo_path(worker_entry["queue"])
    journal_path = repo_path(worker_entry["journal"])
    queue = jsonl_rows(queue_path, inputs)
    if not queue:
        raise ValueError(f"{pool_id}: empty queue")
    case_path_values = {str(row.get("source")) for row in queue}
    if len(case_path_values) != 1:
        raise ValueError(f"{pool_id}: queue contains multiple IR source paths")
    cases_path = repo_path(next(iter(case_path_values)))
    case_rows = jsonl_rows(cases_path, inputs)
    proof_path = cases_path.parent / "selection-proof.jsonl"
    proof_rows = jsonl_rows(proof_path, inputs)

    plan_hash_map = plan.get("artifact_hashes", plan.get("pins", {}))
    plan_hash_map = {str(Path(k).resolve()): v for k, v in plan_hash_map.items()} if isinstance(plan_hash_map, dict) else {}
    actual_plan_sha = inputs[str(plan_path.resolve())]["sha256"]
    expected_plan_sha = config.get("plan_sha256")
    root_approved = plan.get("root_approved") is True or plan.get("approved") is True
    pinned_ir = plan_hash_map.get(str(cases_path.resolve()))
    pinned_queue = plan_hash_map.get(str(queue_path.resolve()), plan.get("queue_sha256"))
    pinned_proof = plan_hash_map.get(str(proof_path.resolve()))
    actual_ir = pin(cases_path, inputs)
    actual_queue = pin(queue_path, inputs)
    actual_proof = pin(proof_path, inputs)
    attempts = terminal_counts(queue_path, journal_path, cases_path, inputs)
    pin_checks = {
        "root_approved_plan": root_approved,
        "plan_matches_registered_review_sha256": isinstance(expected_plan_sha, str) and expected_plan_sha == actual_plan_sha,
        "selected_ir_matches_root_plan_pin": pinned_ir == actual_ir,
        "queue_matches_root_plan_pin": pinned_queue == actual_queue,
        "selection_proof_matches_root_plan_pin": pinned_proof == actual_proof,
        "assignment_case_count_matches_queue": worker_entry.get("cases") == len(queue),
        "queue_case_count_matches_ir": len(queue) == len(case_rows),
        "selection_proof_row_count_matches_ir": len(proof_rows) == len(case_rows),
        "queue_entries_link_exactly_to_ir_indices": not attempts.get("queue_ir_link_errors"),
        "local_finished_job_identities_match_indexed_ir": not attempts.get("finished_job_ir_identity_link_errors"),
    }

    launch_receipt_path_value = config.get("launch_receipt_path")
    launch_receipt_plan_matches = False
    if isinstance(launch_receipt_path_value, str):
        launch_receipt_path = repo_path(launch_receipt_path_value)
        try:
            launch_receipt = read_json(launch_receipt_path, inputs)
            launch_receipt_plan_matches = (
                launch_receipt.get("plan_sha256") == actual_plan_sha
                and repo_path(launch_receipt.get("plan", "")).resolve() == plan_path.resolve()
            )
        except (OSError, ValueError, KeyError, TypeError):
            launch_receipt_plan_matches = False

    proof_ids = [row.get("id") for row in proof_rows]
    case_ids = [row.get("id") for row in case_rows]
    proof_case_join_matches = (
        len(set(case_ids)) == len(case_ids)
        and set(proof_ids) == set(case_ids)
        and len(proof_ids) == len(case_rows)
        and all(row.get("id") == case_rows[index].get("id")
                and row.get("source_ids") == case_rows[index].get("source_ids")
                for index, row in enumerate(proof_rows))
    )
    pin_checks["selection_proof_ids_match_selected_ir"] = proof_case_join_matches
    pin_checks["launch_receipt_present"] = isinstance(launch_receipt_path_value, str) and str(repo_path(launch_receipt_path_value).resolve()) in inputs
    pin_checks["launch_receipt_binds_reviewed_plan"] = launch_receipt_plan_matches
    metrics = source_metrics(case_rows)
    return {
        "pool_id": pool_id,
        "provider": provider,
        "cohort_status": config.get("cohort_status", "registered_snapshot"),
        "root_plan": str(plan_path.resolve()),
        "root_plan_sha256": actual_plan_sha,
        "launch_receipt": str(repo_path(launch_receipt_path_value).resolve()) if isinstance(launch_receipt_path_value, str) else None,
        "root_plan_status": plan.get("status"),
        "coverage_claim": "assigned provider-attempt coverage; source IDs/groups may overlap other registered pools",
        "target_counts": {"assigned_cases": len(queue)},
        "selection_proof_duplicate_target_ids": len(proof_ids) - len(set(proof_ids)),
        "selection_proof_duplicate_payload_signatures": len([row.get("payload_signature") for row in proof_rows]) - len({row.get("payload_signature") for row in proof_rows}),
        "source_coverage_for_assigned_cases": metrics,
        "attempts_and_results": attempts,
        "validation": {"checks": pin_checks, "issues": [name for name, ok in pin_checks.items() if not ok]},
    }


def build_report(registry_path: Path) -> dict[str, Any]:
    inputs: dict[str, dict[str, Any]] = {}
    registry_path = registry_path.resolve()
    registry = read_json(registry_path, inputs)
    if registry.get("schema") != "natlang.generation_backlog_registry/1":
        raise ValueError(f"unsupported registry schema: {registry.get('schema')}")
    if not isinstance(registry.get("pools"), list) or not registry["pools"]:
        raise ValueError("registry must contain a nonempty pools list")
    pools = []
    for config in registry["pools"]:
        kind = config.get("kind")
        if kind == "qwen_alternates":
            pools.append(qwen_pool(config, inputs))
        elif kind == "reviewed_local_queue":
            pools.append(local_pool(config=config, inputs=inputs))
        else:
            raise ValueError(f"unsupported registered pool kind {kind!r}")
    errors = [
        f"{pool['pool_id']}: {issue}"
        for pool in pools
        for issue in pool.get("validation", {}).get("issues", [])
    ]
    qwen_registered = next((pool for pool in pools if "source_inventory" in pool), None)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "schema": "natlang.generation_backlog_status/1",
        "generated_at_utc": now,
        "scope": {
            "completeness": "partial_explicit_registry_cohort",
            "registry_path": str(registry_path),
            "registry_id": registry.get("registry_id"),
            "cohort_label": registry.get("cohort_label"),
            "cohort_status": registry.get("cohort_status"),
            "historical_cohort_warning": registry.get("historical_cohort_warning"),
            "registered_target_pools": [pool["pool_id"] for pool in pools],
            "unregistered_source_and_repair_pools": "unknown; no system-wide total claimed",
            "registered_pools_are_not_implicitly_current": True,
            "source_coverage_is_not_attempt_count": True,
            "local_result_admission_caveat": "No result admission ledger was registered for local Luna/Bunny pools; only queue/journal/output-accounting attempts are counted, and local result acceptance is left null.",
        },
        "pools": pools,
        "cross_pool_source_coverage": {
            "unique_source_ids": None,
            "unique_source_groups": None,
            "reason": "This report does not perform a complete recursive alias join across pools; per-pool source coverage counts are provided without adding them.",
        },
        "holds_and_unregistered_backlog": {
            "qwen_composition_nontrain_or_protected_candidates": (
                qwen_registered["target_counts"]["nontrain_or_protected_candidates"] if qwen_registered else None
            ),
            "qwen_source_inventory_nonprimitive_components_excluded": (
                qwen_registered["source_inventory"]["excluded_nonprimitive_components"] if qwen_registered else None
            ),
            "other_source_review_holds": None,
            "repair_candidates": None,
            "note": "The counts above are from this Qwen composition selection report only; other source-review and repair inventories are not joined in this partial report.",
        },
        "content_pins": inputs,
        "validation": {"passed": not errors, "errors": errors},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("plans/generation-backlog-registry-20261003-v1.json"),
        help="explicit registered pool cohort JSON; default is the preserved v3/v44 cohort",
    )
    parser.add_argument("--output", type=Path, help="new report path (default: timestamped file under runs/generation-backlog-status)")
    args = parser.parse_args()
    try:
        registry_path = args.registry if args.registry.is_absolute() else REPO / args.registry
        report = build_report(registry_path)
        if args.output:
            output_path = args.output if args.output.is_absolute() else REPO / args.output
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = REPO / "runs/generation-backlog-status" / f"report-{stamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        report_hash = sha256_file(output_path)
        digest_path = output_path.with_suffix(output_path.suffix + ".sha256")
        with digest_path.open("x", encoding="ascii") as stream:
            stream.write(f"{report_hash}  {output_path.name}\n")
        print(json.dumps({"report": str(output_path.resolve()), "sha256": report_hash, "sha256_sidecar": str(digest_path.resolve()), "validation": report["validation"], "pools": [{"id": pool["pool_id"], "target_counts": pool["target_counts"], "attempts": pool["attempts_and_results"]} for pool in report["pools"]]}, indent=2))
        return 0 if report["validation"]["passed"] else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"generation backlog report failed closed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
