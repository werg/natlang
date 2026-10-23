"""Shared scoring and exact checkpoint retention for protected authored task packets."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from ..common.hashing import sha256_file_hex
from ..common.jsonio import read_jsonl

TASK_OUTCOMES = (
    "task_success",
    "invalid_action",
    "repetition",
    "budget_exhaustion",
    "infrastructure_error",
    "model_incomplete",
    "semantic_mismatch",
    "unresolved",
)


def _text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).casefold()


def _native_terminal_outcome(accounting: dict[str, Any]) -> str | None:
    """Recognize exact engine termination evidence, not generated action text."""
    states = accounting.get("job_states")
    if (accounting.get("manifest_valid") is not True
            or accounting.get("expected_jobs") != 1
            or not isinstance(states, list) or len(states) != 1):
        return None
    state = states[0]
    if not isinstance(state, dict) or state.get("state") != "failed":
        return None
    error = state.get("terminal_error")
    if (not isinstance(error, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(error.get("sha256", "")))):
        return None
    source_digest = accounting.get('source_sha256')
    record_digest = state.get('digest')
    if (error.get('record_binding_verified') is not True
            or not isinstance(source_digest,str) or re.fullmatch(r'[0-9a-f]{64}',source_digest) is None
            or not isinstance(record_digest,str) or re.fullmatch(r'[0-9a-f]{64}',record_digest) is None
            or error.get('program_ir_sha256') != record_digest
            or error.get('source_sha256') != source_digest):
        return None
    message = error.get("error")
    prefix = "Error: model stream error: "
    if not isinstance(message, str) or not message.startswith(prefix):
        return None
    try:
        payload = json.loads(message[len(prefix):])
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("code") != "neuralese-native-state-open":
        return None
    return {
        "neuralese-native-state-open: generation ended inside a produce span": "budget_exhaustion",
        "neuralese-native-state-open: generation stopped inside a produce span": "model_incomplete",
    }.get(payload.get("message") if isinstance(payload.get("message"), str) else None)


def classify_case(result: dict[str, Any] | None, finish: dict[str, Any] | None) -> str:
    """Use explicit terminal evidence; repeated action-shape counts alone are not a failure label."""
    if finish and finish.get('status') == 'repetition' and finish.get('resource_limit_reason') == 'identical_turn_limit':
        return 'repetition'
    evidence = (finish or {}).get('model_incomplete_evidence') or {}
    requests = evidence.get('request_sha256') or []
    if (finish and finish.get('status') == 'model_incomplete'
            and finish.get('resource_limit_reason') == 'consecutive_truncated_no_action'
            and evidence.get('reason') == 'consecutive_truncated_no_action'
            and isinstance(evidence.get('invocation_id'), str) and evidence['invocation_id']
            and type(evidence.get('limit')) is int and evidence['limit'] > 0
            and type(evidence.get('completed_no_action_replies')) is int
            and evidence['completed_no_action_replies'] >= evidence['limit']
            and isinstance(requests, list) and len(requests) == evidence['completed_no_action_replies']
            and all(isinstance(item,str) and re.fullmatch('[0-9a-f]{64}',item) for item in requests)
            and not (result and (result.get('outcome') or {}).get('status') == 'done'
                     and (result.get('outcome') or {}).get('accepted') is True)):
        return 'model_incomplete'
    if result is None:
        accounting = (finish or {}).get('output_accounting') or {}
        states = accounting.get('job_states') or []
        if any(isinstance(state, dict) and state.get('state') == 'transport_failed' for state in states):
            return 'infrastructure_error'
        if finish and finish.get("resource_limit_reason") in {"case_wall_budget", "model_request_budget"}:
            return "budget_exhaustion"
        if finish and finish.get("status") in {"timeout", "no_observation_limit"}:
            return "budget_exhaustion"
        if finish and finish.get("status") in {"complete", "complete_with_skips"}:
            return "infrastructure_error"
        native_outcome = _native_terminal_outcome(accounting)
        if native_outcome is not None:
            return native_outcome
        return "unresolved"

    outcome = result.get("outcome") if isinstance(result.get("outcome"), dict) else {}
    if outcome.get("status") == "done" and outcome.get("accepted") is True:
        return "task_success"

    if finish and finish.get("resource_limit_reason") in {"case_wall_budget", "model_request_budget"}:
        return "budget_exhaustion"
    if outcome.get("resource_limit_reason") or outcome.get("resource_limit"):
        return "budget_exhaustion"

    incomplete_details = [outcome.get("detail")]
    failed_invocation_details: list[str] = []
    invocations = outcome.get("invocation_ledger")
    if isinstance(invocations, list):
        for invocation in invocations:
            if not isinstance(invocation, dict):
                continue
            detail = invocation.get("completion_detail")
            incomplete_details.append(detail)
            if invocation.get("completion_status") == "failed" and isinstance(detail, str):
                failed_invocation_details.append(detail.casefold())
    if any(re.search(r"\b(episode turn, token, or wall-clock budget exhausted|case wall(?:-clock)? budget exhausted|model request budget exhausted)\b",
                     str(detail).casefold()) for detail in incomplete_details if isinstance(detail, str)):
        return "budget_exhaustion"
    # A recovered collector finish can be complete even when a nested runtime
    # invocation was interrupted by the host memory floor. The invocation ledger
    # retains the abort while the queue journal records the recovered exact output.
    if any(detail == "this operation was aborted" for detail in failed_invocation_details):
        return "infrastructure_error"
    if any(isinstance(detail, str) and
           re.match(r"^model_incomplete:\s*(terminated_without_result|truncated_without_action)\b", detail.casefold())
           for detail in incomplete_details):
        return "model_incomplete"

    details = _text({
        "failure_reason": outcome.get("failure_reason"),
        "rejection_reasons": outcome.get("rejection_reasons"),
        "scope_failures": outcome.get("scope_failures"),
        "action_ledger": outcome.get("action_ledger"),
        "invocation_ledger": outcome.get("invocation_ledger"),
        "detail": outcome.get("detail"),
        "error": result.get("error"),
        "finish": finish,
    })
    if re.search(r"(econnrefused|econnreset|connection refused|connection reset|provider request retry exhausted|http 5\d\d|server unavailable)", details):
        return "infrastructure_error"
    scope_failures = outcome.get("scope_failures")
    if isinstance(scope_failures, list):
        for failure in scope_failures:
            if not isinstance(failure, dict):
                continue
            message = str(failure.get("message", "")).casefold()
            if message.strip() == "terminated":
                return "infrastructure_error"
            if re.search(r"\b(syntaxerror|typeerror|referenceerror|rangeerror|not a function|cannot read properties|undefined variable|unknown identifier)\b", message):
                return "invalid_action"
    if re.search(r"\b(repeated|repetition|stuck in a loop|same action.*again)\b", details):
        return "repetition"
    actions = outcome.get("action_ledger")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            diagnostics = action.get("diagnostics")
            if action.get("outcome") in {"invalid", "rejected"} or (isinstance(diagnostics, list) and diagnostics):
                return "invalid_action"
    if outcome.get("status") in {"failed", "quiesced"} and scope_failures:
        # A deliberate terminal `return_result({status:"blocked"})` can leave a
        # scope-failure trace when the task expected an answer. The case has an
        # exact model outcome, so it is a scored failure, not an unresolved run.
        last_action = actions[-1] if isinstance(actions, list) and actions else None
        if (finish and finish.get("status") in {"complete", "complete_with_skips"}
                and isinstance(last_action, dict)
                and last_action.get("name") == "return_result"
                and last_action.get("outcome") == "blocked"
                and isinstance(last_action.get("arguments"), dict)
                and last_action["arguments"].get("status") == "blocked"):
            return "semantic_mismatch"
        return "unresolved"
    if outcome.get("status") not in {"done", "failed", "quiesced"}:
        return "infrastructure_error"
    if finish and finish.get("status") not in {"complete", "complete_with_skips"}:
        return "infrastructure_error"
    return "semantic_mismatch"


def score_rows(source_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]],
               finishes: dict[str, dict[str, Any]], queue_keys: dict[int, str]) -> dict[str, Any]:
    """Score only exact case identities and collector terminal outcomes."""
    source_by_id = {row.get("id"): row for row in source_rows}
    if None in source_by_id or len(source_by_id) != len(source_rows):
        raise ValueError("source packet IDs must be present and unique")
    result_by_id: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        program = (row.get("task") or {}).get("program_ir") or {}
        identity = program.get("id")
        if identity not in source_by_id or identity in result_by_id:
            raise ValueError(f"unexpected or duplicate task result identity: {identity!r}")
        result_by_id[identity] = row

    case_rows = []
    for index, row in enumerate(source_rows):
        identity = row["id"]
        result = result_by_id.get(identity)
        finish = finishes.get(queue_keys.get(index, ""))
        outcome_class = classify_case(result, finish)
        result_hash = None
        if result is not None:
            result_hash = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=False).encode()).hexdigest()
        task_outcome = result.get("outcome", {}) if result else {}
        case_rows.append({
            "id": identity,
            "source_ids": row.get("source_ids", []),
            "source_groups": row.get("source_groups", []),
            "family": row.get("family") or (row.get("curriculum") or {}).get("family"),
            "outcome": outcome_class,
            "collector_status": task_outcome.get("status") if isinstance(task_outcome, dict) else None,
            "accepted": task_outcome.get("accepted") if isinstance(task_outcome, dict) else None,
            "rejection_reasons": task_outcome.get("rejection_reasons", []) if isinstance(task_outcome, dict) else [],
            "checks": task_outcome.get("checks") if isinstance(task_outcome, dict) else None,
            "result_sha256": result_hash,
        })

    counts = {name: sum(item["outcome"] == name for item in case_rows) for name in TASK_OUTCOMES}
    by_family: dict[str, dict[str, int]] = {}
    for case in case_rows:
        family = str(case.get("family") or "unknown")
        group = by_family.setdefault(family, {name: 0 for name in TASK_OUTCOMES})
        group[case["outcome"]] += 1
    complete = len(result_by_id) == len(source_rows) and counts["unresolved"] == 0
    return {
        "schema": "natlang.direct-task-packet-score/1",
        "cases_expected": len(source_rows),
        "cases_with_exact_terminal_result": len(result_by_id),
        "complete": complete,
        "eligible_for_best": complete and counts["infrastructure_error"] == 0,
        "task_successes": counts["task_success"],
        "task_success_rate": counts["task_success"] / len(source_rows) if source_rows else None,
        "outcome_counts": counts,
        "by_family": by_family,
        "cases": case_rows,
    }


def accounted(score: dict) -> bool:
    """Every selected case has a known outcome; only infrastructure/unresolved
    outcomes make scoring ineligible. Result-export completeness is separate.
    """
    if not isinstance(score, dict):
        return False
    counts = score.get('outcome_counts')
    expected = score.get('cases_expected', score.get('cases'))
    return (type(expected) is int and expected > 0
            and isinstance(counts, dict)
            and {'task_success', 'infrastructure_error', 'unresolved'} <= set(counts) <= set(TASK_OUTCOMES)
            and all(type(value) is int and value >= 0 for value in counts.values())
            and counts['infrastructure_error'] == 0 and counts['unresolved'] == 0
            and sum(counts.values()) == expected)


def retain_best_checkpoint(checkpoint: str | Path, evaluation: dict[str, Any],
                           evaluation_sha256: str, destination: str | Path) -> dict[str, Any]:
    """Publish the highest task-success count among complete, comparable evaluations."""
    checkpoint, destination = Path(checkpoint).resolve(strict=True), Path(destination)
    if not evaluation.get("eligible_for_best"):
        return {"promoted": False, "reason": "evaluation is incomplete or has infrastructure errors"}
    comparison_key = evaluation.get("comparison_key")
    if not isinstance(comparison_key, str) or not re.fullmatch(r"[0-9a-f]{64}", comparison_key):
        raise ValueError("evaluation requires a stable comparison_key SHA-256")
    successes = evaluation.get("task_successes")
    if isinstance(successes, bool) or not isinstance(successes, int) or successes < 0:
        raise ValueError("task_successes must be a nonnegative integer")
    digest = sha256_file_hex(checkpoint)
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".task-best.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        best_path = destination / "best-task-checkpoint.pt"
        receipt_path = destination / "best-task-checkpoint.json"
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if previous and previous.get("comparison_key") != comparison_key:
            raise ValueError("refusing to compare task scores from different evaluation configurations")
        if previous and previous.get("task_successes", -1) >= successes:
            return {"promoted": False, "reason": "existing task-success count is equal or higher",
                    "previous_task_successes": previous.get("task_successes")}
        pending = best_path.with_name(best_path.name + ".pending")
        pending.unlink(missing_ok=True)
        os.link(checkpoint, pending)
        pending.replace(best_path)
        receipt = {
            "schema": "natlang.direct-task-best-checkpoint/1",
            "checkpoint_path": str(best_path.resolve()),
            "checkpoint_sha256": digest,
            "task_successes": successes,
            "task_success_rate": evaluation.get("task_success_rate"),
            "comparison_key": comparison_key,
            "evaluation_sha256": evaluation_sha256,
            "cases_expected": evaluation.get("cases_expected"),
            "outcome_counts": evaluation.get("outcome_counts"),
            "by_family": evaluation.get("by_family"),
            "tie_policy": "keep-earlier-checkpoint",
        }
        temp = receipt_path.with_name(receipt_path.name + ".pending")
        temp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        temp.replace(receipt_path)
        return {"promoted": True, "task_successes": successes, "checkpoint_sha256": digest,
                "path": str(best_path)}
