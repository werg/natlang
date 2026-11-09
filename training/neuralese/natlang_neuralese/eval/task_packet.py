"""Shared scoring and exact checkpoint retention for protected authored task packets."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any

from ..common.hashing import sha256_file_hex
from ..common.jsonio import read_jsonl

_CLASSIFIER_SOURCE = Path(__file__).resolve()
_CLASSIFIER_SHA256 = hashlib.sha256(_CLASSIFIER_SOURCE.read_bytes()).hexdigest()


def classification_binding():
    if hashlib.sha256(_CLASSIFIER_SOURCE.read_bytes()).hexdigest() != _CLASSIFIER_SHA256:
        raise ValueError('outcome classifier changed after import; restart from a frozen runtime')
    return {'version': 'host-evidence-terminal-babi-article/3', 'path': str(_CLASSIFIER_SOURCE),
            'sha256': _CLASSIFIER_SHA256}

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


def _accepted_terminal(outcome: dict[str, Any], finish: dict[str, Any] | None) -> bool:
    """An expected, evidenced stop is a correct task outcome too."""
    if outcome.get('accepted') is not True:
        return False
    if outcome.get('status') == 'done':
        return True
    checks = outcome.get('checks')
    return (outcome.get('status') in {'failed', 'quiesced'}
            and (finish or {}).get('status') in {'complete', 'complete_with_skips'}
            and not (finish or {}).get('resource_limit_reason')
            and not outcome.get('resource_limit_reason') and not outcome.get('resource_limit')
            and not outcome.get('rejection_reasons')
            and (outcome.get('oracle') or {}).get('accepted') is True
            and isinstance(checks, dict)
            and all(checks.get(key) is True for key in ('expected_status', 'honest_stop', 'stop_evidence'))
            and all(value is True for value in checks.values()))


def accepted_babi_article_answer(result, finish=None):
    """Repair the historical bAbI location article alias, retaining every other check.

    This is deliberately narrower than general QA normalization: archived task
    packets declared only generic normalization. Never waive a world/effect,
    terminal-status or non-answer failure, or infer facts from model prose.
    """
    if not isinstance(result, dict):
        return False
    task = result.get('task')
    if not isinstance(task, dict) or not isinstance(task.get('program_ir'), dict):
        return False
    program = task['program_ir']
    semantics, outcome = program.get('semantics'), result.get('outcome')
    if not isinstance(semantics, dict) or not isinstance(outcome, dict):
        return False
    checks = outcome.get('checks')
    if not isinstance(checks, dict) or not isinstance(program.get('source_ids'), list):
        return False
    if (program.get('source') != 'external-rl:babi'
            or not re.fullmatch(r'ext_babi_task\d{2}', str(program.get('family', '')))
            or not any(str(x).startswith('facebook/babi_qa:') for x in program.get('source_ids', []))
            or semantics.get('oracle') != 'normalized'
            or outcome.get('status') != 'done'
            or outcome.get('accepted') is not False
            or set(outcome.get('rejection_reasons') or []) != {'answer'}
            or checks.get('answer') is not False or not checks
            or any(value is not True for key, value in checks.items() if key != 'answer')
            or (finish and (finish.get('status') != 'complete' or finish.get('resource_limit_reason')))
            or outcome.get('resource_limit_reason') or outcome.get('resource_limit')):
        return False
    expected, actual = semantics.get('expected'), outcome.get('value')
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    def normalize(value):
        return ' '.join(unicodedata.normalize('NFKC', value).lower().split())
    expected, actual = normalize(expected), normalize(actual)
    # Only the redundant leading definite article is proved by this repair.
    return bool(expected) and actual == 'the ' + expected and not expected.startswith('the ')


_MODEL_TERMINAL = re.compile(r"^(?:blocked|error): ")
# Host-written interruption details are plain lowercase phrases. A model's own stop reason is always prefixed
# ("blocked: " / "error: ", ts-host runtime.ts return_result), and a `done` detail is the dumped return value.
_HOST_WALL_DETAIL = re.compile(r"[a-z ,-]{0,60}wall(?:-clock)? budget exhausted")
_HOST_REQUEST_DETAIL = re.compile(
    r"[a-z ,-]{0,60}(?:model request budget exhausted|episode (?:turn|token) budget exhausted)(?: before [a-z ]+)?")
_TRANSPORT = re.compile(
    r"(econnrefused|econnreset|connection refused|connection reset|provider request retry exhausted|http 5\d\d|server unavailable)")
_COMPLETED_FINISH = {"complete", "complete_with_skips"}


def _host_incomplete_details(outcome: dict[str, Any]) -> list[str]:
    """Interruption details written by the host runtime, never the model's terminal reason or return value.

    A `done` root's `detail` and a `done` invocation's `completion_detail` are dumped return values, and an explicit
    blocked/failed return carries the model's own reason; none can evidence a limit or a transport fault."""
    details: list[Any] = [] if outcome.get("status") == "done" else [outcome.get("detail")]
    ledger = outcome.get("invocation_ledger")
    if isinstance(ledger, list):
        for invocation in ledger:
            if isinstance(invocation, dict) and invocation.get("completion_status") != "done":
                details.append(invocation.get("completion_detail"))
    return [detail.strip().casefold() for detail in details
            if isinstance(detail, str) and not _MODEL_TERMINAL.match(detail.strip())]


def _completed_oracle_rejection(outcome: dict[str, Any], finish: dict[str, Any] | None) -> bool:
    """A finished episode whose task oracle rejected the final answer: a verified model failure."""
    return (outcome.get("status") == "done" and outcome.get("accepted") is False
            and (finish or {}).get("status") in _COMPLETED_FINISH
            and isinstance(outcome.get("checks"), dict)
            and any(value is False for value in outcome["checks"].values()))


def _host_transport_failure(result: dict[str, Any], finish: dict[str, Any] | None) -> bool:
    """Transport faults from the host's own records: the collector's error field and the finish accounting."""
    if _TRANSPORT.search(_text(result.get("error"))):
        return True
    states = (((finish or {}).get("output_accounting") or {}).get("job_states")) or []
    for state in states if isinstance(states, list) else []:
        if not isinstance(state, dict):
            continue
        error = state.get("terminal_error")
        if state.get("state") == "transport_failed" or (
                isinstance(error, dict) and _TRANSPORT.search(_text(error.get("error")))):
            return True
    return False


def _host_aborted_invocation(outcome: dict[str, Any]) -> bool:
    """The host aborted a nested invocation (memory floor), recorded by the runtime's own invocation trace."""
    ledger = outcome.get("invocation_ledger")
    return isinstance(ledger, list) and any(
        isinstance(invocation, dict) and invocation.get("completion_status") == "failed"
        and invocation.get("completion_source") == "runtime_invocation_trace"
        and isinstance(invocation.get("completion_detail"), str)
        and invocation["completion_detail"].strip().casefold() == "this operation was aborted"
        for invocation in ledger)


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
        if finish and (finish.get("resource_limit_reason") == "case_wall_budget"
                       or finish.get("status") in {"timeout", "no_observation_limit"}):
            # Host time spent waiting/decoding is not a completed model answer.
            # Preserve it as unscored even when useful partial actions exist.
            return "unresolved"
        if finish and finish.get("resource_limit_reason") == "model_request_budget":
            return "budget_exhaustion"
        if finish and finish.get("status") in {"complete", "complete_with_skips"}:
            return "infrastructure_error"
        native_outcome = _native_terminal_outcome(accounting)
        if native_outcome is not None:
            return native_outcome
        return "unresolved"

    outcome = result.get("outcome") if isinstance(result.get("outcome"), dict) else {}
    if _accepted_terminal(outcome, finish):
        return "task_success"
    if accepted_babi_article_answer(result, finish):
        return "task_success"

    if finish and (finish.get("resource_limit_reason") == "case_wall_budget"
                   or finish.get("status") in {"timeout", "no_observation_limit"}):
        return "unresolved"
    if outcome.get('resource_limit_reason') in ('case_wall_budget', 'wall_clock_budget'):
        return "unresolved"
    if finish and finish.get("resource_limit_reason") == "model_request_budget":
        return "budget_exhaustion"
    if outcome.get("resource_limit_reason") or outcome.get("resource_limit"):
        return "budget_exhaustion"

    # Host-recorded interruption of a nested invocation (the collector recovered the exact output) is not a
    # model answer, even when the final result was rejected.
    if _host_aborted_invocation(outcome):
        return "infrastructure_error"
    # A finished episode whose oracle rejected the answer is a verified model failure. Nothing the model wrote
    # (code, arguments, return values, stop reasons) can turn it into an unscored outcome, so no text heuristic
    # below may run first.
    if _completed_oracle_rejection(outcome, finish):
        return "semantic_mismatch"
    if _host_transport_failure(result, finish):
        return "infrastructure_error"

    host_details = _host_incomplete_details(outcome)
    if any(_HOST_WALL_DETAIL.fullmatch(detail) for detail in host_details):
        # Older mixed-limit messages do not establish whether a deterministic
        # turn/token limit or the host's wall-clock timeout actually fired.
        return "unresolved"
    if any(_HOST_REQUEST_DETAIL.fullmatch(detail) for detail in host_details):
        return "budget_exhaustion"
    if any(re.match(r"^model_incomplete:\s*(terminated_without_result|truncated_without_action)\b", detail)
           for detail in host_details):
        return "model_incomplete"

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
    if any(detail.startswith("model_incomplete: repeated_failed_action") for detail in host_details):
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
                # The runtime maps return_result status "failed" to the same blocked terminal.
                and last_action["arguments"].get("status") in {"blocked", "failed"}):
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
