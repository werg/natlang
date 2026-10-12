"""Shared scoring and exact checkpoint retention for protected authored task packets."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


TASK_OUTCOMES = (
    "task_success",
    "invalid_action",
    "repetition",
    "budget_exhaustion",
    "infrastructure_error",
    "semantic_mismatch",
    "unresolved",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{number}")
            rows.append(value)
    return rows


def _text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).casefold()


def classify_case(result: dict[str, Any] | None, finish: dict[str, Any] | None) -> str:
    """Use explicit terminal evidence; repeated action-shape counts alone are not a failure label."""
    if result is None:
        if finish and finish.get("resource_limit_reason") in {"case_wall_budget", "model_request_budget"}:
            return "budget_exhaustion"
        if finish and finish.get("status") in {"timeout", "no_observation_limit"}:
            return "budget_exhaustion"
        if finish and finish.get("status") in {"complete", "complete_with_skips"}:
            return "infrastructure_error"
        return "unresolved"

    outcome = result.get("outcome") if isinstance(result.get("outcome"), dict) else {}
    if outcome.get("status") == "done" and outcome.get("accepted") is True:
        return "task_success"

    if finish and finish.get("resource_limit_reason") in {"case_wall_budget", "model_request_budget"}:
        return "budget_exhaustion"
    if outcome.get("resource_limit_reason") or outcome.get("resource_limit"):
        return "budget_exhaustion"

    details = _text({
        "failure_reason": outcome.get("failure_reason"),
        "rejection_reasons": outcome.get("rejection_reasons"),
        "scope_failures": outcome.get("scope_failures"),
        "detail": outcome.get("detail"),
        "error": result.get("error"),
        "finish": finish,
    })
    if re.search(r"\b(repeated|repetition|stuck in a loop|same action.*again)\b", details):
        return "repetition"
    actions = outcome.get("action_ledger")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("outcome") in {"invalid", "rejected", "error"} or action.get("diagnostics"):
                return "invalid_action"
    if outcome.get("status") in {"failed", "quiesced"} and outcome.get("scope_failures"):
        return "invalid_action"
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


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


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
    digest = sha256_file(checkpoint)
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


def export_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Build the strict LFM direct-serving artifact from a full training checkpoint."""
    from ..train.direct_export import LFM2_ADAPTER, _file_inventory, export_direct_checkpoint
    from ..model.lfm2_port import DEFAULT_BASE

    snapshot = Path(args.base_snapshot).resolve(strict=True)
    launch_path = Path(args.launch_receipt).resolve(strict=True)
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    hub_cache = Path(launch["hf_hub_cache"]).resolve(strict=True)
    weights = sorted((name, digest) for name, digest in _file_inventory(snapshot).items()
                     if name.endswith(".safetensors"))
    if not weights:
        raise ValueError("base snapshot has no safetensors weights")
    proof = {
        "backbone_adapter": LFM2_ADAPTER,
        "model_id": DEFAULT_BASE,
        "revision": snapshot.name,
        "snapshot_path": str(snapshot),
        "hf_hub_cache": str(hub_cache),
        "model_safetensors_sha256": weights[0][1],
        "launch_receipt_path": str(launch_path),
        "launch_receipt_sha256": sha256_file(launch_path),
        "runtime_image": launch["runtime_image"],
    }
    checkpoint_sha = sha256_file(args.checkpoint)
    artifact = export_direct_checkpoint(
        args.checkpoint, args.output, base_snapshot=snapshot, base_source_proof=proof,
        backbone_adapter=LFM2_ADAPTER,
    )
    if sha256_file(args.checkpoint) != checkpoint_sha:
        raise ValueError("source checkpoint changed during strict export")
    return artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export", help="strictly export a direct-SFT checkpoint for task evaluation")
    exporter.add_argument("--checkpoint", type=Path, required=True)
    exporter.add_argument("--output", type=Path, required=True)
    exporter.add_argument("--base-snapshot", type=Path, required=True)
    exporter.add_argument("--launch-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        print(json.dumps(export_checkpoint(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
