#!/usr/bin/env python3
"""Prepare a reviewed Luna folder-campaign plan; never launch or admit data.

The source and authored-reference replay are read-only inputs. This command
checks their identities, training splits, admission and capture visibility,
then writes fresh worker queues and a plan compatible with
start_reviewed_luna_campaign.py. Owner review must set root_approved and seal
the resulting plan hash before the printed start command can run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
from datetime import datetime, timezone
from typing import Any

from freeze_training_runtime import tree_identity


ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "scripts/start_reviewed_luna_campaign.py"
SUPERVISOR = ROOT / "scripts/run_bonsai_queue.py"

# Operational per-case defaults shared by recent reviewed Luna collectors.
# Dataset sizes, source indices, item totals, and worker distribution are
# always derived from the supplied files and arguments.
CASE_SECONDS = 1800
CONTEXT_TOKENS = 32768
MAX_MODEL_REQUESTS = 384
MAX_TURNS = 60
TRANSPORT_RETRIES = 0
REQUEST_RETRIES = 3


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for physical_index, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{physical_index + 1}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{physical_index + 1}")
            rows.append((physical_index, row))
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def source_identity(row: dict[str, Any]) -> str:
    value = row.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("Every source case must have a nonempty string id")
    return value


def replay_identity(row: dict[str, Any]) -> str:
    """Read the program id from supported reference-result envelopes."""
    admission = row.get("admission")
    if isinstance(admission, dict) and isinstance(admission.get("program_id"), str):
        return admission["program_id"]
    result_row = row.get("row")
    if isinstance(result_row, dict):
        task = result_row.get("task")
        if isinstance(task, dict):
            program = task.get("program_ir")
            if isinstance(program, dict) and isinstance(program.get("id"), str):
                return program["id"]
    for key in ("program_id", "case_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Reference row has no supported source-case identity")


def replay_identities(row: dict[str, Any]) -> set[str]:
    identities: set[str] = set()
    admission = row.get("admission")
    if isinstance(admission, dict) and isinstance(admission.get("program_id"), str):
        identities.add(admission["program_id"])
    result_row = row.get("row")
    if isinstance(result_row, dict):
        task = result_row.get("task")
        if isinstance(task, dict):
            program = task.get("program_ir")
            if isinstance(program, dict) and isinstance(program.get("id"), str):
                identities.add(program["id"])
    for key in ("program_id", "case_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            identities.add(value)
    if not identities:
        identity = row.get("id")
        if isinstance(identity, str) and identity:
            identities.add(identity)
    if len(identities) > 1:
        raise ValueError(f"Reference envelope identities disagree: {sorted(identities)}")
    return identities


def check_replay_visibility(row: dict[str, Any], identity: str) -> list[str]:
    """Validate one of the supported authored-reference gate envelopes.

    Legacy replays carry explicit admission and capture_visibility records. The
    static annotation adapter instead carries its actual reference result and
    visibility assertions; those are not synthesized into admission flags.
    """
    result_row = row.get("row")
    if "context_visibility" in row or "output_matches_annotation" in row:
        if not isinstance(result_row, dict):
            raise ValueError(f"Authored reference envelope is missing its exact result row: {identity}")
        task = result_row.get("task")
        program = task.get("program_ir") if isinstance(task, dict) else None
        if not isinstance(program, dict) or program.get("id") != identity:
            raise ValueError(f"Authored reference program identity mismatch: {identity}")
        outcome = result_row.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("accepted") is not True:
            raise ValueError(f"Authored reference did not accept its exact program: {identity}")
        if row.get("output_matches_annotation") is not True:
            raise ValueError(f"Authored reference output did not match its annotation: {identity}")
        visibility = row.get("context_visibility")
        if not isinstance(visibility, dict) or visibility.get("visible") is not True:
            raise ValueError(f"Authored reference context is not fully visible: {identity}")
        items = visibility.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError(f"Authored reference visibility has no item evidence: {identity}")
        required = ("exact_typed_input_visible", "question_visible", "complete_label_domain_visible")
        for item in items:
            if not isinstance(item, dict) or any(item.get(field) is not True for field in required):
                raise ValueError(f"Authored reference item visibility is incomplete: {identity}")
        return ["row.outcome.accepted", "output_matches_annotation", "context_visibility.visible",
                "all_items_exact_typed_input_question_label_domain_visible"]

    admission = row.get("admission")
    visibility = row.get("capture_visibility")
    if not isinstance(admission, dict) or admission.get("admitted") is not True:
        raise ValueError(f"Reference case is not admitted by its authored-reference gate: {identity}")
    if not isinstance(visibility, dict) or visibility.get("visible") is not True:
        raise ValueError(f"Reference case capture visibility is not true: {identity}")
    # If item-level detail is present, ensure the aggregate flag is not hiding
    # a missing capture. Older replay formats may omit the detail list.
    items = visibility.get("items")
    if items is not None:
        if not isinstance(items, list) or not items:
            raise ValueError(f"Reference visibility has an invalid or empty item list: {identity}")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Reference visibility item is malformed: {identity}")
            visible_fields = item.get("visible_fields")
            fields = item.get("fields")
            if visible_fields is not None and not isinstance(visible_fields, list):
                raise ValueError(f"Reference visible_fields is malformed: {identity}")
            if fields is not None and not isinstance(fields, list):
                raise ValueError(f"Reference fields is malformed: {identity}")
            if visible_fields is not None and fields is not None:
                if not set(fields).issubset(set(visible_fields)):
                    raise ValueError(f"Reference visibility item has hidden fields: {identity}")
            if item.get("missing_fields") not in (None, []):
                raise ValueError(f"Reference visibility item reports missing fields: {identity}")
    return ["admission.admitted", "capture_visibility.visible"]


def load_predecessors(campaign: Path) -> list[dict[str, Any]]:
    launch_path = campaign / "campaign-launch.json"
    if not launch_path.is_file():
        raise ValueError(f"Predecessor campaign has no campaign-launch.json: {campaign}")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    workers = launch.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError("Predecessor campaign launch has no actual worker records")
    predecessors = []
    seen_queues: set[Path] = set()
    for worker in workers:
        if not isinstance(worker, dict):
            raise ValueError("Malformed predecessor worker record")
        pid = worker.get("pid")
        queue = Path(worker.get("queue", "")).expanduser().resolve()
        journal = Path(worker.get("journal", "")).expanduser().resolve()
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
            raise ValueError("Predecessor worker has no valid recorded PID")
        if not queue.is_file() or not journal.is_file():
            raise ValueError(f"Predecessor worker queue/journal missing: {queue}, {journal}")
        if queue in seen_queues:
            raise ValueError(f"Duplicate predecessor queue: {queue}")
        seen_queues.add(queue)
        predecessors.append({"pid": pid, "queue": str(queue), "journal": str(journal),
                             "queue_sha256": digest(queue)})
    return predecessors


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    cases_path = args.cases.expanduser().resolve(strict=True)
    replay_path = args.reference_replay.expanduser().resolve(strict=True)
    runtime = args.runtime.expanduser().resolve(strict=True)
    out = args.out.expanduser().resolve()
    if out.exists():
        raise ValueError(f"Output path must be fresh and absent: {out}")
    if not 1 <= args.workers <= 5:
        raise ValueError("--workers must be between 1 and 5")
    if isinstance(args.seed_base, bool) or args.seed_base < 0:
        raise ValueError("--seed-base must be a nonnegative integer")
    if args.item_agreement_count is not None and args.item_agreement_count < 0:
        raise ValueError("--item-agreement-count must be nonnegative")

    frozen_path = runtime / "frozen-runtime.json"
    if not frozen_path.is_file():
        raise ValueError(f"Runtime has no frozen-runtime.json: {runtime}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if not isinstance(frozen.get("files"), dict) or tree_identity(runtime) != frozen["files"]:
        raise ValueError("Runtime tree does not match its frozen-runtime.json identity")

    case_rows = read_jsonl(cases_path)
    replay_rows = read_jsonl(replay_path)
    case_ids: list[str] = []
    case_index: dict[str, int] = {}
    source_item_count = 0
    for index, row in case_rows:
        identity = source_identity(row)
        if identity in case_index:
            raise ValueError(f"Duplicate source case id: {identity}")
        if row.get("split") != "train":
            raise ValueError(f"Only training split cases may be prepared: {identity} split={row.get('split')!r}")
        ids = row.get("source_ids", [])
        if not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError(f"Malformed source_ids for case: {identity}")
        case_ids.append(identity)
        case_index[identity] = index  # preserve physical JSONL line index
        source_item_count += len(ids)
    if args.item_agreement_count is not None and args.item_agreement_count > source_item_count:
        raise ValueError("--item-agreement-count cannot exceed the derived source item count")

    replay_index: dict[str, dict[str, Any]] = {}
    gates_by_identity: dict[str, list[str]] = {}
    for _, row in replay_rows:
        identity = replay_identity(row)
        if replay_identities(row) != {identity}:
            raise ValueError(f"Reference envelope does not consistently identify case: {identity}")
        if identity in replay_index:
            raise ValueError(f"Duplicate reference replay case id: {identity}")
        gates_by_identity[identity] = check_replay_visibility(row, identity)
        replay_index[identity] = row
    if len(case_ids) != len(replay_index):
        raise ValueError(f"Source/replay case counts differ: {len(case_ids)} vs {len(replay_index)}")
    if set(case_ids) != set(replay_index):
        missing = sorted(set(case_ids) - set(replay_index))
        extra = sorted(set(replay_index) - set(case_ids))
        raise ValueError(f"Source/replay identities differ; missing={missing[:5]} extra={extra[:5]}")
    for _, case in case_rows:
        identity = source_identity(case)
        replay = replay_index[identity]
        result = replay.get("row")
        task = result.get("task") if isinstance(result, dict) else None
        program = task.get("program_ir") if isinstance(task, dict) else None
        if program is not None and program != case:
            raise ValueError(f"Reference program contents differ from the supplied case: {identity}")
        visibility = replay.get("context_visibility", replay.get("capture_visibility", {}))
        items = visibility.get("items")
        if items is not None:
            visible_ids = [item.get("id") for item in items]
            expected_ids = case.get("source_ids", [])
            if (any(not isinstance(value, str) or not value for value in visible_ids) or
                    len(visible_ids) != len(set(visible_ids)) or
                    set(visible_ids) != set(expected_ids)):
                raise ValueError(f"Reference item visibility does not cover the exact source IDs: {identity}")
    if args.workers > len(case_rows):
        raise ValueError("Worker count cannot exceed the number of source cases")

    predecessor_campaign = args.predecessor_campaign.expanduser().resolve() if args.predecessor_campaign else None
    predecessors = load_predecessors(predecessor_campaign) if predecessor_campaign else []
    now = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects prior attempts and makes every queue/log path fresh.
    out.mkdir(mode=0o750)

    review = {
        "schema": "natlang.root-folder-generation-review/1",
        "time_utc": now,
        "source_path": str(cases_path),
        "source_sha256": digest(cases_path),
        "reference_replay_path": str(replay_path),
        "reference_replay_sha256": digest(replay_path),
        "runtime_path": str(runtime),
        "runtime_manifest_sha256": digest(frozen_path),
        "programs": len(case_rows),
        "source_items": source_item_count,
        "source_case_ids": case_ids,
        "source_splits": {"train": len(case_rows)},
        "reference_replay_cases": len(replay_rows),
        "reference_gates": {"checked_gate_sets": sorted({tuple(gates) for gates in gates_by_identity.values()}),
                            "per_case": gates_by_identity},
        "source_quality": "held_for_independent_quality_review",
        "training_admission": False,
        "independent_new_worlds": 0,
        "related_worlds": "Source lineage and split/group metadata are preserved verbatim; this preparation adds no independent-world credit.",
        "authored_reference_replay": "Gate evidence only; authored-reference acceptance is not teacher reasoning or semantic admission.",
        "item_agreement_count": args.item_agreement_count,
        "item_agreement_count_status": "caller-supplied review note only; not independently validated" if args.item_agreement_count is not None else "not supplied",
        "max_new_workers": args.workers,
        "predecessor_campaign": str(predecessor_campaign) if predecessor_campaign else None,
        "predecessor_worker_count": len(predecessors),
        "reasoning_effort": args.reasoning_effort,
        "prepared_only": True,
    }
    review_path = out / "root-generation-source-review-v1.json"
    review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    artifact_hashes: dict[str, str] = {
        str(cases_path): digest(cases_path),
        str(replay_path): digest(replay_path),
        str(review_path): digest(review_path),
        str(frozen_path): digest(frozen_path),
        str(SUPERVISOR): digest(SUPERVISOR),
        str(STARTER): digest(STARTER),
    }
    worker_records = []
    all_cases = list(case_rows)
    for worker_number in range(1, args.workers + 1):
        worker_dir = out / f"worker-{worker_number:02d}"
        worker_dir.mkdir(mode=0o750)
        queue_path = worker_dir / "queue.jsonl"
        entries = []
        for source_index, case in all_cases[worker_number - 1::args.workers]:
            case_id = source_identity(case)
            seed = args.seed_base + source_index
            key_hash = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]
            case_dir = worker_dir / f"case-{source_index:06d}"
            entries.append({
                "case_seconds": CASE_SECONDS,
                "context_tokens": CONTEXT_TOKENS,
                "count": 1,
                "index": source_index,
                "jobs": str(case_dir / "jobs"),
                "key": f"luna-folder-{key_hash}-{source_index}-seed{seed}",
                "log": str(case_dir / "collector.log"),
                "max_model_requests": MAX_MODEL_REQUESTS,
                "max_turns": MAX_TURNS,
                "output": str(case_dir / "results.jsonl"),
                "request_retries": REQUEST_RETRIES,
                "seed": seed,
                "source": str(cases_path),
                "text_neuralese_emulation": True,
                "transport_retries": TRANSPORT_RETRIES,
            })
        queue_path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
        journal = worker_dir / "journal.jsonl"
        worker_log = worker_dir / "worker.log"
        worker_records.append({"number": worker_number, "queue": str(queue_path),
                              "journal": str(journal), "log": str(worker_log)})
        artifact_hashes[str(queue_path)] = digest(queue_path)

    plan = {
        "schema": "natlang.reviewed-fresh-luna-campaign/1",
        "root_approved": False,
        "preparation_only": True,
        "cwd": str(ROOT),
        "runtime": str(runtime),
        "supervisor": str(SUPERVISOR),
        "reasoning_effort": args.reasoning_effort,
        "training_admission": False,
        "independent_new_worlds": 0,
        "workers": worker_records,
        "artifact_hashes": artifact_hashes,
        "launch_record": str(out / "campaign-launch.json"),
        "predecessors": predecessors,
        "approval_scope": "Preparation only. Source/reference gates passed, but source quality and execution remain subject to owner review; no training admission or independent-world credit is conferred.",
    }
    plan_path = out / "reviewed-plan-v1.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan, plan_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="train-only source cases JSONL")
    parser.add_argument("--reference-replay", required=True, type=Path,
                        help="authored-reference replay JSONL with admission and capture visibility")
    parser.add_argument("--runtime", required=True, type=Path, help="frozen runtime directory")
    parser.add_argument("--out", required=True, type=Path, help="fresh output directory; must not exist")
    parser.add_argument("--seed-base", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int, help="one to five Luna workers")
    parser.add_argument("--predecessor-campaign", type=Path,
                        help="optional campaign root; actual worker queues become pinned predecessors")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--item-agreement-count", type=int,
                        help="optional caller-supplied count recorded as an unverified review note")
    args = parser.parse_args()
    plan, path = build_plan(args)
    plan_sha = digest(path)
    quoted_plan = shlex.quote(str(path))
    # The candidate is deliberately unapproved. Hash it at launch time so the
    # command remains exact after an owner has reviewed and sealed the plan.
    manual_command = (f"python3 scripts/start_reviewed_luna_campaign.py {quoted_plan} "
                      f'--sha256 "$(sha256sum {quoted_plan} | cut -d \' \' -f 1)"')
    print(json.dumps({
        "prepared": True,
        "root_approved": False,
        "case_count": len(read_jsonl(args.cases.expanduser().resolve(strict=True))),
        "worker_count": len(plan["workers"]),
        "plan": str(path),
        "plan_sha256": plan_sha,
        "source_review": str(path.parent / "root-generation-source-review-v1.json"),
        "manual_start_command_after_owner_review": manual_command,
        "note": "Preparation only. The plan currently has root_approved=false. After owner review seals root_approved=true, run the command; the starter will verify the current plan hash. This command did not launch jobs.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
