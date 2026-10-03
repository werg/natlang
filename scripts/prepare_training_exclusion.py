#!/usr/bin/env python3
"""Create a read-only, checkpoint-bound source-exclusion transition proposal.

This command never changes the input corpus or checkpoint. Its manifest records
how a future copied-checkpoint run can omit reviewed row IDs while retaining the
original split and exact remaining shuffled permutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from pathlib import Path
import tempfile
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.corpus import digest, file_digest, index_pairs, records, split_programs
from scripts.training_exclusion import (exclusion_split_manifest, exclusion_target,
                                        filter_training_order, pending_source_review_entry,
                                        post_exclusion_mix_summary, transition_state)
from scripts.training_readiness import validate_training_audit, validate_training_inventory_audit, validate_training_mix_audit

REVIEWED_ADDITIONAL_SOURCE_PINS = {
    "2hop__568389_161223": {
        "row_id": "teacher-program:0a89f8bbbd02e9d7bed9:decision:0000",
        "source_id": "2hop__568389_161223",
        "source_program_id": "inline-curriculum:source_musique:a306e9c880f6a18c081c:v1",
        "source_snapshot_sha256": "fd9ca87bfd1ad07569ce8e28365d6344515ab5bfd1805085e4b1fbe5d550c348",
    }
}


def checkpoint_file_hashes(directory: Path) -> dict[str, str]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint contains unsupported symlink: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = file_digest(path)
    if "state.json" not in files:
        raise ValueError("checkpoint has no state.json")
    return files


def write_new_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_object(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def make_transition(*, data_path: Path, split_path: Path, checkpoint_dir: Path,
                    exposure_path: Path, source_review_path: Path,
                    required_source_id: str, required_program_id: str,
                    additional_exclusions_path: Path | None = None,
                    mix_path: Path, inventory_ready_path: Path, inventory_policy_path: Path,
                    active_plan_path: Path, model: str, model_revision: str, max_len: int) -> dict:
    data_path, split_path, checkpoint_dir = map(Path, (data_path, split_path, checkpoint_dir))
    state_path = checkpoint_dir / "state.json"
    before = checkpoint_file_hashes(checkpoint_dir)
    state = load_object(state_path)
    state_sha = file_digest(state_path)
    exposure = load_object(exposure_path)
    split_saved = load_object(split_path)
    source_sha = file_digest(source_review_path)
    source_root = Path(__file__).resolve().parent
    code_pins = {}
    source_runtime_root = Path(source_review_path).resolve().parents[1]
    if Path(source_review_path).name != "source-review.ts" or not (source_runtime_root / "controller").is_dir():
        raise ValueError("production source-exclusion manifests require a sealed candidate runtime source-review path")
    for name in ("training_exclusion.py", "prepare_training_exclusion.py",
                 "apply_training_exclusion_transition.py", "train_lora.py"):
        code_path = source_runtime_root / "controller" / name
        if not code_path.is_file() or file_digest(code_path) != file_digest(source_root / name):
            raise ValueError(f"sealed runtime controller does not match source code: {name}")
        code_pins[name] = {"path": str(code_path.resolve()), "sha256": file_digest(code_path)}
    active_plan_sha = file_digest(active_plan_path)
    data_sha = file_digest(data_path)
    training_audit = validate_training_audit(data_path, max_len, model, model_revision)
    mix_identity = validate_training_mix_audit(mix_path, data_path, 0.25, training_audit)
    inventory_identity = validate_training_inventory_audit(inventory_ready_path, inventory_policy_path)
    corpus = state.get("corpus")
    if not isinstance(corpus, dict) or corpus.get("data_sha256") != data_sha:
        raise ValueError("checkpoint and source corpus identities differ")
    if exposure.get("data_sha256") != data_sha:
        raise ValueError("exposure receipt names different source corpus bytes")
    if exposure.get("ordering") != "actual scripts.corpus.split_programs(index_pairs, holdout=200, seed=42)":
        raise ValueError("unsupported exposure ordering receipt")

    args = state.get("args", {})
    seed, holdout = int(corpus["seed"]), int(args["holdout"])
    pairs = index_pairs(data_path)
    held, train, split = split_programs(pairs, holdout, seed)
    if split != split_saved or digest(split) != corpus.get("split_sha256"):
        raise ValueError("saved split differs from the checkpoint's original split")
    expected_gates = {**mix_identity, **inventory_identity}
    if corpus.get("joint_gate_identity") != expected_gates:
        raise ValueError("training gate identities do not match the checkpoint's pinned original inputs")
    if corpus.get("data_order") == "source":
        train.sort(key=lambda row: row["offset"])
    elif corpus.get("data_order") != "shuffle":
        raise ValueError("unsupported checkpoint training order")

    exposure_rows = exposure.get("rows")
    if not isinstance(exposure_rows, list) or not exposure_rows:
        raise ValueError("exposure receipt does not contain row identities")
    excluded = [str(row["row_id"]) for row in exposure_rows]
    if any(row.get("program_id") != required_program_id for row in exposure_rows):
        raise ValueError("exposure rows do not all belong to the exact held source program")
    source_review_text = Path(source_review_path).read_text(encoding="utf-8")
    if required_source_id not in source_review_text or '"status": "pending"' not in source_review_text:
        raise ValueError("current source-review file does not contain the pending source hold")

    cursor = state.get("cursor")
    trained_examples = state.get("trained_examples")
    if state.get("skipped", 0) != 0 or cursor != trained_examples:
        raise ValueError("current checkpoint has skips or cursor/example divergence; manual review required")
    if exposure.get("step", 0) > state.get("step", -1):
        raise ValueError("exposure receipt is newer than the selected checkpoint")
    for row in exposure_rows:
        if not isinstance(row.get("position_1based"), int):
            raise ValueError("exposure receipt row is missing an order position")

    additional = []
    additional_descriptor_sha = None
    if additional_exclusions_path is not None:
        additional_descriptor_sha = file_digest(additional_exclusions_path)
        extra_doc = load_object(additional_exclusions_path)
        if extra_doc.get("schema") != "natlang.training_source_exclusion_additional/1":
            raise ValueError("unsupported additional exclusion descriptor")
        extra_rows = extra_doc.get("rows")
        if not isinstance(extra_rows, list) or not extra_rows:
            raise ValueError("additional exclusion descriptor must contain rows")
        indexed = {str(row["id"]): row for row in train}
        raw_records = {}
        raw_row_hashes = {}
        wanted = {str(item.get("row_id")) for item in extra_rows}
        for row in records(data_path):
            if str(row.get("id")) in wanted:
                raw_records[str(row["id"])] = row
                raw_row_hashes[str(row["id"])] = hashlib.sha256(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if set(raw_records) != wanted:
            raise ValueError("additional exclusion row is missing from the pinned corpus")
        for item in extra_rows:
            row_id = str(item.get("row_id"))
            source_id = str(item.get("source_id"))
            program_id = str(item.get("source_program_id"))
            pinned = REVIEWED_ADDITIONAL_SOURCE_PINS.get(source_id)
            if pinned is None or any(item.get(key) != value for key, value in pinned.items()):
                raise ValueError(f"additional exclusion is not an exact reviewed source pin: {source_id}")
            base = indexed.get(row_id)
            raw = raw_records[row_id]
            if (base is None or source_id not in raw.get("source_ids", []) or
                    program_id not in raw.get("source_groups", []) or
                    program_id not in base.get("source_groups", [])):
                raise ValueError(f"additional exclusion identity does not match corpus row {row_id}")
            if item.get("corpus_row_sha256") != raw_row_hashes[row_id]:
                raise ValueError(f"additional exclusion row bytes changed: {row_id}")
            if item.get("source_snapshot_sha256") != item.get("expected_source_snapshot_sha256"):
                raise ValueError(f"additional source snapshot pin mismatch for {source_id}")
            if not pending_source_review_entry(source_review_text, source_id):
                raise ValueError(f"additional source review is not pending for {source_id}")
            if item.get("source_review_path") != str(Path(source_review_path).resolve()):
                try:
                    same_review_bytes = file_digest(Path(item.get("source_review_path", ""))) == source_sha
                except OSError:
                    same_review_bytes = False
                if not same_review_bytes:
                    raise ValueError("additional source review does not match the pinned source-review bytes")
            additional.append({**item, "corpus_row_id": row_id,
                               "training_position_1based": next(i for i, r in enumerate(train, 1)
                                                                 if str(r["id"]) == row_id)})
            excluded.append(row_id)
    if len(set(excluded)) != len(excluded):
        raise ValueError("source exclusion list contains duplicate row IDs")

    order = filter_training_order(train, excluded, cursor)
    positions = {row["id"]: index + 1 for index, row in enumerate(train)}
    for receipt_row in exposure_rows:
        actual_position = positions.get(receipt_row["row_id"])
        if actual_position != receipt_row["position_1based"]:
            raise ValueError(f"exposure order position mismatch for {receipt_row['row_id']}")
    consumed_receipt = {str(row["row_id"]) for row in exposure.get("consumed_rows", [])}
    receipt_cursor = int(exposure.get("cursor", -1))
    if receipt_cursor < 0 or receipt_cursor > cursor:
        raise ValueError("exposure receipt cursor is invalid or newer than the checkpoint")
    receipt_expected_consumed = {str(row["row_id"]) for row in exposure_rows
                                 if int(row["position_1based"]) <= receipt_cursor}
    if consumed_receipt != receipt_expected_consumed:
        raise ValueError("pinned historical exposure receipt is internally inconsistent")

    target = exclusion_target(int(corpus["target_examples"]), int(trained_examples),
                              len(order.future_removed_ids))
    accumulation = int(args["accum"])
    optimizer_steps_required = int(state["step"]) + math.ceil(
        (target - int(trained_examples)) / accumulation)
    preserved_scheduler_steps = int(corpus["steps"])
    active_split = exclusion_split_manifest(split, order.rows, excluded)
    mix_target = float(load_object(mix_path)["target_reducer_share"])
    filtered_mix = post_exclusion_mix_summary(data_path, train, excluded, mix_target)
    if not filtered_mix["target_met"]:
        raise ValueError("source exclusion would violate the required reducer mix target")
    manifest_identity = {
        "schema": "natlang.training_source_exclusion/1",
        "status": "review_candidate_not_applied",
        "procedure_code": {"files": code_pins,
                           "runtime_requirement": "apply and resumed training require a separately frozen runtime containing these reviewed code versions and the current source-review/compiled-policy pair"},
        "active_plan": {"path": str(Path(active_plan_path).resolve()), "sha256": active_plan_sha},
        "data": {"path": str(data_path.resolve()), "sha256": data_sha,
                 "rows": len(pairs), "held_rows": len(held), "training_rows": len(train)},
        "split": {"path": str(split_path.resolve()), "sha256": file_digest(split_path),
                  "identity_sha256": digest(split), "preserved": True,
                  "active_identity_sha256": digest(active_split),
                  "active_manifest_preview": active_split},
        "source_review": {"path": str(Path(source_review_path).resolve()), "sha256": source_sha,
                          "source_id": required_source_id, "program_id": required_program_id,
                          "status": "pending",
                          "additional_pending_sources": additional,
                          "additional_exclusions_descriptor": ({"path": str(Path(additional_exclusions_path).resolve()),
                                                                 "sha256": additional_descriptor_sha}
                                                                if additional_exclusions_path else None)},
        "input_gates": {
            "training_audit": {"path": str(data_path.with_name(data_path.name + ".manifest.json").resolve()),
                               "sha256": file_digest(data_path.with_name(data_path.name + ".manifest.json")),
                               "dataset_sha256": training_audit["sha256"], "ready": True},
            "base_mix_audit": {"path": str(Path(mix_path).resolve()), "sha256": file_digest(mix_path),
                               "policy_sha256": mix_identity["mix_policy_sha256"],
                               "target_reducer_share": mix_target,
                               "scope": "unchanged base corpus; original audit remains historical and is not claimed as the post-exclusion stream"},
            "post_exclusion_mix": filtered_mix,
            "inventory_ready": {"path": str(Path(inventory_ready_path).resolve()),
                                "sha256": file_digest(inventory_ready_path),
                                "report_sha256": inventory_identity["report_sha256"],
                                "policy_path": str(Path(inventory_policy_path).resolve()),
                                "policy_sha256": inventory_identity["policy_sha256"],
                                "scope": "unchanged base data inventory; separate exact source-hold exclusion overlays this historical receipt"},
            "source_hold_overlay": {"separate_from_old_ready_and_mix_receipts": True,
                                    "excluded_ids_sha256": digest(excluded),
                                    "source_review_sha256": source_sha},
        },
        "exposure_receipt": {"path": str(Path(exposure_path).resolve()),
                             "sha256": file_digest(exposure_path),
                             "receipt_step": exposure.get("step"),
                             "receipt_cursor": receipt_cursor,
                             "checkpoint_step": state.get("step"),
                             "receipt_consumed_rows": sorted(consumed_receipt),
                             "checkpoint_consumed_rows": list(order.consumed_removed_ids),
                             "checkpoint_future_rows": list(order.future_removed_ids),
                             "scope": "primary source receipt is historical; checkpoint exposure is recomputed from actual current cursor for all held source IDs"},
        "excluded_row_ids": excluded,
        "excluded_row_ids_sha256": digest(excluded),
        "training_order": {
            "previous_ids_sha256": order.previous_order_sha256,
            "filtered_ids_sha256": order.filtered_order_sha256,
            "previous_cursor": order.previous_cursor,
            "cursor": order.cursor,
            "remaining_order_preserved": True,
        },
        "checkpoint": {
            "path": str(checkpoint_dir.resolve()), "state_sha256": state_sha,
            "files": before, "step": state["step"], "cursor": cursor,
            "trained_examples": trained_examples, "skipped": state.get("skipped", 0),
            "optimizer": corpus.get("optimizer"), "scheduler_steps": corpus.get("steps"),
            "parent_corpus_identity": corpus,
            "optimizer_scheduler_rng_weights_preserved": True,
        },
        "target": {
            "previous_examples": corpus["target_examples"], "examples": target,
            "future_removed_rows": len(order.future_removed_ids),
            "steps_horizon_preserved": preserved_scheduler_steps,
            "optimizer_steps_required_at_this_boundary": optimizer_steps_required,
            "scheduler_horizon_steps_after_expected_finish": preserved_scheduler_steps - optimizer_steps_required,
            "adjustment": "subtract only excluded training rows not yet consumed; retain one already-consumed harmless exposure in history",
        },
        "state_update_preview": transition_state(state, exclusion_manifest_sha256=None,
                                          order=order, target_examples=target,
                                          split_sha256=digest(active_split),
                                          parent_state_sha256=state_sha),
        "state_update_manifest_binding": "The applying tool must add the final manifest file SHA-256 to state.corpus.exclusion_manifest_sha256 and state.exclusion_transition.manifest_sha256 after the immutable manifest is written.",
        "application": {
            "required": "copy checkpoint to a new run directory, verify all pinned files, rewrite state.json only, and resume with a future runtime that validates this manifest",
            "active_run_mutated": False,
            "ready_to_apply": False,
            "reason": "manifest is bound to a moving live checkpoint boundary and must be regenerated at an approved graceful stop",
        },
    }
    after = checkpoint_file_hashes(checkpoint_dir)
    if after != before or file_digest(state_path) != state_sha:
        raise ValueError("checkpoint changed while preparing exclusion review; discard and retry from a stable boundary")
    return manifest_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--additional-exclusions", type=Path)
    parser.add_argument("--mix-audit", type=Path, required=True)
    parser.add_argument("--inventory-ready", type=Path, required=True)
    parser.add_argument("--inventory-policy", type=Path, required=True)
    parser.add_argument("--active-plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-len", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = make_transition(data_path=args.data, split_path=args.split,
                            checkpoint_dir=args.checkpoint, exposure_path=args.exposure,
                            source_review_path=args.source_review, required_source_id=args.source_id,
                            required_program_id=args.program_id,
                            additional_exclusions_path=args.additional_exclusions,
                            mix_path=args.mix_audit,
                            inventory_ready_path=args.inventory_ready,
                            inventory_policy_path=args.inventory_policy, active_plan_path=args.active_plan,
                            model=args.model,
                            model_revision=args.model_revision, max_len=args.max_len)
    write_new_json(args.output, value)
    print(json.dumps({"path": str(args.output.resolve()), "sha256": file_digest(args.output),
                      "status": value["status"], "checkpoint_step": value["checkpoint"]["step"],
                      "old_cursor": value["training_order"]["previous_cursor"],
                      "new_cursor": value["training_order"]["cursor"],
                      "consumed_removed": len(value["exposure_receipt"]["checkpoint_consumed_rows"]),
                      "future_removed": len(value["exposure_receipt"]["checkpoint_future_rows"]),
                      "target_examples": value["target"]["examples"]}, sort_keys=True))


if __name__ == "__main__":
    main()
