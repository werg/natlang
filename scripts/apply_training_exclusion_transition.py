#!/usr/bin/env python3
"""Stage an approved exclusion transition into a new, separate training run.

Never edits the source corpus, source checkpoint, active run, or service. The
caller must provide a root-generated quiescence receipt for the exact checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.corpus import digest, file_digest, index_pairs, split_programs
from scripts.training_exclusion import (bind_transition_manifest_sha, exclusion_split_manifest,
                                        filter_training_order, pending_source_review_entry,
                                        post_exclusion_mix_summary)
from scripts.training_readiness import (validate_training_audit,
                                        validate_training_inventory_audit,
                                        validate_training_mix_audit)


def reject_overlapping_run_paths(target_run: Path, source_run: Path) -> None:
    """Prevent any temporary output or cleanup path from overlapping active history."""
    target = target_run.resolve(strict=False)
    source = source_run.resolve(strict=True)
    if target == source or target in source.parents or source in target.parents:
        raise ValueError("new target run must be disjoint from the source run")


def checkpoint_hashes(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source checkpoint has a symlink: {path}")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = file_digest(path)
    return result


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def apply_transition(*, manifest_path: Path, manifest_sha256: str,
                     quiescence_path: Path, target_run: Path) -> dict:
    manifest_path, quiescence_path, target_run = map(Path, (manifest_path, quiescence_path, target_run))
    if not manifest_path.is_absolute() or manifest_path.resolve(strict=True) != manifest_path:
        raise ValueError("manifest path must be absolute and canonical")
    actual_manifest_sha = file_digest(manifest_path)
    if actual_manifest_sha != manifest_sha256:
        raise ValueError("approved exclusion manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "natlang.training_source_exclusion/1" or manifest.get("status") != "approved":
        raise ValueError("source-exclusion manifest is not root-approved")

    quiescence = json.loads(quiescence_path.read_text(encoding="utf-8"))
    if (quiescence.get("schema") != "natlang.training_quiescence/1" or
            quiescence.get("state") not in {"stopped", "paused"}):
        raise ValueError("root quiescence receipt is missing or not stopped/paused")
    if quiescence.get("active_plan_sha256") != manifest.get("active_plan", {}).get("sha256"):
        raise ValueError("quiescence receipt is for a different active training plan")
    active_plan_path = Path(manifest["active_plan"]["path"])
    if file_digest(active_plan_path) != manifest["active_plan"]["sha256"]:
        raise ValueError("active training plan changed since candidate preparation")

    checkpoint = manifest["checkpoint"]
    source_checkpoint = Path(checkpoint["path"])
    if not source_checkpoint.is_absolute() or source_checkpoint.resolve(strict=True) != source_checkpoint:
        raise ValueError("source checkpoint path must be absolute and canonical")
    if quiescence.get("checkpoint_state_sha256") != checkpoint.get("state_sha256"):
        raise ValueError("quiescence checkpoint differs from the approved transition boundary")
    if checkpoint_hashes(source_checkpoint) != checkpoint.get("files"):
        raise ValueError("source checkpoint artifacts changed since the candidate was reviewed")

    gates = manifest["input_gates"]
    data = Path(manifest["data"]["path"])
    if file_digest(data) != manifest["data"]["sha256"]:
        raise ValueError("base corpus changed since exclusion review")
    audit_path = Path(gates["training_audit"]["path"])
    if file_digest(audit_path) != gates["training_audit"]["sha256"]:
        raise ValueError("base token-audit manifest changed")
    corpus = json.loads((source_checkpoint / "state.json").read_text()).get("corpus", {})
    parent_state = json.loads((source_checkpoint / "state.json").read_text())
    if corpus != checkpoint.get("parent_corpus_identity"):
        raise ValueError("checkpoint corpus identity differs from the approved parent identity")
    split_path = Path(manifest["split"]["path"])
    if file_digest(split_path) != manifest["split"]["sha256"]:
        raise ValueError("original split manifest changed since exclusion review")
    saved_split = json.loads(split_path.read_text(encoding="utf-8"))
    pairs = index_pairs(data)
    held_rows, train_rows, actual_split = split_programs(
        pairs, int(parent_state["args"]["holdout"]), int(corpus["seed"]))
    if actual_split != saved_split or digest(actual_split) != manifest["split"]["identity_sha256"]:
        raise ValueError("original split no longer matches the reviewed base split")
    if corpus.get("data_order") == "source":
        train_rows.sort(key=lambda row: row["offset"])
    elif corpus.get("data_order") != "shuffle":
        raise ValueError("unsupported data order in parent corpus")
    excluded_ids = manifest["excluded_row_ids"]
    order = filter_training_order(train_rows, excluded_ids, int(checkpoint["cursor"]))
    if order.filtered_order_sha256 != manifest["training_order"]["filtered_ids_sha256"]:
        raise ValueError("filtered row order differs from the reviewed transition")
    active_split = exclusion_split_manifest(actual_split, order.rows, excluded_ids)
    if active_split != manifest["split"]["active_manifest_preview"]:
        raise ValueError("active split assignment differs from the reviewed transition")
    filtered_mix = post_exclusion_mix_summary(data, train_rows, excluded_ids,
                                              float(manifest["input_gates"]["post_exclusion_mix"]["target_reducer_share"]))
    if filtered_mix != manifest["input_gates"]["post_exclusion_mix"]:
        raise ValueError("post-exclusion training mix differs from the reviewed transition")
    audit = validate_training_audit(data, int(corpus["max_len"]), corpus["model"], corpus["model_revision"])
    mix_path = Path(gates["base_mix_audit"]["path"])
    if file_digest(mix_path) != gates["base_mix_audit"]["sha256"]:
        raise ValueError("base reducer-mix receipt changed")
    mix_identity = validate_training_mix_audit(mix_path, data,
                                              float(gates["base_mix_audit"]["target_reducer_share"]), audit)
    ready = Path(gates["inventory_ready"]["path"])
    policy = Path(gates["inventory_ready"]["policy_path"])
    if (file_digest(ready) != gates["inventory_ready"]["sha256"] or
            file_digest(policy) != gates["inventory_ready"]["policy_sha256"]):
        raise ValueError("base inventory receipt or policy changed")
    inventory_identity = validate_training_inventory_audit(ready, policy)
    if corpus.get("joint_gate_identity") != {**mix_identity, **inventory_identity}:
        raise ValueError("base training gates no longer match the parent checkpoint")

    review = manifest["source_review"]
    review_path = Path(review["path"])
    if file_digest(review_path) != review["sha256"]:
        raise ValueError("source-review hold changed since exclusion review")
    review_text = review_path.read_text(encoding="utf-8")
    if not pending_source_review_entry(review_text, str(review.get("source_id"))):
        raise ValueError("exact source review is no longer pending")
    additional_descriptor = review.get("additional_exclusions_descriptor")
    additional_sources = review.get("additional_pending_sources", [])
    if additional_sources:
        if not isinstance(additional_descriptor, dict):
            raise ValueError("additional source holds lack a pinned descriptor")
        descriptor_path = Path(additional_descriptor["path"])
        if file_digest(descriptor_path) != additional_descriptor["sha256"]:
            raise ValueError("additional exclusion descriptor changed")
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        descriptor_rows = {str(item.get("source_id")): item for item in descriptor.get("rows", [])}
        if descriptor.get("schema") != "natlang.training_source_exclusion_additional/1":
            raise ValueError("additional exclusion descriptor schema changed")
        excluded_set = set(map(str, manifest.get("excluded_row_ids", [])))
        if set(descriptor_rows) != {str(item.get("source_id")) for item in additional_sources}:
            raise ValueError("additional source holds differ from the pinned descriptor")
        for item in additional_sources:
            source_id = str(item.get("source_id"))
            described = descriptor_rows.get(source_id, {})
            if (described.get("row_id") != item.get("row_id") or
                    described.get("source_snapshot_sha256") != item.get("source_snapshot_sha256") or
                    described.get("source_program_id") != item.get("source_program_id") or
                    item.get("row_id") not in excluded_set):
                raise ValueError(f"additional exclusion row pin changed: {source_id}")
            if not pending_source_review_entry(review_text, source_id):
                raise ValueError(f"additional source review is no longer pending: {source_id}")
    if manifest["input_gates"]["post_exclusion_mix"].get("target_met") is not True:
        raise ValueError("post-exclusion training mix does not meet its target")

    target_run = target_run.resolve(strict=False)
    if not target_run.is_absolute() or target_run.exists() or not target_run.parent.is_dir():
        raise ValueError("new target run must be an unused path under an existing directory")
    source_run = Path(manifest["active_plan"]["path"]).parent.resolve(strict=True)
    reject_overlapping_run_paths(target_run, source_run)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target_run.name}.pending-", dir=target_run.parent))
    try:
        target_checkpoint = temporary / "trainer-output" / "checkpoint"
        target_checkpoint.mkdir(parents=True)
        for relative in checkpoint["files"]:
            if relative == "state.json":
                continue
            source = source_checkpoint / relative
            destination = target_checkpoint / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        update = bind_transition_manifest_sha(manifest["state_update_preview"], actual_manifest_sha)
        if update.get("corpus", {}).get("split_sha256") != digest(active_split):
            raise ValueError("state update split hash differs from the exact filtered split")
        update.setdefault("args", {})["out"] = str((target_run / "trainer-output").resolve())
        atomic_json(target_checkpoint / "state.json", update)
        atomic_json(temporary / "split.json", manifest["split"]["active_manifest_preview"])

        expected = {name: sha for name, sha in checkpoint["files"].items() if name != "state.json"}
        copied = checkpoint_hashes(target_checkpoint)
        copied.pop("state.json", None)
        if copied != expected:
            raise ValueError("copied weights/optimizer/scheduler/RNG bytes differ from source checkpoint")
        if checkpoint_hashes(source_checkpoint) != checkpoint["files"]:
            raise ValueError("source checkpoint changed during transition staging")
        application = {
            "schema": "natlang.training_source_exclusion_application/1",
            "status": "staged_for_runtime_review",
            "manifest_path": str(manifest_path), "manifest_sha256": actual_manifest_sha,
            "quiescence_receipt_path": str(quiescence_path.resolve()),
            "quiescence_receipt_sha256": file_digest(quiescence_path),
            "source_checkpoint": str(source_checkpoint),
            "source_state_sha256": checkpoint["state_sha256"],
            "target_run": str(target_run),
            "target_state_sha256": file_digest(target_checkpoint / "state.json"),
            "unchanged_checkpoint_files": copied,
            "original_checkpoint_mutated": False,
            "runtime_plan_review_required": True,
            "provider_or_gpu_used": False,
        }
        atomic_json(temporary / "source-exclusion-application.json", application)
        source_fd = os.open(target_checkpoint, os.O_RDONLY)
        os.fsync(source_fd)
        os.close(source_fd)
        temporary_fd = os.open(temporary, os.O_RDONLY)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        os.rename(temporary, target_run)
        parent_fd = os.open(target_run.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return application


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--quiescence-receipt", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    args = parser.parse_args()
    result = apply_transition(manifest_path=args.manifest, manifest_sha256=args.sha256,
                              quiescence_path=args.quiescence_receipt, target_run=args.target_run)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
