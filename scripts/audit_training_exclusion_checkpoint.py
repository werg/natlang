#!/usr/bin/env python3
"""Audit a read-only checkpoint copy and hypothetical state-only exclusion transition."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.corpus import digest, file_digest, index_pairs, split_programs
from scripts.training_exclusion import (bind_transition_manifest_sha, exclusion_split_manifest,
                                        exclusion_target, filter_training_order,
                                        post_exclusion_mix_summary, transition_state)


def hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint contains symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_digest(path)
    return result


def audit(source_checkpoint: Path, candidate_path: Path, data_path: Path,
          split_path: Path, mix_path: Path, output: Path) -> dict:
    source_checkpoint = source_checkpoint.resolve(strict=True)
    candidate_path = candidate_path.resolve(strict=True)
    data_path = data_path.resolve(strict=True)
    split_path = split_path.resolve(strict=True)
    mix_path = mix_path.resolve(strict=True)
    output = output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"preserving existing audit output: {output}")
    parent_copy = output / "parent-copy"
    hypothetical = output / "hypothetical-transition"
    output.mkdir(parents=True)
    try:
        before = hashes(source_checkpoint)
        shutil.copytree(source_checkpoint, parent_copy, copy_function=shutil.copy2)
        after = hashes(source_checkpoint)
        copied = hashes(parent_copy)
        if before != after or before != copied:
            raise ValueError("live checkpoint changed during copy or copied bytes differ")
        if any((source_checkpoint / name).stat().st_ino == (parent_copy / name).stat().st_ino
               for name in before):
            raise ValueError("checkpoint copy shares file inodes with the live source")
        parent_state_path = parent_copy / "state.json"
        parent_state = json.loads(parent_state_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        if candidate.get("status") != "review_candidate_not_applied":
            raise ValueError("candidate status is not a review-only candidate")
        if (candidate.get("checkpoint", {}).get("state_sha256") != before.get("state.json") or
                candidate.get("checkpoint", {}).get("step") != parent_state.get("step") or
                candidate.get("checkpoint", {}).get("cursor") != parent_state.get("cursor") or
                candidate.get("checkpoint", {}).get("files") != before):
            raise ValueError("candidate and physical checkpoint copy are from different checkpoint boundaries")
        if file_digest(data_path) != candidate["data"]["sha256"]:
            raise ValueError("candidate and data corpus identities differ")
        if parent_state.get("corpus", {}).get("data_sha256") != file_digest(data_path):
            raise ValueError("checkpoint and source corpus identities differ")
        saved_split = json.loads(split_path.read_text(encoding="utf-8"))
        pairs = index_pairs(data_path)
        held, train, original_split = split_programs(
            pairs, int(parent_state["args"]["holdout"]), int(parent_state["corpus"]["seed"]))
        if original_split != saved_split:
            raise ValueError("source split does not match the parent checkpoint")
        if parent_state["corpus"].get("data_order") == "source":
            train.sort(key=lambda row: row["offset"])
        excluded = candidate["excluded_row_ids"]
        order = filter_training_order(train, excluded, int(parent_state["cursor"]))
        if order.previous_order_sha256 != candidate["training_order"]["previous_ids_sha256"]:
            raise ValueError("saved candidate was prepared against a different original permutation")
        target = exclusion_target(int(parent_state["corpus"]["target_examples"]),
                                  int(parent_state["trained_examples"]), len(order.future_removed_ids))
        active_split = exclusion_split_manifest(original_split, order.rows, excluded)
        mix_target = float(json.loads(mix_path.read_text(encoding="utf-8"))["target_reducer_share"])
        mix = post_exclusion_mix_summary(data_path, train, excluded, mix_target)
        state_sha = file_digest(parent_state_path)
        # This synthetic digest is for a throwaway local state-difference proof only.
        synthetic_binding = hashlib.sha256(("NONAPPROVED-CONTINUITY-AUDIT:" + state_sha).encode()).hexdigest()
        updated = transition_state(parent_state, exclusion_manifest_sha256=None, order=order,
                                   target_examples=target, split_sha256=digest(active_split),
                                   parent_state_sha256=state_sha)
        updated = bind_transition_manifest_sha(updated, synthetic_binding)
        shutil.copytree(parent_copy, hypothetical, copy_function=shutil.copy2)
        temp = hypothetical / "state.json.pending"
        with temp.open("xb") as stream:
            stream.write((json.dumps(updated, sort_keys=True, indent=2) + "\n").encode())
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, hypothetical / "state.json")
        updated_hashes = hashes(hypothetical)
        unchanged_files = {name: value for name, value in before.items() if name != "state.json"}
        if {name: value for name, value in updated_hashes.items() if name != "state.json"} != unchanged_files:
            raise ValueError("hypothetical transition changed non-state checkpoint bytes")
        changed = [key for key in set(parent_state) | set(updated)
                   if parent_state.get(key) != updated.get(key)]
        if set(changed) != {"cursor", "corpus", "exclusion_transition"}:
            raise ValueError(f"unexpected state fields changed: {changed}")
        corpus_changes = [key for key in set(parent_state["corpus"]) | set(updated["corpus"])
                          if parent_state["corpus"].get(key) != updated["corpus"].get(key)]
        if set(corpus_changes) != {"split_sha256", "target_examples", "exclusion_manifest_sha256"}:
            raise ValueError(f"unexpected corpus state fields changed: {corpus_changes}")
        import torch
        optimizer = torch.load(parent_copy / "optimizer.pt", map_location="cpu", weights_only=False)
        scheduler = torch.load(parent_copy / "scheduler.pt", map_location="cpu", weights_only=False)
        rng = torch.load(parent_copy / "rng.pt", map_location="cpu", weights_only=False)
        if (optimizer.get("format") != "natlang.muon-adamw/1" or
                not isinstance(optimizer.get("muon"), dict) or
                len(optimizer["muon"].get("state", {})) != 184 or optimizer.get("adamw") is not None):
            raise ValueError("actual checkpoint does not contain the expected Muon-only slot partition")
        if not isinstance(scheduler.get("last_epoch"), int) or not rng:
            raise ValueError("checkpoint scheduler or RNG state is incomplete")
        proof = {
            "schema": "natlang.training_muon_exclusion_continuity_proof/1",
            "status": "offline_checkpoint_copy_proven_no_training",
            "audit_input_checkpoint_path": str(source_checkpoint),
            "candidate_original_checkpoint_path": candidate["checkpoint"]["path"],
            "copied_checkpoint_path": str(parent_copy),
            "hypothetical_transition_path": str(hypothetical),
            "copy_started_sha256": before["state.json"],
            "copy_finished_live_sha256": after["state.json"],
            "copied_parent_state_sha256": file_digest(parent_state_path),
            "source_copy_bytes_identical_before_after": before == after == copied,
            "physical_copy_no_shared_inodes": True,
            "parent_checkpoint_step": parent_state["step"],
            "parent_checkpoint_cursor": parent_state["cursor"],
            "optimizer": {"format": optimizer["format"], "muon_state_entries": 184,
                          "adamw_partition": "none", "file_sha256": before["optimizer.pt"],
                          "unchanged": True},
            "scheduler": {"last_epoch": scheduler["last_epoch"],
                          "file_sha256": before["scheduler.pt"], "unchanged": True},
            "rng": {"file_sha256": before["rng.pt"], "unchanged": True},
            "weights_sha256": {key: value for key, value in before.items() if key.startswith("weights/")},
            "changed_checkpoint_state_fields_only": changed,
            "changed_corpus_state_fields_only": corpus_changes,
            "source_exclusion_ids": excluded,
            "consumed_exclusion_ids_at_this_checkpoint": list(order.consumed_removed_ids),
            "future_exclusion_ids_at_this_checkpoint": list(order.future_removed_ids),
            "cursor_before": order.previous_cursor,
            "cursor_after": order.cursor,
            "exact_remaining_order_sha256": order.filtered_order_sha256,
            "split_heldout_assignment_preserved": active_split["held_programs"] == original_split["held_programs"],
            "target_examples_before": parent_state["corpus"]["target_examples"],
            "target_examples_after": target,
            "scheduler_steps_preserved": parent_state["corpus"]["steps"],
            "mix_after_exclusion": mix,
            "synthetic_binding": "nonapproved audit-only digest; do not pass to trainer",
            "gpu_training_provider_calls": False,
        }
        return proof
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--mix-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--proof", type=Path, required=True)
    a = p.parse_args()
    proof = audit(a.checkpoint, a.candidate, a.data, a.split, a.mix_audit, a.output)
    a.proof.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(proof, sort_keys=True, indent=2) + "\n").encode()
    with a.proof.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": proof["status"], "step": proof["parent_checkpoint_step"],
                      "muon_state_entries": proof["optimizer"]["muon_state_entries"],
                      "proof": str(a.proof.resolve()),
                      "proof_sha256": file_digest(a.proof)}, sort_keys=True))

if __name__ == "__main__":
    main()
