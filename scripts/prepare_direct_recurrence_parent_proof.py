#!/usr/bin/env python3
"""Create a read-only provenance packet for one stable recurrence checkpoint.

The checkpoint is never copied or rewritten. The packet is written only after
the stage plan, launch intent, executed launch receipt, recipe, parent manifest,
and checkpoint identity have all been checked against their pinned hashes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training" / "neuralese"))

from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402
from natlang_neuralese.train.direct_export import (  # noqa: E402
    RECURRENCE_PARENT_PROOF_SCHEMA,
    direct_parent_binding,
    verify_recurrence_parent_proof,
)


def stable_file_binding(path: Path) -> dict:
    """Hash one file and reject replacement or in-place modification during the read."""
    path = path.resolve(strict=True)
    before = path.stat()
    digest = sha256_file_hex(path)
    after = path.stat()
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if signature(before) != signature(after):
        raise ValueError(f"file changed while hashing: {path}")
    return {"path": str(path), "sha256": digest}


def pinned_file(path: str, expected_sha: str, role: str) -> dict:
    row = stable_file_binding(Path(path))
    if row["sha256"] != expected_sha:
        raise ValueError(f"{role} hash differs from its declared pin: {row['path']}")
    return row


def build_packet(*, stage_plan_path: Path, launch_intent_path: Path,
                 launch_receipt_path: Path, recipe_path: Path,
                 checkpoint_path: Path, checkpoint_provenance_path: Path,
                 parent_artifact: Path | None,
                 expected_checkpoint_sha256: str) -> dict:
    stage_plan_path = stage_plan_path.resolve(strict=True)
    launch_intent_path = launch_intent_path.resolve(strict=True)
    launch_receipt_path = launch_receipt_path.resolve(strict=True)
    recipe_path = recipe_path.resolve(strict=True)
    checkpoint_path = checkpoint_path.resolve(strict=True)
    checkpoint_provenance_path = Path(os.path.normpath(str(checkpoint_provenance_path)))

    plan_binding = stable_file_binding(stage_plan_path)
    plan = json.loads(stage_plan_path.read_text())
    intent_binding = stable_file_binding(launch_intent_path)
    intent = json.loads(launch_intent_path.read_text())
    launch_binding = stable_file_binding(launch_receipt_path)
    recipe_binding = stable_file_binding(recipe_path)

    if plan.get("schema") != "natlang.neuralese-direct-stage-plan/1":
        raise ValueError("unsupported direct-stage plan schema")
    recipe_pin = plan.get("recipe_sha256")
    pinned_file(str(recipe_path), recipe_pin, "recipe")
    if os.path.normpath(str(plan.get("recipe_path", ""))) != os.path.normpath(str(recipe_path)):
        raise ValueError("recipe path differs from the exact path declared by the stage plan")
    if (intent.get("schema") != "natlang.neuralese-direct-stage-launch-intent/1" or
            os.path.normpath(str(intent.get("plan_path", ""))) != os.path.normpath(str(stage_plan_path)) or
            intent.get("plan_sha256") != plan_binding["sha256"]):
        raise ValueError("launch intent does not bind the exact stage plan")

    for name, row in (plan.get("inputs") or {}).items():
        if name == "direct_artifact_manifest":
            continue  # verified below through the explicitly resolved parent artifact
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise ValueError("stage plan has a malformed pinned input: " + str(name))
        pinned_file(row["path"], row["sha256"], "stage input " + str(name))

    parent_input = (plan.get("inputs") or {}).get("direct_artifact_manifest") or {}
    if not isinstance(parent_input.get("path"), str) or not parent_input.get("sha256"):
        raise ValueError("stage plan lacks its pinned direct artifact manifest input")
    historical_parent_manifest = Path(parent_input["path"])
    current_parent_artifact = parent_artifact or historical_parent_manifest.parent
    parent_binding = direct_parent_binding(current_parent_artifact)
    if parent_binding["manifest_file_sha256"] != parent_input["sha256"]:
        raise ValueError("current direct artifact does not match the stage plan's parent manifest content")
    # Preserve the original stage-plan path in the proof packet. The exporter
    # resolves current storage from its explicit parent artifact argument.
    parent_ref = {"path": str(historical_parent_manifest), "sha256": parent_input["sha256"]}

    checkpoint_binding = stable_file_binding(checkpoint_path)
    checkpoint_sha = checkpoint_binding["sha256"]
    if checkpoint_sha != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA differs from the operator-supplied stable-checkpoint pin")
    stat_before = checkpoint_path.stat()
    checkpoint_state = __import__("torch").load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    stat_after = checkpoint_path.stat()
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if signature(stat_before) != signature(stat_after):
        raise ValueError("checkpoint changed while reading its identity")
    checkpoint_identity = checkpoint_state.get("identity") or {}
    if checkpoint_state.get("schema") != "natlang.neuralese_recurrence_checkpoint/1":
        raise ValueError("selected file is not a native recurrence checkpoint")

    proof = {
        "schema": RECURRENCE_PARENT_PROOF_SCHEMA,
        "checkpoint": {"path": str(checkpoint_provenance_path), "sha256": checkpoint_sha},
        "stage_plan": plan_binding,
        "launch_intent": intent_binding,
        "launch_receipt": launch_binding,
        "recipe": recipe_binding,
        "parent_manifest": parent_ref,
    }
    verify_recurrence_parent_proof(
        proof,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        current_parent=parent_binding,
        checkpoint_identity=checkpoint_identity,
    )
    # A second content hash catches same-size writes even if filesystem timestamp
    # granularity or a writer's replacement behavior makes the stat check weak.
    if sha256_file_hex(checkpoint_path) != checkpoint_sha:
        raise ValueError("checkpoint content changed while constructing the proof packet")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-plan", type=Path, required=True)
    parser.add_argument("--launch-intent", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True,
                        help="executed launch.json, not prepared-launch metadata")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="explicit stable checkpoint path; never copied or modified")
    parser.add_argument("--expected-checkpoint-sha256", required=True,
                        help="independently recorded SHA-256 of the selected stable checkpoint")
    parser.add_argument("--checkpoint-provenance-path", type=Path,
                        help="historical stage path to record if the checkpoint was relocated")
    parser.add_argument("--parent-artifact", type=Path,
                        help="current parent artifact location when storage was relocated")
    parser.add_argument("--out", type=Path, required=True,
                        help="new proof-packet path; existing files are never overwritten")
    args = parser.parse_args()
    provenance_path = args.checkpoint_provenance_path or args.checkpoint
    proof = build_packet(
        stage_plan_path=args.stage_plan,
        launch_intent_path=args.launch_intent,
        launch_receipt_path=args.launch_receipt,
        recipe_path=args.recipe,
        checkpoint_path=args.checkpoint,
        checkpoint_provenance_path=provenance_path,
        parent_artifact=args.parent_artifact,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with out.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"proof_path": str(out), "proof_sha256": sha256_file_hex(out),
                      "checkpoint_sha256": proof["checkpoint"]["sha256"],
                      "parent_manifest_sha256": proof["parent_manifest"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
