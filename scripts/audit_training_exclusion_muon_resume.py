#!/usr/bin/env python3
"""CPU-only native Muon save/reload proof for exclusion-aware continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import torch
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha  # noqa: E402


def tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha(p)
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "state.json"}


def save_checkpoint(root: Path, model, optimizer, scheduler, state: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), root / "weights.pt")
    torch.save(optimizer.state_dict(), root / "optimizer.pt")
    torch.save(scheduler.state_dict(), root / "scheduler.pt")
    torch.save(torch.get_rng_state(), root / "rng.pt")
    with (root / "state.json").open("w", encoding="utf-8") as stream:
        json.dump(state, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def build():
    from scripts.training_optimizers import MuonWithAdamW
    model = torch.nn.Linear(3, 3, bias=True, device="cpu")
    optimizer = MuonWithAdamW(model.named_parameters(), lr=0.025, momentum=0.9, ns_steps=3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    return model, optimizer, scheduler


def train_step(model, optimizer, scheduler, row_index: int) -> None:
    values = torch.tensor([row_index / 10, (row_index + 1) / 7, (row_index - 2) / 5], dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    loss = model(values).square().mean() + model(values.roll(1)).abs().mean() * 0.1
    loss.backward()
    optimizer.step()
    scheduler.step()


def state_files(directory: Path) -> dict[str, str]:
    return tree_hashes(directory)


def run(runtime_root: Path, output: Path) -> dict:
    sys.path.insert(0, str(runtime_root.resolve(strict=True)))
    from scripts.training_exclusion import (bind_transition_manifest_sha,
                                             filter_training_order,
                                             transition_state,
                                             validate_exclusion_checkpoint_resume)
    torch.set_num_threads(1)
    torch.manual_seed(812)
    random.seed(812)
    root = output.resolve(strict=False)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    stage = root / "transition-stage"
    branch = root / "uninterrupted"
    saved = root / "after-one-resumed-step"
    resumed = root / "resumed"

    parent_model, parent_optimizer, parent_scheduler = build()
    train_step(parent_model, parent_optimizer, parent_scheduler, 0)
    parent_state = {"step": 1, "cursor": 8, "trained_examples": 8, "skipped": 0,
                    "heldout_before": 1.234,
                    "corpus": {"target_examples": 32, "steps": 8, "optimizer": "muon",
                               "muon": {"momentum": 0.9, "ns_steps": 3,
                                        "adjust_lr_fn": "match_rms_adamw", "partition_version": 1}},
                    "args": {"holdout": 1, "accum": 8}}
    save_checkpoint(stage, parent_model, parent_optimizer, parent_scheduler, parent_state)
    parent_files = {**state_files(stage), "state.json": sha(stage / "state.json")}
    parent_state_sha = parent_files["state.json"]

    ordered = [{"id": f"r{i}"} for i in range(32)]
    order = filter_training_order(ordered, ["r0", "r30"], cursor=8)
    manifest_sha = "a" * 64
    transition = transition_state(parent_state, exclusion_manifest_sha256=None, order=order,
                                  target_examples=31, split_sha256="b" * 64,
                                  parent_state_sha256=parent_state_sha)
    transition = bind_transition_manifest_sha(transition, manifest_sha)
    manifest = {"checkpoint": {"files": parent_files, "state_sha256": parent_state_sha,
                                "step": parent_state["step"], "cursor": parent_state["cursor"],
                                "trained_examples": parent_state["trained_examples"]}}
    (stage / "state.json").write_text(json.dumps(transition, sort_keys=True) + "\n")
    transition_file_hashes = state_files(stage)
    if transition_file_hashes != {key: value for key, value in parent_files.items() if key != "state.json"}:
        raise AssertionError("staging changed Muon/model/scheduler/RNG checkpoint bytes")
    if not validate_exclusion_checkpoint_resume(
            manifest, transition, transition_file_hashes, accumulation=8,
            manifest_sha256=manifest_sha):
        raise AssertionError("exact transition boundary did not validate")

    # The uninterrupted branch takes two clean steps after the transition.
    torch.manual_seed(812)
    reference_model, reference_optimizer, reference_scheduler = build()
    reference_model.load_state_dict(torch.load(stage / "weights.pt", map_location="cpu", weights_only=True))
    reference_optimizer.load_state_dict(torch.load(stage / "optimizer.pt", map_location="cpu", weights_only=False))
    reference_scheduler.load_state_dict(torch.load(stage / "scheduler.pt", map_location="cpu", weights_only=False))
    reference_state = json.loads((stage / "state.json").read_text())
    for index in range(7, 15):
        train_step(reference_model, reference_optimizer, reference_scheduler, index)
    reference_state.update(step=2, cursor=15, trained_examples=16)
    save_checkpoint(branch, reference_model, reference_optimizer, reference_scheduler, reference_state)

    # A second process starts from that saved post-transition checkpoint.
    torch.manual_seed(999)
    interrupted_model, interrupted_optimizer, interrupted_scheduler = build()
    interrupted_model.load_state_dict(torch.load(stage / "weights.pt", map_location="cpu", weights_only=True))
    interrupted_optimizer.load_state_dict(torch.load(stage / "optimizer.pt", map_location="cpu", weights_only=False))
    interrupted_scheduler.load_state_dict(torch.load(stage / "scheduler.pt", map_location="cpu", weights_only=False))
    interrupted_state = json.loads((stage / "state.json").read_text())
    torch.set_rng_state(torch.load(stage / "rng.pt", map_location="cpu", weights_only=True))
    for index in range(7, 15):
        train_step(interrupted_model, interrupted_optimizer, interrupted_scheduler, index)
    interrupted_state.update(step=2, cursor=15, trained_examples=16)
    save_checkpoint(saved, interrupted_model, interrupted_optimizer, interrupted_scheduler, interrupted_state)
    after_first_resume_hashes = state_files(saved)
    if validate_exclusion_checkpoint_resume(
            manifest, interrupted_state, after_first_resume_hashes, accumulation=8,
            manifest_sha256=manifest_sha):
        raise AssertionError("later checkpoint was misclassified as untouched transition boundary")

    torch.manual_seed(31415)
    resumed_model, resumed_optimizer, resumed_scheduler = build()
    resumed_model.load_state_dict(torch.load(saved / "weights.pt", map_location="cpu", weights_only=True))
    resumed_optimizer.load_state_dict(torch.load(saved / "optimizer.pt", map_location="cpu", weights_only=False))
    resumed_scheduler.load_state_dict(torch.load(saved / "scheduler.pt", map_location="cpu", weights_only=False))
    resumed_state = json.loads((saved / "state.json").read_text())
    torch.set_rng_state(torch.load(saved / "rng.pt", map_location="cpu", weights_only=True))
    if validate_exclusion_checkpoint_resume(
            manifest, resumed_state, after_first_resume_hashes, accumulation=8,
            manifest_sha256=manifest_sha):
        raise AssertionError("second resume unexpectedly required original parent file hashes")

    for index in range(15, 23):
        train_step(reference_model, reference_optimizer, reference_scheduler, index)
        train_step(resumed_model, resumed_optimizer, resumed_scheduler, index)
    reference_state.update(step=3, cursor=23, trained_examples=24)
    resumed_state.update(step=3, cursor=23, trained_examples=24)
    save_checkpoint(branch, reference_model, reference_optimizer, reference_scheduler, reference_state)
    save_checkpoint(resumed, resumed_model, resumed_optimizer, resumed_scheduler, resumed_state)
    resumed_hashes = state_files(resumed)
    validate_exclusion_checkpoint_resume(manifest, resumed_state, resumed_hashes,
                                         accumulation=8, manifest_sha256=manifest_sha)
    for expected, actual in zip(reference_model.parameters(), resumed_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for filename in ("optimizer.pt", "scheduler.pt", "rng.pt", "weights.pt"):
        if sha(branch / filename) != sha(resumed / filename):
            raise AssertionError(f"resumed continuation diverged in {filename}")

    proof = {"schema": "natlang.training_exclusion_muon_cpu_resume_proof/1",
             "status": "passed_native_muon_save_reload_continue",
             "runtime_root": str(runtime_root.resolve()),
             "torch_version": torch.__version__, "device": "cpu",
             "optimizer_format": torch.load(resumed / "optimizer.pt", map_location="cpu",
                                            weights_only=False).get("format"),
             "transition_manifest_sha256": manifest_sha,
             "transition_boundary_files_preserved": True,
             "first_post_transition_checkpoint_validated": True,
             "second_resume_validated_without_parent_hash_equality": True,
             "uninterrupted_and_reloaded_continuations_bitwise_equal": True,
             "final_step": resumed_state["step"], "final_cursor": resumed_state["cursor"],
             "final_trained_examples": resumed_state["trained_examples"],
             "parent_muon_state_entries": len(torch.load(stage / "optimizer.pt", map_location="cpu",
                                                         weights_only=False)["muon"]["state"]),
             "gpu_or_provider_used": False}
    output_file = root / "proof.json"
    with output_file.open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    proof["proof_path"] = str(output_file)
    proof["proof_sha256"] = sha(output_file)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.runtime_root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
