#!/usr/bin/env python3
"""Prepare (but never activate) a pinned resume plan after a reviewed exclusion is staged."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.corpus import file_digest


def atomic_create(path: Path, value: dict) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"preserving existing plan artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(fd)
    finally: os.close(fd)
    return hashlib.sha256(payload).hexdigest()


def rewrite_training_argv(argv: list[str], *, old_run: Path, target_run: Path,
                         runtime_root: Path) -> list[str]:
    """Rebase output paths and substitute the exact sealed inventory mount."""
    rewritten = list(argv)
    old_policy_mount = (str(old_run / "frozen-inventory-policy/data_sources.json") +
                        ":/home/werg/natlang/training/data_sources.json:ro")
    runtime_policy = str(runtime_root / "inventory-policy/data_sources.json")
    old_trainer = None
    for i, arg in enumerate(rewritten):
        if arg == old_policy_mount:
            rewritten[i] = runtime_policy + ":/home/werg/natlang/training/data_sources.json:ro"
        elif arg.endswith("/scripts/train_lora.py"):
            old_trainer = arg
            rewritten[i] = str(runtime_root / "code-snapshot/scripts/train_lora.py")
        elif str(old_run) in arg:
            rewritten[i] = arg.replace(str(old_run), str(target_run))
    if old_trainer is None:
        raise ValueError("base training command has no frozen trainer entrypoint")
    expected = runtime_policy + ":/home/werg/natlang/training/data_sources.json:ro"
    if [value for value in rewritten if "/home/werg/natlang/training/data_sources.json:ro" in value] != [expected]:
        raise ValueError("new training command does not mount exactly the sealed inventory policy")
    return rewritten


def prepare(*, base_plan_path: Path, manifest_path: Path, manifest_sha256: str,
            application_path: Path, quiescence_path: Path, runtime_manifest_path: Path,
            supervisor_path: Path, supervisor_sha256: str,
            output_plan: Path, service_draft: Path) -> dict:
    paths = [base_plan_path, manifest_path, application_path, quiescence_path,
             runtime_manifest_path, supervisor_path]
    for path in paths:
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise ValueError(f"input path must be absolute/canonical: {path}")
    if file_digest(manifest_path) != manifest_sha256:
        raise ValueError("approved exclusion manifest SHA mismatch")
    supervisor_module_name = "_sealed_training_supervisor_for_plan_preflight"
    spec = importlib.util.spec_from_file_location(supervisor_module_name, supervisor_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the exact pinned supervisor for plan preflight")
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "natlang.training_source_exclusion/1" or manifest.get("status") != "approved":
        raise ValueError("source exclusion does not have root approval")
    base, base_sha = supervisor.load_plan(base_plan_path)
    if manifest.get("active_plan") != {"path": str(base_plan_path), "sha256": base_sha}:
        raise ValueError("approved exclusion is not bound to this base training plan")
    application = json.loads(application_path.read_text())
    if (application.get("schema") != "natlang.training_source_exclusion_application/1" or
            application.get("status") != "staged_for_runtime_review" or
            application.get("manifest_sha256") != manifest_sha256 or
            application.get("runtime_plan_review_required") is not True):
        raise ValueError("staged checkpoint application receipt is invalid")
    quiescence = json.loads(quiescence_path.read_text())
    if (quiescence.get("schema") != "natlang.training_quiescence/1" or
            quiescence.get("state") not in {"stopped", "paused"} or
            quiescence.get("active_plan_sha256") != base_sha):
        raise ValueError("root quiescence receipt is invalid")
    if (application.get("quiescence_receipt_path") != str(quiescence_path) or
            application.get("quiescence_receipt_sha256") != file_digest(quiescence_path) or
            application.get("source_checkpoint") != manifest.get("checkpoint", {}).get("path") or
            application.get("source_state_sha256") != manifest.get("checkpoint", {}).get("state_sha256")):
        raise ValueError("application receipt does not bind this quiescence and source checkpoint")
    runtime_manifest = json.loads(runtime_manifest_path.read_text())
    if (runtime_manifest.get("schema") != "natlang.training_exclusion_runtime_candidate/1" or
            runtime_manifest.get("status") != "prepared_not_approved_or_active"):
        raise ValueError("candidate runtime is not an inactive reviewed package")
    runtime_root = Path(runtime_manifest["candidate_path"]).resolve(strict=True)
    sealed_supervisor = runtime_root / runtime_manifest.get("supervisor_relative_path", "")
    if (not sealed_supervisor.is_file() or
            supervisor_path != sealed_supervisor.resolve(strict=True) or
            supervisor_sha256 != runtime_manifest.get("supervisor_sha256") or
            file_digest(supervisor_path) != supervisor_sha256):
        raise ValueError("proposed supervisor differs from the root-reviewed runtime supervisor pin")
    if file_digest(runtime_root / "source-review/source-review.ts") != manifest["source_review"]["sha256"]:
        raise ValueError("runtime source-review TypeScript differs from approved source review")
    if Path(manifest["source_review"]["path"]).resolve(strict=True) != (runtime_root / "source-review/source-review.ts").resolve(strict=True):
        raise ValueError("approved transition points to a mutable source-review path outside its sealed runtime")
    compiled_policy = runtime_root / "source-review/dist/teacher/source-review.js"
    if file_digest(compiled_policy) != runtime_manifest.get("source_review_compiled_sha256"):
        raise ValueError("runtime compiled source-review differs from its sealed manifest")
    compiled_text = compiled_policy.read_text(encoding="utf-8")
    for source_id in [manifest["source_review"]["source_id"],
                      *[str(item["source_id"]) for item in manifest["source_review"].get("additional_pending_sources", [])]]:
        if source_id not in compiled_text:
            raise ValueError(f"compiled source-review does not include held ID {source_id}")
    for name, pin in manifest.get("procedure_code", {}).get("files", {}).items():
        if (Path(pin["path"]).resolve(strict=True) != (runtime_root / "controller" / name).resolve(strict=True) or
                file_digest(runtime_root / "controller" / name) != pin["sha256"]):
            raise ValueError(f"approved transition procedure code is not sealed in the candidate runtime: {name}")
    if file_digest(runtime_root / "inventory-policy/data_sources.json") != manifest["input_gates"]["inventory_ready"]["policy_sha256"]:
        raise ValueError("runtime inventory policy differs from reviewed base policy")

    target_run = Path(application["target_run"]).resolve(strict=True)
    checkpoint_state = target_run / "trainer-output/checkpoint/state.json"
    if file_digest(checkpoint_state) != application.get("target_state_sha256"):
        raise ValueError("staged checkpoint state changed after application receipt")
    state = json.loads(checkpoint_state.read_text())
    if state.get("exclusion_transition", {}).get("manifest_sha256") != manifest_sha256:
        raise ValueError("staged checkpoint is not bound to the approved manifest")
    if state.get("corpus", {}).get("exclusion_manifest_sha256") != manifest_sha256:
        raise ValueError("staged checkpoint corpus identity has a different exclusion manifest")
    if state.get("corpus", {}).get("target_examples") != manifest["target"]["examples"]:
        raise ValueError("staged checkpoint target differs from approved manifest")

    plan = copy.deepcopy(base)
    old_run = Path(base["run_dir"])
    runtime_policy = str(runtime_root / "inventory-policy/data_sources.json")
    argv = rewrite_training_argv(plan["argv"], old_run=old_run,
                                 target_run=target_run, runtime_root=runtime_root)
    name_idx = argv.index("--name") + 1
    argv[name_idx] = f"natlang-lfm25-muon-source-exclusion-{manifest_sha256[:12]}"
    triton_idx = next((i for i, value in enumerate(argv) if value.startswith("TRITON_CACHE_DIR=")), None)
    if triton_idx is None:
        raise ValueError("base command lacks a pinned Triton cache path")
    argv[triton_idx] = f"TRITON_CACHE_DIR={target_run}/triton-cache"
    if "--exclusion-manifest" in argv:
        raise ValueError("base command unexpectedly already has an exclusion manifest")
    argv.extend(["--exclusion-manifest", str(manifest_path)])
    plan["argv"] = argv
    plan["phase"] = "reviewed-source-exclusion-resume"
    plan["run_dir"] = str(target_run)
    plan["checkpoint_state"] = str(checkpoint_state)
    plan["stdout_log"] = str(target_run / "supervisor.stdout.log")
    plan["status_file"] = str(target_run / "supervisor-status.json")
    plan["lock_file"] = str(target_run / "supervisor.lock")
    plan["operator_stop_file"] = str(target_run / "operator-stop")
    plan["checkpoint_identity"] = state["corpus"]
    plan["completion"] = {"trained_examples": int(manifest["target"]["examples"]),
                          "final_evaluation_required": True}
    plan["artifact_sha256"] = dict(plan["artifact_sha256"])
    for path in (base_plan_path, manifest_path, application_path, quiescence_path,
                 runtime_manifest_path, supervisor_path, checkpoint_state,
                 Path(manifest["source_review"]["path"]),
                 Path(manifest["data"]["path"]), Path(manifest["split"]["path"]),
                 Path(manifest["input_gates"]["base_mix_audit"]["path"]),
                 Path(manifest["input_gates"]["inventory_ready"]["path"]),
                 Path(manifest["input_gates"]["inventory_ready"]["policy_path"])):
        if path == checkpoint_state:
            # The checkpoint mutates during training; only the applied transition receipt/corpus identity pins it.
            continue
        plan["artifact_sha256"][str(path)] = file_digest(path)
    plan["artifact_sha256"].pop(str(supervisor_path), None)
    for item in runtime_manifest["files"]:
        p = runtime_root / item["path"]
        plan["artifact_sha256"][str(p)] = item["sha256"]
    plan["artifact_sha256"][str(supervisor_path)] = runtime_manifest["supervisor_sha256"]
    plan["artifact_sha256"][str(supervisor_path)] = file_digest(supervisor_path)
    # Confirm every input hash before write; supervisor path is staged separately next to the plan.
    supervisor.verify_pins(plan)
    if output_plan.exists() or service_draft.exists():
        raise FileExistsError("preserving an existing plan or service candidate")
    plan_sha = atomic_create(output_plan, plan)
    unit = ("[Unit]\nDescription=Resumable LFM2.5-350M reviewed source-exclusion phase\n\n"
            "[Service]\nType=simple\nWorkingDirectory=/home/werg/natlang\n"
            f"ExecStartPre=/usr/bin/docker info\nExecStart=/usr/bin/python3 {supervisor_path} run {output_plan} --expected-plan-sha256 {plan_sha}\n"
            "Restart=no\nTimeoutStopSec=infinity\nKillMode=mixed\nSendSIGKILL=no\n"
            "StandardOutput=journal\nStandardError=journal\n\n[Install]\nWantedBy=default.target\n")
    unit_sha = hashlib.sha256(unit.encode()).hexdigest()
    with service_draft.open("xb") as stream:
        stream.write(unit.encode()); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(service_draft.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(fd)
    finally: os.close(fd)
    return {"schema": "natlang.training_exclusion_resume_plan_candidate/1",
            "status": "prepared_not_approved_or_activated", "plan_path": str(output_plan),
            "plan_sha256": plan_sha, "service_unit_draft": str(service_draft),
            "service_unit_sha256": unit_sha, "runtime_manifest_sha256": file_digest(runtime_manifest_path),
            "exclusion_manifest_sha256": manifest_sha256,
            "supervisor_sha256": supervisor_sha256,
            "checkpoint_state_path": str(checkpoint_state), "target_examples": manifest["target"]["examples"],
            "completion_requires_final_eval": True, "active_run_mutated": False,
            "systemd_installed_or_enabled": False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-plan", "manifest", "application", "quiescence", "runtime-manifest",
                 "supervisor", "output-plan", "service-draft"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--supervisor-sha256", required=True)
    a = p.parse_args()
    result = prepare(base_plan_path=a.base_plan, manifest_path=a.manifest,
                     manifest_sha256=a.manifest_sha256, application_path=a.application,
                     quiescence_path=a.quiescence, runtime_manifest_path=a.runtime_manifest,
                     supervisor_path=a.supervisor, supervisor_sha256=a.supervisor_sha256,
                     output_plan=a.output_plan,
                     service_draft=a.service_draft)
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
