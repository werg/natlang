#!/usr/bin/env python3
"""Run one declared FP8 recipe through a bounded checkpoint interruption and exact resume.

This coordinates the shared declared-stage launcher. It does not implement or
alter training: the same recipe and output are used for both invocations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "run_neuralese_recipe.py"
sys.path.insert(0, str(REPO / "training" / "neuralese"))
from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha256



def train_steps(path: Path) -> list[int]:
    steps: list[int] = []
    if not path.is_file():
        return steps
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("step") if isinstance(row, dict) else None
            if isinstance(step, int) and not isinstance(step, bool):
                steps.append(step)
    return steps


def expected_checkpoint_interrupt(returncode: int, report: dict, requested_step: int) -> bool:
    """True only for the declared SIGTERM exit after a durable checkpoint."""
    return (returncode == 128 + signal.SIGTERM and
            report.get("status") == "checkpointed_on_signal" and
            int(report.get("steps", -1)) >= requested_step)


def file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns


def atomic_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def start_launcher(args, *, phase: str, resume: bool, wrapper_log: Path):
    launch_dir = args.receipt.parent / "launches"
    metadata = launch_dir / f"{phase}.launch-metadata.json"
    container_log = launch_dir / f"{phase}.container.log"
    if metadata.exists() or container_log.exists() or wrapper_log.exists():
        raise FileExistsError(f"refusing to reuse {phase} launch evidence")
    command = [str(args.python), str(LAUNCHER), "--recipe", str(args.recipe),
               "--artifact-root", str(args.artifact_root), "--hf-hub-cache", str(args.hf_hub_cache),
               "--device", args.device, "--launch-metadata", str(metadata), "--log", str(container_log)]
    if resume:
        command.append("--resume-existing")
    wrapper_log.parent.mkdir(parents=True, exist_ok=True)
    atomic_receipt(args.receipt, {**args.receipt_state, "phase": phase, "command": command,
                                  "started_at_unix": time.time(), "status": "running"})
    with wrapper_log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
    args.active_process = child
    args.launch_context[phase] = {
        "command": command,
        "wrapper_log": str(wrapper_log),
        "launch_metadata": str(metadata),
        "container_log": str(container_log),
    }
    return child


def launch_control(args) -> dict:
    """Run an uninterrupted same-seed reference whose only recipe difference is its output directory."""
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    output = Path(recipe["outputs"]["directory"]).expanduser()
    if not output.is_absolute():
        output = REPO / output
    control_output = (args.receipt.parent / "uninterrupted-output").resolve()
    try:
        control_output.relative_to(args.artifact_root)
    except ValueError:
        raise ValueError("control output must be under the declared artifact root") from None
    if control_output.exists():
        raise FileExistsError(f"refusing to reuse uninterrupted control output: {control_output}")
    control_output.parent.mkdir(parents=True, exist_ok=True)
    recipe["outputs"] = {"directory": str(control_output)}
    control_recipe = args.receipt.parent / "uninterrupted-control.recipe.json"
    atomic_receipt(control_recipe, recipe)
    control_stage = control_output / recipe["stage"]["id"]
    wrapper_log = args.receipt.parent / "launches" / "uninterrupted.wrapper.log"
    metadata = args.receipt.parent / "launches" / "uninterrupted.launch-metadata.json"
    container_log = args.receipt.parent / "launches" / "uninterrupted.container.log"
    command = [str(args.python), str(LAUNCHER), "--recipe", str(control_recipe),
               "--artifact-root", str(args.artifact_root), "--hf-hub-cache", str(args.hf_hub_cache),
               "--device", args.device, "--launch-metadata", str(metadata), "--log", str(container_log)]
    with wrapper_log.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, check=False,
                                   timeout=args.phase_timeout_seconds)
    report_path = control_stage / "report.json"
    train_path = control_stage / "train.jsonl"
    report = read_report(report_path)
    steps = train_steps(train_path)
    return {
        "recipe_path": str(control_recipe), "recipe_sha256": sha256(control_recipe),
        "original_recipe_sha256": sha256(args.recipe),
        "only_recipe_difference": "outputs.directory",
        "output": str(control_output), "stage_output": str(control_stage),
        "command": command, "returncode": completed.returncode,
        "wrapper_log": str(wrapper_log), "wrapper_log_sha256": sha256(wrapper_log),
        "launch_metadata": str(metadata), "launch_metadata_sha256": sha256(metadata) if metadata.is_file() else None,
        "container_log": str(container_log), "container_log_sha256": sha256(container_log) if container_log.is_file() else None,
        "report": report, "report_sha256": sha256(report_path), "train_steps": steps,
        "checkpoint": str(control_stage / "checkpoint.pt"),
        "checkpoint_sha256": sha256(control_stage / "checkpoint.pt")
        if (control_stage / "checkpoint.pt").is_file() else None,
    }


def finish_launcher(args, *, phase: str, child, timeout: float) -> int:
    return_code = child.wait(timeout=timeout)
    args.active_process = None
    context = args.launch_context[phase]
    wrapper_log = Path(context["wrapper_log"])
    metadata = Path(context["launch_metadata"])
    container_log = Path(context["container_log"])
    args.receipt_state.setdefault("launches", {})[phase] = {
        **context,
        "returncode": return_code,
        "wrapper_log_sha256": sha256(wrapper_log),
        "launch_metadata_sha256": sha256(metadata) if metadata.is_file() else None,
        "container_log_sha256": sha256(container_log) if container_log.is_file() else None,
    }
    return return_code


def wait_for_checkpoint_and_interrupt(args, train_log: Path, checkpoint: Path) -> int:
    original_signature = file_signature(checkpoint)
    deadline = time.monotonic() + args.phase_timeout_seconds
    stable_signature = None
    stable_polls = 0
    while time.monotonic() < deadline:
        child = args.active_process
        if child is None or child.poll() is not None:
            raise RuntimeError("launcher exited before the requested durable training step")
        steps = train_steps(train_log)
        reached = max(steps, default=0)
        if reached >= args.checkpoint_step:
            signature = file_signature(checkpoint)
            if signature is not None and signature != original_signature:
                if signature == stable_signature:
                    stable_polls += 1
                else:
                    stable_signature, stable_polls = signature, 0
                if stable_polls >= args.stable_polls:
                    args.receipt_state["requested_interrupt_step"] = args.checkpoint_step
                    args.receipt_state["observed_training_steps_before_signal"] = steps
                    args.receipt_state["checkpoint_at_signal"] = {
                        "path": str(checkpoint), "bytes": signature[0],
                        "mtime_ns": signature[1], "sha256": sha256(checkpoint),
                    }
                    child.send_signal(signal.SIGTERM)
                    return reached
        time.sleep(args.poll_seconds)
    child = args.active_process
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=args.phase_timeout_seconds)
        except subprocess.TimeoutExpired:
            pass
        args.active_process = None
    raise TimeoutError("timed out waiting for the declared checkpoint and training step; sent a cooperative stop")


def read_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"trainer report was not written: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"trainer report is not an object: {path}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("/srv/storage"))
    parser.add_argument("--hf-hub-cache", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python 3.12 interpreter used for the shared recipe launcher")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint-step", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--stable-polls", type=int, default=2)
    parser.add_argument("--phase-timeout-seconds", type=float, default=7200)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--resume-interrupted-receipt", type=Path,
                        help="recover only from a prior receipt with a verified signal checkpoint")
    args = parser.parse_args(argv)
    args.active_process = None
    args.launch_context = {}
    args.receipt_state = {}

    recipe_path = args.recipe.expanduser().resolve()
    args.recipe = recipe_path
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.hf_hub_cache = args.hf_hub_cache.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.receipt = args.receipt.expanduser().resolve()
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite watcher receipt: {args.receipt}")
    if not recipe_path.is_file() or not args.python.is_file():
        raise FileNotFoundError("declared recipe or Python interpreter is missing")
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    if recipe.get("schema") != "natlang.neuralese-declared-direct-stage/1" or \
            recipe.get("stage", {}).get("kind") != "neuralese_direct":
        raise ValueError("resume exercise requires a declared neuralese_direct recipe")
    parameters = recipe.get("parameters", {})
    if parameters.get("fp8_training") != "te-fp8":
        raise ValueError("this bounded FP8 resume exercise requires fp8_training=te-fp8")
    if (int(parameters.get("steps", 0)) != 12 or int(parameters.get("checkpoint_every", 0)) != 3 or
            int(parameters.get("eval_every", 0)) != 3):
        raise ValueError("declared recipe must use exactly 12 steps, eval_every=3, and checkpoint_every=3")
    if not 1 <= args.checkpoint_step <= 3:
        raise ValueError("interrupt step must be a checkpoint boundary in [1, 3]")
    if args.poll_seconds <= 0 or args.stable_polls < 1 or args.phase_timeout_seconds <= 0:
        raise ValueError("polling and timeout parameters must be positive")
    output = Path(recipe["outputs"]["directory"]).expanduser()
    if not output.is_absolute():
        output = REPO / output
    output = output.resolve()
    try:
        output.relative_to(args.artifact_root)
    except ValueError:
        raise ValueError("recipe output must be under the declared artifact root") from None
    stage_output = output / recipe["stage"]["id"]
    train_log, checkpoint = stage_output / "train.jsonl", stage_output / "checkpoint.pt"
    prior_receipt = None
    if args.resume_interrupted_receipt is not None:
        prior_path = args.resume_interrupted_receipt.expanduser().resolve()
        prior_receipt = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_initial = prior_receipt.get("initial", {})
        prior_report = prior_initial.get("report", {})
        if prior_receipt.get("schema") != "natlang.fp8-same-recipe-checkpoint-resume/1" or \
                prior_receipt.get("recipe_sha256") != sha256(recipe_path) or \
                prior_receipt.get("recipe_path") != str(recipe_path):
            raise ValueError("recovery receipt does not bind this exact declared recipe")
        if prior_report.get("status") != "checkpointed_on_signal" or \
                int(prior_report.get("steps", -1)) < args.checkpoint_step or \
                prior_initial.get("returncode") != 128 + signal.SIGTERM:
            raise ValueError("recovery receipt does not contain the expected signal-checkpoint outcome")
        if not checkpoint.is_file() or sha256(checkpoint) != prior_initial.get("checkpoint_sha256_before_resume"):
            raise ValueError("terminal checkpoint changed since the failed watcher receipt")
        if not output.is_dir():
            raise FileNotFoundError("recovery output directory is missing")
    elif output.exists():
        raise FileExistsError(f"fresh direct-stage output required for first phase: {output}")
    if not output.is_relative_to(args.artifact_root):
        raise ValueError("declared training output must be under the artifact root")
    args.receipt_state = {
        "schema": "natlang.fp8-same-recipe-checkpoint-resume/1",
        "recipe_path": str(recipe_path), "recipe_sha256": sha256(recipe_path),
        "outputs_directory": str(output), "stage_output": str(stage_output),
        "train_log": str(train_log), "checkpoint": str(checkpoint),
        "declared_steps": parameters["steps"], "checkpoint_every": parameters["checkpoint_every"],
        "fp8_training": parameters["fp8_training"], "fp8_recipe": parameters.get("fp8_recipe"),
        "artifact_root": str(args.artifact_root), "hf_hub_cache": str(args.hf_hub_cache),
        "python": str(args.python), "device": args.device,
        "watcher_script": str(Path(__file__).resolve()), "watcher_sha256": sha256(Path(__file__).resolve()),
        "status": "prepared",
    }
    if prior_receipt is not None:
        args.receipt_state["recovered_from_receipt"] = {
            "path": str(prior_path), "sha256": sha256(prior_path),
            "checkpoint_sha256": prior_initial["checkpoint_sha256_before_resume"],
            "terminal_step": prior_report.get("steps"),
        }
    atomic_receipt(args.receipt, args.receipt_state)
    print(json.dumps({"event": "resume_exercise_started", "receipt": str(args.receipt),
                      "recipe_sha256": args.receipt_state["recipe_sha256"]}), flush=True)

    if prior_receipt is not None:
        initial_report = prior_report
        args.receipt_state["initial"] = prior_initial
    else:
        first_log = args.receipt.parent / "launches" / "initial.wrapper.log"
        first_child = start_launcher(args, phase="initial", resume=False, wrapper_log=first_log)
        requested_reached = wait_for_checkpoint_and_interrupt(args, train_log, checkpoint)
        first_code = finish_launcher(args, phase="initial", child=first_child,
                                     timeout=args.phase_timeout_seconds)
        initial_report = read_report(stage_output / "report.json")
        args.receipt_state["initial"] = {
            "returncode": first_code, "report": initial_report,
            "report_sha256": sha256(stage_output / "report.json"),
            "train_steps": train_steps(train_log), "watcher_step_when_signaled": requested_reached,
            "interrupt_observed_step": initial_report.get("steps"),
            "checkpoint_sha256_before_resume": sha256(checkpoint) if checkpoint.is_file() else None,
        }
        if initial_report.get("status") != "checkpointed_on_signal" or \
                int(initial_report.get("steps", -1)) < args.checkpoint_step:
            args.receipt_state["status"] = "interrupted_without_expected_full_state"
            atomic_receipt(args.receipt, args.receipt_state)
            raise RuntimeError(f"initial run did not stop on a complete signal checkpoint: {initial_report}")
        expected_signal_exit = expected_checkpoint_interrupt(
            first_code, initial_report, args.checkpoint_step)
        if first_code != 0 and not expected_signal_exit:
            args.receipt_state["status"] = "initial_launcher_failed"
            atomic_receipt(args.receipt, args.receipt_state)
            raise RuntimeError(f"initial launcher returned {first_code}; refusing resume")
        args.receipt_state["initial"]["expected_signal_exit"] = expected_signal_exit

    resume_log = args.receipt.parent / "launches" / "resume.wrapper.log"
    resume_child = start_launcher(args, phase="resume", resume=True, wrapper_log=resume_log)
    resume_code = finish_launcher(args, phase="resume", child=resume_child,
                                  timeout=args.phase_timeout_seconds)
    final_report = read_report(stage_output / "report.json")
    final_steps = train_steps(train_log)
    args.receipt_state["resume"] = {
        "returncode": resume_code, "report": final_report,
        "report_sha256": sha256(stage_output / "report.json"), "train_steps": final_steps,
        "checkpoint_sha256_after_resume": sha256(checkpoint) if checkpoint.is_file() else None,
        "fp8_runtime_inventory_sha256": sha256(stage_output / "fp8-runtime-inventory.jsonl")
        if (stage_output / "fp8-runtime-inventory.jsonl").is_file() else None,
    }
    complete = resume_code == 0 and final_report.get("status") == "complete" and \
        final_report.get("steps") == parameters["steps"] and final_steps == list(range(1, parameters["steps"] + 1))
    if not complete:
        args.receipt_state["status"] = "resume_incomplete_or_failed"
        atomic_receipt(args.receipt, args.receipt_state)
        return 1
    control = launch_control(args)
    args.receipt_state["uninterrupted_control"] = control
    control_complete = control["returncode"] == 0 and control["report"].get("status") == "complete" and \
        control["report"].get("steps") == parameters["steps"] and \
        control["train_steps"] == list(range(1, parameters["steps"] + 1))
    complete = control_complete
    args.receipt_state["status"] = "complete_operational_resume_and_control_ready" if complete \
        else "uninterrupted_control_failed"
    args.receipt_state["same_recipe_sha256_both_phases"] = True
    atomic_receipt(args.receipt, args.receipt_state)
    print(json.dumps({"event": "resume_exercise_finished", "status": args.receipt_state["status"],
                      "interrupted_at": initial_report.get("steps"), "final_step": final_report.get("steps"),
                      "receipt": str(args.receipt)}), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
