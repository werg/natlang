#!/usr/bin/env python3
"""Small durable supervisor for an exact, checkpoint-resumable training command.

The immutable plan owns the command, its artifact pins, completion target, and
retry policy. The supervisor never edits those values. SIGTERM means graceful
pause (checkpoint and exit); ``stop`` adds a persistent operator latch.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_plan(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    plan = json.loads(raw)
    if plan.get("schema") != "natlang-resumable-training/1":
        raise ValueError("unsupported training plan schema")
    if not isinstance(plan.get("argv"), list) or not plan["argv"] or not all(isinstance(x, str) for x in plan["argv"]):
        raise ValueError("plan argv must be a nonempty string array")
    for key in ("run_dir", "checkpoint_state", "status_file", "lock_file", "operator_stop_file"):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise ValueError(f"plan requires {key}")
    if not isinstance(plan.get("completion"), dict):
        raise ValueError("plan requires explicit completion target")
    if not isinstance(plan.get("checkpoint_identity"), dict) or not plan["checkpoint_identity"]:
        raise ValueError("plan requires checkpoint_identity fields to verify the resumed checkpoint")
    retries = plan.get("retry", {})
    if not (1 <= int(retries.get("max_attempts", 1)) <= 20):
        raise ValueError("retry.max_attempts must be between 1 and 20")
    if int(retries.get("initial_delay_seconds", 0)) < 1 or int(retries.get("max_delay_seconds", 0)) < 1:
        raise ValueError("retry delays must be positive")
    return plan, hashlib.sha256(raw).hexdigest()


def verify_pins(plan: dict) -> None:
    pins = plan.get("artifact_sha256", {})
    if not isinstance(pins, dict) or not pins:
        raise ValueError("plan must pin at least one artifact")
    for raw_path, expected in sorted(pins.items()):
        p = Path(raw_path)
        if not p.is_file() or sha256_file(p) != expected:
            raise ValueError(f"artifact pin mismatch: {raw_path}")
    runtime = plan.get("runtime", {})
    if runtime:
        if runtime.get("kind") != "docker" or not runtime.get("image_ref") or not runtime.get("image_id"):
            raise ValueError("runtime pin requires kind=docker, image_ref, and image_id")
        result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", runtime["image_ref"]],
                                text=True, capture_output=True)
        if result.returncode or result.stdout.strip() != runtime["image_id"]:
            raise ValueError("Docker image ID differs from pinned training plan")


def checkpoint_state(plan: dict) -> dict | None:
    p = Path(plan["checkpoint_state"])
    if not p.is_file():
        return None
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"checkpoint state is unreadable: {p}: {e}") from e
    if not isinstance(state, dict) or not isinstance(state.get("step"), int):
        raise ValueError(f"checkpoint state has no integer step: {p}")
    checkpoint_dir = p.parent
    required = plan.get("checkpoint_required", ["optimizer.pt", "scheduler.pt", "rng.pt", "weights"])
    for name in required:
        item = checkpoint_dir / name
        if not item.exists():
            raise ValueError(f"checkpoint is incomplete; missing {item}")
        if name == "weights" and not any(child.is_file() for child in item.rglob("*")):
            raise ValueError(f"checkpoint weights directory is empty: {item}")
    return state


def is_complete(plan: dict, state: dict | None) -> bool:
    if state is None:
        return False
    goal = plan["completion"]
    step_goal = goal.get("step")
    example_goal = goal.get("trained_examples")
    if step_goal is None and example_goal is None:
        raise ValueError("completion target must pin step and/or trained_examples")
    if step_goal is not None and state["step"] < int(step_goal):
        return False
    if example_goal is not None and int(state.get("trained_examples", -1)) < int(example_goal):
        return False
    expected_identity = plan.get("checkpoint_identity", {})
    corpus = state.get("corpus")
    if expected_identity and (not isinstance(corpus, dict) or any(corpus.get(k) != v for k, v in expected_identity.items())):
        raise ValueError("checkpoint training identity differs from pinned plan")
    if goal.get("final_evaluation_required", False):
        value = state.get("heldout_after")
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            return False
    return True


def proc_start_time(pid: int) -> str | None:
    """Linux process start tick, used to avoid signaling a stale/reused PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        tail = stat[stat.rfind(")") + 2:].split()
        return tail[19]  # field 22; tail begins with field 3 (state)
    except (OSError, IndexError):
        return None


class Supervisor:
    def __init__(self, plan_path: Path):
        self.plan_path = plan_path.resolve()
        self.plan, self.plan_sha = load_plan(self.plan_path)
        self.run_dir = Path(self.plan["run_dir"])
        self.status_path = Path(self.plan["status_file"])
        self.log_path = Path(self.plan.get("stdout_log", str(self.run_dir / "trainer.stdout.log")))
        self.stop_path = Path(self.plan["operator_stop_file"])
        self.lock_path = Path(self.plan["lock_file"])
        self.child: subprocess.Popen | None = None
        self.term_requested = False
        self.owns_lock = False

    def write_status(self, state: str, **extra) -> None:
        ckpt = checkpoint_state(self.plan)
        atomic_json(self.status_path, {
            "schema": "natlang-resumable-training-status/1", "state": state,
            "plan_sha256": self.plan_sha, "pid": os.getpid(), "updated_unix": time.time(),
            "pid_start_time": proc_start_time(os.getpid()),
            "checkpoint_step": None if ckpt is None else ckpt["step"], **extra,
        })

    def on_signal(self, *_args) -> None:
        self.term_requested = True
        if self.child and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def run(self) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.lock_path.open("a+")
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.owns_lock = True
            except BlockingIOError as e:
                raise RuntimeError("another supervisor owns this training run") from e
            identity = self.run_dir / "supervisor-identity.json"
            if identity.exists():
                existing = json.loads(identity.read_text())
                if existing.get("plan_sha256") != self.plan_sha:
                    raise ValueError("run directory is already bound to a different immutable plan")
            else:
                atomic_json(identity, {"schema": "natlang-training-identity/1", "plan_sha256": self.plan_sha,
                                       "plan_path": str(self.plan_path)})
            if self.plan.get("runtime", {}).get("kind") == "docker":
                deadline = time.monotonic() + int(self.plan.get("runtime_ready_timeout_seconds", 120))
                while True:
                    ready = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL).returncode == 0
                    if ready:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Docker daemon did not become ready before the pinned timeout")
                    time.sleep(2)
            verify_pins(self.plan)
            if self.stop_path.exists():
                latch = json.loads(self.stop_path.read_text())
                if latch.get("plan_sha256") != self.plan_sha:
                    raise ValueError("operator stop latch belongs to a different immutable plan")
                self.write_status("operator_stopped", reason="persistent operator stop latch")
                return 0
            state = checkpoint_state(self.plan)
            if is_complete(self.plan, state):
                self.write_status("complete", reason="checkpoint already reached pinned target")
                return 0
            old_handler_term = signal.signal(signal.SIGTERM, self.on_signal)
            old_handler_int = signal.signal(signal.SIGINT, self.on_signal)
            try:
                retry = self.plan["retry"]
                max_attempts = int(retry["max_attempts"])
                delay = int(retry["initial_delay_seconds"])
                max_delay = int(retry["max_delay_seconds"])
                for attempt in range(1, max_attempts + 1):
                    if self.stop_path.exists():
                        self.write_status("operator_stopped", attempt=attempt - 1)
                        return 0
                    if self.term_requested:
                        self.write_status("paused", attempt=attempt - 1, reason="graceful stop requested")
                        return 0
                    verify_pins(self.plan)
                    state = checkpoint_state(self.plan)
                    if is_complete(self.plan, state):
                        self.write_status("complete", attempts=attempt - 1)
                        return 0
                    env = os.environ.copy()
                    env.update(self.plan.get("env", {}))
                    self.write_status("running", attempt=attempt, max_attempts=max_attempts)
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("ab", buffering=0) as log:
                        log.write((f"\n=== supervisor attempt {attempt} started {time.time():.3f} ===\n").encode())
                        self.child = subprocess.Popen(self.plan["argv"], cwd=self.plan.get("cwd"), env=env,
                                                      stdout=log, stderr=subprocess.STDOUT,
                                                      start_new_session=True)
                    rc = self.child.wait()
                    self.child = None
                    state = checkpoint_state(self.plan)
                    if is_complete(self.plan, state):
                        self.write_status("complete", attempts=attempt, child_returncode=rc)
                        return 0
                    if self.stop_path.exists():
                        self.write_status("operator_stopped", attempt=attempt, child_returncode=rc)
                        return 0
                    if self.term_requested:
                        self.write_status("paused", attempt=attempt, child_returncode=rc,
                                          reason="graceful stop; checkpoint retained for exact resume")
                        return 0
                    if rc == 0:
                        self.write_status("paused_incomplete", attempt=attempt,
                                          reason="trainer exited successfully below target; manual review required")
                        return 1
                    if attempt == max_attempts:
                        self.write_status("failed", attempts=attempt, child_returncode=rc,
                                          reason="retry budget exhausted")
                        return rc or 1
                    self.write_status("retry_wait", attempt=attempt, child_returncode=rc,
                                      next_delay_seconds=delay)
                    deadline = time.monotonic() + delay
                    while time.monotonic() < deadline:
                        if self.stop_path.exists() or self.term_requested:
                            break
                        time.sleep(min(1, deadline - time.monotonic()))
                    if self.stop_path.exists():
                        self.write_status("operator_stopped", attempt=attempt)
                        return 0
                    if self.term_requested:
                        self.write_status("paused", attempt=attempt, reason="stop requested during backoff")
                        return 0
                    delay = min(max_delay, max(1, delay * 2))
                return 1
            finally:
                signal.signal(signal.SIGTERM, old_handler_term)
                signal.signal(signal.SIGINT, old_handler_int)
        except BaseException as e:
            # Fail closed: record the infrastructure/configuration error only
            # after the exclusive lock is held, never launch an unpinned command.
            if self.owns_lock:
                try:
                    self.write_status("paused_setup", error=f"{type(e).__name__}: {e}")
                except Exception:
                    pass
            raise
        finally:
            if self.owns_lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()


def control(plan_path: Path, action: str) -> int:
    plan, plan_sha = load_plan(plan_path.resolve())
    status_path = Path(plan["status_file"])
    stop_path = Path(plan["operator_stop_file"])
    identity_path = Path(plan["run_dir"]) / "supervisor-identity.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        if identity.get("plan_sha256") != plan_sha:
            raise ValueError("control plan does not match this run's immutable identity")
    if action == "stop":
        atomic_json(stop_path, {"schema": "natlang-training-operator-stop/1", "plan_sha256": plan_sha,
                                "requested_unix": time.time()})
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        pid = status.get("pid")
        if (status.get("plan_sha256") == plan_sha and isinstance(pid, int) and pid > 1
                and status.get("pid_start_time") == proc_start_time(pid)):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        print(f"operator stop latched for {plan['run_dir']}")
        return 0
    if action == "resume":
        if stop_path.exists():
            latch = json.loads(stop_path.read_text())
            if latch.get("plan_sha256") != plan_sha:
                raise ValueError("operator stop latch belongs to a different immutable plan")
            stop_path.unlink()
            fsync_dir(stop_path.parent)
        print("operator stop latch cleared; start or restart the pinned service to resume")
        return 0
    if action == "status":
        if status_path.exists():
            print(status_path.read_text(), end="")
        else:
            print(json.dumps({"state": "not_started", "plan_sha256": plan_sha}))
        return 0
    raise ValueError(action)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    for name in ("run", "stop", "resume", "status"):
        p = sub.add_parser(name)
        p.add_argument("plan", type=Path)
    args = ap.parse_args(argv)
    if args.action == "run":
        return Supervisor(args.plan).run()
    return control(args.plan, args.action)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"training supervisor: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
