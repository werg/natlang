#!/usr/bin/env python3
"""Durable sequential job runner. Commands are argv arrays, never shell strings.

SIGINT/SIGTERM asks the current process group to stop and waits for its checkpoint.
Re-run the identical config/run directory to resume. No automatic SIGKILL timeout.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def process_identity(pid):
    try:
        # Field 22, after removing the parenthesized command (which may contain spaces).
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def expand(value, root, repo):
    return value.replace("${run}", str(root)).replace("${repo}", str(repo))


def fingerprints(paths):
    result = {}
    for path in paths:
        if path.is_dir():
            files = {str(item.relative_to(path)): digest_file(item) for item in sorted(path.rglob("*")) if item.is_file()}
            result[str(path)] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        else:
            result[str(path)] = digest_file(path)
    return result


def exec_stage(lock_path, argv):
    """Hold an inherited lock across exec, including the parent spawn/journal crash window."""
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("stage is still running under another process") from None
    os.set_inheritable(descriptor, True)
    os.ftruncate(descriptor, 0)
    os.write(descriptor, json.dumps({"pid": os.getpid(), "process_start": process_identity(os.getpid())}).encode())
    os.fsync(descriptor)
    os.execvpe(argv[0], argv, os.environ)


def run_pipeline(config_path, root, *, until=None):
    config_path, root = Path(config_path).resolve(), Path(root).resolve()
    config = json.loads(config_path.read_text())
    if config.get("version") != "natlang.training_pipeline/1":
        raise ValueError("unsupported pipeline config")
    if config.get('run_directory') and Path(config['run_directory']).resolve() != root:
        raise ValueError('pipeline recipe is bound to a different run directory')
    stages = config["stages"]
    ids = [s["id"] for s in stages]
    if len(set(ids)) != len(ids) or not stages or any(not i.replace("-", "").replace("_", "").isalnum() for i in ids):
        raise ValueError("stage ids must be distinct simple names")
    if until and until not in ids:
        raise ValueError("unknown --until stage")
    repo = Path(config.get("repository", config_path.parent)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another runner owns this pipeline") from None
        state_path = root / "pipeline-state.json"
        identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "version": config["version"], "config_sha256": identity, "stages": {}, "status": "pending"}
        if state["config_sha256"] != identity:
            raise ValueError("pipeline config changed; use a new run directory")
        for stage_id in ids:
            with (root / f".stage-{stage_id}.lock").open("a+") as stage_lock:
                try:
                    fcntl.flock(stage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError(f"stage {stage_id} is still running; resume only after it stops") from None
                fcntl.flock(stage_lock, fcntl.LOCK_UN)
        for entry in state["stages"].values():
            if entry.get("pid") and entry.get("process_start") == process_identity(entry["pid"]):
                raise RuntimeError(f"stage process {entry['pid']} still exists; let it finish or stop its process group before resuming")
        stopped = False
        child = None

        def stop(signum, frame):
            nonlocal stopped
            stopped = True
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for stage in stages:
                if stopped:
                    break
                name = stage["id"]
                argv = [expand(v, root, repo) for v in stage["command"]]
                if not argv or not all(isinstance(v, str) for v in argv):
                    raise ValueError(f"{name}: command must be an argv array")
                inputs = [Path(expand(v, root, repo)) for v in stage.get("inputs", [])]
                outputs = [Path(expand(v, root, repo)) for v in stage.get("outputs", [])]
                if not outputs:
                    raise ValueError(f"{name}: declare at least one committed output")
                before = fingerprints(inputs)
                entry = state["stages"].get(name, {})
                if entry.get("inputs") is not None and entry["inputs"] != before:
                    raise ValueError(f"{name}: input content changed; refusing unsafe resume")
                if entry.get("status") == "complete":
                    if fingerprints(outputs) != entry["outputs"]:
                        raise ValueError(f"{name}: completed output changed or is missing")
                    print(f"{name}: already complete", flush=True)
                else:
                    if stage.get("min_free_vram_mib"):
                        available = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
                        # The current trainer selects cuda:0. Do not silently move another GPU job.
                        free = int(available.strip().splitlines()[0])
                        if free < stage["min_free_vram_mib"]:
                            state["status"] = "resource_wait"
                            state["resource_wait"] = {"stage": name, "free_mib": free, "required_mib": stage["min_free_vram_mib"]}
                            atomic_json(state_path, state)
                            print(f"{name}: needs {stage['min_free_vram_mib']} MiB free on GPU 0; currently {free}. Resume the same command when available.", flush=True)
                            return 75
                    log_path = root / f"{name}.log"
                    entry = {"status": "running", "inputs": before, "started": time.time(),
                             "attempts": entry.get("attempts", 0) + 1, "log": str(log_path)}
                    state["stages"][name] = entry
                    state["status"] = "running"
                    # Persist intent before spawning; an interrupted stage must itself be resumable.
                    atomic_json(state_path, state)
                    print(f"{name}: running; log {log_path}", flush=True)
                    with log_path.open("ab", buffering=0) as log:
                        wrapped = [sys.executable, str(Path(__file__).resolve()), "--stage-exec", str(root / f".stage-{name}.lock"), *argv]
                        child = subprocess.Popen(wrapped, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                        entry.update(pid=child.pid, process_start=process_identity(child.pid))
                        atomic_json(state_path, state)
                        if stopped:
                            stop(signal.SIGTERM, None)
                        code = child.wait()
                        child = None
                    entry.pop("pid", None)
                    entry.pop("process_start", None)
                    entry.update(exit_code=code, finished=time.time())
                    if stopped or code != 0:
                        entry["status"] = "stopped" if stopped or code in (75, 130, 143) else "failed"
                        state["status"] = entry["status"]
                        atomic_json(state_path, state)
                        return 130 if stopped else (code if code > 0 else 1)
                    if fingerprints(inputs) != before:
                        raise ValueError(f"{name}: input changed during execution")
                    entry["outputs"] = fingerprints(outputs)
                    # Trainer exits 0 after a graceful checkpoint too; never advance on partial training.
                    if stage.get("training_state"):
                        training = json.loads(Path(expand(stage["training_state"], root, repo)).read_text())
                        if training.get("trained_examples", 0) < training["corpus"]["target_examples"]:
                            entry["status"] = state["status"] = "stopped"
                            atomic_json(state_path, state)
                            return 75
                    entry["status"] = "complete"
                    atomic_json(state_path, state)
                if name == until:
                    state["status"] = "ready"
                    atomic_json(state_path, state)
                    return 0
            state["status"] = "stopped" if stopped else "complete"
            atomic_json(state_path, state)
            return 130 if stopped else 0
        except BaseException as error:
            state["status"] = "failed"
            state["error"] = type(error).__name__ + ": " + str(error)
            atomic_json(state_path, state)
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--until", help="finish this stage, then stop before subsequent stages")
    args = parser.parse_args()
    return run_pipeline(args.config, args.run, until=args.until)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--stage-exec":
        exec_stage(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(main())
