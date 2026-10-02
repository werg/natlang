#!/usr/bin/env python3
"""Start the two reviewed Fledge Alpha free queues through OpenCode Zen.

This launcher is deliberately fail-closed: a root-approved plan, unchanged
runtime/queue pins, a locally stored key, and live public zero-cost metadata
are required before either supervisor is started. Credentials are read only
into the child environment and never enter argv, plans, status, or logs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

MODEL_ID = "fledge-alpha-free"
PROVIDER = "opencode"
KEY_NAME = "OPENCODE_API_KEY"
DEFAULT_KEY = Path.home() / ".config/natlang/opencode.env"
EXPECTED_CONFIG = {
    "piOptions": {"maxTokens": 8192, "reasoningEffort": "high"},
    "piPayload": {"tool_choice": "auto"},
    "modelOptions": {
        "inherit": "big-pickle", "name": "Fledge Alpha Free",
        "contextWindow": 1048576, "maxTokens": 131072,
        "reasoning": True, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    },
    "omitPayloadKeys": ["seed"],
}
status_writer = None
children: list[subprocess.Popen] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmpname, path)
        dfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            os.unlink(tmpname)
        except FileNotFoundError:
            pass


def ensure_key_file(path: Path) -> str | None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        info = path.lstat()
    except FileNotFoundError:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"{KEY_NAME}=\n")
            stream.flush()
            os.fsync(stream.fileno())
        dfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("Zen key file must be a regular user-owned mode-600 file")
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if len(entries) != 1 or not entries[0].startswith(KEY_NAME + "="):
        raise ValueError("Zen key file must contain exactly one OPENCODE_API_KEY entry")
    secret = entries[0].split("=", 1)[1].strip()
    if len(secret) >= 2 and secret[0] == secret[-1] and secret[0] in "\"'":
        secret = secret[1:-1]
    if not secret or any(ch.isspace() for ch in secret):
        return None
    return secret


def public_metadata() -> dict:
    # No credential is passed to this process or to the public metadata URLs.
    script = r'''const urls=['https://opencode.ai/zen/v1/models','https://models.dev/api.json'];
const out=[]; for (const url of urls) { const r=await fetch(url,{signal:AbortSignal.timeout(20000)});
if(!r.ok) throw new Error('metadata_http_'+r.status); out.push(await r.json()); }
process.stdout.write(JSON.stringify(out));'''
    try:
        completed = subprocess.run(["node", "--input-type=module", "-e", script],
                                   check=True, capture_output=True, text=True, timeout=30,
                                   env={k: v for k, v in os.environ.items() if k != KEY_NAME})
        live, catalog = json.loads(completed.stdout)
    except Exception as exc:
        raise ValueError("public Zen model metadata unavailable") from exc
    live_models = live.get("data", []) if isinstance(live, dict) else []
    if not any(isinstance(row, dict) and row.get("id") == MODEL_ID for row in live_models):
        raise ValueError("Fledge Alpha free model is not currently listed by OpenCode Zen")
    provider = catalog.get("opencode", {}) if isinstance(catalog, dict) else {}
    model = provider.get("models", {}).get(MODEL_ID, {})
    api = provider.get("api")
    cost = model.get("cost", {})
    if (api != "https://opencode.ai/zen/v1" or
            cost.get("input") != 0 or cost.get("output") != 0 or
            model.get("limit", {}).get("context") != 1048576 or model.get("limit", {}).get("output") != 131072 or
            model.get("tool_call") is not True or model.get("reasoning") is not True):
        raise ValueError("public model metadata no longer matches the reviewed free Fledge profile")
    return {"model_id": MODEL_ID, "provider": PROVIDER, "live_listed": True,
            "cost_input": 0, "cost_output": 0, "context_window": 1048576,
            "output_limit": 131072, "reasoning": True, "tool_call": True,
            "checked_at": dt.datetime.now(dt.timezone.utc).isoformat()}


def validate_plan(plan_path: Path, plan: dict) -> tuple[Path, list[dict]]:
    if plan.get("root_approved") is not True or plan.get("launch_enabled") is not True or plan.get("status") != "approved":
        raise PermissionError("plan is not root-approved and launch-enabled")
    if plan.get("provider") != PROVIDER or plan.get("model_id") != MODEL_ID:
        raise ValueError("plan must pin OpenCode Zen and exact model ID fledge-alpha-free")
    if plan.get("model_concurrency") != 1 or plan.get("worker_count") != 2:
        raise ValueError("plan must define exactly two workers with one request each")
    runtime = Path(plan["runtime"]).resolve(strict=True)
    frozen = json.loads((runtime / "frozen-runtime.json").read_text())
    if sha256(runtime / "frozen-runtime.json") != plan.get("runtime_manifest_sha256"):
        raise ValueError("runtime manifest pin changed")
    if len(frozen.get("files", {})) != 630:
        raise ValueError("expected the reviewed 630-file runtime closure")
    for rel, expected in frozen.get("files", {}).items():
        if sha256(runtime / rel) != expected:
            raise ValueError("frozen runtime file changed")
    if plan.get("supervisor_sha256") != sha256(Path(plan["supervisor"]).resolve(strict=True)):
        raise ValueError("reviewed supervisor pin changed")
    if plan.get("supervisor_sha256") != "b70c02e69742b3181e84c4a2c2c58270c6ccd3a67e71a3050cb63fbc3871b6e6":
        raise ValueError("supervisor is not the reviewed Bunny v2 collector")
    handoff = Path(plan["handoff_candidate"]).resolve(strict=True)
    if sha256(handoff) != plan.get("handoff_candidate_sha256"):
        raise ValueError("reviewed repair handoff changed")
    handoff_record = json.loads(handoff.read_text())
    pinned_handoff = plan.get("artifact_hashes", {})
    if pinned_handoff != handoff_record.get("artifact_hashes"):
        raise ValueError("launch plan handoff pins differ from the complete reviewed pin set")
    for path, expected in pinned_handoff.items():
        if sha256(Path(path).resolve(strict=True)) != expected:
            raise ValueError("a source handoff artifact pin changed")
    dependency_path = Path(plan["provider_dependency_pins"]).resolve(strict=True)
    if sha256(dependency_path) != plan.get("provider_dependency_pins_sha256"):
        raise ValueError("provider dependency manifest changed")
    dependency_pins = json.loads(dependency_path.read_text()).get("files", {})
    if len(dependency_pins) != 3284:
        raise ValueError("provider dependency manifest is incomplete")
    for path, expected in dependency_pins.items():
        if sha256(Path(path).resolve(strict=True)) != expected:
            raise ValueError("provider dependency file changed")
    controls_path = Path(plan["provider_request_config"]).resolve(strict=True)
    if sha256(controls_path) != plan.get("provider_request_config_sha256"):
        raise ValueError("provider request controls content pin changed")
    controls = json.loads(controls_path.read_text())
    if controls != EXPECTED_CONFIG:
        raise ValueError("Zen provider request controls differ from the reviewed exact profile")
    if Path(plan.get("key_file", "")).expanduser().resolve() != Path(plan.get("expected_key_file", str(DEFAULT_KEY))).expanduser().resolve():
        raise ValueError("plan key-file path differs from the reviewed Zen key location")
    if plan.get("execution_plans") is not True or plan.get("case_seconds") != 1200:
        raise ValueError("execution plan and per-case budget settings differ from reviewed controls")
    workers = plan.get("workers")
    if not isinstance(workers, list) or len(workers) != 2:
        raise ValueError("plan must contain exactly two worker entries")
    seen_queues = set()
    seen_keys = set()
    program_ids: list[str] = []
    queue_counts = []
    source_ir = Path(plan["candidate_IR"]).resolve(strict=True)
    if sha256(source_ir) != plan.get("candidate_IR_sha256"):
        raise ValueError("candidate IR content pin changed")
    ir_rows = [json.loads(line) for line in source_ir.read_text().splitlines() if line.strip()]
    for item in workers:
        for key in ("queue", "journal", "log", "status"):
            if key not in item:
                raise ValueError("worker entry is missing its durable path")
        queue = Path(item["queue"]).resolve(strict=True)
        if queue in seen_queues or sha256(queue) != item.get("queue_sha256"):
            raise ValueError("queue is duplicated or its content pin changed")
        seen_queues.add(queue)
        journal = Path(item["journal"])
        if journal.exists() and journal.stat().st_size:
            raise ValueError("refusing to reuse a queue with an existing journal")
        old_status = Path(item["status"])
        if old_status.exists():
            previous = json.loads(old_status.read_text())
            for pid_key in ("supervisor_pid", "launcher_pid"):
                pid = previous.get(pid_key)
                if isinstance(pid, int) and pid > 1:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise ValueError("worker status records a live prior process")
            if previous.get("state") in {"running", "starting"}:
                raise ValueError("worker status shows a possibly-live prior launcher")
        rows = [json.loads(line) for line in queue.read_text().splitlines() if line.strip()]
        if not rows or any(row.get("key") in seen_keys for row in rows):
            raise ValueError("worker queues must be nonempty and disjoint")
        seen_keys.update(row.get("key") for row in rows)
        queue_counts.append(len(rows))
        worker_programs = []
        for row in rows:
            if Path(row.get("source", "")).resolve() != source_ir:
                raise ValueError("queue points at a different source IR")
            index = row.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(ir_rows):
                raise ValueError("queue index is outside the pinned candidate IR")
            program_id = ir_rows[index].get("id")
            if not isinstance(program_id, str) or program_id in program_ids:
                raise ValueError("queue program identities are missing or overlap")
            worker_programs.append(program_id)
            program_ids.append(program_id)
        if worker_programs != item.get("program_ids"):
            raise ValueError("worker program ID inventory differs from queue order")
        if any(row.get("count", 1) != 1 or row.get("max_turns") != 40 or
               row.get("max_model_requests") != 384 for row in rows):
            raise ValueError("queue per-case budgets differ from the reviewed repair plan")
    if queue_counts != [34, 33] or len(program_ids) != 67:
        raise ValueError("the approved repair scope must remain exactly disjoint 34/33 queues")
    return runtime, workers


def terminate_and_reap() -> list[tuple[int, int]]:
    for child in children:
        if child.poll() is None:
            child.terminate()
    out = []
    for child in children:
        try:
            code = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            code = child.wait()
        out.append((child.pid, code))
    return out


def main() -> int:
    global status_writer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="root-approved Zen worker plan JSON")
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY)
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    base = plan_path.parent
    lock_path = base / "zen-launch.lock"
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another reviewed Zen launcher owns this plan", file=sys.stderr)
        return 2

    def status(state: str, **extra):
        value = {"state": state, "provider": PROVIDER, "model_id": MODEL_ID,
                 "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "launcher_pid": os.getpid(), **extra}
        atomic_json(base / "zen-launch-status.json", value)

    status_writer = status
    active_pids: list[int] = []
    try:
        runtime, workers = validate_plan(plan_path, plan)
        secret = ensure_key_file(args.key_file)
        if secret is None:
            status("waiting_for_key", key_file=str(args.key_file), active_workers=0)
            return 0
        metadata = public_metadata()
        config_path = str(Path(plan["provider_request_config"]).resolve(strict=True))
        status("starting", model_metadata=metadata, active_workers=0)
        def stop(_sig, _frame):
            raise KeyboardInterrupt("launcher asked to stop")
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        for worker in workers:
            log_path = Path(worker["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            journal = Path(worker["journal"])
            journal.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, plan["supervisor"], worker["queue"], str(journal),
                       "--runtime", str(runtime), "--provider", PROVIDER, "--model-id", MODEL_ID,
                       "--model-concurrency", "1", "--case-seconds", "1200",
                       "--reasoning-effort", "high", "--provider-request-config", config_path]
            if plan.get("execution_plans") is True:
                command.append("--execution-plans")
            output = log_path.open("a", encoding="utf-8")
            env = dict(os.environ)
            env.pop("OPENROUTER_API_KEY", None)
            env[KEY_NAME] = secret
            child = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
            children.append(child)
            active_pids.append(child.pid)
            output.close()
            atomic_json(Path(worker["status"]), {"state": "running", "provider": PROVIDER,
                        "model_id": MODEL_ID, "supervisor_pid": child.pid,
                        "queue": worker["queue"], "journal": str(journal),
                        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            status("running", active_workers=len(children), supervisor_pids=active_pids.copy(),
                   model_metadata=metadata)
        secret = None
        codes = []
        pending = {child.pid: child for child in children}
        while pending:
            for child in list(pending.values()):
                code = child.poll()
                if code is None:
                    continue
                child.wait()
                codes.append((child.pid, code))
                worker_index = next(i for i, item in enumerate(workers)
                                    if Path(item["status"]).exists() and
                                    json.loads(Path(item["status"]).read_text()).get("supervisor_pid") == child.pid)
                worker = workers[worker_index]
                finish_rows = [row for line in
                               (Path(worker["journal"]).read_text().splitlines() if Path(worker["journal"]).exists() else [])
                               for row in [json.loads(line)] if row.get("event") == "finish"]
                finished = {row.get("key"): row for row in finish_rows}
                queue_rows = [json.loads(line) for line in Path(worker["queue"]).read_text().splitlines() if line.strip()]
                keys = {row["key"] for row in queue_rows}
                complete = (code == 0 and len(finish_rows) == len(keys) and set(finished) == keys and all(
                    finished[key].get("status") in {"complete", "complete_with_skips"} and
                    finished[key].get("output_accounting", {}).get("complete") is True
                    for key in keys))
                result_state = "finished" if complete else "paused_worker_error"
                atomic_json(Path(worker["status"]), {"state": result_state, "provider": PROVIDER,
                            "model_id": MODEL_ID, "supervisor_pid": child.pid, "exit_code": code,
                            "accounted_keys": len(set(finished) & keys), "expected_keys": len(keys),
                            "queue": worker["queue"], "journal": worker["journal"],
                            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
                pending.pop(child.pid)
                if not complete:
                    # A supervisor can exit zero after a queue pause; do not label
                    # that outcome finished, and stop the sibling safely.
                    results = terminate_and_reap()
                    for other in workers:
                        status_path = Path(other["status"])
                        if status_path.exists() and json.loads(status_path.read_text()).get("state") == "running":
                            atomic_json(status_path, {"state": "stopped_sibling_failure", "reaped": results})
                    pending.clear()
                    break
            status("running", active_workers=len(pending), supervisor_pids=active_pids.copy(),
                   completed_supervisors=codes.copy(), model_metadata=metadata)
            if pending:
                time.sleep(0.5)
        end_state = "finished" if len(codes) == 2 and all(
            json.loads(Path(w["status"]).read_text()).get("state") == "finished" for w in workers
        ) else "paused_worker_error"
        status(end_state, active_workers=0, supervisor_results=codes, model_metadata=metadata)
        return 0 if end_state == "finished" else 1
    except KeyboardInterrupt:
        results = terminate_and_reap()
        status("stopped", active_workers=0, reaped_supervisors=results)
        return 130
    except PermissionError as exc:
        status("paused_setup_failure", error_type=type(exc).__name__, reason=str(exc), active_workers=0)
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        results = terminate_and_reap()
        state = "paused_model_availability" if "metadata" in str(exc).lower() or "listed" in str(exc).lower() else "paused_setup_failure"
        status(state, error_type=type(exc).__name__, active_workers=0, reaped_supervisors=results)
        print(f"Zen launch paused: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
