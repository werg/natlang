#!/usr/bin/env python3
"""Run the fixed protected task packet against a strict direct-SFT checkpoint.

The evaluator exports through the shared strict direct exporter, serves through
the existing Neuralese HTTP engine, and executes authored task roots with the
frozen TS collector. It is CPU-only and does not train or admit data.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
from natlang_neuralese.eval.task_packet import (  # noqa: E402
    read_jsonl,
    retain_best_checkpoint,
    score_rows,
)
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    pending.replace(path)


def snapshot_file(source: Path, destination: Path) -> str:
    """Copy and verify immutable provenance bytes before an evaluator consumes them."""
    before = source.stat()
    digest = sha256_file_hex(source)
    shutil.copyfile(source, destination)
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    after = source.stat()
    copied = sha256_file_hex(destination)
    if (digest != copied or before.st_ino != after.st_ino or before.st_size != after.st_size or
            before.st_mtime_ns != after.st_mtime_ns):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"source changed while immutable evaluator snapshot was copied: {source}")
    return copied


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("could not read host MemAvailable")


def verify_plan(plan_path: Path) -> tuple[dict, Path, Path]:
    plan_path = plan_path.resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["plan_path"] = str(plan_path)
    if plan.get("schema") != "natlang.direct-held-task-evaluation/1":
        raise ValueError("unsupported direct held-task evaluation plan")
    plan["prompt_rendering"] = plan.get("prompt_rendering", "full")
    if plan["prompt_rendering"] not in {"full", "program-header"}:
        raise ValueError("plan declares an unsupported prompt_rendering mode")
    packet = Path(plan["derived_packet"]).resolve(strict=True)
    if sha256_file_hex(packet) != plan["derived_packet_sha256"]:
        raise ValueError("derived held-task packet hash changed")
    source = Path(plan["source_packet"]).resolve(strict=True)
    if sha256_file_hex(source) != plan["source_packet_sha256"]:
        raise ValueError("protected held-task packet hash changed")
    runtime = Path(plan["runtime"]).resolve(strict=True)
    if sha256_file_hex(runtime / "frozen-runtime.json") != plan["runtime_manifest_sha256"]:
        raise ValueError("frozen collector runtime manifest changed")
    queue_template = Path(plan["queue"]).resolve(strict=True)
    if sha256_file_hex(queue_template) != plan["queue_sha256"]:
        raise ValueError("reviewed queue template hash changed")
    supervisor = Path(plan["supervisor"]).resolve(strict=True)
    if sha256_file_hex(supervisor) != plan["supervisor_sha256"]:
        raise ValueError("pinned queue supervisor changed")
    reference_results = Path(plan["reference_results"]).resolve(strict=True)
    if sha256_file_hex(reference_results) != plan["reference_results_sha256"]:
        raise ValueError("held-task reference results changed")
    rows = read_jsonl(packet)
    if len(rows) != plan.get("derived_rows") or len(rows) != 48:
        raise ValueError("held-task packet must contain its exact 48 pinned rows")
    for row in rows:
        contract = (row.get("curriculum") or {}).get("trajectory_contract") or {}
        root_eval = next((args for kind, args in (row.get("curriculum", {}).get("reference", {}).get("root") or [])
                          if kind == "eval"), None)
        if (contract.get("authored_root_eval") is not True or not isinstance(root_eval, dict) or
                hashlib.sha256(root_eval.get("code", "").encode()).hexdigest() !=
                contract.get("authored_root_sha256")):
            raise ValueError(f"authored root contract is invalid: {row.get('id')}")
    return plan, packet, runtime


def comparison_key(plan: dict, image: str) -> str:
    """Bind best-checkpoint comparisons to one packet/runtime/inference contract."""
    threads = int(plan.get("server_threads", 2))
    from natlang_neuralese.data.prompt_rendering import prompt_rendering_identity
    contract = {
        "schema": "natlang.direct-task-comparison/1",
        "derived_packet_sha256": plan["derived_packet_sha256"],
        "protected_packet_sha256": plan["source_packet_sha256"],
        "reference_results_sha256": plan["reference_results_sha256"],
        "runtime_manifest_sha256": plan["runtime_manifest_sha256"],
        "base_snapshot_revision": plan["base_snapshot_dirname"],
        "runtime_image": image,
        "serving_precision": f"{plan.get('serving_dtype', 'float32').upper()} on CPU",
        "threads": threads,
        "server_cpu_count": int(plan.get("server_cpu_count", 2)),
        "server_memory_gib": int(plan.get("server_memory_gib", 4)),
        "sampling": plan["sampling"],
        "queue_entries": plan["queue_entries"],
        "max_case_seconds": plan["max_case_seconds"],
        "execution_plans": plan["execution_plans"],
        "prompt_rendering": prompt_rendering_identity(plan["prompt_rendering"]),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_run_queue(plan: dict, packet: Path, run_dir: Path) -> tuple[Path, list[dict]]:
    template_queue = Path(plan["queue"])
    template_rows = read_jsonl(template_queue)
    if len(template_rows) != 48:
        raise ValueError("prepared queue does not contain 48 single-case entries")
    queue_dir = run_dir / "worker-01"
    queue_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, original in enumerate(template_rows):
        if original.get("index") != index or original.get("source") != str(packet) or original.get("count") != 1:
            raise ValueError("prepared queue identities or single-case shape changed")
        case = queue_dir / f"case-{index:04d}"
        entry = dict(original)
        entry.update({
            "jobs": str(case / "jobs"),
            "log": str(case / "collector.log"),
            "output": str(case / "results.jsonl"),
        })
        entries.append(entry)
    queue = queue_dir / "queue.jsonl"
    queue.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
    return queue, entries


def _queue_entry_contract(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if key not in {"jobs", "log", "output"}}


def collect_existing(plan: dict, packet: Path, queue_path: Path, journal_path: Path,
                     entries: list[dict], out: Path) -> dict:
    journal = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
    finishes = {row["key"]: row for row in journal if row.get("event") == "finish"}
    rows = read_jsonl(packet)
    result_rows, queue_keys = [], {}
    exact_accounting = True
    for index, entry in enumerate(entries):
        queue_keys[index] = entry["key"]
        finish = finishes.get(entry["key"])
        if not finish or finish.get("status") not in {"complete", "complete_with_skips"}:
            exact_accounting = False
        accounting = finish.get("output_accounting") if finish else None
        if not isinstance(accounting, dict) or accounting.get("complete") is not True:
            exact_accounting = False
        output = Path(entry["output"])
        if output.is_file():
            result_rows.extend(read_jsonl(output))
    score = score_rows(rows, result_rows, finishes, queue_keys)
    score["supervisor_accounting_complete"] = exact_accounting
    score["eligible_for_best"] = score["eligible_for_best"] and exact_accounting
    queue_digest = sha256_file_hex(queue_path)
    journal_digest = sha256_file_hex(journal_path)
    report = {
        **score,
        "plan_sha256": sha256_file_hex(Path(plan["plan_path"])) if plan.get("plan_path") else None,
        "derived_packet_sha256": plan["derived_packet_sha256"],
        "protected_packet_sha256": plan["source_packet_sha256"],
        "reference_results_sha256": plan["reference_results_sha256"],
        "runtime_manifest_sha256": plan["runtime_manifest_sha256"],
        "queue_sha256": queue_digest,
        "journal_sha256": journal_digest,
        "expected_values_model_visible": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "task-score.json"
    atomic_json(report_path, report)
    return report


def export_in_container(checkpoint: Path, artifact: Path, base_snapshot: Path,
                        launch_receipt: Path, image: str, log_path: Path) -> None:
    container_name = "natlang-direct-task-export-" + hashlib.sha256(
        f"{checkpoint.resolve()}:{artifact.resolve()}".encode()).hexdigest()[:12]
    command = [
        "docker", "run", "--name", container_name, "--label", "org.natlang.direct-task-evaluation=held48",
        "--pull=never", "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g", "--memory=4g", "--memory-swap=4g", "--cpus=2",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "CUDA_VISIBLE_DEVICES=", "-e", "HOME=/tmp",
        "-e", f"PYTHONPATH={ROOT / 'training/neuralese'}",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-v", f"{ROOT}:{ROOT}:ro", "-v", "/srv/storage/natlang-artifacts:/srv/storage/natlang-artifacts:rw",
        "-w", str(ROOT), image, "python", "-m", "natlang_neuralese.eval.task_packet", "export",
        "--checkpoint", str(checkpoint), "--output", str(artifact),
        "--base-snapshot", str(base_snapshot), "--launch-receipt", str(launch_receipt),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                if available_memory_bytes() < 1024 ** 3:
                    subprocess.run(["docker", "stop", "--timeout", "10", container_name], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                    raise RuntimeError("host available RAM fell below 1 GiB during strict checkpoint export")
                time.sleep(5)
            if child.returncode != 0:
                raise subprocess.CalledProcessError(child.returncode, command)
    finally:
        state = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", container_name],
                               check=False, text=True, capture_output=True)
        if state.returncode == 0:
            if state.stdout.strip() == "true":
                subprocess.run(["docker", "stop", "--timeout", "10", container_name], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["docker", "rm", container_name], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_server(name: str, port: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    while True:
        if available_memory_bytes() < 1024 ** 3:
            raise RuntimeError("host available RAM fell below 1 GiB while waiting for task server")
        state = subprocess.run(["docker", "inspect", name, "--format", "{{.State.Running}}"],
                               check=False, text=True, capture_output=True)
        if state.returncode != 0 or state.stdout.strip() != "true":
            raise RuntimeError("direct task-evaluation server exited before becoming ready")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)


def verify_server_rendering(port: int, mode: str) -> None:
    from natlang_neuralese.data.prompt_rendering import prompt_rendering_identity
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/neuralese/info", timeout=5) as response:
        info = json.loads(response.read())
    if (info.get("prompt_rendering") != mode or
            info.get("prompt_rendering_identity_sha256") !=
            prompt_rendering_identity(mode)["identity_sha256"]):
        raise RuntimeError("serving prompt-rendering contract differs from the reviewed evaluation plan")


def available_floor_monitor(process: subprocess.Popen, floor_bytes: int,
                            server_name: str | None = None) -> tuple[int, str | None]:
    """Wait for the queue and stop it gracefully if host memory crosses the declared floor."""
    while process.poll() is None:
        available = available_memory_bytes()
        if available < floor_bytes:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return available, "host_memory_floor"
        if server_name is not None:
            state = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", server_name],
                                   check=False, text=True, capture_output=True)
            if state.returncode != 0 or state.stdout.strip() != "true":
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return available, "serving_container_exited"
        time.sleep(5)
    return available_memory_bytes(), None


class ContainerMemorySampler:
    """Sample a named evaluator's cgroup and attribute samples to queue entries."""

    def __init__(self, container: str, journal: Path, output: Path, interval: float = 1.0):
        self.container, self.journal, self.output, self.interval = container, journal, output, interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.cgroup: Path | None = None
        self.offset = 0
        self.active_key: str | None = None
        self.by_case: dict[str, dict] = {}
        self.error: str | None = None

    def _locate(self) -> None:
        inspected = subprocess.run(["docker", "inspect", "--format", "{{.State.Pid}}", self.container],
                                   check=True, text=True, capture_output=True)
        pid = int(inspected.stdout.strip())
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                self.cgroup = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                break
        if self.cgroup is None or not (self.cgroup / "memory.current").is_file():
            raise RuntimeError("could not locate evaluator container's cgroup v2 memory files")

    def start(self) -> None:
        try:
            self._locate()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            return
        self.thread = threading.Thread(target=self._sample, name="direct-eval-memory-sampler", daemon=True)
        self.thread.start()

    def _journal_events(self) -> None:
        if not self.journal.is_file():
            return
        with self.journal.open("r", encoding="utf-8") as stream:
            stream.seek(self.offset)
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "start":
                    self.active_key = event.get("key")
                elif event.get("event") in {"finish", "cancel"} and event.get("key") == self.active_key:
                    self.active_key = None
            self.offset = stream.tell()

    def _sample(self) -> None:
        assert self.cgroup is not None
        while not self.stop_event.is_set():
            try:
                self._journal_events()
                current = int((self.cgroup / "memory.current").read_text().strip())
                peak = int((self.cgroup / "memory.peak").read_text().strip())
                event_values = {}
                for line in (self.cgroup / "memory.events").read_text().splitlines():
                    name, value = line.split()
                    event_values[name] = int(value)
                key = self.active_key or "idle"
                case = self.by_case.setdefault(key, {"sample_count": 0, "sampled_peak_current_bytes": 0})
                case["sample_count"] += 1
                case["sampled_peak_current_bytes"] = max(case["sampled_peak_current_bytes"], current)
                case["cgroup_high_water_bytes"] = peak
                case["last_memory_events"] = event_values
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.error = f"{type(error).__name__}: {error}"
                break
            self.stop_event.wait(self.interval)

    def finish(self) -> dict:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        result = {"container": self.container, "cgroup": str(self.cgroup) if self.cgroup else None,
                  "sampling_interval_seconds": self.interval, "per_queue_entry_sampled_memory": self.by_case}
        if self.error:
            result["sampling_error"] = self.error
        if self.cgroup and self.cgroup.exists():
            try:
                result["final_memory_events"] = {
                    name: int(value) for name, value in
                    (line.split() for line in (self.cgroup / "memory.events").read_text().splitlines())
                }
                result["cgroup_peak_bytes"] = int((self.cgroup / "memory.peak").read_text().strip())
            except (OSError, ValueError):
                pass
        atomic_json(self.output, result)
        return result


def run_reviewed_queue(plan: dict, packet: Path, runtime: Path, run_dir: Path,
                       server_name: str, port: int) -> dict:
    """Execute the same fixed, single-case queue against one already-ready server."""
    queue, entries = make_run_queue(plan, packet, run_dir)
    journal = queue.parent / "journal.jsonl"
    sampler = ContainerMemorySampler(server_name, journal, run_dir / "memory-telemetry.json")
    sampler.start()
    command = [
        sys.executable, str(ROOT / "scripts/run_bonsai_queue.py"), str(queue), str(journal),
        "--runtime", str(runtime), "--server", f"http://127.0.0.1:{port}",
        "--chat-completions-url", f"http://127.0.0.1:{port}/v1/chat/completions",
        "--model-concurrency", "1", "--min-free-mib", "1024",
    ]
    try:
        child = subprocess.Popen(command, cwd=ROOT)
        available, stop_reason = available_floor_monitor(child, 1024 ** 3, server_name)
        report = collect_existing(plan, packet, queue, journal, entries, run_dir)
        report["collector_exit_code"] = child.returncode
        report["available_ram_after_collection_bytes"] = available
        report["queue_stop_reason"] = stop_reason
        return report
    finally:
        sampler.finish()


def run_candidate(args: argparse.Namespace) -> dict:
    plan, packet, runtime = verify_plan(args.plan)
    launch = args.launch_receipt.resolve(strict=True)
    base_snapshot = args.base_snapshot.resolve(strict=True)
    if base_snapshot != Path(plan["base_snapshot"]).resolve(strict=True):
        raise ValueError("candidate base snapshot differs from the reviewed evaluation plan")
    serving_dtype = plan.get("serving_dtype", "float32")
    if serving_dtype not in {"float32", "bfloat16"}:
        raise ValueError("evaluation plan has an unsupported CPU serving dtype")
    server_threads = int(plan.get("server_threads", 2))
    server_cpu_count = int(plan.get("server_cpu_count", 2))
    if server_threads < 1 or server_cpu_count < 1:
        raise ValueError("evaluation plan CPU thread and container CPU limits must be positive")
    server_memory_gib = int(plan.get("server_memory_gib", args.server_memory_gib))
    if args.server_memory_gib is not None and args.server_memory_gib != server_memory_gib:
        raise ValueError("server memory limit differs from the reviewed evaluation plan")
    run_dir = args.out.resolve()
    if run_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {run_dir}")
    if available_memory_bytes() < args.start_floor_gib * (1024 ** 3):
        raise RuntimeError("available RAM is below the task-evaluation start floor")
    run_dir.mkdir(parents=True)
    launch_snapshot = run_dir / "launch-receipt.snapshot.json"
    launch_sha = snapshot_file(launch, launch_snapshot)
    copied_checkpoint = run_dir / "candidate-checkpoint.pt"
    source_stat = args.checkpoint.stat()
    checkpoint_sha = sha256_file_hex(args.checkpoint)
    shutil.copyfile(args.checkpoint, copied_checkpoint)
    copied_sha = sha256_file_hex(copied_checkpoint)
    source_after = args.checkpoint.stat()
    if (checkpoint_sha != copied_sha or source_stat.st_ino != source_after.st_ino or
            source_stat.st_size != source_after.st_size or source_stat.st_mtime_ns != source_after.st_mtime_ns):
        raise RuntimeError("rolling checkpoint changed while its immutable eval snapshot was copied")

    artifact = run_dir / "direct-artifact"
    export_in_container(copied_checkpoint, artifact, base_snapshot, launch_snapshot, args.image,
                        run_dir / "direct-export.log")
    if available_memory_bytes() < 1024 ** 3:
        raise RuntimeError("host available RAM fell below 1 GiB after strict checkpoint export")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_binding = manifest.get("prompt_rendering") or {}
    if prompt_binding.get("mode") != plan["prompt_rendering"]:
        raise ValueError("checkpoint text source rendering mode differs from evaluation plan")
    port = free_port()
    server_name = "natlang-direct-task-eval-" + hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]
    server = [
        "docker", "run", "-d", "--name", server_name, "--label", "org.natlang.direct-task-evaluation=held48",
        "--network", "host",
        f"--memory={server_memory_gib}g", f"--memory-swap={server_memory_gib}g", f"--cpus={server_cpu_count}",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "CUDA_VISIBLE_DEVICES=", "-e", "HOME=/tmp",
        "-e", f"PYTHONPATH={ROOT / 'training/neuralese'}",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-e", f"HF_HOME={Path(plan['base_snapshot']).parents[3]}",
        "-e", f"HF_HUB_CACHE={Path(plan['base_snapshot']).parents[3] / 'hub'}",
        "-v", f"{ROOT}:{ROOT}:ro", "-v", "/srv/storage/natlang-artifacts:/srv/storage/natlang-artifacts:ro",
        "-w", str(ROOT), args.image, "python", "-m", "natlang_neuralese.serve",
        "--direct-artifact", str(artifact), "--direct-base-snapshot", str(base_snapshot),
        "--device", "cpu", "--dtype", serving_dtype, "--threads", str(server_threads), "--port", str(port),
    ]
    try:
        subprocess.run(server, check=True, text=True, capture_output=True)
        wait_server(server_name, port)
        verify_server_rendering(port, plan["prompt_rendering"])
        report = run_reviewed_queue(plan, packet, runtime, run_dir, server_name, port)
        runner_code = report["collector_exit_code"]
        report.update({
            "checkpoint_sha256": copied_sha,
            "launch_receipt_snapshot_sha256": launch_sha,
            "training_step": (manifest.get("training_checkpoint") or {}).get("step"),
            "direct_artifact_manifest_sha256": sha256_file_hex(manifest_path),
            "base_snapshot": str(base_snapshot),
            "base_snapshot_revision": base_snapshot.name,
            "serving_precision": f"{serving_dtype.upper()} on CPU",
            "prompt_rendering": prompt_binding.get("mode"),
            "runtime_image": args.image,
            "threads": server_threads,
            "server_cpu_count": server_cpu_count,
            "comparison_key": comparison_key(plan, args.image),
        })
        atomic_json(run_dir / "task-score.json", report)
        if runner_code == 0:
            score_sha = sha256_file_hex(run_dir / "task-score.json")
            report["score_sha256"] = score_sha
            promotion = retain_best_checkpoint(copied_checkpoint, report, score_sha, args.best_dir)
            report["best_checkpoint_update"] = promotion
            atomic_json(run_dir / "task-score.json", report)
        else:
            report["eligible_for_best"] = False
            report["best_checkpoint_update"] = {"promoted": False, "reason": "collector did not close the full queue"}
            atomic_json(run_dir / "task-score.json", report)
        return report
    finally:
        inspected = subprocess.run(["docker", "inspect", "--format", "{{json .State}}", server_name],
                                   check=False, text=True, capture_output=True)
        state = None
        if inspected.returncode == 0:
            try:
                state = json.loads(inspected.stdout)
            except json.JSONDecodeError:
                state = {"inspect_error": inspected.stdout}
        if isinstance(state, dict) and state.get("Running"):
            subprocess.run(["docker", "stop", "--timeout", "30", server_name], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logs = subprocess.run(["docker", "logs", server_name], check=False, text=True, capture_output=True)
        if logs.stdout or logs.stderr:
            (run_dir / "server.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
        atomic_json(run_dir / "server-state.json", {"container": server_name, "state": state,
                                                    "docker_inspect_exit_code": inspected.returncode})
        subprocess.run(["docker", "rm", server_name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # The strict export is only needed while evaluating. Keep the exact
        # optimizer-bearing checkpoint and report; re-export from those pins
        # when the retained candidate is served later.
        shutil.rmtree(artifact, ignore_errors=True)
        best_path = Path(args.best_dir) / "best-task-checkpoint.pt"
        if not best_path.exists() or sha256_file_hex(best_path) != copied_sha:
            copied_checkpoint.unlink(missing_ok=True)


def run_candidate_serialized(args: argparse.Namespace) -> dict:
    plan, _packet, _runtime = verify_plan(args.plan)
    lock_path = Path(plan["derived_packet"]).resolve().parent / ".task-evaluation-run.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return run_candidate(args)


def run_base(args: argparse.Namespace) -> dict:
    """Score the pinned HF base through the same serve engine and reviewed queue."""
    plan, packet, runtime = verify_plan(args.plan)
    base_snapshot = Path(plan["base_snapshot"]).resolve(strict=True)
    serving_dtype = plan.get("serving_dtype", "float32")
    if serving_dtype not in {"float32", "bfloat16"}:
        raise ValueError("evaluation plan has an unsupported CPU serving dtype")
    server_threads = int(plan.get("server_threads", 2))
    server_cpu_count = int(plan.get("server_cpu_count", 2))
    if server_threads < 1 or server_cpu_count < 1:
        raise ValueError("evaluation plan CPU thread and container CPU limits must be positive")
    server_memory_gib = int(plan.get("server_memory_gib", args.server_memory_gib))
    if args.server_memory_gib is not None and args.server_memory_gib != server_memory_gib:
        raise ValueError("server memory limit differs from the reviewed evaluation plan")
    run_dir = args.out.resolve()
    if run_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {run_dir}")
    if available_memory_bytes() < args.start_floor_gib * (1024 ** 3):
        raise RuntimeError("available RAM is below the task-evaluation start floor")
    run_dir.mkdir(parents=True)
    port = free_port()
    server_name = "natlang-direct-task-base-" + hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]
    command = [
        "docker", "run", "-d", "--name", server_name, "--label", "org.natlang.direct-task-evaluation=held48",
        "--network", "host",
        f"--memory={server_memory_gib}g", f"--memory-swap={server_memory_gib}g", f"--cpus={server_cpu_count}",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "CUDA_VISIBLE_DEVICES=", "-e", "HOME=/tmp",
        "-e", f"PYTHONPATH={ROOT / 'training/neuralese'}",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-e", f"HF_HOME={base_snapshot.parents[3]}",
        "-e", f"HF_HUB_CACHE={base_snapshot.parents[3] / 'hub'}",
        "-v", f"{ROOT}:{ROOT}:ro", "-v", "/srv/storage/natlang-artifacts:/srv/storage/natlang-artifacts:ro",
        "-w", str(ROOT), args.image, "python", "-m", "natlang_neuralese.serve",
        "--base", str(base_snapshot), "--device", "cpu", "--dtype", serving_dtype,
        "--threads", str(server_threads), "--port", str(port),
        "--prompt-rendering", plan["prompt_rendering"],
    ]
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
        wait_server(server_name, port)
        verify_server_rendering(port, plan["prompt_rendering"])
        report = run_reviewed_queue(plan, packet, runtime, run_dir, server_name, port)
        report.update({
            "evaluation_kind": "pinned-base-fixed-held-task-packet",
            "plan_sha256": sha256_file_hex(Path(plan["plan_path"])),
            "base_snapshot": str(base_snapshot),
            "base_snapshot_revision": base_snapshot.name,
            "serving_precision": f"{serving_dtype.upper()} on CPU",
            "prompt_rendering": plan["prompt_rendering"],
            "runtime_image": args.image,
            "threads": server_threads,
            "server_cpu_count": server_cpu_count,
            "comparison_key": comparison_key(plan, args.image),
        })
        report_path = run_dir / "task-score.json"
        atomic_json(report_path, report)
        report["score_sha256"] = sha256_file_hex(report_path)
        atomic_json(report_path, report)
        return report
    finally:
        inspected = subprocess.run(["docker", "inspect", "--format", "{{json .State}}", server_name],
                                   check=False, text=True, capture_output=True)
        state = None
        if inspected.returncode == 0:
            try:
                state = json.loads(inspected.stdout)
            except json.JSONDecodeError:
                state = {"inspect_error": inspected.stdout}
        if isinstance(state, dict) and state.get("Running"):
            subprocess.run(["docker", "stop", "--timeout", "30", server_name], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logs = subprocess.run(["docker", "logs", server_name], check=False, text=True, capture_output=True)
        if logs.stdout or logs.stderr:
            (run_dir / "server.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
        atomic_json(run_dir / "server-state.json", {"container": server_name, "state": state,
                                                    "docker_inspect_exit_code": inspected.returncode})
        subprocess.run(["docker", "rm", server_name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_base_serialized(args: argparse.Namespace) -> dict:
    plan, _packet, _runtime = verify_plan(args.plan)
    lock_path = Path(plan["derived_packet"]).resolve().parent / ".task-evaluation-run.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return run_base(args)


def score_existing(args: argparse.Namespace) -> dict:
    plan, packet, _runtime = verify_plan(args.plan)
    queue = args.queue.resolve(strict=True)
    entries = read_jsonl(queue)
    reviewed_entries = read_jsonl(Path(plan["queue"]))
    if len(entries) != plan["queue_entries"]:
        raise ValueError("existing queue does not match the reviewed fixed packet size")
    for index, entry in enumerate(entries):
        if (entry.get("index") != index or entry.get("source") != str(packet) or entry.get("count") != 1 or
                _queue_entry_contract(entry) != _queue_entry_contract(reviewed_entries[index])):
            raise ValueError("existing queue identity differs from the reviewed packet")
    report = collect_existing(plan, packet, queue, args.journal.resolve(strict=True), entries, args.out)
    report["comparison_key"] = comparison_key(plan, args.image)
    report["evaluation_kind"] = "fixed-packet-existing-run"
    atomic_json(args.out / "task-score.json", report)
    return report


def _latest_eval_step(path: Path) -> int:
    if not path.is_file():
        return 0
    last = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
    step = (last or {}).get("step", 0)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"invalid step in training evaluation log: {path}")
    return step


def _phase_complete(report_path: Path) -> bool:
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(report, dict) and report.get("status") == "complete"


def _append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def watch_phases(args: argparse.Namespace) -> dict:
    """Serially evaluate only the newest stable checkpoint in each declared phase."""
    plan, _packet, _runtime = verify_plan(args.plan)
    if not args.phase:
        raise ValueError("at least one --phase NAME CHECKPOINT REPORT EVAL_LOG is required")
    if args.poll_seconds < 1 or args.minimum_step_interval < 1 or args.retry_base_seconds < 1:
        raise ValueError("poll interval, checkpoint cadence, and retry base must be positive")
    if args.maximum_retry_seconds < args.retry_base_seconds:
        raise ValueError("maximum retry delay must be at least the retry base delay")
    phase_names = [phase[0] for phase in args.phase]
    if len(set(phase_names)) != len(phase_names):
        raise ValueError("phase names must be unique")
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "watch-state.json"
    event_path = out_root / "watch-events.jsonl"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema") != "natlang.direct-task-checkpoint-watch/1":
            raise ValueError("unsupported task-evaluation watcher state")
    else:
        state = {"schema": "natlang.direct-task-checkpoint-watch/1", "phase_index": 0,
                 "phases": {}, "candidate_count": 0}
        atomic_json(state_path, state)

    def save_state() -> None:
        state["updated_at"] = time.time()
        atomic_json(state_path, state)

    while state["phase_index"] < len(args.phase):
        name, checkpoint_text, report_text, eval_text = args.phase[state["phase_index"]]
        checkpoint, report_path, eval_path = map(Path, (checkpoint_text, report_text, eval_text))
        phase_state = state["phases"].setdefault(name, {
            "last_successful_step": 0, "last_evaluated_signature": None,
            "retry_count": 0, "next_retry_at": 0.0,
        })
        if not checkpoint.is_file():
            time.sleep(args.poll_seconds)
            continue

        stat = checkpoint.stat()
        signature = f"{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"
        last_seen = phase_state.get("last_seen_signature")
        if signature == last_seen:
            phase_state["stable_polls"] = int(phase_state.get("stable_polls", 0)) + 1
        else:
            phase_state["last_seen_signature"] = signature
            phase_state["stable_polls"] = 1
            save_state()
        if phase_state["stable_polls"] < 2:
            time.sleep(args.poll_seconds)
            continue

        is_final = _phase_complete(report_path)
        current_step = _latest_eval_step(eval_path)
        already_evaluated = phase_state.get("last_evaluated_signature") == signature
        periodic_due = current_step >= int(phase_state.get("last_successful_step", 0)) + args.minimum_step_interval
        if already_evaluated:
            if is_final:
                state["phase_index"] += 1
                save_state()
                continue
            time.sleep(args.poll_seconds)
            continue
        if not (is_final or periodic_due):
            time.sleep(args.poll_seconds)
            continue
        now = time.time()
        if now < float(phase_state.get("next_retry_at", 0.0)):
            time.sleep(min(args.poll_seconds, float(phase_state["next_retry_at"]) - now))
            continue
        if available_memory_bytes() < args.start_floor_gib * (1024 ** 3):
            _append_event(event_path, {"event": "deferred_low_memory", "phase": name,
                                       "available_ram_bytes": available_memory_bytes(),
                                       "required_floor_gib": args.start_floor_gib,
                                       "checkpoint_signature": signature, "time": time.time()})
            time.sleep(args.poll_seconds)
            continue

        state["candidate_count"] += 1
        candidate_no = state["candidate_count"]
        candidate_out = out_root / f"candidate-{candidate_no:04d}-{name}"
        _append_event(event_path, {"event": "evaluation_started", "phase": name,
                                   "candidate": str(candidate_out), "checkpoint_signature": signature,
                                   "training_eval_step": current_step, "final_phase_snapshot": is_final,
                                   "time": time.time()})
        save_state()
        candidate_args = argparse.Namespace(
            plan=args.plan, checkpoint=checkpoint, launch_receipt=args.launch_receipt,
            base_snapshot=args.base_snapshot, out=candidate_out, best_dir=args.best_dir,
            image=args.image, server_memory_gib=args.server_memory_gib,
            start_floor_gib=args.start_floor_gib,
        )
        try:
            evaluation = run_candidate_serialized(candidate_args)
            if not evaluation.get("eligible_for_best"):
                raise RuntimeError("candidate evaluation did not produce a complete, comparable task score")
            evaluated_step = evaluation.get("training_step")
            if isinstance(evaluated_step, bool) or not isinstance(evaluated_step, int) or evaluated_step < 0:
                raise RuntimeError("strict export did not report an exact integer training step")
            phase_state.update({
                "last_successful_step": evaluated_step,
                "last_evaluated_signature": signature,
                "retry_count": 0,
                "next_retry_at": 0.0,
            })
            _append_event(event_path, {"event": "evaluation_complete", "phase": name,
                                       "candidate": str(candidate_out), "training_step": evaluated_step,
                                       "task_successes": evaluation.get("task_successes"),
                                       "outcome_counts": evaluation.get("outcome_counts"),
                                       "best_checkpoint_update": evaluation.get("best_checkpoint_update"),
                                       "time": time.time()})
            save_state()
            if is_final:
                state["phase_index"] += 1
                save_state()
        except Exception as error:
            shutil.rmtree(candidate_out / "direct-artifact", ignore_errors=True)
            (candidate_out / "candidate-checkpoint.pt").unlink(missing_ok=True)
            failures = int(phase_state.get("retry_count", 0)) + 1
            delay = min(args.maximum_retry_seconds, args.retry_base_seconds * (2 ** min(failures - 1, 8)))
            phase_state.update({"retry_count": failures, "next_retry_at": time.time() + delay})
            _append_event(event_path, {"event": "evaluation_retry", "phase": name,
                                       "candidate": str(candidate_out), "failure_count": failures,
                                       "retry_after_seconds": delay,
                                       "error": f"{type(error).__name__}: {error}", "time": time.time()})
            save_state()
            time.sleep(min(args.poll_seconds, delay))

    summary = {"schema": state["schema"], "complete": True,
               "phases_completed": [phase[0] for phase in args.phase],
               "candidate_count": state["candidate_count"], "state_path": str(state_path),
               "event_path": str(event_path)}
    atomic_json(out_root / "watch-complete.json", summary)
    return summary


def main() -> None:
    def clean_exit(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, clean_exit)
    signal.signal(signal.SIGINT, clean_exit)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidate = sub.add_parser("evaluate-checkpoint", help="evaluate a strict direct-SFT checkpoint")
    candidate.add_argument("--plan", type=Path, required=True)
    candidate.add_argument("--checkpoint", type=Path, required=True)
    candidate.add_argument("--launch-receipt", type=Path, required=True)
    candidate.add_argument("--base-snapshot", type=Path, required=True)
    candidate.add_argument("--out", type=Path, required=True, help="fresh candidate evaluation directory")
    candidate.add_argument("--best-dir", type=Path, required=True)
    candidate.add_argument("--image", default="sha256:6b337ae8eb936191c4cba64641aeb0dfcb0658ae3fadb8ede6be03c8a6ccb3de")
    candidate.add_argument("--server-memory-gib", type=int, default=None)
    candidate.add_argument("--start-floor-gib", type=int, default=4)
    base = sub.add_parser("evaluate-base", help="score the pinned HF base on the fixed held-task packet")
    base.add_argument("--plan", type=Path, required=True)
    base.add_argument("--out", type=Path, required=True)
    base.add_argument("--image", default="sha256:6b337ae8eb936191c4cba64641aeb0dfcb0658ae3fadb8ede6be03c8a6ccb3de")
    base.add_argument("--server-memory-gib", type=int, default=None)
    base.add_argument("--start-floor-gib", type=int, default=4)
    baseline = sub.add_parser("score-existing", help="score an already completed fixed-packet queue run")
    baseline.add_argument("--plan", type=Path, required=True)
    baseline.add_argument("--queue", type=Path, required=True)
    baseline.add_argument("--journal", type=Path, required=True)
    baseline.add_argument("--out", type=Path, required=True)
    baseline.add_argument("--image", default="sha256:6b337ae8eb936191c4cba64641aeb0dfcb0658ae3fadb8ede6be03c8a6ccb3de")
    watcher = sub.add_parser("watch-phases", help="evaluate newest stable checkpoints across ordered direct-SFT phases")
    watcher.add_argument("--plan", type=Path, required=True)
    watcher.add_argument("--launch-receipt", type=Path, required=True)
    watcher.add_argument("--base-snapshot", type=Path, required=True)
    watcher.add_argument("--out-root", type=Path, required=True)
    watcher.add_argument("--best-dir", type=Path, required=True)
    watcher.add_argument("--phase", action="append", nargs=4,
                         metavar=("NAME", "CHECKPOINT", "REPORT", "EVAL_LOG"), required=True)
    watcher.add_argument("--image", default="sha256:6b337ae8eb936191c4cba64641aeb0dfcb0658ae3fadb8ede6be03c8a6ccb3de")
    watcher.add_argument("--server-memory-gib", type=int, default=None)
    watcher.add_argument("--start-floor-gib", type=int, default=4)
    watcher.add_argument("--poll-seconds", type=int, default=30)
    watcher.add_argument("--minimum-step-interval", type=int, default=1000)
    watcher.add_argument("--retry-base-seconds", type=int, default=60)
    watcher.add_argument("--maximum-retry-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.command == "evaluate-checkpoint":
        result = run_candidate_serialized(args)
    elif args.command == "evaluate-base":
        result = run_base_serialized(args)
    elif args.command == "score-existing":
        result = score_existing(args)
    else:
        result = watch_phases(args)
    output_path = getattr(args, "out", result.get("state_path", result.get("event_path")))
    print(json.dumps({"out": str(output_path), "task_successes": result.get("task_successes"),
                      "outcome_counts": result.get("outcome_counts"),
                      "best_checkpoint_update": result.get("best_checkpoint_update")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
