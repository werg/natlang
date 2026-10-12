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
import socket
import subprocess
import sys
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
    contract = {
        "schema": "natlang.direct-task-comparison/1",
        "derived_packet_sha256": plan["derived_packet_sha256"],
        "protected_packet_sha256": plan["source_packet_sha256"],
        "reference_results_sha256": plan["reference_results_sha256"],
        "runtime_manifest_sha256": plan["runtime_manifest_sha256"],
        "base_snapshot_revision": plan["base_snapshot_dirname"],
        "runtime_image": image,
        "serving_precision": "FP32 on CPU",
        "threads": 2,
        "sampling": plan["sampling"],
        "queue_entries": plan["queue_entries"],
        "max_case_seconds": plan["max_case_seconds"],
        "execution_plans": plan["execution_plans"],
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
                        launch_receipt: Path, image: str) -> None:
    command = [
        "docker", "run", "--rm", "--pull=never", "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g", "--memory=2g", "--cpus=2",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "CUDA_VISIBLE_DEVICES=", "-e", "HOME=/tmp",
        "-e", f"PYTHONPATH={ROOT / 'training/neuralese'}",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-v", f"{ROOT}:{ROOT}:ro", "-v", "/srv/storage/natlang-artifacts:/srv/storage/natlang-artifacts:rw",
        "-w", str(ROOT), image, "python", "-m", "natlang_neuralese.eval.task_packet", "export",
        "--checkpoint", str(checkpoint), "--output", str(artifact),
        "--base-snapshot", str(base_snapshot), "--launch-receipt", str(launch_receipt),
    ]
    subprocess.run(command, check=True)


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


def available_floor_monitor(process: subprocess.Popen, floor_bytes: int) -> int:
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
            return available
        time.sleep(5)
    return available_memory_bytes()


def run_candidate(args: argparse.Namespace) -> dict:
    plan, packet, runtime = verify_plan(args.plan)
    launch = args.launch_receipt.resolve(strict=True)
    base_snapshot = args.base_snapshot.resolve(strict=True)
    if base_snapshot != Path(plan["base_snapshot"]).resolve(strict=True):
        raise ValueError("candidate base snapshot differs from the reviewed evaluation plan")
    run_dir = args.out.resolve()
    if run_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {run_dir}")
    if available_memory_bytes() < args.start_floor_gib * (1024 ** 3):
        raise RuntimeError("available RAM is below the task-evaluation start floor")
    run_dir.mkdir(parents=True)
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
    export_in_container(copied_checkpoint, artifact, base_snapshot, launch, args.image)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    port = free_port()
    server_name = "natlang-direct-task-eval-" + hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]
    server = [
        "docker", "run", "-d", "--rm", "--name", server_name, "--network", "host",
        f"--memory={args.server_memory_gib}g", "--cpus=2", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "CUDA_VISIBLE_DEVICES=", "-e", "HOME=/tmp",
        "-e", f"PYTHONPATH={ROOT / 'training/neuralese'}",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-e", f"HF_HOME={Path(plan['base_snapshot']).parents[3]}",
        "-e", f"HF_HUB_CACHE={Path(plan['base_snapshot']).parents[3] / 'hub'}",
        "-v", f"{ROOT}:{ROOT}:ro", "-v", "/srv/storage/natlang-artifacts:/srv/storage/natlang-artifacts:ro",
        "-w", str(ROOT), args.image, "python", "-m", "natlang_neuralese.serve",
        "--direct-artifact", str(artifact), "--direct-base-snapshot", str(base_snapshot),
        "--device", "cpu", "--threads", "2", "--port", str(port),
    ]
    subprocess.run(server, check=True, text=True, capture_output=True)
    try:
        wait_server(server_name, port)
        queue, entries = make_run_queue(plan, packet, run_dir)
        journal = queue.parent / "journal.jsonl"
        command = [
            sys.executable, str(ROOT / "scripts/run_bonsai_queue.py"), str(queue), str(journal),
            "--runtime", str(runtime), "--server", f"http://127.0.0.1:{port}",
            "--chat-completions-url", f"http://127.0.0.1:{port}/v1/chat/completions",
            "--model-concurrency", "1", "--min-free-mib", "1024",
        ]
        child = subprocess.Popen(command, cwd=ROOT)
        available = available_floor_monitor(child, 1024 ** 3)
        runner_code = child.returncode
        report = collect_existing(plan, packet, queue, journal, entries, run_dir)
        report.update({
            "checkpoint_sha256": copied_sha,
            "training_step": (manifest.get("training_checkpoint") or {}).get("step"),
            "direct_artifact_manifest_sha256": sha256_file_hex(manifest_path),
            "base_snapshot": str(base_snapshot),
            "base_snapshot_revision": base_snapshot.name,
            "serving_precision": "FP32 on CPU",
            "runtime_image": args.image,
            "threads": 2,
            "comparison_key": comparison_key(plan, args.image),
            "collector_exit_code": runner_code,
            "available_ram_after_collection_bytes": available,
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
        subprocess.run(["docker", "stop", "--time", "30", server_name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # The strict export is only needed while evaluating. Keep the exact
        # optimizer-bearing checkpoint and report; re-export from those pins
        # when the retained candidate is served later.
        shutil.rmtree(artifact, ignore_errors=True)
        best_path = Path(args.best_dir) / "best-task-checkpoint.pt"
        if not best_path.exists() or sha256_file_hex(best_path) != copied_sha:
            copied_checkpoint.unlink(missing_ok=True)


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


def main() -> None:
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
    candidate.add_argument("--server-memory-gib", type=int, default=2)
    candidate.add_argument("--start-floor-gib", type=int, default=3)
    baseline = sub.add_parser("score-existing", help="score an already completed fixed-packet queue run")
    baseline.add_argument("--plan", type=Path, required=True)
    baseline.add_argument("--queue", type=Path, required=True)
    baseline.add_argument("--journal", type=Path, required=True)
    baseline.add_argument("--out", type=Path, required=True)
    baseline.add_argument("--image", default="sha256:6b337ae8eb936191c4cba64641aeb0dfcb0658ae3fadb8ede6be03c8a6ccb3de")
    args = parser.parse_args()
    if args.command == "evaluate-checkpoint":
        args.best_dir.mkdir(parents=True, exist_ok=True)
        with (args.best_dir / ".task-evaluation-run.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = run_candidate(args)
    else:
        result = score_existing(args)
    print(json.dumps({"out": str(args.out), "task_successes": result.get("task_successes"),
                      "outcome_counts": result.get("outcome_counts"),
                      "best_checkpoint_update": result.get("best_checkpoint_update")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
