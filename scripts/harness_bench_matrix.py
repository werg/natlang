#!/usr/bin/env python3
"""The harness-bench matrix (plans/neuralese/HARNESS_BENCH.md §6.3): pi's eval tasks with a student agent, alone and
with the text companion, on one student checkpoint.

Crisp plumbing and measurement only: it starts the executor (llama.cpp, a GGUF) and the student
(natlang_neuralese.serve) on free ports, runs `natlang run applications/pi -- eval` once per configuration, reads the
per-task results and session databases, and writes `matrix.json` and `matrix.md` under --out.

Run it under the memory ledger on DGX, e.g.

  python3 scripts/memory_ledger.py run --unit harness-matrix-base --budget-gb 14 --reserve-gb 2 -- \\
    python3 scripts/harness_bench_matrix.py --label lfm-350m-base \\
      --student-base ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/<rev> \\
      --out <data_nvme>/harness-bench/matrix-20261012/lfm-350m-base --minutes 10

`--student-artifact DIR` serves a direct export (with --student-base its pinned base snapshot). `--student-endpoint`
and `--executor-endpoint` use running servers instead of starting them. `--summarize DIR` only rebuilds the tables.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/neuralese"))
from natlang_neuralese.common import paths  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex as sha256  # noqa: E402

TASKS = ROOT / "applications/pi/tasks"
LLAMA = Path(os.environ.get("NATLANG_LLAMA_SERVER") or paths.resolve("llama_cpp", "build-cuda", "bin", "llama-server"))
DEFAULT_EXECUTOR = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B-GGUF/snapshots/84022ce711b28455e8c4fc364ce68c00cf995875/LFM2.5-2.6B-Q6_K.gguf"
CONFIGS = {"student-alone": [], "student-text-companion": ["--companion"]}
PENDING = {
    "teacher-alone": "no teacher endpoint on DGX (Qwen servers :8082/:8083 down; training holds the memory)",
    "student-neuralese-companion-views": "direct writer not qualified (export status direct-writer-pending)",
    "student-oracle-companion": "oracle companion not implemented",
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, proc: subprocess.Popen | None, seconds: float, log: Path) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited ({proc.returncode}); see {log}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"{url} not healthy after {seconds}s; see {log}")


def model_id(endpoint: str) -> str:
    with urllib.request.urlopen(endpoint.rstrip("/") + "/v1/models", timeout=10) as response:
        return json.load(response)["data"][0]["id"]


def session_stats(path: Path) -> dict:
    """Assistant turns, tool calls and the largest prompt the agent was sent, from a pi session database."""
    stats: dict = {"assistant_turns": None, "tool_calls": None, "max_input_tokens": None, "output_tokens": None}
    if not path.exists():
        return stats
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        messages = []
        for (table,) in db.execute("select name from sqlite_master where type='table'"):
            columns = [row[1] for row in db.execute(f"pragma table_info('{table}')")]
            for column in columns:
                for (value,) in db.execute(f"select \"{column}\" from \"{table}\""):
                    if isinstance(value, (bytes, str)) and '"role"' in (value.decode("utf-8", "replace") if isinstance(value, bytes) else value):
                        try:
                            messages.extend(find_messages(json.loads(value)))
                        except (ValueError, TypeError):
                            pass
        seen, turns, calls, inputs, outputs = set(), 0, 0, [], 0
        for message in messages:
            key = json.dumps(message, sort_keys=True)[:4000]
            if key in seen:
                continue
            seen.add(key)
            if message.get("role") != "assistant":
                continue
            turns += 1
            content = message.get("content")
            if isinstance(content, list):
                calls += sum(1 for part in content if isinstance(part, dict) and part.get("type") == "toolCall")
            usage = message.get("usage") or {}
            if isinstance(usage, dict):
                if usage.get("input") is not None:
                    inputs.append(int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0))
                outputs += int(usage.get("output") or 0)
        stats.update(assistant_turns=turns, tool_calls=calls, max_input_tokens=max(inputs) if inputs else None,
                     output_tokens=outputs)
    finally:
        db.close()
    return stats


def find_messages(value) -> list[dict]:
    if isinstance(value, dict):
        if value.get("role") in ("assistant", "user", "toolResult") and "content" in value:
            return [value]
        return [m for v in value.values() for m in find_messages(v)]
    if isinstance(value, list):
        return [m for v in value for m in find_messages(v)]
    return []


def proof_mounts(args) -> list[tuple[str, str]]:
    """(host, container) mounts that put a direct export's LFM source proof where its manifest says, or [] when the
    proof paths exist here already."""
    proof = json.loads((args.student_artifact / "manifest.json").read_text())["base_binding"].get("source_proof") or {}
    receipt, cache = proof.get("launch_receipt_path"), proof.get("hf_hub_cache")
    if not receipt or Path(receipt).exists():
        return []
    if not args.launch_receipt or sha256(args.launch_receipt) != proof.get("launch_receipt_sha256"):
        raise SystemExit(f"the export's launch receipt {receipt} is not here; pass --launch-receipt with its exact copy")
    hub = Path(args.student_base).resolve().parents[2]
    snapshot = Path(proof["snapshot_path"])
    if Path(args.student_base).resolve().name != snapshot.name:
        raise SystemExit("--student-base is not the export's base revision")
    args.student_base = snapshot   # served by its proof path inside the container
    return [(str(args.launch_receipt.resolve()), receipt), (str(hub), cache)]


def summarize(out: Path, meta: dict | None = None) -> dict:
    meta = meta or (json.loads((out / "matrix.json").read_text())["meta"] if (out / "matrix.json").exists() else {})
    rows, summary = [], {}
    for config in CONFIGS:
        results = out / config / "results.json"
        if not results.exists():
            continue
        passed = 0
        for row in json.loads(results.read_text()):
            stats = session_stats(out / config / f"{row['task']}.sqlite")
            rows.append({"config": config, "task": row["task"], "passed": row["passed"], "status": row.get("status"),
                         "reason": row.get("reason"), "error": row.get("error"),
                         "seconds": round((row.get("ms") or 0) / 1000, 1), **stats})
            passed += bool(row["passed"])
        mine = [r for r in rows if r["config"] == config]
        summary[config] = {"passed": passed, "tasks": len(mine),
                           "mean_seconds": round(sum(r["seconds"] for r in mine) / max(1, len(mine)), 1),
                           "mean_tool_calls": round(sum(r["tool_calls"] or 0 for r in mine) / max(1, len(mine)), 2),
                           "statuses": {s: sum(1 for r in mine if r["status"] == s) for s in {r["status"] for r in mine}}}
    report = {"meta": meta, "summary": summary, "pending": PENDING, "rows": rows}
    (out / "matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"# Harness-bench matrix: {meta.get('label', out.name)}", "",
             "| configuration | passed | mean s | mean tool calls | statuses |", "| --- | --- | --- | --- | --- |"]
    for config, s in summary.items():
        lines.append(f"| {config} | {s['passed']}/{s['tasks']} | {s['mean_seconds']} | {s['mean_tool_calls']} | "
                     + ", ".join(f"{k}: {v}" for k, v in sorted(s["statuses"].items(), key=str)) + " |")
    for config, why in PENDING.items():
        lines.append(f"| {config} | pending | | | {why} |")
    lines += ["", "| configuration | task | pass | status | s | turns | tool calls | max input tok |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['task']} | {'yes' if r['passed'] else 'no'} | {r['status'] or r['error'] and 'error'} "
                     f"| {r['seconds']} | {r['assistant_turns']} | {r['tool_calls']} | {r['max_input_tokens']} |")
    (out / "matrix.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summarize", action="store_true", help="only rebuild matrix.json/matrix.md from --out")
    parser.add_argument("--student-base", type=Path, help="HF snapshot of the student (or the direct export's base)")
    parser.add_argument("--student-artifact", type=Path, help="direct-SFT serving export")
    parser.add_argument("--launch-receipt", type=Path, help="local copy of the export's LFM source-proof launch receipt")
    parser.add_argument("--student-image", default="natlang-neuralese:26.09")
    parser.add_argument("--student-endpoint", help="use a running student server")
    parser.add_argument("--student-ctx", type=int, default=32768)
    parser.add_argument("--student-memory-gb", type=float, default=6.0)
    parser.add_argument("--executor-gguf", type=Path, default=DEFAULT_EXECUTOR)
    parser.add_argument("--executor-endpoint", help="use a running executor (OpenAI-compatible) server")
    parser.add_argument("--executor-ctx", type=int, default=32768)
    parser.add_argument("--executor-slots", type=int, default=2)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--tasks", default="", help="comma-separated task names (default: all)")
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--extra", default="", help="extra pi options, space-separated")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.summarize:
        print(json.dumps(summarize(out)["summary"], indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    containers: list[str] = []
    try:
        executor = args.executor_endpoint
        if not executor:
            port, log = free_port(), out / "executor.log"
            procs.append(subprocess.Popen([str(LLAMA), "-m", str(args.executor_gguf), "--host", "127.0.0.1", "--port", str(port),
                                           "-ngl", "99", "-c", str(args.executor_ctx * args.executor_slots), "--parallel", str(args.executor_slots),
                                           "--jinja", "--metrics", "--no-webui"],
                                          stdout=log.open("w"), stderr=subprocess.STDOUT))
            executor = f"http://127.0.0.1:{port}"
            wait_http(executor + "/health", procs[-1], 600, log)
        student = args.student_endpoint
        if not student:
            port, log = free_port(), out / "student.log"
            mounts = proof_mounts(args) if args.student_artifact else []
            serve = [str(ROOT / ".venv-neuralese/bin/python"), "-m", "natlang_neuralese.serve", "--device", "cuda",
                     "--dtype", "bfloat16", "--port", str(port), "-c", str(args.student_ctx), "--memory-gb", str(args.student_memory_gb)]
            serve += (["--direct-artifact", str(args.student_artifact), "--direct-base-snapshot", str(args.student_base)]
                      if args.student_artifact else ["--base", str(args.student_base)])
            env = {**os.environ, "PYTHONPATH": str(ROOT / "training/neuralese"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            if mounts:
                # The export's source proof names paths of the machine it was trained on: serve it in a container
                # that has the same files (this machine's HF hub, the copied launch receipt) at those paths.
                name = os.environ.get("HARNESS_STUDENT_CONTAINER", "harness-matrix-student")
                container = ["docker", "run", "--rm", "--name", name, "--gpus", "all", "--network", "host",
                             "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                             *[item for key in ("PYTHONPATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE") for item in ("-e", f"{key}={env[key]}")],
                             "-v", f"{ROOT}:{ROOT}:ro", *[item for source, target in mounts for item in ("-v", f"{source}:{target}:ro")],
                             "-w", str(ROOT), args.student_image, "python", *serve[1:]]
                serve = container
                procs.append(subprocess.Popen(serve, stdout=log.open("w"), stderr=subprocess.STDOUT))
                containers.append(name)
            else:
                procs.append(subprocess.Popen(serve, cwd=ROOT, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT))
            student = f"http://127.0.0.1:{port}"
            wait_http(student + "/health", procs[-1], 900, log)
        student_model, executor_model = model_id(student), model_id(executor)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        meta = {"label": args.label or out.name, "commit": commit, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "student_model": student_model, "student_base": str(args.student_base or ""),
                "student_artifact": str(args.student_artifact or ""),
                "student_artifact_manifest_sha256": sha256(args.student_artifact / "manifest.json") if args.student_artifact else None,
                "executor_model": executor_model, "executor_gguf": str(args.executor_gguf) if not args.executor_endpoint else None,
                "minutes_per_task": args.minutes, "student_ctx": args.student_ctx, "max_tokens": args.max_tokens}
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        tasks = [t for t in args.tasks.split(",") if t] or sorted(p.name for p in TASKS.iterdir() if (p / "task.json").exists())
        env = {**os.environ, "NATLANG_SERVER": executor, "NATLANG_MODEL": executor_model}
        env.pop("NATLANG_PROFILE", None)
        for config in [c for c in args.configs.split(",") if c]:
            command = ["natlang", "run", str(ROOT / "applications/pi"), "--", "eval", *tasks,
                       "--agent-endpoint", student, "--agent-model", student_model, "--minutes", str(args.minutes),
                       "--context-window", str(args.student_ctx), "--max-tokens", str(args.max_tokens),
                       "--out", str(out / config), *CONFIGS[config], *args.extra.split()]
            print("$", " ".join(command), flush=True)
            with (out / f"{config}.log").open("w") as log:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"{config}: exit {code}", flush=True)
            summarize(out, meta)
        report = summarize(out, meta)
        print(json.dumps(report["summary"], indent=2))
        return 0
    finally:
        for name in containers:
            subprocess.run(["docker", "stop", "-t", "10", name], capture_output=True)
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
