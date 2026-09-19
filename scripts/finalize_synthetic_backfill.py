#!/usr/bin/env python3
"""Wait for a frozen-IR teacher pass, then audit, refresh, and materialize it.

The output name includes the reference-bank hash. An incomplete teacher pass
cannot silently publish a snapshot, and remaining template gold is reported
explicitly in the final summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from audit_provisional_leaves import audit  # noqa: E402


def audit_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as stream:
        return sum(bool(line.strip()) for line in stream)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_start(pid: int) -> str | None:
    """Linux process start tick, so a recycled PID cannot satisfy the wait."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return stat.rsplit(") ", 1)[1].split()[19]


def wait_for_pass(pid: int, start_tick: str, audit_path: Path,
                  expected_attempts: int, poll_seconds: int) -> None:
    initial = audit_rows(audit_path)
    if initial > expected_attempts:
        raise RuntimeError(f"teacher audit already has {initial} rows, expected {expected_attempts}")
    observed_start = process_start(pid)
    if observed_start is not None and observed_start != start_tick:
        raise RuntimeError("teacher PID was reused")
    print(f"watching teacher PID {pid} for {expected_attempts} audited attempts", flush=True)
    while process_start(pid) == start_tick:
        time.sleep(poll_seconds)
    actual = audit_rows(audit_path)
    if actual != expected_attempts:
        raise RuntimeError(f"teacher pass stopped after {actual}/{expected_attempts} attempts; no snapshot published")
    print(f"teacher pass finished with {actual} audited attempts", flush=True)


def run(command: list[str]) -> None:
    print("running " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def retry_missing_without_turn_cap(src: Path, references: Path, retry_audit: Path) -> int:
    """Give still-missing keys one pass using only the episode budget."""
    missing = audit(src, references)["summary"]["missing_keys"]
    if not missing:
        print("no missing references to retry", flush=True)
        return 0
    if retry_audit.exists() or retry_audit.with_suffix(".trajectory.ir.jsonl").exists():
        raise FileExistsError(f"retry audit already exists: {retry_audit}")
    run([sys.executable, str(ROOT / "scripts/teacher_leaves.py"),
         "--ir", str(src), "--thinking", "0", "--audit-out", str(retry_audit)])
    attempts = audit_rows(retry_audit)
    if attempts != missing:
        raise RuntimeError(f"uncapped retry stopped after {attempts}/{missing} attempts")
    print(f"uncapped retry audited {attempts} missing keys", flush=True)
    return attempts


def finalize(src: Path, references: Path, prefix: Path, workers: int,
             shard_size: int, teacher_audits: list[Path] | None = None) -> dict:
    bank_before = file_hash(references)
    report = audit(src, references)
    bank_after_audit = file_hash(references)
    if bank_before != bank_after_audit:
        raise ValueError("reference bank changed during audit; no snapshot published")
    issues = report["references"]
    for name in ("duplicate_keys", "conflicting_keys", "invalid_key_rows"):
        if issues[name]:
            raise ValueError(f"reference bank has {len(issues[name])} {name}; no snapshot published")
    bank_hash = bank_after_audit
    named = Path(f"{prefix}-{bank_hash[:12]}")
    ir = Path(f"{named}.ir.jsonl")
    shards = Path(f"{named}-shards")
    summary = Path(f"{named}.summary.json")
    if ir.exists() or shards.exists() or summary.exists():
        raise FileExistsError(f"snapshot destination exists: {named}")
    named.parent.mkdir(parents=True, exist_ok=True)
    refresh = [sys.executable, str(ROOT / "scripts/refresh_synthetic_references.py"),
               str(src), str(ir), "--references", str(references)]
    if report["summary"]["missing_keys"] == 0:
        refresh.append("--require-complete")
    run(refresh)
    run([sys.executable, str(ROOT / "scripts/audit_program_ir.py"), str(ir),
         "--out", f"{named}.ir.audit.json"])
    run([sys.executable, str(ROOT / "scripts/materialize_ir_shards.py"), str(ir),
         str(shards), "--workers", str(workers), "--shard-size", str(shard_size)])
    ir_manifest = json.loads(ir.with_suffix(ir.suffix + ".manifest.json").read_text())
    if ir_manifest["reference_bank_sha256"] != bank_hash:
        raise ValueError("reference bank changed during refresh; refusing mismatched snapshot")
    if file_hash(ir) != ir_manifest.get("ir_sha256"):
        raise ValueError("refreshed IR hash does not match its manifest")
    shard_manifest = json.loads((shards / "manifest.json").read_text())
    identity = shard_manifest.get("identity", {})
    if identity.get("ir_sha256") != hashlib.sha256(ir.read_bytes()).hexdigest():
        raise ValueError("shard manifest is for a different IR")
    if shard_manifest.get("programs_in_ir") != ir_manifest.get("counts", {}).get("programs"):
        raise ValueError("shard and IR program counts disagree")
    if shard_manifest.get("new_counts", {}).get("programs") != shard_manifest.get("programs_in_ir"):
        raise ValueError("shard output is missing programs")
    final_bank_before = file_hash(references)
    final_report = audit(ir, references)
    if final_bank_before != file_hash(references) or final_bank_before != bank_hash:
        raise ValueError("reference bank changed during final audit; refusing snapshot")
    if final_report["summary"]["missing_keys"] != report["summary"]["missing_keys"]:
        raise ValueError("refreshed IR missing-key count disagrees with the frozen-source audit")
    if final_report["summary"]["provisional_programs"] != ir_manifest["counts"]["provisional_programs"]:
        raise ValueError("refreshed IR provisional count disagrees with its manifest")
    remaining = report["summary"]["missing_keys"]
    result = {"base_ir": str(src), "reference_bank_sha256": bank_hash,
              "teacher_audits": [str(path) for path in (teacher_audits or [])],
              "teacher_key_audit": report["summary"], "ir": str(ir),
              "ir_counts": ir_manifest["counts"], "shards": str(shards),
              "trace_counts": shard_manifest["new_counts"],
              "complete": remaining == 0 and ir_manifest["counts"].get("provisional_programs", 0) == 0,
              "final_teacher_key_audit": final_report["summary"]}
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-pid", type=int, required=True)
    ap.add_argument("--teacher-start-tick", required=True,
                    help="field 22 from /proc/PID/stat, captured when scheduling")
    ap.add_argument("--expected-attempts", type=int, required=True)
    ap.add_argument("--teacher-audit", type=Path, required=True)
    ap.add_argument("--retry-audit", type=Path,
                    help="after this pass, retry remaining keys without a per-turn cap")
    ap.add_argument("--src", type=Path, default=Path("data/external_pilot/synthetic-all-current.ir.jsonl"))
    ap.add_argument("--references", type=Path, default=Path("data/leaf_references.jsonl"))
    ap.add_argument("--out-prefix", type=Path,
                    default=Path("data/external_pilot/synthetic-all-after-teacher-pass1"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--poll-seconds", type=int, default=30)
    args = ap.parse_args()
    if min(args.expected_attempts, args.workers, args.shard_size, args.poll_seconds) < 1:
        ap.error("expected-attempts, workers, shard-size and poll-seconds must be positive")
    wait_for_pass(args.teacher_pid, args.teacher_start_tick, args.teacher_audit,
                  args.expected_attempts, args.poll_seconds)
    audits = [args.teacher_audit]
    if args.retry_audit:
        retry_missing_without_turn_cap(args.src, args.references, args.retry_audit)
        if args.retry_audit.exists():
            audits.append(args.retry_audit)
    finalize(args.src, args.references, args.out_prefix, args.workers,
             args.shard_size, teacher_audits=audits)


if __name__ == "__main__":
    main()
