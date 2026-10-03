#!/usr/bin/env python3
"""Build an inactive, content-pinned code/policy package for a reviewed exclusion resume."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.clone_runtime_tree import clone_runtime_tree
from scripts.corpus import file_digest


def write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def entries(root: Path) -> list[dict]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"sealed candidate contains symlink: {path}")
        if path.is_file():
            result.append({"path": path.relative_to(root).as_posix(),
                           "sha256": file_digest(path), "bytes": path.stat().st_size,
                           "mode": oct(path.stat().st_mode & 0o777)})
    return result


def prepare(*, base_snapshot: Path, output: Path, source_review_ts: Path,
            training_policy: Path, data_sha256: str, source_hold_ids: list[str],
            supervisor_path: Path, supervisor_sha256: str) -> dict:
    base_snapshot = base_snapshot.resolve(strict=True)
    output = output.resolve(strict=False)
    source_review_ts = source_review_ts.resolve(strict=True)
    training_policy = training_policy.resolve(strict=True)
    supervisor_path = supervisor_path.resolve(strict=True)
    if file_digest(supervisor_path) != supervisor_sha256:
        raise ValueError("supervisor bytes differ from the separately reviewed supervisor pin")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"preserving existing candidate: {output}")
    if file_digest(training_policy) != "6bbbcd6f8b08e33647a587dfc66fda325925a76502baced213a0a77cd0ba512f":
        # The current input policy must be reviewed afresh; this intentionally fails closed.
        raise ValueError("training data-source policy differs from the reviewed base policy pin")
    base_manifest = json.loads((base_snapshot / "snapshot-manifest.json").read_text())
    if base_manifest.get("schema") != "lfm-training-code-snapshot/1":
        raise ValueError("unsupported base training code snapshot")

    output.mkdir(parents=True)
    try:
        code = output / "code-snapshot"
        clone_runtime_tree(base_snapshot, code)
        scripts = code / "scripts"
        overlays = {
            "train_lora.py": ROOT / "scripts/train_lora.py",
            "training_exclusion.py": ROOT / "scripts/training_exclusion.py",
            "training_append.py": ROOT / "scripts/training_append.py",
        }
        for name, source in overlays.items():
            target = scripts / name
            target.write_bytes(source.read_bytes())
            target.chmod(0o444)
        (scripts / "train_lora.py").chmod(0o444)
        (scripts / "training_exclusion.py").chmod(0o444)
        (scripts / "training_append.py").chmod(0o444)

        controller = output / "controller"
        controller.mkdir()
        for name in ("prepare_training_exclusion.py", "apply_training_exclusion_transition.py",
                     "prepare_training_exclusion_plan.py", "audit_training_exclusion_checkpoint.py",
                     "training_exclusion.py", "train_lora.py"):
            source = ROOT / "scripts" / name
            target = controller / name
            shutil.copy2(source, target)
            target.chmod(0o444)

        supervisor_dir = output / "supervisor"
        supervisor_dir.mkdir()
        supervisor_copy = supervisor_dir / "resumable_training_supervisor.py"
        shutil.copy2(supervisor_path, supervisor_copy)
        supervisor_copy.chmod(0o444)

        source_policy = output / "source-review"
        source_policy.mkdir()
        source_dest = source_policy / "source-review.ts"
        shutil.copy2(source_review_ts, source_dest)
        source_dest.chmod(0o444)
        policy_source_dir = source_policy / "src" / "teacher"
        policy_source_dir.mkdir(parents=True)
        for name in ("tatqa-unit-contract.ts", "musique-oklahoma-event-contract.ts"):
            source = ROOT / "ts-host/src/teacher" / name
            target = policy_source_dir / name
            shutil.copy2(source, target)
            target.chmod(0o444)
        policy_copy = output / "inventory-policy" / "data_sources.json"
        policy_copy.parent.mkdir()
        shutil.copy2(training_policy, policy_copy)
        policy_copy.chmod(0o444)
        policy_dist = source_policy / "dist"
        policy_dist.mkdir()
        command = [str(ROOT / "ts-host/node_modules/.bin/tsc"),
                   "src/teacher/source-review.ts",
                   "src/teacher/tatqa-unit-contract.ts",
                   "src/teacher/musique-oklahoma-event-contract.ts",
                   "--target", "ES2022", "--module", "NodeNext", "--moduleResolution", "NodeNext",
                   "--strict", "--declaration", "--skipLibCheck", "--rootDir", "src",
                   "--outDir", str(policy_dist), "--typeRoots", "node_modules/@types"]
        tsc = subprocess.run(command, cwd=ROOT / "ts-host", text=True, capture_output=True)
        if tsc.returncode:
            raise RuntimeError(f"isolated source-review compilation failed: {tsc.stderr[-2000:]}")
        policy_entries = entries(policy_dist)
        required = {"teacher/source-review.js", "teacher/source-review.d.ts",
                    "teacher/tatqa-unit-contract.js", "teacher/musique-oklahoma-event-contract.js"}
        if not required.issubset({item["path"] for item in policy_entries}):
            raise ValueError("isolated compiler output omitted a pinned source-review module")
        js = policy_dist / "teacher/source-review.js"
        if "2hop__568389_161223" not in js.read_text(encoding="utf-8"):
            raise ValueError("compiled source-review module does not contain the new Bugabula hold")

        for py in scripts.glob("*.py"):
            ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        # This package is inactive; directory permissions and final hashes are sealing evidence.
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_file():
                path.chmod(0o444)
            elif path.is_dir():
                path.chmod(0o555)
        output.chmod(0o555)
        manifest = {
            "schema": "natlang.training_exclusion_runtime_candidate/1",
            "status": "prepared_not_approved_or_active",
            "base_snapshot_path": str(base_snapshot),
            "base_snapshot_manifest_sha256": file_digest(base_snapshot / "snapshot-manifest.json"),
            "candidate_path": str(output),
            "training_data_sha256": data_sha256,
            "inventory_policy_sha256": file_digest(training_policy),
            "source_review_typescript_sha256": file_digest(source_dest),
            "source_review_compiled_sha256": file_digest(js),
            "source_review_compiled_modules": policy_entries,
            "source_hold_ids": source_hold_ids,
            "docker_image": "sha256:eabc88d83cba79860c368bd2c7758e0e6914f602e7de6b9e20446fb624de1cdc",
            "supervisor_source_path": str(supervisor_path),
            "supervisor_source_sha256": supervisor_sha256,
            "supervisor_relative_path": "supervisor/resumable_training_supervisor.py",
            "supervisor_sha256": file_digest(supervisor_copy),
            "node_or_provider_used": False,
            "gpu_or_training_used": False,
            "files": entries(output),
            "file_count": len(entries(output)),
            "runtime_scope": "Python trainer in the pinned Docker image; source-review TS/compiled pair is included as current policy provenance, not imported by the trainer",
        }
        return manifest
    except BaseException:
        # The destination is a new review-only candidate; do not touch its parent or any source.
        for path in output.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
            elif path.exists():
                path.chmod(0o644)
        output.chmod(0o755)
        shutil.rmtree(output)
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source-review-ts", type=Path, required=True)
    ap.add_argument("--training-policy", type=Path, required=True)
    ap.add_argument("--data-sha256", required=True)
    ap.add_argument("--source-hold-id", action="append", required=True)
    ap.add_argument("--supervisor", type=Path, required=True)
    ap.add_argument("--supervisor-sha256", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    manifest = prepare(base_snapshot=args.base_snapshot, output=args.output,
                       source_review_ts=args.source_review_ts,
                       training_policy=args.training_policy, data_sha256=args.data_sha256,
                       source_hold_ids=args.source_hold_id, supervisor_path=args.supervisor,
                       supervisor_sha256=args.supervisor_sha256)
    write_immutable(args.manifest, (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"candidate": str(args.output.resolve()), "manifest": str(args.manifest.resolve()),
                      "manifest_sha256": file_digest(args.manifest), "files": manifest["file_count"],
                      "status": manifest["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
