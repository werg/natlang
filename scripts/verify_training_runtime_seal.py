#!/usr/bin/env python3
"""Verify a self-contained frozen training runtime and current policy identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def verify(runtime: Path) -> dict:
    runtime = runtime.resolve(strict=True)
    manifest_path = runtime / "frozen-runtime.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("version") != "natlang.frozen_training_runtime/2":
        raise ValueError("unsupported training runtime seal version")
    if (runtime / "node_modules").is_symlink() or not (runtime / "node_modules").is_dir():
        raise ValueError("runtime dependencies must be physically present in the sealed runtime")
    listed = manifest.get("files")
    modes = manifest.get("file_modes")
    if not isinstance(listed, dict) or set(listed) != set(modes or {}):
        raise ValueError("runtime file and mode inventories differ")
    actual_paths = {p.relative_to(runtime).as_posix() for p in runtime.rglob("*")
                    if p.is_file() and not p.is_symlink()
                    and p.relative_to(runtime).as_posix() != "frozen-runtime.json"}
    if actual_paths != set(listed):
        missing, extra = sorted(set(listed) - actual_paths), sorted(actual_paths - set(listed))
        raise ValueError(f"runtime closure file set changed; missing={missing[:5]} extra={extra[:5]}")
    for relative, expected in listed.items():
        path = runtime / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"runtime file hash mismatch: {relative}")
        if path.stat().st_nlink != 1:
            raise ValueError(f"runtime file is not physically isolated (link count {path.stat().st_nlink}): {relative}")
        if stat.S_IMODE(path.stat().st_mode) != modes[relative]:
            raise ValueError(f"runtime file mode mismatch: {relative}")
    actual_links = {}
    for path in runtime.rglob("*"):
        if path.is_symlink():
            resolved = path.resolve(strict=False)
            if not resolved.is_relative_to(runtime):
                raise ValueError(f"external runtime symlink: {path} -> {resolved}")
            actual_links[path.relative_to(runtime).as_posix()] = os.readlink(path)
    expected_links = {row["path"]: row["target"] for row in manifest.get("symlinks", [])}
    if actual_links != expected_links:
        raise ValueError("runtime symlink map changed")
    required_policy = {
        "curriculum": "dist/teacher/curriculum.js",
        "curriculum_policy": "dist/teacher/curriculum-policy.js",
        "source_conversion": "dist/teacher/source-conversion.js",
        "source_review": "dist/teacher/source-review.js",
        "native_materializer": "dist/teacher/native-materializer.js",
        "admission_dispositions": "scripts/admission-dispositions.mjs",
        "static_bundle_input": "scripts/inline-curriculum/static-bundle-input.mjs",
    }
    identity = manifest.get("current_policy_identity", {})
    for name, relative in required_policy.items():
        if identity.get(name) != listed.get(relative):
            raise ValueError(f"runtime current-policy pin does not match closure: {name}")
    return {"schema": "natlang.training_runtime_seal_check/1",
            "runtime_path": str(runtime),
            "runtime_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "runtime_files": len(listed), "runtime_symlinks": len(actual_links),
            "current_policy_identity": identity,
            "task_contract_identity": manifest.get("task_contract_identity", {}),
            "verified": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.runtime)
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
