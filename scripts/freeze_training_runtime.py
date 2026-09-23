#!/usr/bin/env python3
"""Freeze built Node runtime/scripts so concurrent development cannot change replay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_pipeline import atomic_json, digest_file


def tree_identity(root):
    return {str(path.relative_to(root)): digest_file(path)
            for folder in ("dist", "scripts") for path in sorted((root / folder).rglob("*")) if path.is_file()} | {
                name: digest_file(root / name) for name in ("prelude.js", "package.json", "package-lock.json") if (root / name).exists()}


def freeze(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    before = tree_identity(source)
    if not before or not (source / "dist/native/runtime.js").exists():
        raise ValueError("build the Node runtime before freezing it")
    manifest = output / "frozen-runtime.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if tree_identity(output) != data["files"]:
            raise ValueError("frozen runtime changed")
        # Reuse the frozen revision even if the developer subsequently changes the source tree.
        return data
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="runtime-building-", dir=output.parent))
    for name in ("dist", "scripts"):
        shutil.copytree(source / name, staging / name)
    for name in ("prelude.js", "package.json", "package-lock.json"):
        if (source / name).exists():
            shutil.copy2(source / name, staging / name)
    (staging / "node_modules").symlink_to(source / "node_modules", target_is_directory=True)
    copied = tree_identity(staging)
    if before != copied or before != tree_identity(source):
        raise ValueError("runtime changed while freezing; retry after the build finishes")
    data = {"version": "natlang.frozen_runtime/1", "files": copied,
            "node_modules": str(source / "node_modules"),
            "note": "Runtime and scripts copied; installed Node dependencies are shared and must not be changed during a run."}
    atomic_json(staging / "frozen-runtime.json", data)
    staging.rename(output)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps({"files": len(freeze(args.source, args.output)["files"]), "output": str(args.output)}))
