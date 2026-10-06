#!/usr/bin/env python3
"""Freeze built Node runtime/scripts so concurrent development cannot change replay."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_pipeline import atomic_json, digest_file


def tree_identity(root, compiled_dist=None):
    folders = {name: root / name for name in ("dist", "scripts", "src")}
    if compiled_dist is not None:
        folders["dist"] = compiled_dist
    return {str(Path(name) / path.relative_to(folder)): digest_file(path)
            for name, folder in folders.items() for path in sorted(folder.rglob("*")) if path.is_file()} | {
                name: digest_file(root / name) for name in ("prelude.js", "package.json", "package-lock.json") if (root / name).exists()}


def check_prompt_features(root, compiled_dist=None):
    """Refuse a new snapshot whose prompt advertises an absent eval feature."""
    native = (compiled_dist if compiled_dist is not None else root / 'dist') / 'native'
    prompt = (native / 'prompt.js').read_text()
    if re.search(r'\bfinish\s*:\s*true\b', prompt):
        agent = (native / 'agent.js').read_text()
        runtime = (native / 'runtime.js').read_text()
        if (not re.search(r"\bfinish\s*:\s*\{\s*type\s*:\s*['\"]boolean['\"]", agent)
                or not re.search(r'\bargs\.finish\s*===\s*true\b', runtime)):
            raise ValueError('Prompt advertises eval finish:true without its compiled tool schema and runtime implementation')


def freeze(source, output, compiled_dist=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    compiled_dist = Path(compiled_dist).resolve() if compiled_dist is not None else source / 'dist'
    # World bridges resolve vendor beside ts-host. Preserve that layout for frozen copies.
    source_vendor, frozen_vendor = source.parent / 'vendor', output.parent / 'vendor'
    if source_vendor.exists() and not frozen_vendor.exists():
        frozen_vendor.parent.mkdir(parents=True, exist_ok=True)
        frozen_vendor.symlink_to(source_vendor, target_is_directory=True)
    if source_vendor.exists() and frozen_vendor.resolve() != source_vendor.resolve():
        raise ValueError('frozen runtime vendor path points at a different dependency tree')
    before = tree_identity(source, compiled_dist)
    if not before or not (compiled_dist / "native/runtime.js").exists():
        raise ValueError("build the Node runtime before freezing it")
    manifest = output / "frozen-runtime.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if tree_identity(output) != data["files"]:
            raise ValueError("frozen runtime changed")
        # Reuse the frozen revision even if the developer subsequently changes the source tree.
        return data
    check_prompt_features(source, compiled_dist)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="runtime-building-", dir=output.parent))
    for name in ("dist", "scripts", "src"):
        shutil.copytree(compiled_dist if name == 'dist' else source / name, staging / name)
    for name in ("prelude.js", "package.json", "package-lock.json"):
        if (source / name).exists():
            shutil.copy2(source / name, staging / name)
    (staging / "node_modules").symlink_to(source / "node_modules", target_is_directory=True)
    copied = tree_identity(staging)
    if before != copied or before != tree_identity(source, compiled_dist):
        raise ValueError("runtime changed while freezing; retry after the build finishes")
    data = {"version": "natlang.frozen_runtime/1", "files": copied,
            "compiled_dist_source": str(compiled_dist),
            "node_modules": str(source / "node_modules"),
            "note": "Runtime and scripts copied; installed Node dependencies are shared and must not be changed during a run."}
    atomic_json(staging / "frozen-runtime.json", data)
    staging.rename(output)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compiled-dist", type=Path,
                        help="Use an isolated tsc output without rebuilding or staging live shared Node packages")
    args = parser.parse_args()
    print(json.dumps({"files": len(freeze(args.source, args.output, args.compiled_dist)["files"]), "output": str(args.output)}))
