from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.neuralese_data.build_luna_exact_native_conversion import (
    _protected_runtime_binding, _runtime_row_digest,
)
from natlang_neuralese.common import runtime_inventory
from natlang_neuralese.common.runtime_inventory import immutable_native_runtime_tree


def test_materializer_row_digest_uses_exact_separately_bound_runtime_asset(tmp_path):
    module = tmp_path / "dist/teacher/native-materializer.js"
    module.parent.mkdir(parents=True)
    module.write_text('export function nativeRowDigest(_row) { return "a".repeat(64); }\n')
    module_sha = hashlib.sha256(module.read_bytes()).hexdigest()
    receipt = {"identity": {"runtime": {"dist/teacher/native-materializer.js": module_sha}}}

    assert _runtime_row_digest(tmp_path, {"id": "source-bound-row"}, receipt) == "a" * 64

    receipt["identity"]["runtime"]["dist/teacher/native-materializer.js"] = "0" * 64
    with pytest.raises(ValueError, match="separately bound runtime"):
        _runtime_row_digest(tmp_path, {"id": "source-bound-row"}, receipt)


def test_protected_runtime_input_closure_requires_complete_pool_seal(tmp_path):
    runtime = tmp_path / "teacher-runtime"
    module = runtime / "dist/teacher/native-materializer.js"
    module.parent.mkdir(parents=True)
    module.write_text('export function nativeRowDigest(_row) { return "a".repeat(64); }\n')
    bound_source = tmp_path / "bound-source.txt"
    bound_source.write_text("reviewed source bytes\n")
    (runtime / "runtime-input-closure.json").write_text(__import__("json").dumps({
        "schema": "natlang.protected-runtime-input-closure/1",
        "destination": str(runtime.resolve()), "commit": "1" * 40,
        "bound_files": [{"path": str(bound_source),
                         "sha256": hashlib.sha256(bound_source.read_bytes()).hexdigest()}],
    }))
    pool = tmp_path / "pool"
    inventory = []
    logical_bytes = 0
    for path in (runtime / "runtime-input-closure.json", module):
        path.chmod(0o444)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        blob = pool / digest[:2] / f"{digest}.444"
        blob.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, blob)
        inventory.append(path)
        logical_bytes += path.stat().st_size
    storage = {"schema": "natlang.immutable-runtime-file-pool/1", "pool": str(pool),
               "files": len(inventory), "logical_bytes": logical_bytes}
    (runtime.with_name(runtime.name + ".storage.json")).write_text(__import__("json").dumps(storage))
    receipt_runtime = {"dist/teacher/native-materializer.js": hashlib.sha256(module.read_bytes()).hexdigest()}

    result = _protected_runtime_binding(runtime, receipt_runtime)
    assert result["kind"] == "natlang.immutable-native-runtime-tree/1"
    assert result["commit"] == "1" * 40
    assert result["file_count"] == storage["files"]
    assert immutable_native_runtime_tree(runtime)["pool_verified"] is True

    # The first check authenticates every runtime byte. Repeated rows in one
    # loader process may reuse that proof only while the five-field file/pool
    # snapshot, closure, storage receipt, and bound inputs remain unchanged.
    original_hash = runtime_inventory.sha256_file_hex_cached
    checked_runtime_files = []
    with runtime_inventory._NATIVE_RUNTIME_CACHE_LOCK:
        runtime_inventory._NATIVE_RUNTIME_CACHE.pop(str(runtime.resolve()), None)

    def count_runtime_hashes(path, **kwargs):
        candidate = Path(path)
        if candidate.is_relative_to(runtime):
            checked_runtime_files.append(candidate)
        return original_hash(path, **kwargs)

    runtime_inventory.sha256_file_hex_cached = count_runtime_hashes
    try:
        first_cached = immutable_native_runtime_tree(runtime)
        first_runtime_hash_count = len(checked_runtime_files)
        checked_runtime_files.clear()
        second_cached = immutable_native_runtime_tree(runtime)
        assert first_cached["tree_sha256"] == second_cached["tree_sha256"]
        assert first_runtime_hash_count == storage["files"]
        assert checked_runtime_files == []
    finally:
        runtime_inventory.sha256_file_hex_cached = original_hash

    storage_path = runtime.with_name(runtime.name + ".storage.json")
    storage_path.write_text(__import__("json").dumps({**storage, "logical_bytes": storage["logical_bytes"] + 1}))
    with pytest.raises(ValueError, match="storage receipt"):
        immutable_native_runtime_tree(runtime)
    storage_path.write_text(__import__("json").dumps(storage))

    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (runtime / "escape-link").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes its root"):
        immutable_native_runtime_tree(runtime)
    (runtime / "escape-link").unlink()

    module.chmod(0o644)
    with pytest.raises(ValueError, match="writable file"):
        _protected_runtime_binding(runtime, receipt_runtime)
    module.chmod(0o444)
    bound_source.write_text("changed source bytes\n")
    with pytest.raises(ValueError, match="bound source input changed"):
        immutable_native_runtime_tree(runtime)
    bound_source.write_text("reviewed source bytes\n")
    module.chmod(0o644)
    module.write_text('export function nativeRowDigest(_row) { return "b".repeat(64); }\n')
    module.chmod(0o444)
    with pytest.raises(ValueError, match="absent from its declared pool"):
        immutable_native_runtime_tree(runtime)
