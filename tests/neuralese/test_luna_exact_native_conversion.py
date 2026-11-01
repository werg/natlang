from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.neuralese_data.build_luna_exact_native_conversion import (
    _protected_runtime_binding, _runtime_row_digest,
)
from scripts.runtime_file_pool import seal_runtime_tree


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
    storage = seal_runtime_tree(runtime, pool)
    (runtime.with_name(runtime.name + ".storage.json")).write_text(__import__("json").dumps(storage))
    receipt_runtime = {"dist/teacher/native-materializer.js": hashlib.sha256(module.read_bytes()).hexdigest()}

    result = _protected_runtime_binding(runtime, receipt_runtime)
    assert result["kind"] == "protected-runtime-input-closure"
    assert result["commit"] == "1" * 40
    assert result["files"] == storage["files"]

    module.chmod(0o644)
    with pytest.raises(ValueError, match="writable file"):
        _protected_runtime_binding(runtime, receipt_runtime)
    module.chmod(0o444)
    bound_source.write_text("changed source bytes\n")
    with pytest.raises(ValueError, match="bound input changed"):
        _protected_runtime_binding(runtime, receipt_runtime)
