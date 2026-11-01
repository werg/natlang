from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.neuralese_data.build_luna_exact_native_conversion import _runtime_row_digest


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
