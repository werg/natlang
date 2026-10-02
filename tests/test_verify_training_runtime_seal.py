import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.verify_training_runtime_seal import verify


POLICY_PATHS = {
    "curriculum": "dist/teacher/curriculum.js",
    "curriculum_policy": "dist/teacher/curriculum-policy.js",
    "source_conversion": "dist/teacher/source-conversion.js",
    "source_review": "dist/teacher/source-review.js",
    "native_materializer": "dist/teacher/native-materializer.js",
    "admission_dispositions": "scripts/admission-dispositions.mjs",
    "static_bundle_input": "scripts/inline-curriculum/static-bundle-input.mjs",
}


def write_runtime(root: Path, *, nested_manifest=True):
    files = {name: b"sealed" for name in POLICY_PATHS.values()}
    if nested_manifest:
        files["nested/frozen-runtime.json"] = b'{"nested":true}\n'
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "node_modules").mkdir()
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    manifest = {
        "version": "natlang.frozen_training_runtime/2",
        "files": hashes,
        "file_modes": {name: (root / name).stat().st_mode & 0o777 for name in hashes},
        "symlinks": [],
        "current_policy_identity": {name: hashes[path] for name, path in POLICY_PATHS.items()},
    }
    (root / "frozen-runtime.json").write_text(json.dumps(manifest))
    return manifest


def test_verifier_hashes_nested_frozen_runtime_manifest(tmp_path):
    manifest = write_runtime(tmp_path)
    result = verify(tmp_path)
    assert result["verified"] is True
    assert "nested/frozen-runtime.json" in manifest["files"]


def test_verifier_rejects_nested_manifest_mutation(tmp_path):
    write_runtime(tmp_path)
    (tmp_path / "nested/frozen-runtime.json").write_text("mutated\n")
    with pytest.raises(ValueError, match="hash mismatch: nested/frozen-runtime.json"):
        verify(tmp_path)


def test_verifier_rejects_external_hardlink(tmp_path):
    manifest = write_runtime(tmp_path, nested_manifest=False)
    relative = POLICY_PATHS["curriculum"]
    external = tmp_path.parent / f"external-{tmp_path.name}.js"
    try:
        os.link(tmp_path / relative, external)
        with pytest.raises(ValueError, match="not physically isolated"):
            verify(tmp_path)
    finally:
        external.unlink(missing_ok=True)

