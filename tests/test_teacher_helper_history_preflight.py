import hashlib
from pathlib import Path

from scripts.coreweave.check_teacher_helper_history import check_reference


def test_preflight_accepts_synthetic_exact_path_sha_and_byte_identity(tmp_path: Path):
    payload = b"synthetic fixture; not a historical artifact"
    artifact = tmp_path / "fixture.bin"
    artifact.write_bytes(payload)
    binding = {"path": str(artifact), "sha256": hashlib.sha256(payload).hexdigest(),
               "bytes": len(payload)}

    assert check_reference(binding) is None
    assert "identity mismatch" in check_reference({**binding, "sha256": "0" * 64})
    assert "identity mismatch" in check_reference({**binding, "bytes": len(payload) + 1})
    assert "missing regular file" in check_reference({**binding, "path": str(tmp_path / "absent")})
