"""Golden digests pinning the shared hashing/JSON helpers behind corpus manifests."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/neuralese"))

from natlang_neuralese.common.hashing import (  # noqa: E402
    canonical_json_sha256_hex,
    sha256_file_hex,
    sha256_hex,
)
from natlang_neuralese.common.jsonio import (  # noqa: E402
    canonical_json_bytes,
    canonical_json_str,
    utc_now_iso,
)

NESTED = {"b": [1, {"z": 1, "a": "é☃"}], "a": None, "é": 1.5}
NESTED_REORDERED = {"é": 1.5, "a": None, "b": [1, {"a": "é☃", "z": 1}]}
CANONICAL = '{"a":null,"b":[1,{"a":"é☃","z":1}],"é":1.5}'


def test_sha256_hex_golden():
    assert sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_hex("héllo☃".encode()) == "0ab11e8b13ecc7b64a30cae577e853d1d4b4960e62624cdbaff0afff2e1a6b71"
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_file_hex_golden_and_chunking(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"a\xc3\xa9\n\x00\xff")
    expected = sha256_hex(b"a\xc3\xa9\n\x00\xff")
    assert expected == "af660dd5b9da759b6a5d02e26dee58011329900a6a341f4d9b56ba80be1e8217"
    assert sha256_file_hex(path) == expected
    assert sha256_file_hex(str(path)) == expected
    assert sha256_file_hex(path, chunk_size=1) == expected
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (3 * 1024 * 1024 + 7))
    assert sha256_file_hex(big) == sha256_hex(b"x" * (3 * 1024 * 1024 + 7))


def test_canonical_json_golden():
    assert canonical_json_str(NESTED) == CANONICAL
    assert canonical_json_str(NESTED_REORDERED) == CANONICAL
    assert canonical_json_bytes(NESTED) == CANONICAL.encode("utf-8")
    assert not canonical_json_str(NESTED).endswith("\n")
    assert canonical_json_str("é") == '"é"'  # ensure_ascii=False


def test_canonical_json_sha256_hex_golden():
    assert canonical_json_sha256_hex(NESTED) == "767457078e9052a5eabede57ab810287b716aff4fb56c4a2e4ea9c8a22c58f7f"
    assert canonical_json_sha256_hex(NESTED_REORDERED) == canonical_json_sha256_hex(NESTED)
    assert canonical_json_sha256_hex(NESTED) == sha256_hex(CANONICAL.encode("utf-8"))


def test_utc_now_iso_shape():
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?\+00:00", utc_now_iso())
