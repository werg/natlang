"""Golden digests pinning the shared hashing/JSON helpers behind corpus manifests."""
import re
import sys
import json
import subprocess
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/neuralese"))

from natlang_neuralese.common.hashing import (  # noqa: E402
    canonical_json_sha256_hex,
    sha256_file_hex,
    sha256_hex,
    sha256_reference,
)
from natlang_neuralese.common.jsonio import (  # noqa: E402
    canonical_json_bytes,
    canonical_json_str,
    utc_now_iso,
    write_json_durable,
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
    assert sha256_file_hex(path, drop_cache=True) == expected
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


def test_sha256_reference_accepts_exact_reference_shapes():
    digest = "a" * 64
    assert sha256_reference({"sha256": digest, "bytes": 12}) == digest
    assert sha256_reference(digest) == digest
    assert sha256_reference({"bytes": 12}) is None
    assert sha256_reference("short") is None
    assert sha256_reference(None) is None


def test_utc_now_iso_shape():
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?\+00:00", utc_now_iso())


def test_write_json_durable_replaces_whole_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("old")
    write_json_durable(path, {"b": 1, "a": [2]})
    assert path.read_text() == '{\n  "a": [\n    2\n  ],\n  "b": 1\n}'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_zstd_jsonl_preserves_exact_lines_and_all_frames(tmp_path):
    from natlang_neuralese.common.jsonio import open_jsonl_text, read_jsonl
    values = [{'text': 'one\u2028two\u2029three'}, {'text': 'x' * 200000}]
    lines = [(json.dumps(v, ensure_ascii=False) + '\r\n').encode() for v in values]
    plain = tmp_path / 'rows.jsonl'
    compressed = tmp_path / 'rows.jsonl.zst'
    plain.write_bytes(b''.join(lines))
    compressed.write_bytes(b''.join(subprocess.run(['zstd', '-q', '-c'], input=line,
                               capture_output=True, check=True).stdout for line in lines))
    for path in (plain, compressed):
        with open_jsonl_text(path) as stream:
            assert [s.encode() for s in stream] == lines
        assert read_jsonl(path) == values


def test_zstd_complete_read_rejects_truncation_after_a_valid_row(tmp_path):
    from natlang_neuralese.common.jsonio import open_jsonl_text
    data = subprocess.run(['zstd', '-q', '-c', '--check'], input=b'{"a":1}\n',
                          capture_output=True, check=True).stdout
    path = tmp_path / 'truncated.jsonl.zst'
    path.write_bytes(data[:-2])
    with pytest.raises(RuntimeError, match='zstd failed'):
        with open_jsonl_text(path) as stream:
            stream.read()


def test_zstd_registry_file_symlink_reads_target_and_rejects_corruption(tmp_path):
    from natlang_neuralese.common.jsonio import open_jsonl_text, read_jsonl
    target = tmp_path / 'physical-data'
    data = subprocess.run(['zstd', '-q', '-c', '--check'], input=b'{"a":1}\r\n',
                          capture_output=True, check=True).stdout
    target.write_bytes(data)
    registered = tmp_path / 'registered.jsonl.zst'
    registered.symlink_to(target.name)
    with open_jsonl_text(registered) as stream:
        assert stream.read().encode() == b'{"a":1}\r\n'
    assert read_jsonl(registered) == [{'a': 1}]
    target.write_bytes(data[:-2])
    with pytest.raises(RuntimeError, match='zstd failed'):
        read_jsonl(registered)
    target.unlink()
    with pytest.raises(FileNotFoundError, match='registered.jsonl.zst'):
        read_jsonl(registered)
    with pytest.raises(FileNotFoundError, match='missing.jsonl.zst'):
        read_jsonl(tmp_path / 'missing.jsonl.zst')


def test_zstd_partial_reader_closes_without_masking_consumer_errors(tmp_path):
    from natlang_neuralese.common.jsonio import open_jsonl_text
    path = tmp_path / 'large.jsonl.zst'
    path.write_bytes(subprocess.run(['zstd', '-q', '-c'], input=b'{"a":1}\n' * 100000,
                                   capture_output=True, check=True).stdout)
    with open_jsonl_text(path) as stream:
        assert stream.readline() == '{"a":1}\n'
    with pytest.raises(LookupError, match='consumer'):
        with open_jsonl_text(path) as stream:
            stream.readline()
            raise LookupError('consumer')
def test_split_scan_preserves_escaped_json_and_avoids_unselected_token_arrays(tmp_path, monkeypatch):
    import json
    from natlang_neuralese.common import jsonio
    payload = '\n'.join([
        json.dumps({'split': 'train', 'token_ids': list(range(10000))}),
        '{"split":"test","id":"plain","text":"literal\u2028separator"}',
        r'{"sp\u006cit":"te\u0073t","id":"escaped"}',
        '{"split":"train","metadata":{"split":"test"}}',
        json.dumps({'split': 'train', 'text': 'quoted "split": "test"'}),
    ]) + '\n'
    path = tmp_path / 'rows.jsonl'; path.write_text(payload)
    loads = jsonio.json.loads
    decoded = []
    def track(line):
        decoded.append(line)
        return loads(line)
    monkeypatch.setattr(jsonio.json, 'loads', track)
    retained = list(jsonio.iter_jsonl_rows(path, split='test'))
    assert [row['id'] for _, row in retained] == ['plain', 'escaped']
    assert retained[0][1]['text'] == 'literal\u2028separator'
    assert len(decoded) == 3  # actual tests plus the conservative nested-field candidate
    assert all('token_ids' not in line for line in decoded)

