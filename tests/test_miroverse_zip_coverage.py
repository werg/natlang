import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.neuralese_data.audit_miroverse_zip_coverage import audit


def _fixture(tmp_path: Path, source_rows: bytes, archive_rows: bytes):
    root = tmp_path / 'source'
    (root / 'jsonl_sft').mkdir(parents=True)
    (root / 'zip_sft').mkdir()
    source = root / 'jsonl_sft' / 'part.jsonl'
    source.write_bytes(source_rows)
    archive_path = root / 'zip_sft' / 'MiroVerse-v0_1-SFT.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('jsonl_sft/part.jsonl', archive_rows)
    files = []
    for path in (source, archive_path):
        data = path.read_bytes()
        files.append({'path': str(path.relative_to(root)), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    receipt = root / 'acquisition.json'
    receipt.write_text(json.dumps({'schema': 'natlang.external-source-acquisition/1', 'ref': 'D24',
        'repo': 'miromind-ai/MiroVerse-v0.1', 'revision': 'abc123', 'files': files}))
    return root, receipt


def test_zip_coverage_compares_canonical_row_multisets_not_order(tmp_path):
    root, receipt = _fixture(tmp_path,
        b'{"id":"a","value":1}\n{"id":"b","value":2}\n',
        b'{ "id": "b", "value": 2 }\n{"id":"a","value":1}\n')
    result = audit(receipt, root, tmp_path / 'coverage.json')
    assert result['all_paired_rows_exact'] is True
    assert result['pairs'][0]['archive_rows'] == result['pairs'][0]['source_rows'] == 2
    assert result['pairs'][0]['exact_row_multiset_match'] is True
    assert result['pairs'][0]['archive_member_sha256'] != result['pairs'][0]['source_sha256']


def test_zip_coverage_reports_same_identity_different_payload(tmp_path):
    root, receipt = _fixture(tmp_path,
        b'{"id":"a","value":1}\n',
        b'{"id":"a","value":2}\n')
    result = audit(receipt, root, tmp_path / 'coverage.json')
    pair = result['pairs'][0]
    assert result['all_paired_rows_exact'] is False
    assert pair['exact_row_multiset_match'] is False
    assert pair['same_identity_different_payload_sample']


def test_zip_coverage_refuses_source_bytes_outside_acquisition_receipt(tmp_path):
    root, receipt = _fixture(tmp_path, b'{"id":"a"}\n', b'{"id":"a"}\n')
    (root / 'jsonl_sft' / 'part.jsonl').write_bytes(b'{"id":"changed"}\n')
    with pytest.raises(ValueError, match='differs from immutable receipt'):
        audit(receipt, root, tmp_path / 'coverage.json')
