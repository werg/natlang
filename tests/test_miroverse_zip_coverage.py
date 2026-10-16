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


def _array_fixture(tmp_path: Path, source_files: list[tuple[str, bytes]], archive_files: list[tuple[str, bytes]]):
    root = tmp_path / 'source'
    (root / 'jsonl_sft').mkdir(parents=True)
    (root / 'zip_sft').mkdir()
    source_paths = []
    for name, body in source_files:
        path = root / 'jsonl_sft' / name
        path.write_bytes(body)
        source_paths.append(path)
    archive_path = root / 'zip_sft' / 'MiroVerse-v0_1-SFT.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in archive_files:
            archive.writestr(name, body)
    files = []
    for path in (*source_paths, archive_path):
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


def test_zip_coverage_matches_split_json_arrays_to_combined_jsonl_sources(tmp_path):
    root, receipt = _array_fixture(tmp_path,
        [('a.jsonl', b'{"id":"a","v":1}\n{"id":"b","v":2}\n'),
         ('b.jsonl', b'{"id":"c","v":3}\n')],
        [('MiroVerse_v0_1/split_0_2.json', b' [ {"id":"b","v":2}, {"id":"a","v":1} ] '),
         ('MiroVerse_v0_1/split_2_3.json', b'[{"id":"c","v":3}]')])
    result = audit(receipt, root, tmp_path / 'coverage.json')
    assert result['archive_json_array_member_count'] == 2
    assert [x['rows'] for x in result['archive_json_array_members']] == [2, 1]
    assert result['global_archive_vs_jsonl_source']['exact_row_multiset_match'] is True
    assert result['global_archive_vs_jsonl_source']['archive_rows'] == 3
    assert result['complete_jsonl_coverage'] is True
    assert result['all_archive_payloads_accounted_for'] is True


@pytest.mark.parametrize('body', [b'[{}] trailing', b'[{},{},]', b'[{} {}]'])
def test_streaming_json_array_rejects_invalid_or_trailing_content(tmp_path, body):
    root, receipt = _array_fixture(tmp_path, [('a.jsonl', b'{"id":"a"}\n')],
        [('MiroVerse_v0_1/split_0_1.json', body)])
    with pytest.raises(ValueError):
        audit(receipt, root, tmp_path / 'coverage.json')
