import io
import json

import pytest
from scripts.audit_recovered_sources import array_items, checked_input


def test_streaming_source_json_preserves_large_and_unicode_rows():
    values = [{'question': 'x' * 70000, 'unicode': '工单'}, {'question': 'next'}]
    stream = io.StringIO(json.dumps({'meta': {}, 'questions': values}, ensure_ascii=False))
    assert list(array_items(stream, 'questions')) == values


def test_truncated_source_array_is_not_a_successful_join():
    with pytest.raises(ValueError, match='truncated'):
        list(array_items(io.StringIO('{"questions":[{"id":1}'), 'questions'))


def test_preserved_ir_hash_is_required(tmp_path):
    path = tmp_path / 'source.jsonl'
    path.write_text('{}\n')
    (tmp_path / 'source.jsonl.manifest.json').write_text(json.dumps({'ir_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='checksum mismatch'):
        checked_input(path)
