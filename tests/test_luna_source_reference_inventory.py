"""The dispatcher supports actual references that read/apply before evaluating."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from luna_source_inventory import inventory, verify


def source(tmp_path, root):
    path = tmp_path / 'source.jsonl'
    path.write_text(json.dumps({'id': 'real-read-first-shape', 'source_groups': ['train-world'],
        'split': 'train', 'curriculum': {'reference': {'root': root}},
        'semantics': {'root': 'main.nl', 'files': {'main.nl': 'Read the file, then return its contents.'}}}) + '\n')
    return path


@pytest.mark.parametrize('root', [
    [['read_file', {'path': 'input.json'}], ['eval', {'code': 'return 3;', 'finish': True}]],
    [['apply', {'function': 'bump.nl', 'folder': 'package'}], ['reply', {'text': 'done'}]],
])
def test_read_or_apply_first_reference_is_fully_bound(tmp_path, root):
    path = source(tmp_path, root)
    cases = [{'index': 0, 'seed': 42}]
    result = inventory(path, cases)
    entry = result['entries'][0]
    assert entry['root_reference_format'] == 'tool-calls-v1'
    assert entry['root_reference_sha256'] == hashlib.sha256(json.dumps(
        root, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    pin = tmp_path / 'inventory.json'
    pin.write_text(json.dumps(result))
    assert verify(pin, path, cases) == result
    changed = json.loads(path.read_text())
    changed['curriculum']['reference']['root'][0][1]['path'] = 'different.json'
    path.write_text(json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='inventory does not match'):
        verify(pin, path, cases)


def test_existing_eval_first_recovery_inventory_keeps_its_exact_contract(tmp_path):
    path = source(tmp_path, [['eval', {'code': 'return 3;'}]])
    entry = inventory(path, [{'index': 0, 'seed': 42}])['entries'][0]
    assert entry['root_code_sha256'] == hashlib.sha256(b'return 3;').hexdigest()
    assert 'root_reference_format' not in entry


def test_malformed_reference_is_rejected(tmp_path):
    path = source(tmp_path, [['read_file', 'invalid arguments']])
    with pytest.raises(ValueError, match='malformed tool calls'):
        inventory(path, [{'index': 0, 'seed': 42}])


def test_authored_native_project_is_bound_without_inventing_a_reference(tmp_path):
    path = source(tmp_path, [])
    row = json.loads(path.read_text())
    row.update(version='natlang.program/2', kind='lambda_graph')
    row['curriculum'].pop('reference')
    row['semantics']['inputs'] = {'question': 'actual task'}
    path.write_text(json.dumps(row) + '\n')
    cases = [{'index': 0, 'seed': 42}]
    result = inventory(path, cases)
    entry = result['entries'][0]
    assert entry['root_source_format'] == 'native-project-v1'
    assert 'root_code_sha256' not in entry and 'root_reference_sha256' not in entry
    pin = tmp_path / 'inventory.json'
    pin.write_text(json.dumps(result))
    for member, value in [('files', {'main.nl': 'Changed executable source.'}),
                          ('inputs', {'question': 'different task'})]:
        changed = json.loads(json.dumps(row))
        changed['semantics'][member] = value
        path.write_text(json.dumps(changed) + '\n')
        with pytest.raises(ValueError, match='inventory does not match'):
            verify(pin, path, cases)
    for name in ('../outside.nl', '/outside.nl', 'folder\\outside.nl'):
        changed = json.loads(json.dumps(row))
        changed['semantics']['files'][name] = 'Unsafe source member.'
        path.write_text(json.dumps(changed) + '\n')
        with pytest.raises(ValueError, match='complete native source project'):
            inventory(path, cases)
    changed = json.loads(json.dumps(row))
    changed['curriculum']['reference'] = {}
    path.write_text(json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='canonical root program'):
        inventory(path, cases)
