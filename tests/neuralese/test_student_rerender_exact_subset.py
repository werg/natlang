import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('student_rerender', Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/student_rerender.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def doc(i, split='train', groups=('g',)):
    return {'id': i, 'split': split, 'source_groups': list(groups), 'source_record_ids': [i]}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_eligibility_is_limited_to_split_group_changes():
    item = {'kind': 'direct'}
    ok = {'problems': ['program-header: split_or_group_changed 3'], 'target_parse': {'program-header:exact': 9}}
    assert mod.subset_eligible(ok, item)
    assert not mod.subset_eligible({**ok, 'target_parse': {'program-header:differ': 1}}, item)
    assert not mod.subset_eligible({**ok, 'problems': ok['problems'] + ['program-header: decode_mismatch 1']}, item)
    assert not mod.subset_eligible({'problems': [], 'target_parse': {}}, item)
    parts = {'problems': ok['problems'], 'target_parse': 'not checked (parts render: sources are streamed part inputs)'}
    assert mod.subset_eligible(parts, {'kind': 'parts'})
    assert not mod.subset_eligible(parts, item)


def test_exact_subset_keeps_identical_documents_and_drops_held_connected(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    item = {'id': 'x', 'out': 'new', 'path': 'old', 'kind': 'direct', 'views': ['program-header']}
    queue = {'student': 'lfm3', 'reference': 'lfm25'}
    write(tmp_path / 'old/program-header/lfm25/text-data.jsonl',
          [doc('a', groups=('g1',)), doc('b', groups=('g2',)), doc('c', groups=('g3',)), doc('h', 'test', ('g4',))])
    write(tmp_path / 'new/program-header/lfm3/text-data.jsonl',
          [doc('a', groups=('g1',)),                 # identical: kept
           doc('b', 'test', ('g2',)),                # split changed: dropped, and its group is held now
           doc('c', groups=('g3', 'g2')),            # groups changed: dropped
           doc('d', groups=('g5',)),                 # only in the student render: dropped
           doc('h', 'test', ('g4',))])               # identical held: kept
    report = {'problems': ['program-header: split_or_group_changed 2'], 'target_parse': {}}
    result = mod.write_exact_subset(item, queue, report)
    view = result['views']['program-header']
    assert view['counts'] == {'train': 1, 'test': 1, 'excluded_nonidentical': 3}
    assert sorted(view['excluded_ids']) == ['b', 'c', 'd']
    path = mod.serving_view_file(item, 'program-header', 'lfm3')
    assert path == tmp_path / 'new' / view['file']
    assert path.suffix == '.zst'


def test_exact_subset_drops_train_rows_in_newly_held_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    item = {'id': 'x', 'out': 'new', 'path': 'old', 'kind': 'direct', 'views': ['program-header']}
    queue = {'student': 'lfm3', 'reference': 'lfm25'}
    write(tmp_path / 'old/program-header/lfm25/text-data.jsonl', [doc('a', groups=('g1',)), doc('b', groups=('g1',))])
    write(tmp_path / 'new/program-header/lfm3/text-data.jsonl', [doc('a', groups=('g1',)), doc('b', 'test', ('g1',))])
    view = mod.write_exact_subset(item, queue, {'problems': []})['views']['program-header']
    assert view['counts'] == {'excluded_held_group': 1, 'excluded_nonidentical': 1}


def test_reference_free_verification_checks_overlap_and_decode(tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    queue = {'student': 'lfm3', 'reference': 'lfm25', 'items': {'x': {
        'id': 'x', 'out': 'new', 'path': None, 'kind': 'parts', 'views': ['program-header']}}}
    (tmp_path / 'q.json').write_text(json.dumps(queue))
    (tmp_path / 'new').mkdir()
    (tmp_path / 'new/receipt.json').write_text('{}')

    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return ''.join(chr(i) for i in ids)
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tok())
    import sys
    monkeypatch.setitem(sys.modules, 'student_tokenizers', types.SimpleNamespace(resolve=lambda s: {'path': 'tok'}))
    rows = [{**doc('a', groups=('g1',)), 'text': 'ab', 'token_ids': [97, 98]},
            {**doc('b', 'test', ('g2',)), 'text': 'c', 'token_ids': [99]}]
    write(tmp_path / 'new/program-header/lfm3/text-data.jsonl', rows)
    mod.verify(types.SimpleNamespace(queue=str(tmp_path / 'q.json'), id='x'))
    report = json.loads((tmp_path / 'new/student-verification.json').read_text())
    assert report['passed'] and report['reference'] is None
    rows[1]['source_groups'] = ['g1']
    write(tmp_path / 'new/program-header/lfm3/text-data.jsonl', rows)
    mod.verify(types.SimpleNamespace(queue=str(tmp_path / 'q.json'), id='x'))
    report = json.loads((tmp_path / 'new/student-verification.json').read_text())
    assert not report['passed'] and report['problems'] == ['program-header: train_test_group_overlap 1']
