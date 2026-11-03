"""compact-params converter over inline S1-shaped fixture rows (scripts/neuralese_data/operator_cases_compact_params.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from neuralese_data import operator_cases_compact_params as cp  # noqa: E402
from neuralese_data.records import VERSION, group_key, license_, seal, source, validate  # noqa: E402


def row(index, text, *, sources=None, split='train', label='gold', question='Who wrote it?'):
    sources = sources or [source('document', text)]
    return seal({
        'version': VERSION, 'id': f'bgkit:fixture:{index}', 'family': 'qa_narrative', 'task': 'consume',
        'sources': sources,
        'writer': {'instructions': 'Read the text so that this question can be answered: ' + question,
                   'result_type': 'Neuralese<Document>', 'context': []},
        'consumer': {'context': [{'role': 'user', 'content': question}], 'withheld': ['sources']},
        'target': {'kind': 'text', 'value': 'Ada', 'alternatives': []},
        'outcome': {'label': label, 'checked': 'reference-answer'},
        'lineage': {'project': 'bgkit', 'store': 'fixture', 'store_version': None, 'row': index, 'upstream': 'x',
                    'upstream_id': str(index), 'upstream_revision': None, 'teacher': None, 'converter': 'test',
                    'sha256': ''},
        'license': license_('MIT'), 'split': split, 'split_groups': [group_key('doc', str(index))]})


def test_converter_rewrites_purpose_and_drops_by_reason(tmp_path, monkeypatch):
    s1 = tmp_path / 's1'
    s1.mkdir()
    text = lambda n: ('paragraph %d about Ada. ' % n) * 400  # noqa: E731
    rows = [row(0, text(0)), row(1, text(1), split='test'), row(2, text(2), split='validation'),
            row(3, 'short'), row(4, text(4), sources=[source('document', text(4)), source('document', text(5)), source('document', 'x' * 300)]),
            row(7, text(7), sources=[source('document', text(i)) for i in range(9)]),
            row(8, text(8), sources=[source('document', text(8)), source('document', 'tiny')]),
            row(9, text(9), sources=[source('document', text(9))] * 2 + [source('document', 'y' * 95_000)]),
            row(5, text(6), label='teacher'), row(6, text(6) * 40)]
    (s1 / 'qa_narrative.port-records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    monkeypatch.setattr(cp, 'S1', s1)
    summary = cp.convert_family('qa_narrative', 's1-qa-narrative', 10, tmp_path / 'out', None)
    assert summary['counts'] == {'train': 2, 'test': 2}
    out = tmp_path / 'out' / 'compact-params'
    cases = [json.loads(line) for line in (out / 's1-qa-narrative.port-records.jsonl').read_text().splitlines()]
    assert len(cases) == 4 and all(validate(case) == [] for case in cases)
    train = next(case for case in cases if case['split'] == 'train' and len(case['sources']) == 1)
    assert train['writer']['instructions'] == 'The call that receives this document answers: Who wrote it?'
    multi = next(case for case in cases if len(case['sources']) == 3)
    assert multi['writer']['instructions'] == 'The call that receives these 3 arguments (document) answers: Who wrote it?'
    assert train['consumer'] == rows[0]['consumer'] and train['target'] == rows[0]['target']
    assert train['split_groups'] == rows[0]['split_groups']
    manifest = json.loads((out / 's1-qa-narrative.manifest.json').read_text())
    assert manifest['rejects'] == {'sources-above-8': 1, 'below-4096-chars': 1, 'label': 1, 'above-120000-chars': 1,
                                  'source-below-200-chars': 1, 'total-above-110000-chars': 1}
