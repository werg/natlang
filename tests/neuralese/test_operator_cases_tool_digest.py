"""compact-value tool-digest converter (scripts/neuralese_data/operator_cases_tool_digest.py) on inline fixtures."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from neuralese_data import bgkit  # noqa: E402
from neuralese_data import operator_cases_tool_digest as td  # noqa: E402


def digest_row(workspace, lines, split='train', purpose='find the failing import', key='k'):
    context = '\n'.join(f'src/{workspace}/mod{i}.py:{i + 10}:     import thing{i}  # see https://example.org/{workspace}/{i}'
                        for i in range(lines))
    return {'family': 'tool_digest', 'framing': 'tool_response', 'context': context,
            'prompt': f'Compress the search result to digest: {purpose}\n', 'instruction': '', 'tool_name': 'search',
            'tool_args': '{"query": "thing"}', 'target': 'A digest written by a teacher.', 'split': split,
            'meta': json.dumps({'key': f'{workspace}-{key}'})}


def s1_record(index, row):
    text_row = dict(row, context=row['context'] + f'\n/workspace/org__{index}__1.0/x.py')
    return bgkit.convert_tool_digest('tool_digest_v2', index, text_row, {})


def test_retarget_refs_and_lines_are_exact_and_ordered():
    record = s1_record(0, digest_row('a', 60))
    refs = td.retarget(record, 'refs')
    assert refs['id'].endswith(':refs') and refs['outcome']['label'] == 'checked'
    assert refs['outcome']['checked'] == 'code:exact_refs_from' and refs['lineage']['teacher'] is None
    text = record['sources'][0]['text']
    value = refs['target']['value']
    assert len(value) == td.MAX_LISTED_REFS and all(item in text for item in value)
    assert [text.find(item) for item in value] == sorted(text.find(item) for item in value)
    question = refs['consumer']['context'][0]['content']
    assert question.startswith('From the search result read for "find the failing import", list the exact file paths')
    assert 'List at most the first 24.' in question and refs['consumer']['withheld'] == ['sources']
    lines = td.retarget(record, 'lines')
    stats = lines['target']['value']
    assert stats['lines'] == 61 and stats['first'].startswith('src/a/mod0.py:10:') and stats['last'] == '/workspace/org__0__1.0/x.py'
    seed = refs['sources'][0]['meta']['operator_seed']
    assert seed == {'text': '\n'.join(value), 'provenance': 'checked', 'origin': 'code:exact_refs_from'}
    assert 'operator_seed' not in record['sources'][0].get('meta', {})
    line_seed = lines['sources'][0]['meta']['operator_seed']
    assert line_seed['origin'] == 'code:line-stats' and line_seed['text'].startswith('61 lines\nfirst: src/a/mod0.py:10:')
    assert lines['outcome']['checked'] == 'code:line-stats' and lines['family'] == 'tool_digest_lines'


def test_convert_selects_closes_groups_and_reports_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(bgkit, 'TOOL_DIGEST_EXTRA_STORES', {})
    rows = [digest_row('a', 60, key='1'), digest_row('b', 60, 'test', key='2'),
            digest_row('c', 3, key='3'), digest_row('d', 60, key='4', purpose='x')]
    records = [s1_record(index, row) for index, row in enumerate(rows)]
    # the fourth row belongs to the second row's repository world: it must follow it into test
    records[3]['split_groups'] = list(records[1]['split_groups'])
    s1 = tmp_path / 'tool_digest.port-records.jsonl'
    s1.write_text(''.join(json.dumps(record) + '\n' for record in records))
    manifest = td.convert(tmp_path / 'out', s1_path=s1, tasks_dir=tmp_path, train_cap=5, test_cap=5)
    cases = [json.loads(line) for line in (tmp_path / 'out' / 'compact-value' / 'tool-digest.port-records.jsonl')
             .read_text().splitlines()]
    by_id = {case['id']: case['split'] for case in cases}
    assert by_id['bgkit:tool_digest_v2:0:refs'] == 'train' and by_id['bgkit:tool_digest_v2:0:lines'] == 'train'
    assert by_id['bgkit:tool_digest_v2:1:refs'] == 'test' and by_id['bgkit:tool_digest_v2:3:refs'] == 'test'
    assert not any(case_id.startswith('bgkit:tool_digest_v2:2:') for case_id in by_id)
    assert manifest['counts'] == {'train': 2, 'test': 4}
    assert manifest['rejects']['source-size'] == 1
