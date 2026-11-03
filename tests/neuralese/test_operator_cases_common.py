"""The shared learned-operator case contract (scripts/neuralese_data/operator_cases_common.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from neuralese_data.operator_cases_common import CaseSink, operator_problems  # noqa: E402
from neuralese_data.records import VERSION, group_key, license_, seal, source  # noqa: E402


def record(index, text, label='gold'):
    return seal({
        'version': VERSION, 'id': f'test:value:{index}', 'family': 'tool_digest_refs', 'task': 'consume',
        'sources': [source('tool_output', text, meta={'tool': 'bash'})],
        'writer': {'instructions': 'Read this bash result for: find the failing test', 'result_type': 'Neuralese<ToolResult>',
                   'context': []},
        'consumer': {'context': [{'role': 'user', 'content': 'List the exact paths.'}], 'withheld': ['sources']},
        'target': {'kind': 'json', 'value': ['src/a.py:3'], 'alternatives': []},
        'outcome': {'label': label, 'checked': 'code:exact_refs_from'},
        'lineage': {'project': 'bgkit', 'store': 'tool_digest_v2', 'store_version': 'v2', 'row': index, 'upstream': None,
                    'upstream_id': None, 'upstream_revision': None, 'teacher': None, 'converter': 'test', 'sha256': '',
                    'notes': {}},
        'license': license_('MIT'), 'split': 'train', 'split_groups': [group_key('repo', 'a/b')]})


def test_sink_validates_dedupes_and_writes_manifest(tmp_path):
    long_text = 'src/a.py:3 failed\n' * 200
    with CaseSink('compact-value', 'unit', converter=__file__, inputs=[__file__], out_root=tmp_path) as sink:
        assert sink.offer(record(0, long_text))
        assert not sink.offer(record(1, long_text))           # same sources and target
        assert not sink.offer(record(2, 'short'))             # below the runtime cut threshold
        assert not sink.offer(record(3, long_text + 'x', label='teacher'))
    lines = (tmp_path / 'compact-value' / 'unit.port-records.jsonl').read_text().splitlines()
    assert [json.loads(line)['id'] for line in lines] == ['test:value:0']
    manifest = json.loads((tmp_path / 'compact-value' / 'unit.manifest.json').read_text())
    assert manifest['counts'] == {'train': 1} and manifest['operator_version'] == 2
    assert manifest['seed_provenance'] == {'mechanical:identifiers-outline': 1}
    assert json.loads(lines[0])['sources'][0]['meta']['operator_seed']['provenance'] == 'mechanical:identifiers-outline'
    assert sum(manifest['rejects'].values()) == 3


def test_extract_requires_several_documents():
    problems = operator_problems(record(0, 'x' * 3000), 'extract')
    assert any('sources: 1 not in [2, 8]' in problem for problem in problems)


def test_compact_params_accepts_several_arguments_with_one_over_the_cut_floor():
    rec = record(0, 'x' * 5000)
    rec['sources'] = [{**rec['sources'][0], 'role': 'passage'}, {**rec['sources'][0], 'role': 'passage', 'text': 'y' * 300}]
    assert operator_problems(rec, 'compact-params') == []
    rec['sources'] = [{**rec['sources'][0], 'text': 'y' * 300}] * 2
    assert any('compaction floor' in problem for problem in operator_problems(rec, 'compact-params'))
    rec['sources'] = [{**rec['sources'][0], 'text': 'z' * 60_000}] * 2
    assert any('above 110000' in problem for problem in operator_problems(rec, 'compact-params'))


def test_filled_seeds_keep_records_valid(tmp_path):
    from neuralese_data.records import validate
    with CaseSink('compact-value', 'unit', converter=__file__, inputs=[__file__], out_root=tmp_path) as sink:
        assert sink.offer(record(0, 'src/a.py:3 failed\n' * 200))
    line = (tmp_path / 'compact-value' / 'unit.port-records.jsonl').read_text().splitlines()[0]
    assert validate(json.loads(line)) == []
