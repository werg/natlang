"""`extract` operator cases from broad graph cases (scripts/neuralese_data/operator_cases_broad_extract.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'neuralese_data'))

import broad_graph_cases as b  # noqa: E402
from neuralese_data import operator_cases_broad_extract as x  # noqa: E402
from neuralese_data.records import validate  # noqa: E402


def doc(title, size, word):
    return {'title': title, 'text': f'# {title}\n' + (word + ' ') * size, 'view': 'No relevant facts.'}


def case(upstream_id, docs, split='train', answer='Paris', source='hotpotqa', oracle='normalized', **extra):
    return b.make_case(source=source, dataset='hotpotqa/hotpot_qa distractor', upstream_id=upstream_id,
                       upstream_split=split, split=split, groups=[f'hotpotqa-q:{upstream_id}'], kind='evidence',
                       request_text='Which city is the capital of France?', answer=answer, documents=docs,
                       license={'spdx': 'CC-BY-SA-4.0'}, revision='r1', files=[], adaptation='test',
                       domain='wiki-multihop', oracle=oracle, **extra)


def test_record_maps_documents_request_and_gold_answer():
    c = case('a1', [doc('France', 150, 'france'), doc('Spain', 150, 'madrid')])
    record = x.to_record(c)
    assert validate(record) == []
    assert [s['role'] for s in record['sources']] == ['document', 'document']
    assert [s['text'] for s in record['sources']] == [d['text'] for d in c['documents']]
    assert record['writer']['instructions'] == x.WRITER_PREFIX + 'Which city is the capital of France?'
    assert record['writer']['result_type'] == 'Neuralese<string>' and record['writer']['context'] == []
    content = record['consumer']['context'][0]['content']
    assert content.startswith(c['consumer']['instructions']) and content.endswith('question: Which city is the capital of France?')
    assert record['consumer']['withheld'] == ['sources']
    assert record['target'] == {'kind': 'text', 'value': 'Paris'}
    assert record['outcome'] == {'label': 'gold', 'checked': 'normalized'}
    assert record['split_groups'] == c['groups'] and record['lineage']['upstream_id'] == 'a1'
    assert 'No relevant facts.' not in json.dumps(record['writer'])   # per-document views are never read


def test_summary_kind_is_text_and_aliases_become_alternatives():
    c = case('a2', [doc('A', 150, 'x'), doc('B', 150, 'y')], answer_kind='summary', alternates=['Paris, France'])
    record = x.to_record(c)
    assert record['target'] == {'kind': 'text', 'value': 'Paris', 'alternatives': ['Paris, France']}
    assert record['outcome']['checked'] == 'normalized'


def test_ineligible_reasons():
    ok = [doc('A', 150, 'x'), doc('B', 150, 'y')]
    assert x.ineligible(case('b1', ok)) is None
    assert x.ineligible(case('b2', ok[:1])) == 'document-count'
    assert x.ineligible(case('b3', ok + [doc('C', 5, 'z')])) == 'short-document'
    assert x.ineligible(case('b4', ok + [doc('C', 5000, 'longword')])) == 'long-document'
    assert x.ineligible(case('b5', ok, answer='  ')) == 'missing-answer'


def test_water_fill_keeps_small_sources_whole():
    assert x.water_fill({'a': 10, 'b': 3}, 100, 8) == {'a': 8, 'b': 3}
    quota = x.water_fill({'a': 50, 'b': 40, 'c': 5}, 60, 50)
    assert quota['c'] == 5 and sum(quota.values()) == 60 and abs(quota['a'] - quota['b']) <= 1


def test_convert_streams_selects_by_rank_and_writes_per_source_outputs(tmp_path):
    cases = [case(f'c{i}', [doc(f'T{i}', 150, f'w{i}'), doc(f'U{i}', 150, f'v{i}')]) for i in range(6)]
    cases.append(case('t0', [doc('T', 150, 'tt'), doc('U', 150, 'uu')], split='test'))
    cases.append(case('bad', [doc('T', 150, 'tt')]))
    path = tmp_path / 'cases.jsonl'
    path.write_text(''.join(json.dumps(c) + '\n' for c in cases))
    report = x.convert(path, tmp_path / 'out', ['hotpotqa'], train_cap=4, total_cap=10, test_cap=5)
    assert report['hotpotqa']['counts'] == {'train': 4, 'test': 1}
    assert report['hotpotqa']['rejects'] == {'document-count': 1}
    out = tmp_path / 'out' / 'extract'
    ids = [json.loads(line)['id'] for line in (out / 'broad-hotpotqa.port-records.jsonl').read_text().splitlines()]
    expected = sorted((c for c in cases[:6]), key=lambda c: x.stable_rank(c['id'], 'broad-hotpotqa'))[:4]
    assert [i for i in ids if i != 'broad-extract:hotpotqa:t0'] == ['broad-extract:' + c['id'] for c in expected]
    manifest = json.loads((out / 'broad-hotpotqa.manifest.json').read_text())
    assert manifest['operator'] == 'extract' and json.loads(manifest['notes'])['omitted_by_cap'] == {'train-cap': 2}
