"""compact-value web_search_r1_v3 converter (scripts/neuralese_data/operator_cases_web_search_r1.py) on inline rows."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from neuralese_data import operator_cases_web_search_r1 as ws  # noqa: E402
from neuralese_data.common import Reject  # noqa: E402

FILLER = ' The article continues with unrelated details about the subject.' * 30


def row(question, answer, titles=('Moist (Canadian band)',), body=FILLER, split='train'):
    pages = [f'[{1000 + i}] {title}\n{title} is the subject. Answer text: {answer}.{body}' for i, title in enumerate(titles)]
    return {'family': 'web_search_r1', 'framing': 'tool_slots', 'context': json.dumps(pages),
            'prompt': json.dumps([f'web search some query: {question}']),
            'target': json.dumps([{'role': 'assistant', 'content': f'ANSWER: {answer}', 'reasoning_content': 'think'}]),
            'split': split, 'meta': json.dumps({'dataset': 'ThornZ/Search-R1-SFT', 'answer_match': 'em'})}


def test_convert_row_maps_spec_fields():
    record = ws.convert_row(7, row('Who founded Moist?', 'David Usher', titles=('Moist', 'Usher')))
    assert record['sources'][0]['role'] == 'tool_output' and record['sources'][0]['text'].count('\n\n') == 1
    assert record['writer']['instructions'] == 'Read these web_search results for: Who founded Moist?'
    assert record['consumer']['context'] == [{'role': 'user', 'content': 'Who founded Moist?\nAnswer with a short phrase.'}]
    assert record['target']['value'] == 'David Usher' and record['outcome'] == {
        'label': 'gold', 'checked': 'dataset-answer-em', 'details': {'answer_match': 'em', 'dataset': 'ThornZ/Search-R1-SFT'}}
    assert 'wiki:moist' in record['split_groups'] and 'wiki:usher' in record['split_groups']
    assert record['split_groups'][0].startswith('search-r1-q:')


def test_gold_aliases_exact_match_and_non_final_rows():
    aliased = row('Who is "The Boss"?', 'Bruce Springsteen')
    aliased['meta'] = json.dumps({'dataset': 'd', 'answer_match': 'em', 'gold': ['bruce springsteen', 'Springsteen']})
    record = ws.convert_row(1, aliased)
    assert record['target']['alternatives'] == ['bruce springsteen', 'Springsteen']
    wrong = row('Q?', 'Springsteen')
    wrong['meta'] = json.dumps({'dataset': 'd', 'answer_match': 'em', 'gold': ['Bruce Springsteen']})
    with pytest.raises(Reject, match='answer-not-em'):
        ws.convert_row(2, wrong)
    non_final = row('Q?', 'x')
    non_final['target'] = json.dumps([{'role': 'assistant', 'content': '', 'reasoning_content': 'think'}])
    with pytest.raises(Reject, match='no-final-answer'):
        ws.convert_row(3, non_final)
    declared = row('Who?', 'A')
    declared['prompt'] = json.dumps(['web search who: what: Who is A?'])
    declared['meta'] = json.dumps({'answer_match': 'em', 'question': 'what: Who is A?'})
    assert ws.convert_row(4, declared)['writer']['instructions'].endswith('for: what: Who is A?')


def test_answer_must_be_in_results():
    bad = row('Q?', 'zzz')
    bad['target'] = json.dumps([{'role': 'assistant', 'content': 'ANSWER: absent'}])
    with pytest.raises(Reject, match='answer-not-in-results'):
        ws.convert_row(0, bad)


def test_convert_caps_dedupes_and_carves_test(tmp_path, monkeypatch):
    rows = [row(f'Question number {i}?', f'answer{i}') for i in range(40)]
    rows.append(rows[0])                                              # duplicate of row 0
    rows.append(row('Short?', 'x', body=''))                          # below the runtime cut threshold
    monkeypatch.setattr(ws, 'parquet_rows', lambda path, limit=None: iter(enumerate(rows)))
    manifest = ws.convert(tmp_path / 'out', train_cap=10, test_cap=1, path=Path(__file__))
    assert manifest['counts']['train'] == 10 and manifest['counts'].get('test', 0) <= 1
    assert manifest['rejects']['duplicate'] == 1 and manifest['rejects']['source-size'] == 1
    cases = [json.loads(line) for line in (tmp_path / 'out' / 'compact-value' / 'web-search-r1.port-records.jsonl')
             .read_text().splitlines()]
    questions = {case['consumer']['context'][0]['content'] for case in cases}
    assert len(questions) == len(cases)
