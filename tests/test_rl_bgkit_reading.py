"""bgkit knowledge-reasoning RL reader sources (scripts/rl_tasks/sources/bgkit_reading.py) on tiny synthetic stores."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
pa = pytest.importorskip('pyarrow')
import pyarrow.parquet as pq  # noqa: E402

from scripts.rl_tasks.sources import bgkit_reading as br  # noqa: E402

FILLER = 'The river ran past the old mill and the town grew around it over many years. '


def _babi_row(i, task='qa2'):
    facts = ['Mary moved to the hallway.', 'John went to the garden.', 'Mary got the apple there.', 'Mary travelled to the office.',
             'John took the milk.', 'Sandra journeyed to the kitchen.', 'Mary dropped the apple.', 'Daniel went to the bedroom.']
    parts = []
    for k, fact in enumerate(facts):
        parts.append(FILLER * 6 + fact)
    context = ' '.join(parts) + ' ' + FILLER * 4
    return {'context': context, 'instruction': 'Where is the apple?', 'target': 'office', 'split': 'train',
            'meta': json.dumps({'doc': f'babi:{task}:{i}:0', 'task': task, 'length': 1000, 'derived': False,
                                'filler_source': 'wikitext', 'n_facts': len(facts)})}


def test_haystack_cases_keep_story_units_and_checked_notes(tmp_path, monkeypatch):
    rows = [_babi_row(i) for i in range(40)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'babi_filler_v3.parquet')
    monkeypatch.setattr(br, 'BGKIT_TASKS', tmp_path)
    cases = br.haystack(train=(6, 6, 6, 6), held=2)
    train = [c for c in cases if c['split'] == 'train']
    assert len(train) == 24 and all(c['id'].startswith('ext:bgkit:') for c in cases)
    assert len({c['id'] for c in cases}) == len(cases)
    shapes = {c['curriculum']['shape'].rsplit('-qa', 1)[0] for c in train}
    assert shapes == {'haystack-state-reducer', 'haystack-fact-finders', 'haystack-plan-then-spot', 'haystack-triage-then-spot'}
    for case in train:
        assert case['semantics']['expected'] == 'office'
        # one story is one split unit: every chunk reader note is a checked sentence list found by the fact pattern
        notes = case['curriculum']['reference']['children']
        assert notes and all(n['provenance'] == 'checked' for n in notes)
        assert 'Mary' in json.dumps([n['value'] for n in notes])
        assert case['source_groups'] and any(g.startswith('babi-story:') for g in case['source_groups'])


def test_searchqa_snippets_and_aliases():
    parsed = br._snippets('[DOC] [TLE] Gehrig bio [PAR] Lou Gehrig, nicknamed the Iron Horse, played first base for the Yankees. '
                          '[DOC] [TLE] short [PAR] x [DOC] [TLE] Columbia [PAR] The Columbia football and lacrosse star later joined the Yankees.')
    assert [s['title'] for s in parsed] == ['Gehrig bio', 'Columbia']
    assert br._alias_forms('(Lou) Gehrig') == ['Gehrig']
    assert br._alias_forms('The Beatles') == ['Beatles']


def test_chunks_never_split_a_sentence_and_cover_the_text():
    text = ' '.join(f'Sentence number {i} has a few words in it.' for i in range(200))
    chunks = br._chunks(text, size=400)
    assert ' '.join(chunks) == text and all(c.endswith('.') for c in chunks)


def test_narrative_items_require_a_verbatim_short_span(tmp_path, monkeypatch):
    summary = ' '.join(f'In scene {i} the heroine Ada walks to the harbour with her brother Tom and sees a ship.' for i in range(30))
    rows = [{'context': summary, 'instruction': 'Who does Ada walk with?', 'target': 'her brother Tom', 'split': 'train',
             'meta': json.dumps({'doc': 'd1', 'title': 't', 'kind': 'gutenberg', 'answers': ['her brother Tom', 'Tom']})},
            {'context': summary, 'instruction': 'Who is the villain?', 'target': 'nobody here', 'split': 'train',
             'meta': json.dumps({'doc': 'd1', 'title': 't', 'kind': 'gutenberg', 'answers': ['an answer the text never says']})}]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'qa_narrativeqa.parquet')
    monkeypatch.setattr(br, 'BGKIT_TASKS', tmp_path)
    items = br.narrative_items(limit=(5, 2))['train']
    assert [i['question'] for i in items] == ['Who does Ada walk with?'] and items[0]['groups'] == ['narrativeqa-doc:d1']


def test_overlap_with_pool_counts_repeated_programs(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts' / 'rl_tasks'))
    from measure_repetition import overlap_with_pool

    def row(ident, text):
        return {'id': ident, 'split': 'train', 'family': 'f', 'generation': {'generator': 'g'},
                'curriculum': {'mode': 'single_call', 'rl': {'level': 1, 'family': 'f'}},
                'semantics': {'root': 'f.nl', 'files': {'f.nl': '---\nargs: {}\nreturns: "string"\n---\n' + text},
                              'inputs': {}, 'expected': 'x', 'oracle': 'exact'}}
    pool, mine = tmp_path / 'pool.jsonl', tmp_path / 'mine.jsonl'
    pool.write_text(json.dumps(row('p', 'Sort the invoices by due date and return the earliest.')) + '\n')
    mine.write_text(''.join(json.dumps(row(i, t)) + '\n' for i, t in (
        ('a', 'Sort the invoices by due date and return the earliest.'), ('b', 'Compute the median latency of the samples.'))))
    out = overlap_with_pool([str(mine)], [str(pool)])
    assert out['rows_with_instruction_in_pool'] == 1 and out['rows_with_functional_program_in_pool'] == 1
    assert out['rows_in_families_absent_from_pool'] == 0
