"""bgkit knowledge-reasoning RL reader sources (scripts/rl_tasks/sources/bgkit_reading.py) on tiny synthetic stores."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
pa = pytest.importorskip('pyarrow')
import pyarrow.parquet as pq  # noqa: E402

from scripts.rl_tasks.sources import bgkit_programs as bp  # noqa: E402
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


STORIES = {
    'qa1': ('Where is Mary?', 'office'), 'qa2': ('Where is the apple?', 'office'),
    'qa3': ('Where was the apple before the office?', 'hallway'),
}


def _babi_store(tmp_path, monkeypatch, stories=96):
    rows = []
    for i in range(stories):
        task = ('qa1', 'qa2', 'qa3')[i % 3]
        row = _babi_row(i, task)
        row['instruction'], row['target'] = STORIES[task]
        if i >= stories - 36:
            row['split'] = 'eval'
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'babi_filler_v3.parquet')
    monkeypatch.setattr(br, 'BGKIT_TASKS', tmp_path)


def _signatures(cases):
    """program name -> set of functional signatures (measure_repetition.functional_signature) of its cases."""
    sys.path.insert(0, str(ROOT / 'scripts' / 'rl_tasks'))
    from measure_repetition import functional_signature
    out = {}
    for case in cases:
        out.setdefault(case['curriculum']['external']['structure']['program'], set()).add(functional_signature(case))
    return out


def _assert_distinct_programs(cases, programs):
    train = [c for c in cases if c['split'] == 'train']
    names = {c['curriculum']['external']['structure']['program'] for c in train}
    assert names == {p.name for p in programs}
    signatures = _signatures(train)
    seen = set()
    for name, group in signatures.items():
        assert not (group & seen), f'{name} shares a functional signature with another program'
        seen |= group
    for case in train:
        structure = case['curriculum']['external']['structure']
        assert structure['smoke']['root'][0][0] == 'eval' and structure['smoke']['root'][0][1]['code'].strip()
        root = case['semantics']['root']
        files = case['semantics']['files']
        assert len(files) == structure['functions'] and root in files
        # decision readouts are leaves that call nothing
        for path, text in files.items():
            if 'readout: decision' in text.split('---')[1]:
                assert path != root and not re.search(r'\b(decide|await)\b', text.split('---', 2)[2])
    return train


def _note_values_vary(train):
    """A reference note is derived from its item: no producer site carries one constant value across items."""
    by_site = {}
    for case in train:
        for entry in case['curriculum']['reference'].get('children') or []:
            assert entry['provenance'] == 'checked' or entry['provenance'].startswith('mechanical:')
            site = (case['curriculum']['external']['structure']['program'], entry['match'][0])
            by_site.setdefault(site, set()).add(json.dumps(entry['value'], sort_keys=True))
    return {site: len(values) for site, values in by_site.items()}


def test_haystack_cases_keep_story_units_and_checked_notes(tmp_path, monkeypatch):
    _babi_store(tmp_path, monkeypatch)
    cases = br.haystack(train=2, held=1)
    assert len({c['id'] for c in cases}) == len(cases) and all(c['id'].startswith('ext:bgkit:') for c in cases)
    train = _assert_distinct_programs(cases, br.HAY_PROGRAMS)
    assert len(br.HAY_PROGRAMS) == 12 and len(train) == 24
    for case in train:
        question = case['semantics']['inputs']['question']
        assert case['semantics']['expected'] == STORIES[next(t for t, (q, _) in STORIES.items() if q == question)][1]
        assert case['source_groups'] and any(g.startswith('babi-story:') for g in case['source_groups'])
        assert all(n['provenance'] == 'checked' for n in case['curriculum']['reference'].get('children') or [])
    # the person-only programs only get questions that name a person
    for case in train:
        if case['curriculum']['external']['structure']['program'] in bp.HAY_PERSON_ONLY:
            assert case['semantics']['inputs']['question'] == 'Where is Mary?'
    ledgers = [c for c in train if c['curriculum']['external']['structure']['program'] == 'haystack-person-ledgers']
    note = ledgers[0]['curriculum']['reference']['children'][0]
    assert note['value']['person'] == 'Mary' and all(m.startswith('Mary ') for m in note['value']['moves'])
    windows = [c for c in train if c['curriculum']['external']['structure']['program'] == 'haystack-window-readers']
    assert all(n['value']['facts'] for c in windows for n in c['curriculum']['reference']['children'])


def test_replay_of_a_hidden_story_answers_all_three_question_forms():
    facts = ['Mary moved to the hallway.', 'John went to the garden.', 'Mary got the apple there.', 'Mary travelled to the office.',
             'John took the milk.', 'Mary dropped the apple.', 'John went back to the kitchen.', 'Mary picked up the apple.',
             'Mary journeyed to the bedroom.']
    assert bp.replay(facts, 'Where is Mary?') == 'bedroom'
    assert bp.replay(facts, 'Where is the apple?') == 'bedroom'  # carried
    assert bp.replay(facts, 'Where is the milk?') == 'kitchen'  # carried by John
    assert bp.replay(facts[:6], 'Where is the apple?') == 'office'  # dropped there
    assert bp.replay(facts, 'Where was the apple before the bedroom?') == 'office'
    assert bp.replay(facts, 'Where was the apple before the hallway?') is None  # never there before: not answerable
    assert bp.replay(['Sandra took the ball.'], 'Where is the ball?') is None  # a person never seen to move


def test_grep_and_count_eligibility_follow_the_generator_text():
    item = {'chunks': ['The tide came in. Mary moved to the hallway. The harbour was quiet.', 'John Smith was born in Ohio.'],
            'facts': ['Mary moved to the hallway.']}
    assert bp.grep_eligible(item) and bp.count_eligible(item)
    assert not bp.grep_eligible({**item, 'chunks': ['It said, Mary moved to the hallway. And so on.'], 'facts': ['Mary moved to the hallway. And so on.']})


def test_narrative_programs_are_distinct_and_notes_are_derived(tmp_path, monkeypatch):
    rows = []
    for d in range(60):
        summary = ' '.join(f'In scene {i} of story {d} the heroine Ada walks to harbour {d + i} with her brother Tom{d} and sees ship {i}.'
                           for i in range(24))
        answer = f'Tom{d}'
        rows.append({'context': summary, 'instruction': f'Who does Ada walk with in story {d}?', 'target': answer,
                     'split': 'train' if d < 36 else 'eval',
                     'meta': json.dumps({'doc': f'doc{d}', 'title': 't', 'kind': 'gutenberg', 'answers': [answer]})})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'qa_narrativeqa.parquet')
    monkeypatch.setattr(br, 'BGKIT_TASKS', tmp_path)
    cases = br.narrative(train=2, held=1)
    train = _assert_distinct_programs(cases, br.NQ_PROGRAMS)
    assert len(br.NQ_PROGRAMS) == 12 and len(train) == 24
    variation = _note_values_vary(train)
    assert variation and all(n >= 1 for n in variation.values())
    timeline = next(c for c in train if c['curriculum']['external']['structure']['program'] == 'story-event-timeline')
    last = timeline['curriculum']['reference']['children'][-1]['value']['events']
    assert last and all(timeline['semantics']['expected'] in e for e in last)
    # who-did-what: the data cannot derive its events, so that producer site has no note (never an invented one)
    events = next(c for c in train if c['curriculum']['external']['structure']['program'] == 'story-who-did-what')
    assert not events['curriculum']['reference'].get('children')


def test_memory_questions_lose_the_asked_on_prefix_and_programs_are_distinct(tmp_path, monkeypatch):
    rows = []
    for e in range(80):
        sessions = [f'[session on 2024/0{1 + k % 9}/1{k} (Mon) 10:00]\nSpeaker 1: Hello.\nSpeaker 2: My favourite colour is colour{e} today.'
                    if k == 2 else f'[session on 2024/0{1 + k % 9}/1{k} (Mon) 10:00]\nSpeaker 1: Hello number {k}.' for k in range(5)]
        rows.append({'context': json.dumps(sessions),
                     'prompt': json.dumps([f'(asked on 2024/05/01 (Wed) 08:02) What is the favourite colour of speaker {e}?']),
                     'target': f'colour{e}', 'split': 'train' if e < 56 else 'eval',
                     'meta': json.dumps({'source': 'chronicles+teacher', 'episode': e, 'qtype': 'single_session',
                                         'asked_on': '(asked on 2024/05/01 (Wed) 08:02)'})})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'memory_qa_v2_pairs.parquet')
    monkeypatch.setattr(br, 'BGKIT_TASKS', tmp_path)
    items = br.memory_items(limit=(5, 2))['train']
    assert items[0]['question'].startswith('What is the favourite colour') and '(asked on' not in items[0]['question']
    cases = br.memory(train=4, held=1)
    train = [c for c in cases if c['split'] == 'train']
    assert {c['curriculum']['external']['structure']['program'] for c in train} == {p.name for p in br.MEM_PROGRAMS}
    assert len(br.MEM_PROGRAMS) == 10
    _assert_distinct_programs(train, br.MEM_PROGRAMS)
    summaries = next(c for c in train if c['curriculum']['external']['structure']['program'] == 'session-summaries-bottleneck')
    for note in summaries['curriculum']['reference']['children']:
        assert note['value'].startswith('[session on') and summaries['semantics']['expected'] in note['value']
    batches = next(c for c in train if c['curriculum']['external']['structure']['program'] == 'session-batches-recall')
    assert [n['value']['relevant'] for n in batches['curriculum']['reference']['children']] == [True, False]


def test_pubmed_programs_are_distinct_and_decision_readouts_are_leaves(tmp_path, monkeypatch):
    rows = []
    for i in range(160):
        label = ('yes', 'no', 'maybe')[i % 3]
        rows.append({'pubid': 1000 + i, 'question': f'Does treatment {i} help?', 'final_decision': label,
                     'long_answer': f'Treatment {i} helped in the trial.',
                     'context': {'labels': ['BACKGROUND', 'METHODS', 'RESULTS', 'CONCLUSIONS'],
                                 'contexts': [f'Background {i}.', f'A trial {i} was run.', f'Treatment {i} helped in the trial. More text {i}.',
                                              f'Treatment {i} helped. We conclude it works.']}})
    root = tmp_path / 'qiaojin__PubMedQA' / 'pqa_labeled' / 'train'
    root.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root / '0000.parquet')
    monkeypatch.setattr(br, 'DECISION_SOURCES', tmp_path)
    cases = br.pubmed(train=12)
    train = _assert_distinct_programs(cases, br.PUB_PROGRAMS)
    assert len(br.PUB_PROGRAMS) == 8
    readout = next(c for c in train if c['curriculum']['external']['structure']['program'] == 'abstract-hedging-rule-table')
    assert any('readout: decision' in text for text in readout['semantics']['files'].values())
    gist = [c for c in train if c['curriculum']['external']['structure']['program'] == 'abstract-gist-then-verdict']
    notes = [n for c in gist for n in c['curriculum']['reference'].get('children') or []]
    assert notes and all(n['provenance'] == 'mechanical:decisive-section-overlap' for n in notes)
    assert all(not c['curriculum']['reference'].get('children') for c in gist if c['semantics']['expected'] == 'maybe')
    debate = next(c for c in train if c['curriculum']['external']['structure']['program'] == 'abstract-yes-no-debate')
    assert not debate['curriculum']['reference'].get('children')


def test_searchqa_programs_are_distinct(tmp_path, monkeypatch):
    base = tmp_path / 'datasets--lucadiliello--searchqa' / 'snapshots' / 'x' / 'data'
    base.mkdir(parents=True)
    for name, count, offset in (('train', 400, 0), ('validation', 160, 10000)):
        rows = []
        for i in range(count):
            n = offset + i
            snippets = ' '.join(f'[DOC] [TLE] Page {k} of {n} [PAR] Item {n} shows that the capital answer{n} was famous in many records of town {k}.'
                                for k in range(6))
            rows.append({'context': snippets, 'question': f'This clue number {n} names which capital?', 'answers': [f'answer{n}'], 'key': f'k{n}'})
        pq.write_table(pa.Table.from_pylist(rows), base / f'{name}-0000.parquet')
    monkeypatch.setattr(br, 'HF_HUB', tmp_path)
    cases = br.searchqa(train=2, held=1)
    train = _assert_distinct_programs(cases, br.SQ_PROGRAMS)
    assert len(br.SQ_PROGRAMS) == 14 and len(train) == 28
    kind = next(c for c in train if c['curriculum']['external']['structure']['program'] == 'answer-type-then-candidates')
    assert any('readout: decision' in text for text in kind['semantics']['files'].values())
    skim = next(c for c in train if c['curriculum']['external']['structure']['program'] == 'skim-then-read-chosen')
    notes = skim['curriculum']['reference']['children']
    assert len(notes) == len(skim['semantics']['inputs']['snippets']) and all(n['provenance'] == 'checked' for n in notes)
    assert any(skim['semantics']['expected'] in n['value'] for n in notes)


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
