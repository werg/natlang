import json
import random
import sqlite3

from scripts.rl_tasks.sources import reading, sql
from scripts.rl_tasks.build_verified_successors import _guided_rows, _sql_rows
from scripts.rl_tasks.assemble import interleave


def test_sql_profile_distinguishes_simple_rows_from_compositional_queries():
    assert sql.complexity('SELECT COUNT(*) FROM answers') == 0
    ordered = sql.sql_profile("SELECT name FROM people WHERE note = 'JOIN' ORDER BY name LIMIT 5")
    assert ordered['order_by'] and not ordered['joins']
    assert ordered['compositional'] is False
    assert sql.complexity("SELECT name FROM people WHERE note = 'JOIN' ORDER BY name LIMIT 5") == 1
    hard = sql.sql_profile(
        'SELECT a.name FROM a JOIN b ON a.id=b.a_id JOIN c ON b.id=c.b_id '
        'WHERE a.active=1 AND c.score>2 GROUP BY a.name HAVING COUNT(*)>1')
    assert hard['joins'] == 2 and hard['group_by'] and hard['having'] and hard['compositional']


def test_hard_sql_candidate_order_round_robins_operation_signatures():
    queries = [
        'SELECT a.id FROM a JOIN b ON a.id=b.id JOIN c ON b.id=c.id',
        'SELECT a.id FROM a WHERE a.id IN (SELECT b.a_id FROM b)',
        'SELECT a.kind FROM a JOIN b ON a.id=b.a_id GROUP BY a.kind HAVING COUNT(*)>1',
        'SELECT a.id FROM a JOIN b ON a.id=b.a_id UNION SELECT c.id FROM c JOIN d ON c.id=d.c_id',
    ]
    items = [('train', 'db', q, q, '', None, str(i)) for i, q in enumerate(queries)]
    ordered = sql.diverse_query_order(items, random.Random(123))
    assert len({sql.operation_signature(item[3]) for item in ordered[:4]}) == 4


def test_sql_builder_requires_ordered_results_for_order_by_and_keeps_upstream_group(tmp_path):
    db = tmp_path / 'fixture.sqlite'
    with sqlite3.connect(db) as con:
        con.executescript('CREATE TABLE t(id INTEGER, name TEXT); INSERT INTO t VALUES (2, "b"), (1, "a");')
    cases = sql._build(
        'spider-hard', [('train', 'toy', 'List names by id.', 'SELECT name FROM t ORDER BY id', '', None, 'q1')],
        lambda _split, _db: db, 4, tmp_path / 'assets', 'spider', 'fixture-v1', 1,
        group_source='spider')
    assert len(cases) == 1
    case = cases[0]
    assert case['semantics']['expected'] == [['a'], ['b']]
    assert case['semantics']['oracle'] == {'level': 'normalized', 'normalization': 'result-set', 'ordered': True}
    assert [['b'], ['a']] not in case['curriculum']['external']['checks']['alias_answers']
    assert 'sql-db:spider:toy' in case['source_groups']
    assert 'sql-db:spider-hard:toy' not in case['source_groups']


def test_focused_sql_builder_filters_simple_cases_and_keeps_db_split_atomic(tmp_path):
    db = tmp_path / 'fixture.sqlite'
    with sqlite3.connect(db) as con:
        con.executescript('''
            CREATE TABLE a(id INTEGER, name TEXT, active INTEGER);
            CREATE TABLE b(id INTEGER, a_id INTEGER);
            CREATE TABLE c(id INTEGER, b_id INTEGER, score INTEGER);
            INSERT INTO a VALUES (1, 'x', 1), (2, 'y', 1);
            INSERT INTO b VALUES (10, 1), (20, 2);
            INSERT INTO c VALUES (100, 10, 3), (200, 20, 4);
        ''')
    items = [
        ('train', 'toy', 'Count rows.', 'SELECT COUNT(*) FROM a', '', None, 'easy'),
        ('train', 'held-toy', 'Count rows through joins.',
         'SELECT COUNT(*) FROM a JOIN b ON a.id=b.a_id JOIN c ON b.id=c.b_id', '', None, 'hard-train'),
        ('dev', 'held-toy', 'Count rows through joins in dev.',
         'SELECT COUNT(*) FROM a JOIN b ON a.id=b.a_id JOIN c ON b.id=c.b_id', '', None, 'hard-dev'),
    ]
    built = sql._focused('spider-hard', items, lambda _split, _db: db, 8, tmp_path / 'assets',
                         'spider', 'fixture-v1', 3, 'spider')
    assert [case['curriculum']['external']['upstream_id'] for case in built] == ['hard-dev']
    case = built[0]
    assert case['split'] in {'dev', 'confirmation'}
    assert 'sql-db:spider:held-toy' in case['source_groups']


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def test_babi_temporal_successor_keeps_long_history_and_adds_awaited_helper(tmp_path, monkeypatch):
    monkeypatch.setattr(reading, 'BASE', tmp_path)
    context = [{'id': i, 'text': f'Person{i % 3} moved to place{i % 5}.'} for i in range(1, 126)]
    row = {'task': 3, 'task_name': 'three-supporting-facts', 'story_id': 'train-3-1', 'context': context,
           'question': 'Where was the milk before the hallway?', 'answer': 'garden',
           'supporting_fact_ids': [8, 27, 81]}
    _write_jsonl(tmp_path / 'babi/train.jsonl', [row])
    _write_jsonl(tmp_path / 'babi/test.jsonl', [])
    [case] = reading.babi_temporal3(per_task=1, eval_per_task=1)
    assert case['source'] == 'external-rl:babi-temporal3'
    assert len(case['semantics']['inputs']['facts']) == 125
    assert case['semantics']['expected'] == 'garden'
    assert case['curriculum']['external']['framing'] == 'neuralese-helper'
    assert 'await nl(facts, question)' in case['semantics']['files']['answer_from_facts.nl']
    assert 'answer_from_facts/track_query_history.nl' in case['semantics']['files']
    assert 'babi:train-3-1' in case['source_groups']


def test_clutrr_guided_successor_keeps_only_multi_hop_and_preserves_world_group(tmp_path, monkeypatch):
    monkeypatch.setattr(reading, 'BASE', tmp_path)
    rows = [
        {'id': 'short', 'query': "('A', 'B')", 'target_text': 'son', 'story': 'A has son B.',
         'edge_types': "['son', 'mother', 'brother']"},
        {'id': 'long', 'query': "('A', 'D')", 'target_text': 'brother', 'story': 'A and D are family.',
         'edge_types': "['son', 'grandmother', 'son', 'brother']"},
    ]
    _write_jsonl(tmp_path / 'clutrr/gen_train234_test2to10__train.jsonl', rows)
    _write_jsonl(tmp_path / 'clutrr/gen_train234_test2to10__test.jsonl', [])
    [case] = reading.clutrr_guided(cap_train=1, cap_eval=1)
    assert case['source'] == 'external-rl:clutrr-guided'
    assert case['curriculum']['external']['upstream_id'] == 'long'
    assert case['semantics']['expected'] == 'brother'
    assert case['curriculum']['external']['framing'] == 'neuralese-helper'
    assert 'await nl(sentence)' in case['semantics']['files']['find_relation.nl']
    assert 'clutrr:long' in case['source_groups']


def test_verified_successor_transform_keeps_groups_and_filters_sql_to_real_composition(tmp_path):
    base = {
        'version': 1, 'id': 'old', 'kind': 'natlang.program/2', 'family': 'ext_spider_sql_qa',
        'source': 'external-rl:spider', 'split': 'train', 'source_ids': ['spider:q1'],
        'source_groups': ['sql-db:spider:db-a', 'upstream-family:a'],
        'source_revisions': ['spider@fixture'], 'curriculum': {
            'family': 'ext_spider_sql_qa', 'shape': 'spider-plain-L2', 'variant': 'plain',
            'rl': {'level': 2}, 'external': {'source': 'spider', 'gold_sql':
                'SELECT a.id FROM a JOIN b ON a.id=b.id JOIN c ON b.id=c.id'}},
        'semantics': {'inputs': {'question': 'Find related ids'}, 'expected': [[1]],
            'root': 'answer_from_database.nl', 'files': {'answer_from_database.nl': '---\n---\nReturn rows.'},
            'oracle': {'level': 'normalized', 'normalization': 'result-set'}}}
    easy = json.loads(json.dumps(base))
    easy['id'] = 'easy'
    easy['curriculum']['external']['gold_sql'] = 'SELECT id FROM a'
    path = tmp_path / 'spider.jsonl'
    _write_jsonl(path, [easy, base])
    [hard] = _sql_rows('spider', path, 4)
    assert hard['source_groups'] == base['source_groups']
    assert hard['source'] == 'external-rl:spider-hard'
    assert hard['curriculum']['external']['sql_profile']['joins'] == 2


def test_verified_guidance_variant_preserves_exact_world_and_answer(tmp_path):
    row = {'id': 'babi:task3', 'source': 'external-rl:babi', 'split': 'train', 'source_ids': ['babi:q'],
           'source_groups': ['babi:story-1'], 'source_revisions': ['babi@fixture'], 'family': 'ext_babi_task03',
           'curriculum': {'family': 'ext_babi_task03', 'shape': 'babi-three-supporting-facts', 'rl': {'level': 2},
               'external': {'source': 'babi', 'framing': 'plain'}},
           'semantics': {'inputs': {'facts': ['1. Anna moved to garden.'], 'question': 'Where was Anna?'},
               'expected': 'garden', 'root': 'answer_from_facts.nl', 'files': {'answer_from_facts.nl':
                   '---\n---\nfacts is a numbered story, in order. Answer question from it, taking into account that later facts can change earlier ones. Return only the answer word or words (comma-separated when there are several).'}}}
    path = tmp_path / 'babi.jsonl'
    _write_jsonl(path, [row])
    [guided] = _guided_rows('babi', path, 48)
    assert guided['source_groups'] == row['source_groups']
    assert guided['semantics']['inputs'] == row['semantics']['inputs']
    assert guided['semantics']['expected'] == row['semantics']['expected']
    assert 'await nl(facts, question)' in guided['semantics']['files']['answer_from_facts.nl']


def test_assembly_interleave_skips_sources_with_no_train_rows():
    assert interleave({'empty-source': [], 'a': [{'id': 'a1', 'family': 'f'}, {'id': 'a2', 'family': 'f'}],
                       'b': [{'id': 'b1', 'family': 'f'}]}) == [
                           {'id': 'a1', 'family': 'f'}, {'id': 'b1', 'family': 'f'},
                           {'id': 'a2', 'family': 'f'}]
