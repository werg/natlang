"""Authored trajectories -> the expert-iteration teacher tier (scripts/neuralese_data/authored_trajectories.py)."""
import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'scripts/neuralese_data', ROOT / 'training/neuralese', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import authored_trajectories as at  # noqa: E402
import sft_sources as s  # noqa: E402


def task(task_id, split='train'):
    return {'version': 'natlang.program/2', 'id': task_id, 'family': 'inv_x', 'split': split,
            'source_groups': ['invented:x'], 'semantics': {'root': 'f.nl'}}


def call_turn(name='return_result', args=None, tools=('eval', 'return_result')):
    return {'invocation_id': 'call/1', 'function': 'f', 'tools': [{'type': 'function', 'function': {'name': t}}
                                                                   for t in tools],
            'messages': [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'You are inside this call: f()'}],
            'target': {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'authored_0_0', 'type': 'function',
                 'function': {'name': name, 'arguments': json.dumps(args or {'status': 'success', 'value': 'T01'})}}]}}


def readout_turn():
    return {'invocation_id': 'call/2', 'function': 'g', 'tools': [],
            'messages': [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'You are inside this call: g()'}],
            'target': {'role': 'assistant', 'content': '"remaining-urn"'}}


@pytest.fixture
def no_held(monkeypatch):
    monkeypatch.setattr(at, 'held_out', lambda tasks: set())


def test_pool_writes_a_teacher_pool_and_refuses_what_must_not_train(tmp_path, monkeypatch):
    """Train rows become teacher_gen-layout episodes; a held split, a held-excluded task, reasoning in a target and a
    row without turns are refused with their reason."""
    monkeypatch.setattr(at, 'held_out', lambda tasks: {'held-sibling'})
    reasoned = call_turn()
    reasoned['target']['reasoning_content'] = 'teacher thinking'
    rows = [at.trajectory_row(task('ok'), source='t', turns=[call_turn(), readout_turn()]),
            at.trajectory_row(task('dev', split='dev'), source='t', turns=[call_turn()]),
            at.trajectory_row(task('held-sibling'), source='t', turns=[call_turn()]),
            at.trajectory_row(task('thinks'), source='t', turns=[reasoned]),
            at.trajectory_row(task('empty'), source='t', turns=[])]
    summary = at.write_pool(rows, tmp_path / 'pool')
    assert summary['episodes'] == 1 and summary['turns'] == 2
    assert set(summary['refused']) == {'dev', 'held-sibling', 'thinks', 'empty'}
    manifest = at.read_jsonl(tmp_path / 'pool' / 'manifest.jsonl')
    assert [m['task_id'] for m in manifest] == ['ok'] and manifest[0]['teacher'] == at.TEACHER
    rollout = at.read_jsonl(manifest[0]['rollouts'])[0]
    assert rollout['outcome'] == 'task_success' and rollout['reward']['quality'] == 1.0
    assert 'decision_reasoning' not in rollout['turns'][0]
    # a decision readout's answer serves its reasoning samples and answer request (ReplayEpisode group repeats)
    assert rollout['turns'][1]['decision_reasoning'] == {'index': 0, 'samples': at.READOUT_REPEATS, 'action': True}
    assert at.read_jsonl(Path(manifest[0]['rollouts']).parent / 'cases.jsonl')[0]['id'] == 'ok'


def test_sft_sources_extracts_the_pool_as_teacher_episodes(tmp_path, monkeypatch, no_held):
    """``sft_sources extract --teacher POOL`` reads the pool unchanged: one teacher episode per trajectory, its
    readout marked as a reasoned group so the replay serves the repeats from the authored answer."""
    at.write_pool([at.trajectory_row(task('ok'), source='t', turns=[call_turn(), readout_turn()])],
                  tmp_path / 'authored' / 'stamp')
    monkeypatch.setattr(s.data_registry, 'require_registered', lambda *a, **k: None)
    monkeypatch.setattr(s, 'record_work', lambda *a, **k: None)
    s.extract(SimpleNamespace(rl=[], teacher=[str(tmp_path / 'authored' / 'stamp')], work=str(tmp_path / 'work'),
                              out_name='episodes', registry_repo=None, registry_ref=None))
    episodes = [json.loads(line) for line in s.zstd_lines(tmp_path / 'work' / 'episodes.jsonl.zst')]
    assert len(episodes) == 1 and episodes[0]['source'] == 'teacher'
    assert episodes[0]['key'].startswith('teacher:authored/stamp:')
    replay = s.ReplayEpisode(episodes[0]['turns'])
    messages = readout_turn()['messages']
    replay.answer({'messages': call_turn()['messages']})
    answers = [replay.answer({'messages': messages})['choices'][0]['message']['content'] for _ in range(3)]
    assert answers == ['"remaining-urn"'] * 3 and replay.served.count(False) == 0


def operator_turn():
    """A learned operator's call (learned compaction): view's site, the authored summary as its reply."""
    return {'invocation_id': 'call/1#operator-1', 'function': 'view', 'tools': [],
            'messages': [{'role': 'system', 'content': 'view prompt'},
                         {'role': 'user', 'content': 'instructions:\nThe call that receives this argument ...\n\nvalue:\n"..."'}],
            'target': {'role': 'assistant', 'content': 'A heartbeat log; every line reports ok.'}}


def test_an_operator_call_is_its_own_turn_not_a_readout_and_its_episode_replays_in_context(tmp_path, no_held):
    """An operator call is one call (no reasoned-readout repeats), and the replay of an authored episode reads Neuralese
    in context as its render did (run_bonsai_queue entry ``in_context_neuralese``)."""
    assert at.operator_turn(operator_turn()) and not at.readout_turn(operator_turn()) and at.readout_turn(readout_turn())
    at.write_pool([at.trajectory_row(task('ok'), source='t', turns=[operator_turn(), call_turn()])], tmp_path / 'pool')
    manifest = at.read_jsonl(tmp_path / 'pool' / 'manifest.jsonl')
    rollout = at.read_jsonl(manifest[0]['rollouts'])[0]
    assert 'decision_reasoning' not in rollout['turns'][0] and rollout['turns'][0]['function'] == 'view'
    assert s.authored_episode({'teacher': manifest[0]['teacher']})
    assert not s.authored_episode({'teacher': {'model_id': 'gpt', 'provider': 'openai'}}) and not s.authored_episode({})


def test_graph_calls_lower_an_operator_capture_as_a_pass_through_import(monkeypatch):
    """A consumer of an operator's summary reads it spliced (engines splice captures): its row imports the state between
    the consume markers, and the producer document is the operator call's prompt, its reply up to the content,
    produce_open, then the content (Engine.capture_plan), which the learner computes in the consumer's forward."""
    import numpy as np
    from natlang_neuralese.model.shared_state import ROLE_MARKERS
    from natlang_neuralese.serve import targets
    ids = {**dict(zip(ROLE_MARKERS, (900, 901, 902, 903))), '</think>': 7, '<|im_end|>': 2}
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda t: ids[t],
                                decode=lambda part, **_: ''.join(' ' if t == 50 else 'x' for t in part))
    rendered = {'op': ([('tok', 1), ('tok', 3)], [('tok', 7), ('tok', 50), ('tok', 11), ('tok', 12), ('tok', 2)]),
                'use': ([('tok', 1), ('block', 'S'), ('tok', 4)], [('tok', 20), ('tok', 2)])}
    monkeypatch.setattr(targets, 'target_items', lambda engine, messages, tools, target: rendered[messages[0]['content']])
    monkeypatch.setattr(s, 'think_context', lambda *a, **k: 0)
    turns = [{'invocation_id': 'c#operator-1', 'messages': [{'role': 'user', 'content': 'op'}], 'tools': [],
              'target': {'role': 'assistant', 'content': 'sum'}, 'captures': [{'id': 'S'}]},
             {'invocation_id': 'c', 'messages': [{'role': 'user', 'content': 'use'}], 'tools': [],
              'target': {'role': 'assistant', 'content': 'ok'}}]
    stats = collections.Counter()
    rows = at.graph_calls(turns, SimpleNamespace(tokenizer=tokenizer), stats)
    source = rows[1]['sources']['S']
    assert source['ids'].tolist() == [1, 3, 7, 50, 900, 11, 12] and (source['start'], source['stop']) == (5, 7)
    assert rows[1]['prompt'].tolist() == [1, 902, 11, 12, 903, 4] and rows[1]['imports'] == [(2, 'S')]
    assert rows[0]['imports'] == [] and rows[0]['sample'].tolist() == [7, 50, 11, 12, 2]
    assert stats['authored_captures'] == 1 and stats['authored_consumer_calls'] == 1
    # a state with no recorded producing call is not lowered
    assert at.graph_calls(turns[1:], SimpleNamespace(tokenizer=tokenizer), stats) is None
    # the operator turn keeps its captures through the pool and extraction
    assert at.rollout_row(at.trajectory_row(task('ok'), source='t', turns=turns))['turns'][0]['captures'] == [{'id': 'S'}]
    assert s.semantic_turn(turns[0], False)['captures'] == [{'id': 'S'}]
    assert s.native_not_authored({'turns': []}) is False


def test_operator_calls_are_not_reworded_matches_of_each_other():
    """A view site's opening has no signature line: two operator calls with different instructions or values must not
    match by their first line (``instructions:``), or one serves the other's summary."""
    one = [{'role': 'system', 'content': 'view'}, {'role': 'user', 'content': 'instructions:\nfor A\n\nvalue:\n"a"'}]
    two = [{'role': 'system', 'content': 'view'}, {'role': 'user', 'content': 'instructions:\nfor B\n\nvalue:\n"b"'}]
    assert s.reworded_key(one) != s.reworded_key(two) and s.signature_key(one) != s.signature_key(two)
    call = [{'role': 'user', 'content': 'You are inside this call: f(x)\nold wording'}]
    assert s.reworded_key(call) == s.reworded_key([{'role': 'user', 'content': 'You are inside this call: f(x)\nnew'}])


def test_a_drifted_success_with_reasoned_readout_repeats_is_admitted():
    """``group-repeat`` entries are a reasoned readout's matched entry answering its repeats (as in ``exact``): they do
    not keep a drifted success out, and the turn count compares with the replay's calls."""
    record = {'outcome': 'task_success', 'exact': False, 'unserved': 0, 'replayed_turns': 23, 'recorded_turns': 23,
              'replay_calls': 23, 'served': {'exact': 22, 'drifted': 1, 'group-repeat': 16, 'scoring-declined': 80}}
    assert s.drifted_success(record)
    assert not s.drifted_success({**record, 'served': {**record['served'], 'missing': 1}})


def test_feed_takes_admitted_train_programs_once_per_digest(tmp_path):
    """Dev/confirmation programs and review-withheld ones never feed; an exported digest is not exported again, a
    re-admitted one (new digest) is."""
    root = tmp_path / 'inv'
    for slug, split, digest in (('a', 'train', 'd1'), ('b', 'dev', 'd2'), ('c', 'train', 'd3'), ('w', 'train', 'd4')):
        home = root / 'admitted' / slug
        home.mkdir(parents=True)
        (home / 'admission.json').write_text(json.dumps({'slug': slug, 'split': split, 'digest': digest,
                                                         'submission': str(tmp_path / 'none')}))
    (root / 'admitted' / 'w' / 'review.json').write_text(json.dumps({'digest': 'd4', 'wrong_expected': ['x']}))
    assert [a['slug'] for a, _ in at.pending_programs(root)] == ['a', 'c']
    (root / 'authored').mkdir()
    (root / 'authored' / 'exported.jsonl').write_text(json.dumps({'slug': 'a', 'digest': 'd1'}) + '\n'
                                                      + json.dumps({'slug': 'c', 'digest': 'old'}) + '\n')
    assert [a['slug'] for a, _ in at.pending_programs(root)] == ['c']


def test_async_driver_extracts_authored_pools_from_the_registry_owner(tmp_path, monkeypatch):
    """The round extraction reads the ``ei-authored-teacher`` role of the registry owner's git layer; only pools with a
    manifest; a registry failure leaves the round without them instead of failing it."""
    import expert_iteration as ei
    from natlang_neuralese.common import data_registry
    good = tmp_path / 'p1'
    good.mkdir()
    (good / 'manifest.jsonl').write_text('')
    seen = {}

    def role_paths(name, **kw):
        seen.update(name=name, **kw)
        return [str(good / 'manifest.jsonl'), str(tmp_path / 'gone')]
    monkeypatch.setattr(data_registry, 'role_paths', role_paths)
    monkeypatch.setattr(data_registry, 'registry_owner_repo', lambda: Path('/owner'))
    assert ei.authored_teacher_pools() == [good]
    assert seen == {'name': ei.AUTHORED_TEACHER_ROLE, 'repo': Path('/owner')}
    monkeypatch.setattr(data_registry, 'role_paths', lambda *a, **k: (_ for _ in ()).throw(ValueError('no git')))
    assert ei.authored_teacher_pools() == []


def test_an_authored_program_feeds_its_trajectories_not_only_its_oracle_cases(tmp_path, monkeypatch):
    """authored_programs.py admit writes every instance to trajectories.jsonl and only the RL instances to cases.jsonl."""
    home = tmp_path / 'p'
    home.mkdir()
    (home / 'trajectories.jsonl').write_text(json.dumps({'id': 't1'}) + '\n' + json.dumps({'id': 't2'}) + '\n')
    (home / 'cases.jsonl').write_text(json.dumps({'id': 't1'}) + '\n')
    seen = []
    monkeypatch.setattr(at, 'render', lambda cases, **kw: seen.extend(c['id'] for c in cases) or [({'id': c['id']}, None) for c in cases])
    rows, refused = at.program_rows({'slug': 'p'}, home, tmp_path)
    assert seen == ['t1', 't2'] and len(rows) == 2 and refused == {}


def test_promotion_runs_in_the_registry_owner_checkout(monkeypatch, tmp_path):
    """A feed run from a pin promotes through the main checkout (promotion commits training/corpora/ where it runs),
    the pool role reading its manifest."""
    from natlang_neuralese.common import data_registry
    calls = []
    monkeypatch.setattr(data_registry, 'registry_owner_repo', lambda: tmp_path / 'main')
    monkeypatch.setattr(at.subprocess, 'run', lambda argv, **kw: calls.append((argv, kw)))
    at.promote('ds', at.POOL_ROLE, 'manifest.jsonl', False)
    argv, kw = calls[0]
    assert argv[1] == str(tmp_path / 'main' / 'scripts/data_registry.py') and kw['cwd'] == tmp_path / 'main'
    assert argv[2:] == ['promote', 'ds', '--role', at.POOL_ROLE, '--role-file', 'manifest.jsonl']
