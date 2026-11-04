import hashlib
import io
import json
import shutil

import pytest

from natlang_neuralese.rl import memetic_stage as stage
from natlang_neuralese.rl import memetic_workspace as ws
from natlang_neuralese.rl.native_replay import group_episode_advantages
from natlang_neuralese.rl.task_rollouts import memetic_outcome

ROOT = '---\nargs:\n  tickets: string[]\nreturns: string\n---\nRoute every ticket. Use intent on each ticket.\n'
CHILD = '---\nargs:\n  request: string\nreturns: string\n---\nWhich intent does request express?\n'


def case(index, family='fam', group=None, inputs=None):
    return {'id': f'{family}-{index}', 'family': family, 'split': 'train',
            'source_groups': [group or f'g{family}{index}'],
            'curriculum': {'split_group': group or f'g{family}{index}'},
            'semantics': {'root': 'route.nl', 'files': {'route.nl': ROOT, 'route/intent.nl': CHILD},
                          'inputs': inputs or {'tickets': [hashlib.sha256(f'{family}{index}{k}'.encode()).hexdigest()
                                                           for k in range(4)]},
                          'expected': f'answer-{index}-xyz', 'oracle': 'exact'}}


def fake_result(outcome, value='v', code='return 1'):
    return {'outcome': {'status': outcome, 'value': value, 'checks': {'answer': outcome == 'task_success'}},
            'trajectory': [{'model_response': {'calls': [['eval', {'code': code}]]},
                            'assistant': {'reasoning': 'think'}},
                           {'context': [{'role': 'tool', 'content': 'out-1'}],
                            'model_response': {'calls': [['return_result', {'status': 'success', 'value': value}]]}}]}


class Evaluator:
    """Success grows with the number of guidance entries; records every call."""
    def __init__(self):
        self.calls = []

    def __call__(self, cases, guidance, adapter, *, name, out=None, samples=None):
        self.calls.append((name, [c['id'] for c in cases], guidance, [c['semantics']['files'] for c in cases], samples))
        rate = min(1., 0.25 + 0.25 * len(guidance or []))
        results = [(c['id'], fake_result('task_success' if i < round(rate * len(cases)) else 'semantic_mismatch'))
                   for i, c in enumerate(cases)]
        return {'success': rate, 'results': results, 'dir': name}


def spec(tmp_path, **extra):
    support = [case(i) for i in range(3)]
    query = [case(i) for i in range(3, 6)]
    return {'schema': ws.SPEC_SCHEMA, 'parent_sha256': 'p' * 64,
            'program': {'id': 'memetic:fam', 'family': 'fam', 'key': 'fam@x', 'files': support[0]['semantics']['files'],
                        'root': 'route.nl', 'functions': ['intent', 'route']},
            'support': support, 'query': query,
            'prepared_path': str(tmp_path / 'stage' / 'programs' / 'fam' / ('prepared-' + 'p' * 16 + '.json')),
            'stage_out': str(tmp_path / 'stage'),
            'settings': {key: stage.DEFAULTS[key] for key in ('min_success', 'max_success', 'query', 'query_rollouts',
                                                              'expensive_seconds', 'samples', 'ablation')},
            'budget': {'evaluations': 3, 'fits': 1}, 'eval': {}, 'cost_weight': 0.05, 'out': str(tmp_path / 'ep'), **extra}


def workspace(tmp_path, **extra):
    return ws.Workspace(spec(tmp_path, **extra), Evaluator(), check=lambda case: [], prepare=stage.ensure_prepared)


def test_typed_changes_are_checked_before_they_get_an_id(tmp_path):
    w = workspace(tmp_path)
    assert w.propose({'kind': 'guidance', 'scope': {'kind': 'function', 'name': 'nope'}, 'text': 'x'})['problems']
    assert w.propose({'kind': 'guidance', 'scope': {'kind': 'inputs', 'when': ''}, 'text': 'x'})['problems']
    assert w.propose({'kind': 'instructions', 'file': 'route.nl', 'find': 'absent', 'replace': 'y'})['problems']
    assert w.propose({'kind': 'skill', 'function': 'intent', 'name': 'Bad Name', 'description': 'd', 'body': 'b',
                      'action': 'create'})['problems']
    assert w.propose({'kind': 'adapter', 'adapter': 'x'})['problems'], 'adapters come only from fit'
    assert w.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'Use intent.'}) == {'id': 'c1'}
    assert w.propose({'kind': 'instructions', 'file': 'route.nl', 'find': 'Route every ticket.',
                      'replace': 'Route every ticket exactly once.'}) == {'id': 'c2'}
    assert w.propose({'kind': 'skill', 'function': 'intent', 'name': 'intent-cues', 'description': 'When an intent is unclear',
                      'body': '1. Read the whole request.', 'action': 'create'}) == {'id': 'c3'}
    assert w.propose({'kind': 'revert', 'id': 'c1'}) == {'id': 'c4'}
    # The compiler check runs on the changed program; its problems block the change.
    rejecting = ws.Workspace(spec(tmp_path / 'r'), Evaluator(), check=lambda case: ['route.nl: bad type'],
                             prepare=stage.ensure_prepared)
    assert rejecting.propose({'kind': 'signature', 'file': 'route.nl', 'field': 'returns', 'type': '???'}) == {
        'problems': ['route.nl: bad type']}


def test_changes_apply_to_files_scope_by_scope_and_skills_sit_with_their_function(tmp_path):
    w = workspace(tmp_path)
    for change in ({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'Call intent per ticket.'},
                   {'kind': 'guidance', 'scope': {'kind': 'function', 'name': 'intent'}, 'text': 'Pick one label.'},
                   {'kind': 'guidance', 'scope': {'kind': 'inputs', 'when': 'a ticket mentions money'}, 'text': 'Check transfers.'},
                   {'kind': 'skill', 'function': 'intent', 'name': 'intent-cues', 'description': 'Unclear intents',
                    'body': 'Read it all.', 'action': 'create'},
                   {'kind': 'signature', 'file': 'route/intent.nl', 'field': 'args', 'name': 'request', 'type': 'string'}):
        w.propose(change)
    candidate = w.candidate(['c1', 'c2', 'c3', 'c4', 'c5'])
    # Program guidance for every call; the intent calls get the program part followed by their own.
    assert candidate['guidance'] == [{'function': None, 'text': 'Call intent per ticket.'},
                                     {'function': 'intent', 'text': 'Call intent per ticket.\n\nPick one label.'}]
    changed = ws.apply_changes(w.spec['support'][0], candidate['files'])
    files = changed['semantics']['files']
    assert 'route/intent/skills/intent-cues/SKILL.md' in files, 'the companion folder of intent'
    # Input-class guidance is a skill of the root: the called model loads it only for inputs its description fits.
    money = [path for path in files if path.startswith('route/skills/for-a-ticket-mentions-money-')]
    assert len(money) == 1 and 'Use for inputs where a ticket mentions money' in files[money[0]]
    assert 'Check transfers.' in files[money[0]]
    assert changed['semantics']['inputs'] == w.spec['support'][0]['semantics']['inputs']
    w.propose({'kind': 'revert', 'id': 'c2'})
    assert w.candidate(['c1', 'c2', 'c6'])['guidance'] == [{'function': None, 'text': 'Call intent per ticket.'}]


def test_evaluate_ablate_budget_commit_and_held_score(tmp_path):
    w = workspace(tmp_path)
    w.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'a'})
    w.propose({'kind': 'guidance', 'scope': {'kind': 'function', 'name': 'route'}, 'text': 'b'})
    first = w.evaluate(['c1'])
    assert first['success'] == 0.5 and first['baseline'] == 0.25 and first['evaluationsLeft'] == 2
    assert w.baseline() == {'attempts': 3, 'success': 0.25}
    prepared = w.prepared
    # Preparation measured support, then the paired query baselines on every reserved world with 4 samples each.
    assert [call[0] for call in w.evaluate_cases.calls[:3]] == ['baseline-support', 'baseline-query-none',
                                                                 'baseline-query-default']
    assert prepared['query_ids'] == ['fam-3', 'fam-4', 'fam-5'] and prepared['query_samples'] == 4
    assert prepared['baselines']['none']['noise'] == stage.binomial_noise(0.25, 12)
    assert all(call[1] == ['fam-0', 'fam-1', 'fam-2'] for call in w.evaluate_cases.calls[3:]), 'only support worlds'
    assert w.ablate(['c1', 'c2']) == [{'id': 'c1', 'without': 0.5}, {'id': 'c2', 'without': 0.5}]
    assert w.evaluate(['c1'])['evaluationsLeft'] == 1, 'an already measured change set costs nothing'
    assert w.evaluate(['c1', 'c2'])['evaluationsLeft'] == 0
    w.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'c'})
    with pytest.raises(ValueError, match='budget is spent'):
        w.evaluate(['c3'])
    assert w.fit({}) == {'problems': ['adapter fits are unavailable in this run (no bank coefficient-fit service is '
                                      'configured); use guidance, skills or patches']}
    w.commit(['c1', 'c2'], 'routes every ticket')
    scored = w.score()['memetic']
    assert scored['query_success'] == 0.75
    assert scored['anchor'] == 0.5 and scored['gain'] == 0.25
    assert scored['quality'] == pytest.approx(0.75 - 0.025)  # the evaluation budget spent, no fit
    query_calls = [call for call in w.evaluate_cases.calls if call[0].startswith('query-')]
    assert query_calls[-1][1] == ['fam-3', 'fam-4', 'fam-5'] and query_calls[-1][4] == 4, 'paired with the baselines'
    assert scored['ablation'] is None, 'the ablation runs host-side after the round'
    assert w.score() == json.loads((tmp_path / 'ep' / 'score.json').read_text()), 'scored once'


def test_uncommitted_episode_and_answer_leak(tmp_path):
    w = workspace(tmp_path)
    scored = w.score()['memetic']
    assert scored['query_success'] == 0.25 and scored['quality'] == pytest.approx(0.25 - 0.05)
    leaky = workspace(tmp_path / 'leak')
    leaky.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'the answer is answer-4-xyz'})
    leaky.commit(['c1'], 'x')
    assert leaky.score()['memetic']['leak_refused'] is True
    assert leaky.score()['memetic']['query_success'] == 0.


def test_anchored_advantages_never_reward_the_trivial_improvement():
    def row(quality, anchor):
        return {'reward': {'quality': quality, 'memetic': {'anchor': anchor, 'gain': quality - anchor, 'cost': 0.,
                                                           'query_success': quality}}}
    # Every candidate below the best trivial improvement: all advantages are negative.
    assert group_episode_advantages([row(.25, .5), row(.5, .5), row(.0, .5)], jolt=True, mode='grpo') == [-.25, 0., -.5]
    # Above it: centered on the group mean.
    values = group_episode_advantages([row(1., .25), row(.5, .25)], jolt=True, mode='grpo')
    assert values == [0.25, -0.25]
    with pytest.raises(ValueError, match='one trivial-improvement anchor'):
        group_episode_advantages([row(1., .25), row(.5, .5)], jolt=False, mode='grpo')


def test_memetic_outcome_reads_only_memetic_worlds():
    task = {'semantics': {'world': {'kind': 'memetic'}}}
    result = {'outcome': {'world': {'memetic': {'schema': 'natlang.memetic-outcome/1', 'quality': .4, 'anchor': .2,
                                                'gain': .2, 'cost': 0., 'query_success': .4}}}}
    assert memetic_outcome(task, result)['quality'] == .4
    assert memetic_outcome({'semantics': {}}, result) is None
    assert memetic_outcome(task, None) is None


def test_program_pool_keeps_related_worlds_together_and_drops_near_duplicates():
    rows = [case(i, group='shared' if i < 3 else None) for i in range(12)]
    rows.append(case(99, inputs=rows[5]['semantics']['inputs']))  # exact repeat of world 5's inputs
    settings = {**stage.DEFAULTS, 'held_fraction': 0., 'support': 3, 'query': 3}
    pool = stage.program_pool(rows, settings=settings)
    assert len(pool) == 1
    key = next(iter(pool))
    assert key.startswith('fam@')
    groups = pool[key]['components']
    assert sum(1 for g in groups if {r['id'] for r in g} >= {'fam-0', 'fam-1', 'fam-2'}) == 1
    ids = [r['id'] for g in groups for r in g]
    assert not ({'fam-5', 'fam-99'} <= set(ids)), 'a near-duplicate world was kept'
    split = stage.split_program(pool[key], settings)
    assert len(split['support']) == 3 and len(split['query']) == 7, 'query reserves every remaining world up to max_query'
    assert not {c['id'] for c in split['support']} & {c['id'] for c in split['query']}
    held = stage.program_pool(rows, settings={**settings, 'held_fraction': 1., 'dev': 3, 'confirmation': 3})
    assert set(stage.split_program(next(iter(held.values())), settings)) == {'support', 'dev', 'confirmation'}
    # Worlds whose sources differ are different programs.
    other = [case(i, family='fam') for i in range(20, 32)]
    for row in other:
        row['semantics']['files'] = {'route.nl': ROOT.replace('Route every', 'Sort every'), 'route/intent.nl': CHILD}
    assert len(stage.program_pool(rows + other, settings=settings)) == 2


def test_build_is_crisp_and_preparation_is_lazy_once_per_parent(tmp_path):
    rows = [case(i) for i in range(10)] + [case(i, family='other') for i in range(10)]
    settings = {'held_fraction': 0., 'support': 3, 'query': 3, 'max_query': 5}
    receipt = stage.build(cases=rows, out=tmp_path, parent_sha256='p' * 64, settings=settings, log=lambda line: None)
    tasks = [json.loads(line) for line in open(receipt['task_source']['path'])]
    assert len(tasks) == 2 and {t['curriculum']['program_family'].split('@')[0] for t in tasks} == {'fam', 'other'}
    task = tasks[0]
    assert task['family'] == stage.FAMILY and task['split'] == 'train'
    assert task['semantics']['world']['kind'] == 'memetic' and task['semantics']['root'] == 'improve_program.nl'
    assert set(task['semantics']['files']) == {'improve_program.nl', 'improve_program/refine.nl'}
    assert 'supportSuccess' not in task['semantics']['inputs']['program'], 'measured later, inside the lease'
    spec_value = json.loads(open(task['semantics']['world']['task']).read())
    assert len(spec_value['support']) == 3 and len(spec_value['query']) == 5
    evaluator = Evaluator()
    prepared = stage.ensure_prepared(spec_value, evaluator)
    assert stage.ensure_prepared(spec_value, Evaluator()) == prepared, 'computed once, then read'
    assert prepared['baselines']['default']['kind'] == 'mechanical-guidance'
    assert 'verified successful run' in prepared['baselines']['default']['text']
    assert prepared['query_samples'] == 4 and len(prepared['query_ids']) == 5
    assert receipt['repetition']['families'] == 2
    # A program without headroom is recorded and skipped for the next stale_parents parents, then measured again.
    blocked = {**spec_value, 'settings': {**spec_value['settings'], 'max_success': 0.1},
               'prepared_path': str(tmp_path / 'blocked' / ('prepared-' + 'p' * 16 + '.json'))}
    assert stage.ensure_prepared(blocked, Evaluator())['admitted'] is False
    memo = tmp_path / 'blocked' / 'measurements.json'
    assert stage._excluded(memo, ['p' * 64, 'q' * 64], {'stale_parents': 3}) is True
    assert stage._excluded(memo, ['p' * 64, 'q' * 64, 'r' * 64, 's' * 64], {'stale_parents': 3}) is False
    with pytest.raises(ValueError, match='no improvement headroom'):
        ws.Workspace(blocked, Evaluator(), check=lambda c: [], prepare=stage.ensure_prepared).prepared


def test_query_plan_sizes_by_cost():
    worlds = list(range(12))
    settings = {**stage.DEFAULTS}
    assert stage.query_plan(worlds, seconds_per_rollout=60, settings=settings) == (worlds, 2)
    assert stage.query_plan(worlds, seconds_per_rollout=900, settings=settings) == (worlds[:4], 4)
    assert stage.query_plan(worlds[:5], seconds_per_rollout=60, settings=settings) == (worlds[:5], 4)


def test_ablation_uses_another_programs_guidance(tmp_path):
    rows = [case(i) for i in range(10)] + [case(i, family='other') for i in range(10)]
    out = tmp_path / 'm'
    receipt = stage.build(cases=rows, out=out, parent_sha256='p' * 64,
                          settings={'held_fraction': 0., 'support': 3, 'query': 3}, log=lambda line: None)
    specs = [json.loads(open(json.loads(line)['semantics']['world']['task']).read())
             for line in open(receipt['task_source']['path'])]
    for value in specs:
        stage.ensure_prepared(value, Evaluator())
    mine = ws.Workspace(specs[0], Evaluator(), check=lambda case: [], prepare=stage.ensure_prepared, run='r1')
    mine.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'own'})
    mine.commit(['c1'], 'r')
    assert mine.score()['memetic']['ablation'] is None
    evaluator = Evaluator()
    done = stage.ablate_round(out, 'p' * 64, endpoint='http://a', evaluator=evaluator)
    assert len(done) == 1
    ablation = json.loads((mine.out / 'ablation.json').read_text())
    assert 'prepared-' in ablation['shuffled_source'] and ablation['shuffled_source'] != specs[0]['prepared_path']
    shuffled_call = [call for call in evaluator.calls if call[0].startswith('ablation-shuffled-')][0]
    assert shuffled_call[2][0]['text'].startswith('A verified successful run')
    assert ablation['actual'] == 0.5 and ablation['none'] == 0.25 and ablation['distinguishable'] is False
    assert stage.ablate_round(out, 'p' * 64, endpoint='http://a', evaluator=evaluator) == [], 'once per score'



def test_promotion_and_distillation():
    assert stage.promotion({'a': .3, 'b': .1}, {'a': .1, 'b': .1})['promote'] is True
    assert stage.promotion({'a': .1}, {'b': .1})['promote'] is False
    rows = stage.distillation_rows([{'task_id': 't', 'sample': 0, 'reward': {'quality': .6, 'memetic': {'gain': .2, 'anchor': .4}}},
                                    {'task_id': 't', 'sample': 1, 'reward': {'quality': .3, 'memetic': {'gain': -.1, 'anchor': .4}}}])
    assert [r['sample'] for r in rows] == [0]


def test_after_collection_distills_and_promotes(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, 'production_evaluator', lambda spec, endpoint: Evaluator())
    out = tmp_path / 'memetic'
    rows = [case(i) for i in range(12)]
    settings = {**stage.DEFAULTS, 'held_fraction': 1., 'support': 3, 'dev': 3, 'confirmation': 3}
    receipt = stage.build(cases=rows, out=out, parent_sha256='p' * 64, settings=settings, log=lambda line: None)
    assert open(receipt['task_source']['path']).read() == '' and receipt['held_programs']
    held_rows = [json.loads(line) for line in open(receipt['held_source']['path'])]
    round_out = tmp_path / 'round'
    round_out.mkdir()
    query_dir = tmp_path / 'q'
    query_dir.mkdir()
    (query_dir / 'guidance-written.json').write_text(json.dumps({'program_id': 'memetic:fam', 'written': {'id': 'x'}}))
    (round_out / 'rollouts.jsonl').write_text(json.dumps({'task_id': 't', 'sample': 0, 'policy': 'p' * 64, 'reward': {
        'quality': .6, 'memetic': {'gain': .2, 'anchor': .4, 'query_dir': str(query_dir)}}}) + '\n')

    def read_admitted(path):
        return [json.loads(line) for line in open(path)]

    def fake_collect(*, cases, out, **kwargs):
        out.mkdir(parents=True)
        (out / 'rollouts.jsonl').write_text(''.join(json.dumps({'task_id': row['id'], 'reward': {'memetic': {'gain': .1}}}) + '\n'
                                                    for row in read_admitted(cases)))
        return {'scored': len(held_rows), 'rollouts': len(held_rows)}
    import natlang_neuralese.rl.memetic_program as program
    original = program.actor_identity
    program.actor_identity = lambda endpoint: {'checkpoint_sha256': 'p' * 64}
    exported = []
    original_distill = stage.distill
    stage.distill = lambda *a, **k: original_distill(*a, **k, export=lambda q, d, w: exported.append((q, w)) or {'rows': 2})
    try:
        record = stage.after_collection({'stage_out': str(out), 'parent_sha256': 'p' * 64, 'promotion_due': True,
                                         'settings': settings}, endpoint='http://a', round_out=round_out,
                                        collect=fake_collect, log=lambda line: None)
    finally:
        program.actor_identity = original
        stage.distill = original_distill
    assert 'distillation_error' not in record, record
    assert record['distillation'] == {'supervisor_episodes': 1, 'guidance_graph_files': 1, 'problems': 0}
    assert exported and exported[0][1]['program_id'] == 'memetic:fam'
    assert record['promotion']['decision']['promote'] is None, 'the first evaluated parent is the reference'
    assert set(record['promotion']['gains']) == {row['curriculum']['program_family'] for row in held_rows}


def test_bridge_protocol_hides_score_and_reports_errors(tmp_path, monkeypatch):
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec(tmp_path)))
    monkeypatch.setenv('NATLANG_MEMETIC_ENDPOINT', 'http://actor')
    monkeypatch.setattr(ws, 'production_evaluator', lambda spec, endpoint: Evaluator())
    monkeypatch.setattr(ws, 'check_program', lambda case: [])
    requests = [{'id': 1, 'op': 'load', 'task': str(path)}, {'id': 2, 'op': 'failures'},
                {'id': 3, 'op': 'propose', 'args': [{'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 't'}]},
                {'id': 4, 'op': 'attempt', 'args': ['missing']}, {'id': 5, 'op': 'commit', 'args': [['c1'], 'r']},
                {'id': 6, 'op': 'score'}]
    out = io.StringIO()
    ws.serve(io.StringIO(''.join(json.dumps(r) + '\n' for r in requests)), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert replies[0]['result']['service'] == 'workspace' and 'score' not in replies[0]['result']['methods']
    assert 'baseline' in replies[0]['result']['methods']
    assert replies[1]['result'][0]['id'] in ('a2', 'a3'), 'the parent failed two of three support worlds'
    assert replies[2]['result'] == {'id': 'c1'}
    assert 'no support attempt' in replies[3]['error']
    assert replies[5]['result']['memetic']['query_success'] == 0.5


@pytest.mark.skipif(shutil.which('node') is None, reason='needs node and the built ts-host dist')
def test_supervisor_package_and_changed_programs_load():
    files = stage.package_files()
    assert ws.check_program({'semantics': {'files': files}}) == []
    bad = dict(files)
    bad['improve_program.nl'] = files['improve_program.nl'].replace('returns: Improvement', 'returns: Nope<')
    assert ws.check_program({'semantics': {'files': bad}})


def test_stage_parameters_and_due_rounds(tmp_path):
    stage.validate_stage_parameters({'rl_stage': 'rl', 'memetic_every': 2, 'group': 4, 'settings': {'support': 3}})
    for bad in ({}, {'rl_stage': 'rl', 'group': 1}, {'rl_stage': 'rl', 'settings': {'nope': 1}},
                {'rl_stage': 'rl', 'promotion_min_gain': 2}):
        with pytest.raises(ValueError):
            stage.validate_stage_parameters(bad)
    source = tmp_path / 'programs.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in [case(i) for i in range(8)]))
    declared = {'id': 'memetic', 'parameters': {'rl_stage': 'rl', 'memetic_every': 2, 'programs_per_round': 3, 'group': 4,
                                                'settings': {'held_fraction': 0., 'support': 3, 'query': 3}},
                'program_source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}
    config = {'out': str(tmp_path), 'cases': 'ordinary.jsonl', 'rows_per_round': 16, 'group': 8}
    assert stage.bind_round(config, declared, {'round': 0}, 'p' * 64, protected=lambda c: False) is None, 'not due'
    state = {'round': 1}
    # The controller builds the parent's task source itself: the stage is live without a manual build.
    bound = stage.bind_round(config, declared, state, 'p' * 64, protected=lambda c: False)
    assert bound['rows_per_round'] == 3 and bound['group'] == 4 and bound['cases'].endswith('memetic-tasks-pppppppppppppppp.jsonl')
    assert bound['memetic_round']['parent_sha256'] == 'p' * 64 and bound['memetic_round']['promotion_due'] is True
    assert stage.bind_round(config, declared, {'round': 3}, 'p' * 64, protected=lambda c: False)[
        'memetic_round']['promotion_due'] is False
    curriculum = stage.bind_round({**config, 'task_curriculum': {'schema': 'x', 'min_rows_per_round': 16, 'min_group': 8}},
                                  declared, {'round': 1}, 'p' * 64, protected=lambda c: False)['task_curriculum']
    assert (curriculum['min_rows_per_round'], curriculum['min_group']) == (3, 4), 'the selector floor keeps the memetic size'
    assert state['interleave']['memetic']['pending'] is None
    other = stage.bind_round(config, declared, {'round': 1}, 'q' * 64, protected=lambda c: False)
    assert other['cases'].endswith('memetic-tasks-qqqqqqqqqqqqqqqq.jsonl'), 'each parent gets its own build'
    with pytest.raises(ValueError, match='program source changed'):
        stage.bind_round(config, {**declared, 'program_source': {**declared['program_source'], 'sha256': '0' * 64}},
                         {'round': 1}, 'r' * 64, protected=lambda c: False)


def test_each_rollout_has_its_own_workspace(tmp_path):
    shared = spec(tmp_path)
    first = ws.Workspace(shared, Evaluator(), check=lambda case: [], run='task-1/1', prepare=stage.ensure_prepared)
    second = ws.Workspace(shared, Evaluator(), check=lambda case: [], run='task-2/1', prepare=stage.ensure_prepared)
    first.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'a'})
    first.commit(['c1'], 'r')
    assert second.state['committed'] is None and second.propose(
        {'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'b'}) == {'id': 'c1'}
    assert first.score()['memetic']['query_success'] != second.score()['memetic']['query_success']


def test_missing_workspace_score_is_a_failure_at_the_group_anchor(tmp_path):
    path = tmp_path / 'spec.json'
    value = spec(tmp_path)
    path.write_text(json.dumps(value))
    task = {'semantics': {'world': {'kind': 'memetic', 'task': str(path)}}}
    fallback = memetic_outcome(task, {'outcome': {'status': 'repetition'}})
    assert fallback['anchor'] == 0., 'no preparation: no measured anchor for any member of the group'
    stage.ensure_prepared(value, Evaluator())
    fallback = memetic_outcome(task, {'outcome': {'status': 'repetition'}})
    assert fallback['quality'] == 0. and fallback['anchor'] == 0.5 and fallback['unscored_workspace'] is True


def test_memetic_rows_carry_no_privileged_facts():
    from natlang_neuralese.rl.privileged_context import case_facts

    class Custody:
        def check(self, case):
            return {'case_id': case['id']}
    receipt, facts = case_facts({'id': 'm', 'semantics': {'expected': None, 'world': {'kind': 'memetic'}}}, custody=Custody())
    assert facts == []


def test_bank_fit_baseline_and_typed_fit(tmp_path):
    class Problem(ValueError):
        problems = ['worlds must be support worlds']

    def fitter(episode, fit_spec):
        if fit_spec.get('worlds') == ['nope']:
            raise Problem('bad')
        return {'adapter': {'schema': 'natlang.memetic-adapter/1', 'adapter_id': 'a'}, 'support_loss': {'before': 2., 'after': 1.}}
    w = ws.Workspace(spec(tmp_path), Evaluator(), check=lambda case: [], prepare=stage.ensure_prepared, fitter=fitter)
    bank = w.prepared['baselines']['bank']
    assert bank['kind'] == 'bank-default-fit' and bank['success'] == 0.25 and bank['adapter']['adapter_id'] == 'a'
    assert w.fit({'worlds': ['nope']}) == {'problems': ['worlds must be support worlds']}
    assert w.fit({}) == {'id': 'c1', 'supportLoss': {'before': 2., 'after': 1.}}
    assert w.candidate(['c1'])['adapter'] == {'schema': 'natlang.memetic-adapter/1', 'adapter_id': 'a'}
    assert 'fit(spec: FitSpec)' in ws.offered(True)['declaration'] and 'fit' in ws.offered(True)['methods']
    assert 'fit(' not in ws.offered(False)['declaration'] and 'fit' not in ws.offered(False)['methods']


def test_withhold_change_edits_the_function_frontmatter(tmp_path):
    w = workspace(tmp_path)
    assert w.propose({'kind': 'withhold', 'file': 'route/intent.nl', 'tools': ['teleport']})['problems']
    assert w.propose({'kind': 'withhold', 'file': 'route/intent.nl', 'tools': ['delegate']}) == {'id': 'c1'}
    changed = ws.apply_changes(w.spec['support'][0], w.candidate(['c1'])['files'])
    meta, _ = ws.frontmatter(changed['semantics']['files']['route/intent.nl'])
    assert meta['withhold'] == ['delegate'] and meta['args'] == {'request': 'string'}


def test_prepare_round_measures_selected_programs_before_collection(tmp_path, monkeypatch):
    rows = [case(i) for i in range(10)]
    receipt = stage.build(cases=rows, out=tmp_path / 'm', parent_sha256='p' * 64,
                          settings={'held_fraction': 0., 'support': 3, 'query': 3}, log=lambda line: None)
    monkeypatch.setattr(ws, 'production_evaluator', lambda spec, endpoint: Evaluator())
    monkeypatch.delenv('NATLANG_BANK_SERVICE', raising=False)
    done = stage.prepare_round(receipt['task_source']['path'], endpoint='http://a', log=lambda line: None)
    assert len(done) == 1 and done[0]['admitted'] is True and done[0]['support_success'] == 0.25
    spec_path = json.loads(open(receipt['task_source']['path']).readline())['semantics']['world']['task']
    assert stage.ensure_prepared(json.loads(open(spec_path).read()), None)['support_success'] == 0.25, 'read, not rerun'


def test_interleaved_stage_declares_the_guidance_writer():
    from natlang_neuralese.rl import interleave
    from natlang_neuralese.rl.production import guidance_collection_options
    base = {'sft_stage': 'sft', 'rl_rounds_per_window': 2}
    interleave.validate_stage_parameters({**base, 'guidance_stage': 'writer'})
    with pytest.raises(ValueError, match='backbone or writer'):
        interleave.validate_stage_parameters({**base, 'guidance_stage': 'native'})
    assert interleave.CONTROLLER_KEYS['guidance_stage'] == 'jolt_guidance_stage'
    assert guidance_collection_options('writer') == {} and guidance_collection_options('backbone')['writer'] == 'off'


def test_unresolved_support_is_no_measurement(tmp_path):
    class TimingOut(Evaluator):
        def __call__(self, cases, guidance, adapter, *, name, out=None, samples=None):
            return {**super().__call__(cases, guidance, adapter, name=name, samples=samples), 'success': 0., 'scored': 0}
    value = spec(tmp_path)
    prepared = stage.ensure_prepared(value, TimingOut())
    assert prepared['unresolved'] is True and prepared['admitted'] is False
    memo = tmp_path / 'stage' / 'programs' / 'fam' / 'measurements.json'
    assert stage._excluded(memo, ['p' * 64], {'stale_parents': 3}) is False, 'measured again at the next parent'
    with pytest.raises(ValueError, match='did not resolve'):
        ws.Workspace(value, Evaluator(), check=lambda c: [], prepare=stage.ensure_prepared).prepared



def test_memetic_rounds_keep_their_own_cursor(tmp_path):
    memetic = {}
    state = {'round': 3, 'offset': 40}
    stage._memetic_cursor(state, memetic)
    assert state['offset'] == 0 and memetic['ordinary_offset'] == 40
    stage._memetic_cursor(state, memetic)  # repeated bind of the same round
    assert state['offset'] == 0 and memetic['ordinary_offset'] == 40
    state.update(round=4, offset=2)  # the memetic round advanced its cursor to 2
    stage._ordinary_cursor(state, memetic)
    assert state['offset'] == 40 and memetic['offset'] == 2 and memetic['active_round'] is None
    state.update(round=7, offset=44)
    stage._memetic_cursor(state, memetic)
    assert state['offset'] == 2 and memetic['ordinary_offset'] == 44


def test_unresolved_memetic_episode_has_no_reward_and_records_carry_distillation_fields():
    from natlang_neuralese.rl.task_rollouts import memetic_reward_record
    record = memetic_reward_record({'anchor': .5, 'gain': .1, 'cost': 0., 'query_success': .6, 'query_dir': '/q',
                                    'committed': {'ids': ['c1'], 'reason': 'r'}})
    assert record['query_dir'] == '/q' and record['committed']['ids'] == ['c1'] and record['unscored_workspace'] is False
    rows = stage.distillation_rows([{'task_id': 't', 'sample': 0, 'reward': {'quality': .6, 'memetic': record}}])
    assert rows[0]['query_dir'] == '/q'


def test_held_packet_withdraws_whole_families_and_never_meta_trains_them(tmp_path):
    rows = [case(i, family=f) for f in ('big', 'small', 'mid') for i in range({'big': 30, 'small': 12, 'mid': 20}[f])]
    settings = {**stage.DEFAULTS, 'support': 3, 'query': 3, 'dev': 3, 'confirmation': 3, 'held_families': 1}
    families, held = stage.held_packet_rows(rows, settings=settings)
    assert families == ['small'] and {r['family'] for r in held} == {'small'} and len(held) == 12
    remaining = [r for r in rows if r['family'] != 'small']
    pool = stage.program_pool(remaining, settings=settings, held_cases=[{**r, 'split': 'dev'} for r in held])
    assert {k.split('@')[0]: v['held'] for k, v in pool.items()} == {'big': False, 'mid': False, 'small': True}
    with pytest.raises(ValueError, match='overlap'):
        stage.program_pool(rows, settings=settings, held_cases=held)
    # The controller refuses an ordinary RL source that still holds them.
    source = tmp_path / 'programs.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in remaining))
    held_path = tmp_path / 'held.jsonl'
    held_path.write_text(''.join(json.dumps({**r, 'split': 'dev'}) + '\n' for r in held))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    declared = {'id': 'm', 'parameters': {'rl_stage': 'rl', 'memetic_every': 1, 'settings': {'support': 3, 'query': 3,
                                                                                            'dev': 3, 'confirmation': 3}},
                'program_source': {'path': str(source), 'sha256': sha(source)},
                'held_program_source': {'path': str(held_path), 'sha256': sha(held_path)}}
    ordinary = tmp_path / 'ordinary.jsonl'
    ordinary.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='still contains memetic held worlds'):
        stage.bind_round({'out': str(tmp_path / 'c1'), 'cases': str(ordinary)}, declared, {'round': 0}, 'p' * 64,
                         protected=lambda c: False)
    bound = stage.bind_round({'out': str(tmp_path / 'c2'), 'cases': str(source)}, declared, {'round': 0}, 'p' * 64,
                             protected=lambda c: False)
    receipt = json.loads(open(bound['memetic_round']['receipt']['path']).read())
    assert receipt['held_programs'] and all(p.startswith('small@') for p in receipt['held_programs'])


def test_parent_selection_is_gated_by_the_memetic_verdict():
    from natlang_neuralese.train.promotion import select_parent, RECEIPT_SCHEMA
    receipt = {'schema': RECEIPT_SCHEMA, 'phase': 'development', 'promoted': True, 'declaration_sha256': 'd',
               'matched': True, 'delta': 1, 'unmatched_reasons': [], 'decision_reasons': [], 'case_transitions': [],
               'candidate': {'checkpoint_sha256': 'c' * 64}, 'reference': {'checkpoint_sha256': 'r' * 64}}
    history = [{'parent_sha256': 'r' * 64, 'gains': {'f': .2}, 'comparisons': {}},
               {'parent_sha256': 'c' * 64, 'gains': {'f': .05},
                'comparisons': {'r' * 64: stage.promotion({'f': .05}, {'f': .2})}}]
    assert select_parent('s', [receipt])['selected_sha256'] == 'c' * 64, 'no memetic stage: behaviour alone'
    blocked = select_parent('s', [receipt], memetic_history=history)
    assert blocked['selected_sha256'] == 'r' * 64 and blocked['memetic']['passed'] is False
    history[1]['comparisons']['r' * 64] = stage.promotion({'f': .18}, {'f': .2})
    assert select_parent('s', [receipt], memetic_history=history)['selected_sha256'] == 'c' * 64
    assert select_parent('s', [receipt], memetic_history=history[:1])['memetic']['reason'].startswith('no paired')


def test_distillation_package_is_an_exact_protected_source_addition(tmp_path, monkeypatch):
    out = tmp_path / 'm'
    graphs = tmp_path / 'g.jsonl'
    rows = [{'id': 'memetic:x:1', 'split': 'train', 'source_groups': ['memetic-program:p']},
            {'id': 'memetic:x:2', 'split': 'train', 'source_groups': ['memetic-program:p']}]
    graphs.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest = out / 'distill' / 'pppp' / 'manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'parent_sha256': 'p' * 64, 'supervisor_episodes': [{'policy': 'p' * 64}],
                                    'guidance_graphs': [{'path': str(graphs),
                                                         'sha256': hashlib.sha256(graphs.read_bytes()).hexdigest()}]}))

    class FakeSource:
        def __init__(self, path, tokenizer, required_split, role_ids):
            self.entries = [{'id': json.loads(l)['id'], 'binding': 'b-' + json.loads(l)['id'], 'communication': True}
                            for l in open(path)]
    import natlang_neuralese.data.native_sources as sources
    monkeypatch.setattr(sources, 'NativeGraphSource', FakeSource)

    class Guard:
        digest = 'd'
        def related(self, row):
            return []
    receipt = stage.package_distillation(out, tokenizer='t', roles=[1, 2, 3, 4], protected=Guard(), log=lambda l: None)
    assert [r['id'] for r in receipt['rows']] == ['memetic:x:1', 'memetic:x:2']
    assert all(r['disposition'] == 'added' for r in receipt['rows']) and receipt['episode_actors'] == ['p' * 64]
    assert stage.package_distillation(out, tokenizer='t', roles=[1, 2, 3, 4], protected=Guard(), log=lambda l: None) is None
