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
    """Success grows with the number of guidance lines; records every call."""
    def __init__(self):
        self.calls = []

    def __call__(self, cases, guidance, adapter, *, name):
        self.calls.append((name, [c['id'] for c in cases], guidance, [c['semantics']['files'] for c in cases]))
        rate = min(1., 0.25 + 0.25 * (guidance or '').count('For '))
        results = [(c['id'], fake_result('task_success' if i < round(rate * len(cases)) else 'semantic_mismatch'))
                   for i, c in enumerate(cases)]
        return {'success': rate, 'results': results, 'dir': name}


def spec(tmp_path, **extra):
    support = [case(i) for i in range(3)]
    query = [case(i) for i in range(3, 6)]
    return {'schema': ws.SPEC_SCHEMA, 'program': {'id': 'memetic:fam', 'family': 'fam', 'files': support[0]['semantics']['files'],
            'root': 'route.nl', 'functions': ['intent', 'route']},
            'support': support, 'query': query,
            'evidence': {'success': 0.25, 'attempts': [{'digest': ws.attempt_digest(fake_result('semantic_mismatch'),
                         attempt_id='a1', world='fam-0'), 'steps': ws.attempt_steps(fake_result('semantic_mismatch'))}]},
            'baselines': {'none': {'success': 0.25}, 'default': {'success': 0.5, 'kind': 'mechanical-guidance'}},
            'budget': {'evaluations': 3, 'fits': 1}, 'eval': {}, 'cost_weight': 0.05, 'out': str(tmp_path / 'ep'), **extra}


def workspace(tmp_path, **extra):
    return ws.Workspace(spec(tmp_path, **extra), Evaluator(), check=lambda case: [])


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
    rejecting = ws.Workspace(spec(tmp_path / 'r'), Evaluator(), check=lambda case: ['route.nl: bad type'])
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
    assert candidate['guidance'] == ('For every call of this program:\nCall intent per ticket.\n\nFor calls of intent:\n'
                                     'Pick one label.\n\nFor inputs where a ticket mentions money:\nCheck transfers.')
    changed = ws.apply_changes(w.spec['support'][0], candidate['files'])
    assert 'route/skills/intent-cues/SKILL.md' in changed['semantics']['files']
    assert changed['semantics']['inputs'] == w.spec['support'][0]['semantics']['inputs']
    w.propose({'kind': 'revert', 'id': 'c2'})
    assert 'Pick one label' not in w.candidate(['c1', 'c2', 'c6'])['guidance']


def test_evaluate_ablate_budget_commit_and_held_score(tmp_path):
    w = workspace(tmp_path)
    w.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'a'})
    w.propose({'kind': 'guidance', 'scope': {'kind': 'function', 'name': 'route'}, 'text': 'b'})
    first = w.evaluate(['c1'])
    assert first['success'] == 0.5 and first['baseline'] == 0.25 and first['evaluationsLeft'] == 2
    assert all(call[1] == ['fam-0', 'fam-1', 'fam-2'] for call in w.evaluate_cases.calls), 'only support worlds'
    assert w.ablate(['c1', 'c2']) == [{'id': 'c1', 'without': 0.5}, {'id': 'c2', 'without': 0.5}]
    with pytest.raises(ValueError, match='budget is spent'):
        w.evaluate(['c1'])
    assert w.fit({}) == {'problems': ['adapter fits are unavailable in this run (no bank coefficient-fit service is '
                                      'configured); use guidance, skills or patches']}
    w.commit(['c1', 'c2'], 'routes every ticket')
    scored = w.score()['memetic']
    assert scored['query_success'] == 0.75
    assert scored['anchor'] == 0.5 and scored['gain'] == 0.25
    assert scored['quality'] == pytest.approx(0.75 - 0.025)  # the evaluation budget spent, no fit
    assert w.evaluate_cases.calls[-1][1] == ['fam-3', 'fam-4', 'fam-5']
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
    groups = pool['fam']['components']
    assert sum(1 for g in groups if {r['id'] for r in g} >= {'fam-0', 'fam-1', 'fam-2'}) == 1
    ids = [r['id'] for g in groups for r in g]
    assert not ({'fam-5', 'fam-99'} <= set(ids)), 'a near-duplicate world was kept'
    split = stage.split_program(pool['fam'], settings)
    assert len(split['support']) == 3 and len(split['query']) == 3
    assert not {c['id'] for c in split['support']} & {c['id'] for c in split['query']}
    held = stage.program_pool(rows, settings={**settings, 'held_fraction': 1., 'dev': 3, 'confirmation': 3})
    assert set(stage.split_program(held['fam'], settings)) == {'support', 'dev', 'confirmation'}


def test_build_prepares_baselines_curriculum_and_episode_rows(tmp_path):
    rows = [case(i) for i in range(10)] + [case(i, family='other') for i in range(10)]
    settings = {'held_fraction': 0., 'support': 3, 'query': 3, 'min_success': 0.0, 'max_success': 0.6}
    receipt = stage.build(cases=rows, out=tmp_path, evaluate_for=lambda d, family: Evaluator(), parent_sha256='p' * 64,
                          settings=settings, log=lambda line: None)
    tasks = [json.loads(line) for line in open(receipt['task_source']['path'])]
    assert len(tasks) == 2 and {t['curriculum']['program_family'] for t in tasks} == {'fam', 'other'}
    task = tasks[0]
    assert task['family'] == stage.FAMILY and task['split'] == 'train'
    assert task['semantics']['world']['kind'] == 'memetic' and task['semantics']['root'] == 'improve_program.nl'
    assert set(task['semantics']['files']) == {'improve_program.nl', 'improve_program/refine.nl'}
    spec = json.loads(open(task['semantics']['world']['task']).read())
    assert spec['baselines']['default']['kind'] == 'mechanical-guidance'
    assert 'verified successful run' in spec['baselines']['default']['text']
    assert receipt['repetition']['families'] == 2
    # Curriculum: a parent that already succeeds too often has no headroom.
    blocked = stage.build(cases=rows, out=tmp_path / 'b', evaluate_for=lambda d, family: Evaluator(), parent_sha256='p' * 64,
                          settings={**settings, 'max_success': 0.1}, log=lambda line: None)
    assert open(blocked['task_source']['path']).read() == ''


def test_promotion_and_distillation():
    assert stage.promotion({'a': .3, 'b': .1}, {'a': .1, 'b': .1})['promote'] is True
    assert stage.promotion({'a': .1}, {'b': .1})['promote'] is False
    rows = stage.distillation_rows([{'task_id': 't', 'sample': 0, 'reward': {'quality': .6, 'memetic': {'gain': .2, 'anchor': .4}}},
                                    {'task_id': 't', 'sample': 1, 'reward': {'quality': .3, 'memetic': {'gain': -.1, 'anchor': .4}}}])
    assert [r['sample'] for r in rows] == [0]


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
    assert replies[1]['result'][0]['id'] == 'a1'
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
    declared = {'id': 'memetic', 'parameters': {'rl_stage': 'rl', 'memetic_every': 2, 'programs_per_round': 3, 'group': 4}}
    config = {'out': str(tmp_path), 'cases': 'ordinary.jsonl', 'rows_per_round': 16, 'group': 8}
    assert stage.bind_round(config, declared, {'round': 0}, 'p' * 64) is None, 'not due'
    state = {'round': 1}
    assert stage.bind_round(config, declared, state, 'p' * 64) is None
    assert state['interleave']['memetic']['pending']['reason'] == 'no memetic task source built for this parent'
    rows = [case(i) for i in range(8)]
    stage.build(cases=rows, out=tmp_path / 'memetic', evaluate_for=lambda d, family: Evaluator(), parent_sha256='p' * 64,
                settings={'held_fraction': 0., 'support': 3, 'query': 3}, log=lambda line: None)
    bound = stage.bind_round(config, declared, state, 'p' * 64)
    assert bound['rows_per_round'] == 3 and bound['group'] == 4 and bound['cases'].endswith('memetic-tasks-pppppppppppppppp.jsonl')
    assert state['interleave']['memetic']['pending'] is None and state['interleave']['memetic']['rounds'][0]['round'] == 1
    assert stage.bind_round(config, declared, {'round': 1}, 'q' * 64) is None, 'another parent needs its own build'
