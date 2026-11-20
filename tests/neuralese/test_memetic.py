"""Memetic "improve program P" episodes on the per-call GRPO stack (plans/neuralese/RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md
section 3): typed changes, the workspace's evaluate/commit/score, anchored rewards, rows and splits, scoring."""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/neuralese_data"))
import grpo  # noqa: E402
from natlang_neuralese.rl import memetic_tasks, memetic_workspace as mw  # noqa: E402
from natlang_neuralese.rl import task_rollouts  # noqa: E402
from natlang_neuralese.rl.task_rollouts import memetic_outcome  # noqa: E402

ROOT = '---\ndescription: Answer the question.\nargs:\n  question: string\nreturns: string\n---\n1. Read question.\n2. Answer it.\n'
FILES = {'answer.nl': ROOT, 'helper.nl': ROOT.replace('Answer the question.', 'Help.')}


def world(i, family='fam', files=FILES, expected='secret-answer-%d'):
    return {'id': f'w{i}', 'family': family, 'split': 'train', 'source_groups': [f'g{i}'],
            'semantics': {'root': 'answer.nl', 'files': dict(files),
                          'inputs': {'question': ' '.join(f'word{i}x{j}' for j in range(40))},
                          'expected': expected % i if '%d' in expected else expected}}


@pytest.fixture(autouse=True)
def steps(monkeypatch):
    """Attempt steps without a recorded trajectory: one clean eval step."""
    monkeypatch.setattr(mw, 'attempt_steps', lambda result, limit=60: [
        {'call': 'root', 'turn': 0, 'tool': 'eval', 'status': 'ok', 'code': 'x = 1', 'output': '1'}])


def spec(tmp_path, evaluations=2):
    worlds = memetic_tasks.split([world(i) for i in range(10)], memetic_tasks.DEFAULTS)
    value = memetic_tasks.episode_spec('fam@abc', worlds, settings={**memetic_tasks.DEFAULTS, 'evaluations': evaluations})
    value['out'] = str(tmp_path / 'memetic')
    return value


class Evaluator:
    """Success 0.25 unchanged, 0.75 with any guidance; records what ran."""

    def __init__(self):
        self.calls = []

    def __call__(self, cases, guidance, *, name, samples=None):
        self.calls.append((name, [c['id'] for c in cases], guidance, [c['semantics']['files'] for c in cases]))
        success = 0.75 if guidance else 0.25
        results = [(case['id'], {'outcome': {'status': 'task_success' if k == 0 else 'wrong_result', 'value': 'v'},
                                 'trajectory': []}) for k, case in enumerate(cases)]
        return {'success': success, 'scored': len(cases), 'rollouts': len(cases), 'results': results, 'dir': name}


def workspace(tmp_path, evaluations=2, run='r0'):
    evaluator = Evaluator()
    w = mw.Workspace(spec(tmp_path, evaluations), evaluator, check=lambda case: [], run=run)
    return w, evaluator


def test_preparation_is_shared_by_the_group_and_anchors_on_the_trivial_improvements(tmp_path):
    w, evaluator = workspace(tmp_path)
    assert w.baseline() == {'attempts': 4, 'success': 0.25}
    names = [name for name, *_ in evaluator.calls]
    assert names == ['baseline-support', 'baseline-query-none', 'baseline-query-default']
    assert evaluator.calls[1][1] == [f'w{i}' for i in range(4, 10)]          # query worlds, never shown as evidence
    other, other_evaluator = workspace(tmp_path, run='r1')
    assert other.baseline()['success'] == 0.25 and other_evaluator.calls == []   # read from the shared preparation
    prepared = json.loads((tmp_path / 'memetic/prepared.json').read_text())
    assert prepared['baselines']['none']['success'] == 0.25 and prepared['baselines']['default']['success'] == 0.75


def test_changes_are_checked_before_they_get_an_id(tmp_path):
    w, _ = workspace(tmp_path)
    assert w.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'TODO'})['problems']
    assert w.propose({'kind': 'instructions', 'file': 'answer.nl', 'find': 'nowhere', 'replace': 'x'})['problems']
    assert w.propose({'kind': 'withhold', 'file': 'answer.nl', 'tools': ['teleport']})['problems']
    assert w.propose({'kind': 'guidance', 'scope': {'kind': 'function', 'name': 'missing'}, 'text': 'Read it.'})['problems']
    assert w.propose({'kind': 'instructions', 'file': 'answer.nl', 'find': '2. Answer it.',
                      'replace': '2. Check the question, then answer it.'}) == {'id': 'c1'}
    assert w.propose({'kind': 'skill', 'function': 'helper', 'name': 'careful-help', 'action': 'create',
                      'description': 'when helping', 'body': 'Read the request twice and answer in one word.'}) == {'id': 'c2'}
    changed = mw.apply_changes(w.spec['support'][0], w.candidate(['c1', 'c2'])['files'])
    files = changed['semantics']['files']
    assert '2. Check the question, then answer it.' in files['answer.nl']
    assert 'helper/skills/careful-help/SKILL.md' in files
    assert w.problems({'kind': 'revert', 'id': 'c1'}) == []
    assert w.propose({'kind': 'revert', 'id': 'c1'}) == {'id': 'c3'}
    assert w.candidate(['c1', 'c3'])['ids'] == []                            # a revert drops its target


def test_evaluate_spends_budget_once_per_distinct_change_set_and_score_is_the_anchored_query_outcome(tmp_path):
    w, evaluator = workspace(tmp_path, evaluations=1)
    change = w.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'Read the question carefully and '
                        'answer with the exact name it asks for.'})['id']
    first = w.evaluate([change])
    assert first['success'] == 0.75 and first['evaluationsLeft'] == 0
    assert w.evaluate([change])['evaluationsLeft'] == 0                     # the same set again is free
    with pytest.raises(ValueError, match='budget is spent'):
        w.evaluate([])
    w.commit([change], 'adds careful reading')
    score = w.score()
    outcome = score['memetic']
    assert score['score'] == 100 and outcome['schema'] == mw.OUTCOME_SCHEMA
    assert outcome['anchor'] == 0.75 and outcome['query_success'] == 0.75
    assert outcome['quality'] == pytest.approx(0.75 - 0.05 * 0.5)            # half the cost weight for a spent budget
    query_runs = [call for call in evaluator.calls if call[0].startswith('query-')]
    assert query_runs and query_runs[0][2] == [{'function': None, 'text': w.state['changes'][change]['text']}]


def test_a_candidate_that_copies_a_query_answer_scores_zero(tmp_path):
    w, _ = workspace(tmp_path)
    change = w.propose({'kind': 'guidance', 'scope': {'kind': 'program'},
                        'text': 'When asked anything, the answer is secret-answer-5 most of the time.'})['id']
    w.commit([change], 'answers')
    outcome = w.score()['memetic']
    assert outcome['leak_refused'] and outcome['query_success'] == 0.0 and outcome['quality'] == 0.0


def test_scoped_guidance_and_input_skills():
    changes = [{'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'A'},
               {'kind': 'guidance', 'scope': {'kind': 'function', 'name': 'helper'}, 'text': 'B'},
               {'kind': 'guidance', 'scope': {'kind': 'inputs', 'when': 'the question is about dates'}, 'text': 'C'}]
    assert mw.compose_guidance(changes) == [{'function': None, 'text': 'A'}, {'function': 'helper', 'text': 'A\n\nB'}]
    skill, = mw.input_skills(changes, 'answer.nl')
    assert skill['function'] == 'answer' and skill['description'].endswith('dates') and skill['body'] == 'C'


def test_memetic_reward_never_rewards_the_trivial_thing():
    baselines = {'none': {'success': 0.5}, 'default': {'success': 0.6}}
    quality, anchor, gain, cost = mw.memetic_reward(0.6, baselines, evaluations=0, budget={'evaluations': 2},
                                                    cost_weight=0.05, committed=True)
    assert anchor == 0.6 and gain == 0 and quality == 0.6
    quality, *_ = mw.memetic_reward(0.6, baselines, evaluations=0, budget={'evaluations': 2}, cost_weight=0.05,
                                    committed=False)
    assert quality == pytest.approx(0.55)


def test_programs_split_related_worlds_apart_and_skip_held_and_small_programs(tmp_path):
    settings = memetic_tasks.DEFAULTS
    cases = [world(i) for i in range(12)]
    cases[1]['source_groups'] = ['g0']                      # related to w0: one component
    cases += [world(i + 100, family='small') for i in range(3)]
    pool = memetic_tasks.programs(cases, settings=settings, excluded=frozenset({'w9'}))
    (key, worlds), = pool.items()
    assert key.startswith('fam@') and len(worlds) == 10
    assert not {'w0', 'w1'} <= {w['id'] for w in worlds} and 'w9' not in {w['id'] for w in worlds}
    support = memetic_tasks.split(worlds, settings)['support']
    assert [c['id'] for c in memetic_tasks.unmeasured(cases, excluded=frozenset({'w9'}))] == [w['id'] for w in support]
    rates = {w['id']: {'successes': 1, 'scored': 2} for w in worlds}
    assert memetic_tasks.unmeasured(cases, excluded=frozenset({'w9'}), pass_rates=rates) == []
    unrated = {**rates, support[0]['id']: {'successes': 1, 'scored': 1}}
    assert memetic_tasks.build(cases, tmp_path / 'none', excluded=frozenset({'w9'}), pass_rates=unrated,
                               log=lambda _: None)['skipped']['no_rate'] == 1, 'every support world needs a rate'
    assert memetic_tasks.merge_rates([{'a': {'successes': 1, 'scored': 2}}, {'a': {'successes': 0, 'scored': 2}}]) == \
        {'a': {'successes': 1, 'scored': 4}}
    receipt = memetic_tasks.build(cases, tmp_path / 'out', excluded=frozenset({'w9'}), pass_rates=rates, log=lambda _: None)
    assert receipt['rows'] == 1
    row = json.loads((tmp_path / 'out/cases.jsonl').read_text())
    assert row['family'] == 'memetic_improve' and row['semantics']['root'] == 'improve_program.nl'
    assert 'improve_program.nl' in row['semantics']['files'] and row['semantics']['world']['kind'] == 'memetic'
    spec_value = json.loads(Path(row['semantics']['world']['task']).read_text())
    assert len(spec_value['support']) == 4 and len(spec_value['query']) == 6
    assert not {w['id'] for w in spec_value['support']} & {w['id'] for w in spec_value['query']}
    hard = {w['id']: {'successes': 0, 'scored': 4} for w in worlds}
    assert memetic_tasks.build(cases, tmp_path / 'out2', pass_rates=hard, log=lambda _: None)['rows'] == 0


def test_task_rollouts_read_the_world_score_and_score_a_crash_against_the_anchor(tmp_path):
    out = tmp_path / 'g' / 'memetic'
    out.mkdir(parents=True)
    (out / 'prepared.json').write_text(json.dumps({'baselines': {'none': {'success': 0.4}, 'default': None}}))
    (tmp_path / 'spec.json').write_text(json.dumps({'out': str(out)}))
    task = {'semantics': {'world': {'kind': 'memetic', 'task': str(tmp_path / 'spec.json')}}}
    scored = {'schema': mw.OUTCOME_SCHEMA, 'quality': 0.7, 'anchor': 0.4, 'gain': 0.3, 'cost': 0.}
    assert memetic_outcome(task, {'outcome': {'world': {'score': 100, 'memetic': scored}}}) == scored
    crashed = memetic_outcome(task, {'outcome': {}})
    assert crashed['quality'] == 0. and crashed['anchor'] == 0.4 and crashed['unscored_workspace']
    assert memetic_outcome({'semantics': {}}, {'outcome': {}}) is None


def test_grpo_centres_memetic_groups_on_the_anchor():
    episodes = [{'reward': r, 'quality': r, 'anchor': 0.6} for r in (0.5, 0.7, 0.3, 0.5)]
    advantages = grpo.episode_advantages(episodes)
    assert advantages[1] > 0 and all(a < 0 for i, a in enumerate(advantages) if i != 1)   # only the one above 0.6 gains
    assert grpo.episode_advantages([{'reward': 0.5, 'quality': 0.5, 'anchor': 0.6}] * 2) is None


def test_grpo_binds_a_memetic_row_to_its_group(tmp_path):
    (tmp_path / 'spec.json').write_text(json.dumps({'schema': mw.SPEC_SCHEMA, 'out': None, 'endpoint': None}))
    row = {'id': 'memetic:x', 'semantics': {'world': {'kind': 'memetic', 'task': str(tmp_path / 'spec.json')}}}
    c = grpo.Collector.__new__(grpo.Collector)
    c.args = type('A', (), {'actor': 'http://actor'})()
    c.runtime, c.work = tmp_path / 'ts-host', tmp_path
    directory = tmp_path / 'groups/g000001'
    directory.mkdir(parents=True)
    bound = c._bind_memetic(row, directory)
    spec_value = json.loads(Path(bound['semantics']['world']['task']).read_text())
    assert spec_value['out'] == str(directory / 'memetic') and spec_value['endpoint'] == 'http://actor'
    assert spec_value['sampling_config'] == str(tmp_path / 'sampling-config.json')
    assert row['semantics']['world']['task'] == str(tmp_path / 'spec.json')                 # the pool row is unchanged


def test_mechanical_guidance_outlines_the_shortest_clean_success():
    steps = [[{'call': 'root', 'tool': 'eval', 'status': 'ok', 'code': 'x = read()', 'output': 'ok'},
              {'call': 'root', 'tool': 'eval', 'status': 'error', 'code': 'boom()', 'output': 'TypeError'},
              {'call': 'root', 'tool': 'return_result', 'status': 'ok', 'arguments': '{"value": 1}', 'output': ''}]]
    text = mw.mechanical_guidance(steps)
    assert '1. eval: x = read()' in text and 'boom' not in text and '2. return_result' in text


NODE = os.environ.get('NATLANG_NODE') or shutil.which('node') or str(
    Path.home() / '.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node')


@pytest.mark.skipif(not Path(NODE).exists() or not (mw.REPO / 'ts-host/dist/index.js').exists(),
                    reason='needs node and a built ts-host')
def test_the_natlang_loader_checks_a_changed_program(monkeypatch):
    monkeypatch.setenv('NATLANG_NODE', NODE)
    assert mw.check_program({'semantics': {'files': FILES}}) == []
    broken = {**FILES, 'answer.nl': '---\nargs: [unclosed\n---\nbody\n'}
    assert mw.check_program({'semantics': {'files': broken}})


def test_only_a_group_whose_support_all_passes_is_skipped():
    assert mw.all_passed({'rollouts': 8, 'scored': 8, 'success': 1.0})
    assert not mw.all_passed({'rollouts': 8, 'scored': 8, 'success': 0.0}), 'all-fail groups are admitted'
    assert not mw.all_passed({'rollouts': 8, 'scored': 6, 'success': 1.0}), 'an unscored sample is no pass'
    assert not mw.all_passed({'rollouts': 0, 'scored': 0, 'success': 0.0})


def test_memetic_groups_log_the_guided_against_unguided_gap(tmp_path):
    record = task_rollouts.memetic_reward_record({'anchor': .4, 'gain': .1, 'cost': 0, 'query_success': .5,
                                                  'baselines': {'none': {'success': .25}, 'default': None}})
    assert record['without'] == .25 and record['default'] is None
    episodes = [{'memetic': {**record, 'query_success': q, 'committed': {'ids': ['c1']}}} for q in (.5, .75)]
    episodes.append({'memetic': {'query_success': 0., 'unscored_workspace': True, 'without': None}})
    group = grpo.memetic_group_record(7, 'memetic:x', episodes)
    assert group['with_guidance'] == .625 and group['without'] == .25 and group['gap'] == .375
    assert group['scored'] == 2 and group['committed'] == 2 and group['samples'] == 3
    assert grpo.memetic_group_record(8, 't', [{'reward': 1}]) is None
    (tmp_path / 'train.jsonl').write_text(json.dumps({'memetic_groups': [{'gap': .125}]}) + '\n')
    history = list(grpo.memetic_gap_history(tmp_path / 'train.jsonl'))
    step = grpo.memetic_step_record([group], history)
    assert step['gap_mean'] == .375 and step['gap_running_mean'] == .25 and step['gap_running_groups'] == 2


class StateEvaluator(Evaluator):
    """As Evaluator, writing the guidance binding file a real evaluation writes; refined states succeed at ``refined``."""

    def __init__(self, root, refined):
        super().__init__()
        self.root, self.refined = root, refined

    def __call__(self, cases, guidance, *, name, samples=None, states=None):
        report = super().__call__(cases, guidance, name=name, samples=samples)
        out = self.root / name
        out.mkdir(parents=True, exist_ok=True)
        if guidance:
            (out / 'program-guidance.json').write_text(json.dumps(
                [{'native_state_id': 'nzs1_' + 't' * 52, 'function': entry['function']} for entry in guidance]))
        self.calls[-1] = (*self.calls[-1], states)
        return {**report, 'dir': str(out), **({'success': self.refined} if states else {})}


@pytest.mark.parametrize('refined, kept', [(1.0, True), (0.5, False)])
def test_refined_guidance_states_are_kept_only_when_the_query_worlds_confirm_them(tmp_path, monkeypatch, refined, kept):
    value = spec(tmp_path, 1)
    value['refine'] = {'checkpoint': '/x.pt', 'steps': 1}
    evaluator = StateEvaluator(tmp_path / 'evals', refined)
    w = mw.Workspace(value, evaluator, check=lambda case: [], run='r0')
    refined_state = {None: {'id': 'nzs1_' + 'r' * 52, 'build': str(tmp_path / 'build')}}
    seen = {}

    def refine(spec_, reports, guidance, out):
        seen.update(guidance=guidance)
        return refined_state, {'rows': 3}
    monkeypatch.setattr(mw, 'refine_guidance', refine)
    change = w.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'Read the question carefully and '
                        'answer with the exact name it asks for.'})['id']
    w.evaluate([change])
    w.commit([change], 'careful reading')
    outcome = w.score()['memetic']
    assert seen['guidance'] == [{'function': None, 'text': w.state['changes'][change]['text']}]
    runs = {call[0]: call for call in evaluator.calls}
    refined_run = next(call for name, call in runs.items() if name.startswith('query-refined-'))
    assert refined_run[1] == [f'w{i}' for i in range(4, 10)] and refined_run[-1] == refined_state, 'judged on the query worlds'
    assert outcome['refine']['kept'] is kept and outcome['refine']['text_success'] == 0.75
    assert outcome['query_success'] == (refined if kept else 0.75)


def test_guidance_rows_are_the_verified_calls_that_imported_the_state_with_its_text_in_place():
    if not (mw.REPO / 'ts-host/dist/tune/pool.js').exists():
        pytest.skip('needs the built ts-host dist')
    state = 'nzs1_' + 'g' * 52
    step = lambda: {'phase': 'action', 'invocation_id': 'i1', 'tools_offered': [], 'assistant': {},
                    'model_response': {'text': 'Sunday'},
                    'context': [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': [
                        {'type': 'text', 'text': 'Instructions: x\n\n<natlang_program_guidance>\n'},
                        {'type': 'native_state', 'id': state}, {'type': 'text', 'text': '\n</natlang_program_guidance>\n'}]}]}
    passed = {'id': 'row1', 'trajectory': [step()], 'outcome': {'status': 'done', 'oracle': {'accepted': True}}}
    report = {'results': [('w1', passed), ('w2', {**passed, 'id': 'row2'})], 'rewards': [1.0, 0.0]}
    guidance = [{'function': None, 'text': 'Check the weekday twice.'}]
    rows = mw.guidance_rows([report], guidance)
    assert len(rows) == 1, 'only the verified success'
    [row] = rows
    assert row['binding'] == 'guidance:*' and row['texts'] == {'instructions': 'Check the weekday twice.'}
    assert row['messages'][1]['content'] == ('Instructions: x\n\n<natlang_program_guidance>\nCheck the weekday twice.\n'
                                             '</natlang_program_guidance>\n')
    assert row['teacher'] == 'rl' and row['evidence']['rootOracle'] is True and row['reply'] == {'text': 'Sunday'}
