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
    assert [name for name, *_ in evaluator.calls] == ['baseline-support']   # the evidence; the anchor only to score
    assert w.baselines['none']['success'] == 0.25 and w.baselines['default']['success'] == 0.75
    calls = {name: worlds for name, worlds, *_ in evaluator.calls}
    assert set(calls) == {'baseline-support', 'baseline-query-none', 'baseline-query-default'}
    assert calls['baseline-query-none'] == [f'w{i}' for i in range(4, 10)]   # query worlds, never shown as evidence
    other, other_evaluator = workspace(tmp_path, run='r1')
    assert other.baseline()['success'] == 0.25 and other.baselines['none']['success'] == 0.25
    assert other_evaluator.calls == []                                       # read from the shared preparation
    assert 'baselines' not in json.loads((tmp_path / 'memetic/prepared.json').read_text())
    assert json.loads((tmp_path / 'memetic/baselines.json').read_text())['default']['success'] == 0.75


def test_a_preparation_recorded_before_the_staged_layout_keeps_its_baselines(tmp_path):
    out = tmp_path / 'memetic'
    out.mkdir()
    (out / 'prepared.json').write_text(json.dumps({'admitted': True, 'support_success': 0.5, 'attempts': [],
                                                   'baselines': {'none': {'success': 0.4}, 'default': None}}))
    w, evaluator = workspace(tmp_path)
    assert w.baselines == {'none': {'success': 0.4}, 'default': None} and evaluator.calls == []


def test_the_staged_layout_fits_a_memetic_group_in_the_episode_wall_clock():
    value = {'budget': {'evaluations': 2}, 'eval': {'case_seconds': 900, 'workers': 4, 'samples': 2}}
    staged = mw.bind_layout(value, episode_seconds=7200)
    # support, two evaluations, the commit's query score and the supervisor's own turns: five shares of 7200 s
    assert staged['layout'] == 'staged' and staged['eval'] == {'case_seconds': 1440, 'workers': 0, 'samples': 2}
    assert value['eval']['case_seconds'] == 900                               # the pool's spec is unchanged
    inline = mw.bind_layout(value, episode_seconds=7200, layout='inline')
    assert inline['eval'] == value['eval'] and inline['layout'] == 'inline'
    assert mw.group_seconds(7200, 2) == 7200 + 1440 and mw.group_seconds(7200, 2, 'inline') == 7200
    with pytest.raises(ValueError):
        mw.bind_layout(value, episode_seconds=7200, layout='sideways')


def test_the_preparation_stage_writes_the_evidence_before_the_anchor(tmp_path, monkeypatch, capsys):
    value = spec(tmp_path)
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(value))
    seen = []

    def evaluator(spec_value):
        def evaluate(cases, guidance, *, name, samples=None):
            if name != 'baseline-support':
                seen.append((name, (tmp_path / 'memetic/prepared.json').exists()))
            return Evaluator()(cases, guidance, name=name, samples=samples)
        return evaluate
    monkeypatch.setattr(mw, 'collection_evaluator', evaluator)
    prepared = mw.prepare_stage(path)
    assert prepared['admitted'] and sorted(seen) == [('baseline-query-default', True), ('baseline-query-none', True)]
    assert (tmp_path / 'memetic/baselines.json').exists()
    events = [json.loads(line)['event'] for line in capsys.readouterr().out.splitlines()]
    assert events == ['memetic prepared', 'memetic baselines']


def test_parallel_runs_every_call_at_once_and_raises_the_first_error():
    import threading
    barrier = threading.Barrier(3, timeout=5)
    assert mw.parallel({k: (lambda k=k: barrier.wait() * 0 + k) for k in range(3)}) == {0: 0, 1: 1, 2: 2}
    with pytest.raises(ZeroDivisionError):
        mw.parallel({'a': lambda: 1, 'b': lambda: 1 / 0})


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
    (out / 'prepared.json').write_text(json.dumps({'admitted': True}))      # the staged layout: the anchor on its own
    (out / 'baselines.json').write_text(json.dumps({'none': {'success': 0.3}, 'default': {'success': 0.6}}))
    assert memetic_outcome(task, {'outcome': {}})['anchor'] == 0.6
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
    (tmp_path / 'spec.json').write_text(json.dumps({'schema': mw.SPEC_SCHEMA, 'out': None, 'endpoint': None,
                                                    'budget': {'evaluations': 2}, 'eval': {'case_seconds': 900}}))
    row['semantics']['world']['case_seconds'] = 7200
    c.args.case_seconds = 3600                                     # the episode runs the larger clock (build_queue)
    staged = json.loads(Path(c._bind_memetic(row, directory)['semantics']['world']['task']).read_text())
    assert staged['layout'] == 'staged' and staged['eval']['case_seconds'] == 1440 and staged['eval']['workers'] == 0
    c.args.memetic_layout = 'inline'
    inline = json.loads(Path(c._bind_memetic(row, directory)['semantics']['world']['task']).read_text())
    assert inline['layout'] == 'inline' and inline['eval'] == {'case_seconds': 900}


def _staged_collector(tmp_path, layout='staged'):
    import collections
    import random
    import threading
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'spec.json').write_text(json.dumps({'schema': mw.SPEC_SCHEMA, 'out': None, 'endpoint': None,
                                                    'budget': {'evaluations': 2}, 'eval': {'case_seconds': 900}}))
    row = {'id': 'memetic:x', 'family': 'memetic_improve', 'split': 'train', 'semantics': {
        'root': 'improve_program.nl', 'files': {}, 'world': {'kind': 'memetic', 'task': str(tmp_path / 'spec.json'),
                                                             'case_seconds': 7200}}}
    c = grpo.Collector.__new__(grpo.Collector)
    c.args = type('A', (), {'actor': 'http://actor', 'memetic_share': 1., 'memetic_group': 4, 'group': 8,
                            'case_seconds': 7200, 'memetic_model_requests': 64, 'max_model_requests': 40,
                            'max_turns': 128, 'seed_base': 0, 'context_tokens': 131072, 'memetic_layout': layout})()
    c.runtime, c.work, c.policy_sha = tmp_path / 'ts-host', tmp_path, 'a' * 64
    c.memetic_rows, c.memetic_cursor, c.rng = [row], 0, random.Random(0)
    c.next_group, c.groups, c.outstanding = 1, {}, 0
    c.pending, c.lock = collections.deque(), threading.Lock()
    c._new_group()
    return c


def test_a_staged_memetic_group_starts_its_supervisors_only_once_its_preparation_admits_it(tmp_path, capsys):
    c = _staged_collector(tmp_path)
    [(group, item)] = list(c.pending)                       # one preparation item, the supervisors held back
    assert item[grpo.MEMETIC_PREPARE] and len(c.groups[group]['held_back']) == 4
    c.pending.popleft()                                     # a slot runs the preparation stage
    c.pending.append((99, {'key': 'other'}))
    path = c.groups[group]['dir'] / 'memetic' / 'prepared.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'admitted': True, 'support_success': 0.5, 'resolved_fraction': 1.0}))
    assert c._memetic_prepared(group, c.groups[group], path)
    assert [g for g, _ in c.pending] == [group] * 4 + [99]   # ahead of other groups
    assert c.groups[group]['held_back'] == [] and c.outstanding == 1

    other = _staged_collector(tmp_path / 'b')
    [(group, _)] = list(other.pending)
    path = other.groups[group]['dir'] / 'memetic' / 'prepared.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'admitted': False, 'support_success': 1.0, 'resolved_fraction': 1.0}))
    other.pending.clear()
    assert other._memetic_prepared(group, other.groups[group], path)
    assert not other.pending and group not in other.groups and other.outstanding == 0
    dropped = [json.loads(line) for line in capsys.readouterr().out.splitlines() if 'dropped' in line]
    assert dropped[-1]['reason'] == 'not admitted'


def test_an_inline_memetic_group_queues_its_supervisors_at_once(tmp_path):
    c = _staged_collector(tmp_path, layout='inline')
    assert len(c.pending) == 4 and not any(item.get(grpo.MEMETIC_PREPARE) for _, item in c.pending)


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


# ---- memetic episodes in expert-iteration collection and the SFT mix (B15) ------------------------------------------
def _outcome(**over):
    return {'schema': mw.OUTCOME_SCHEMA, 'quality': 0.8, 'anchor': 0.5, 'gain': 0.35, 'cost': 0.05,
            'query_success': 0.85, 'committed': {'ids': ['c1'], 'reason': 'r'}, 'leak_refused': False, **over}


def test_a_memetic_success_beats_the_trivial_improvements_on_query_worlds_with_a_leak_free_commit():
    assert task_rollouts.memetic_success(_outcome())
    assert not task_rollouts.memetic_success(_outcome(quality=0.5))           # ties the anchor: no gain
    assert not task_rollouts.memetic_success(_outcome(leak_refused=True))
    assert not task_rollouts.memetic_success(_outcome(committed=None))
    assert not task_rollouts.memetic_success(_outcome(unscored_workspace=True))
    assert not task_rollouts.memetic_success(None)


class GuidedEvaluator(Evaluator):
    """The mechanical default does nothing (0.25); written guidance that names the exact answer form helps (1.0)."""

    def __call__(self, cases, guidance, *, name, samples=None):
        report = super().__call__(cases, guidance, name=name, samples=samples)
        text = ' '.join(g['text'] for g in guidance or ())
        report['success'] = 1.0 if 'exact name' in text else 0.25
        return report


def test_a_workspace_commit_that_beats_the_anchor_is_a_success_and_the_mechanical_default_is_not(tmp_path):
    w = mw.Workspace(spec(tmp_path), GuidedEvaluator(), check=lambda case: [], run='r0')
    change = w.propose({'kind': 'guidance', 'scope': {'kind': 'program'},
                        'text': 'Answer with the exact name the question asks for.'})['id']
    assert w.evaluate([change])['success'] == 1.0
    w.commit([change], 'exact answer form')
    assert task_rollouts.memetic_success(w.score()['memetic'])
    lazy = mw.Workspace(spec(tmp_path), GuidedEvaluator(), check=lambda case: [], run='r1')
    lazy.commit([], 'nothing to change')
    assert not task_rollouts.memetic_success(lazy.score()['memetic'])


def test_a_recorded_only_evaluator_reads_recorded_evaluations_and_never_samples(tmp_path):
    out = tmp_path / 'memetic'
    (out / 'evals' / 'support-x').mkdir(parents=True)
    (out / 'evals' / 'support-x' / 'summary.json').write_text(json.dumps({'success': 0.5}))
    value = {'out': str(out), 'endpoint': 'http://127.0.0.1:9', 'recorded_only': True,
             'eval': {'samples': 2, 'workers': 1, 'case_seconds': 1, 'max_model_requests': 1, 'max_turns': 1,
                      'seed_base': 0}}
    evaluate = mw.collection_evaluator(value)
    assert evaluate([], None, name='support-x') == {'success': 0.5}
    with pytest.raises(ValueError, match='not recorded'):
        evaluate([world(0)], None, name='support-y')
    assert not (out / 'evals' / 'support-y' / 'collection').exists()


def test_grpo_runs_no_privileged_attempts_of_a_memetic_group(tmp_path):
    c = grpo.Collector.__new__(grpo.Collector)
    c.args = type('A', (), {'teacher_attempts': 2})()
    c.groups = {1: {'dir': tmp_path, 'memetic': True}}
    assert c._teachers(1) == 0


def _ei():
    import expert_iteration
    return expert_iteration


class _MemeticBase:
    def __init__(self, rows):
        self.memetic_rows, self.groups, self.args = rows, {}, type('A', (), {'memetic_max_groups': 2})()

    def _memetic_open(self):
        return True


def test_expert_iteration_bounds_memetic_groups_in_flight_and_near_the_deadline():
    import time
    ei = _ei()
    rows = [{'semantics': {'world': {'kind': 'memetic', 'case_seconds': 3600}}}]
    collector = ei.frontier_collector(_MemeticBase)(rows)
    collector.deadline = time.time() + 7200
    assert collector._memetic_open()
    collector.groups = {1: {'memetic': True, 'done': 0, 'size': 4}, 2: {'memetic': True, 'done': 4, 'size': 4},
                        3: {'done': 0, 'size': 4}}
    assert collector._memetic_open()                       # one running, one finished, one ordinary
    collector.groups[4] = {'memetic': True, 'done': 1, 'size': 4}
    assert not collector._memetic_open()                   # two running: the bound
    collector.groups = {}
    collector.deadline = time.time() + 1800
    assert not collector._memetic_open()                   # its wall clock no longer fits before the deadline
    collector.deadline = time.time() + 4000
    assert not collector._memetic_open()                   # nor with its preparation stage (3600 + 3600 / 5)
    collector.args.memetic_layout = 'inline'
    assert collector._memetic_open()


def test_expert_iteration_collection_draws_memetic_groups_by_default_and_opts_out(tmp_path):
    from types import SimpleNamespace
    ei = _ei()
    assert ei.MEMETIC_SHARE == 0.02
    c = ei.Cycle.__new__(ei.Cycle)
    c.a = SimpleNamespace(kv_gb=12, slots=8, fill=None, k=4, collect_hours=1., no_neuralese_variants=True,
                          neuralese_variants=None)
    c.pin, c.collect_dir, c.dir = tmp_path, tmp_path / 'collect', tmp_path
    argv = c.collect_command('u', 'ckpt', 'sha', 8)
    assert argv[argv.index('--memetic-share') + 1] == '0.02'
    c.a.memetic_share = 0.
    argv = c.collect_command('u', 'ckpt', 'sha', 8)
    assert argv[argv.index('--memetic-share') + 1] == '0.0'


def test_frontier_rates_count_a_memetic_episode_by_its_verified_success(tmp_path):
    ei = _ei()
    directory = tmp_path / 'groups' / 'g000000'
    directory.mkdir(parents=True)
    rows = [{'task_id': 'memetic:p', 'reward': 0.7, 'outcome': 'memetic_scored', 'memetic_success': True},
            {'task_id': 'memetic:p', 'reward': 0.3, 'outcome': 'memetic_scored', 'memetic_success': False},
            {'task_id': 't', 'reward': 1.0}]
    (directory / 'rewards.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert ei.group_outcomes([directory]) == {'memetic:p': [1, 2], 't': [1, 1]}


def test_sft_sources_take_a_memetic_success_as_an_own_success_and_replay_it_from_its_recorded_evaluations(tmp_path):
    import hashlib
    import sft_sources
    ok = {'outcome': 'memetic_scored', 'memetic_success': True, 'reward': {'quality': 0.7}}
    assert sft_sources.episode_success(ok)
    assert not sft_sources.episode_success({**ok, 'memetic_success': False})
    assert sft_sources.episode_success({'outcome': 'task_success', 'reward': {'quality': 1.0}})
    group = tmp_path / 'collect' / 'groups' / 'g000001'
    recorded = group / 'memetic'
    for name in ('baseline-support', 'support-abc'):
        (recorded / 'evals' / name / 'collection').mkdir(parents=True)
        (recorded / 'evals' / name / 'summary.json').write_text(json.dumps({'success': 0.5, 'name': name}))
    (recorded / 'prepared.json').write_text(json.dumps({'admitted': True}))
    spec_path = group / 'memetic-spec.json'
    spec_path.write_text(json.dumps({'schema': mw.SPEC_SCHEMA, 'out': str(recorded), 'endpoint': 'http://actor'}))
    task = {'id': 'memetic:p', 'semantics': {'world': {
        'kind': 'memetic', 'task': str(spec_path), 'spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest()}}}
    replay = tmp_path / 'replay' / 'e'
    replay.mkdir(parents=True)
    bound, refusal = sft_sources.memetic_replay_task(task, replay)
    assert refusal is None and sft_sources.is_memetic(bound)
    bound_spec = json.loads(Path(bound['semantics']['world']['task']).read_text())
    assert bound_spec['recorded_only'] and bound_spec['out'] == str(replay / 'memetic')
    assert sorted(p.parent.name for p in (replay / 'memetic' / 'evals').glob('*/summary.json')) == \
        ['baseline-support', 'support-abc']
    assert not (replay / 'memetic' / 'evals' / 'support-abc' / 'collection').exists()   # summaries only
    assert task['semantics']['world']['task'] == str(spec_path)                         # the recording is read only
    spec_path.write_text(json.dumps({'schema': mw.SPEC_SCHEMA, 'out': str(recorded), 'refine': True}))
    assert sft_sources.memetic_replay_task(task, tmp_path / 'r2')[1] == 'memetic_spec_changed'


def test_sft_build_drops_a_memetic_row_whose_worlds_include_a_held_task(tmp_path, monkeypatch):
    import sft_sources
    from natlang_neuralese.rl import held_eval
    for name in ('exclusion_ids', 'training_exclusion_ids'):     # the held tasks themselves: none in this fixture
        monkeypatch.setattr(held_eval, name, lambda *args, **kwargs: [], raising=False)
    tasks = {'memetic:a': {'id': 'memetic:a', 'curriculum': {'world_ids': ['w1', 'held-1']}},
             'memetic:b': {'id': 'memetic:b', 'curriculum': {'world_ids': ['w1', 'w2']}}}
    held = ({'held-1'}, set(), set())
    assert sft_sources.held_excluded_ids('manifest', tasks, held) == {'memetic:a'}


def test_a_guidance_binding_carries_the_runtime_sentinel_around_its_state_id(tmp_path, monkeypatch):
    """guidance/program.ts accepts a binding only when program_guidance is the native-state sentinel of its id; every
    guided evaluation of grpo-v4 run2 failed on that before 2026-11-23 ('invalid natlang.memetic-program-guidance/1')."""
    state = 'nzs1_' + 'a' * 52
    identity = {'architecture': 'lfm3', 'checkpoint_sha256': 'a' * 64, 'tokenizer_fingerprint': 'b' * 64,
                'embedding_weight_sha256': 'c' * 64}
    monkeypatch.setattr(mw, 'request_json', lambda server, path, value=None, timeout=900:
                        {'id': state} if 'write-text' in path else {})
    monkeypatch.setattr(mw, 'native_identity', lambda info: identity)
    (tmp_path / 'states').mkdir()
    (tmp_path / 'states' / f'{state}.safetensors').write_bytes(b'x')
    binding = mw.guidance_binding('http://actor', tmp_path, 'memetic:p', 'Read twice.')
    assert binding['program_guidance'] == '' + state + '' and binding['native_state_id'] == state


def test_an_evaluation_no_run_finished_measures_nothing(tmp_path):
    class Failing(Evaluator):
        def __call__(self, cases, guidance, *, name, samples=None):
            report = super().__call__(cases, guidance, name=name, samples=samples)
            return {**report, 'success': None if guidance else report['success'], 'scored': 0 if guidance else len(cases)}
    w = mw.Workspace(spec(tmp_path), Failing(), check=lambda case: [], run='r0')
    assert w.baselines['default']['success'] is None                       # the anchor ignores it
    change = w.propose({'kind': 'guidance', 'scope': {'kind': 'program'}, 'text': 'Read the question twice.'})['id']
    with pytest.raises(ValueError, match='not counted against the budget'):
        w.evaluate([change])
    assert w.budget()['evaluations'] == 2
    w.commit([change], 'reads twice')
    with pytest.raises(ValueError, match='unmeasured'):
        w.score()
    nothing = mw.Workspace(spec(tmp_path / 'b'), lambda cases, guidance, *, name, samples=None: {
        'success': None, 'scored': 0, 'rollouts': len(cases), 'results': [], 'dir': name}, check=lambda case: [], run='r')
    with pytest.raises(ValueError, match='no evidence'):
        nothing.baseline()


def test_collection_leaves_out_memetic_rows_with_a_held_world(tmp_path):
    rows = []
    for name, ids in (('a', ['w1', 'w2']), ('b', ['w3', 'w4'])):
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps({'support': [{'id': ids[0]}], 'query': [{'id': ids[1]}]}))
        rows.append({'id': f'memetic:{name}', 'semantics': {'world': {'kind': 'memetic', 'task': str(path)}}})
    manifest = tmp_path / 'held.json'
    manifest.write_text(json.dumps({'tasks': [{'id': 'w4'}], 'excluded_split_groups': []}))
    args = type('A', (), {'held_manifest': str(manifest), 'exclude_manifest': []})()
    assert [r['id'] for r in grpo.memetic_rows_held_free(rows, args)] == ['memetic:a']


def test_async_drive_passes_its_memetic_share_to_the_collection(tmp_path, monkeypatch):
    ei = _ei()
    seen = []
    monkeypatch.setattr(ei, 'async_drive', lambda args: seen.append(args) or 0)
    base = ['expert_iteration.py', 'async-drive', '--root', str(tmp_path), '--start', '/p.pt', '--commit', 'c' * 40]
    for extra, share in (([], ei.MEMETIC_SHARE), (['--memetic-share', '0'], 0.)):
        monkeypatch.setattr(sys, 'argv', base + extra)
        ei.main()
        assert seen[-1].memetic_share == share
    driver = ei.AsyncRoundDriver.__new__(ei.AsyncRoundDriver)
    driver.a, driver.root = seen[-1], tmp_path
    assert driver.cycle_args().memetic_share == 0.


# ---- memetic groups under collection load (2026-11-23 aei1) -----------------------------------------------------------
def _journal(path, seconds=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{'event': 'start', 'key': 'k'}] + [{'event': 'activity', 'pad': 'x' * 500}] * 50
    if seconds is not None:
        lines.append({'event': 'finish', 'key': 'k', 'status': 'complete', 'elapsed_seconds': seconds})
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines))


def test_a_memetic_program_runs_only_when_its_family_fits_the_nested_wall_clock(tmp_path):
    assert mw.fits([100.] * 8, 1440) and not mw.fits([100.] * 7, 1440)          # too few measured episodes
    assert not mw.fits([100.] * 7 + [2000.] * 3, 1440)                          # its 0.9 quantile runs past
    _journal(tmp_path / 'done.journal.jsonl', 812.5)
    _journal(tmp_path / 'running.journal.jsonl')
    assert mw.finish_seconds(tmp_path / 'done.journal.jsonl') == 812.5
    assert mw.finish_seconds(tmp_path / 'running.journal.jsonl') is None
    c = _staged_collector(tmp_path / 'c')
    c.args.memetic_fit_quantile = 0.9
    rows = [{**c.memetic_rows[0], 'id': f'memetic:{name}', 'curriculum': {'program_family': name}}
            for name in ('slow', 'fast')]
    c.memetic_rows, c.memetic_cursor = rows, 0
    c.family_seconds = {'slow': [3000.] * 10}
    assert c._memetic_row() is None                                             # nothing fits yet: an ordinary group
    c.family_seconds['fast'] = [600.] * 10
    assert c._memetic_row()['id'] == 'memetic:fast'
    c.args.memetic_fit_quantile = 0
    assert {c._memetic_row()['id'] for _ in range(2)} == {'memetic:slow', 'memetic:fast'}   # opt-out: every program


def test_an_ordinary_groups_episode_seconds_feed_its_family_and_nested_runs_take_slots(tmp_path):
    import time
    c = _staged_collector(tmp_path)
    c.family_seconds = __import__('collections').defaultdict(list)
    directory = tmp_path / 'groups' / 'g000009'
    for k, seconds in enumerate((300., 900.)):
        _journal(directory / 'workers' / f's7-task-000-{k:02d}.journal.jsonl', seconds)
    c._record_seconds(directory, 'fam')
    assert sorted(c.family_seconds['fam']) == [300., 900.]
    [(group, _)] = list(c.pending)
    evals = c.groups[group]['dir'] / 'memetic' / 'evals' / 'support-x' / 'collection' / 'workers'
    _journal(evals / 'a.journal.jsonl')
    _journal(evals / 'b.journal.jsonl')
    _journal(evals / 'c.journal.jsonl', 100.)
    c.nested_seen = (0., 0)
    assert c._nested_slots() == 2                                               # two running, one finished
    _journal(evals / 'd.journal.jsonl')
    assert c._nested_slots() == 2 and c.nested_seen[0] <= time.time()          # cached for 20 s
    c.args.memetic_nested_slots = False
    assert c._nested_slots() == 0


def test_family_seconds_are_seeded_from_earlier_collections(tmp_path):
    c = _staged_collector(tmp_path / 'collect' / 'v0001')
    c.work = tmp_path / 'collect' / 'v0001'
    c.memetic_rows = [{**c.memetic_rows[0], 'curriculum': {'program_family': 'fam'}}]
    earlier = tmp_path / 'collect' / 'v0000' / 'groups' / 'g000001'
    earlier.mkdir(parents=True)
    (earlier / 'cases.jsonl').write_text(json.dumps({'id': 't', 'family': 'fam'}) + '\n')
    _journal(earlier / 'workers' / 's7-task-000-00.journal.jsonl', 444.)
    c.family_seconds = __import__('collections').defaultdict(list)
    c._seed_family_seconds()
    assert c.family_seconds['fam'] == [444.]
