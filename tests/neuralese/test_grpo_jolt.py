"""JOLT+ on the per-call GRPO stack (plans/neuralese/RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md section 1): teacher
attempts for groups whose students failed, trained in student form with a non-zero, never negative advantage."""
import collections
import json
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/neuralese_data"))
import grpo  # noqa: E402
from natlang_neuralese.rl import jolt  # noqa: E402
from natlang_neuralese.rl.hindsight import TrajectoryIndex, build_index, trajectory_value  # noqa: E402


def test_guided_advantages_give_teachers_signal_where_students_had_none():
    student, teacher = jolt.guided_advantages([0.] * 8, [1., 0., 0., 1.])
    assert student is None                       # plain GRPO: an all-fail student group has no student signal
    assert teacher[0] > 0 and teacher[3] == teacher[0] and teacher[1] == teacher[2] == 0.
    # students keep exactly their own plain advantage; a guided failure is never pushed down
    student, teacher = jolt.guided_advantages([1., 0., 0., 0.], [0., 1.])
    assert student == pytest.approx(grpo.group_advantages([1., 0., 0., 0.]))
    assert teacher[0] == 0. and teacher[1] > 0
    assert jolt.guided_advantages([0., 0.], [0., 0.]) == (None, None)
    assert jolt.guided_advantages([1., 1.], [1.]) == (None, None)   # a teacher success where students succeeded


def test_teacher_fact_kinds_exclude_answers_by_default():
    kinds = jolt.teacher_fact_kinds()
    assert not set(kinds) & set(jolt.ANSWER_FACT_KINDS) and 'reference_solution' in kinds
    assert jolt.teacher_fact_kinds('all') is None
    assert jolt.teacher_fact_kinds('decisive,reference_solution') == ('decisive', 'reference_solution')
    with pytest.raises(ValueError):
        jolt.teacher_fact_kinds('answer')


def _capture(prompt, samples=((7, -0.5), (8, -0.25))):
    return {'prompt': prompt, 'policy': {'checkpoint_sha256': 'p'},
            'sampling_law': {'temperature': 1, 'top_p': 1.0, 'probabilities': 'post-processing-normalized'},
            'samples': [{'token_id': t, 'logp': lp, 'input_role': 'ordinary'} for t, lp in samples]}


def _teacher_rollout():
    guided = [{'kind': 'tokens', 'token_ids': [1, 2]},
              {'kind': 'tokens', 'token_ids': [90, 91], 'privileged': True},
              {'kind': 'tokens', 'token_ids': [3]}]
    writer = [{'kind': 'tokens', 'token_ids': [40, 41, 42]}]
    return {'task_id': 't', 'jolt_role': 'teacher', 'reward': {'quality': 1.0}, 'outcome': 'task_success',
            'turns': [{'invocation_id': 'root', 'sampling': {**_capture(guided), 'privileged': {'mode': 'text'}}},
                      {'invocation_id': 'writer', 'sampling': _capture(writer)}]}


def test_student_form_drops_only_the_guidance_and_keeps_the_actor_law():
    row, stats = _teacher_rollout(), collections.Counter()
    converted = jolt.student_form(row, stats)
    root, writer = converted['turns']
    assert [s['token_ids'] for s in root['sampling']['prompt']] == [[1, 2], [3]]
    assert 'privileged' not in root['sampling']
    assert writer['sampling']['prompt'] == row['turns'][1]['sampling']['prompt']      # the writer runs in its own context
    assert [s['logp'] for s in root['sampling']['samples']] == [-0.5, -0.25]          # mu(a | s, guidance) stays
    assert stats['teacher_calls_student_form'] == 1 and stats['teacher_privileged_tokens_removed'] == 2
    assert len(row['turns'][0]['sampling']['prompt']) == 3                            # the input is unchanged


def test_load_group_lowers_teachers_in_student_form(tmp_path):
    student = {'task_id': 't', 'reward': {'quality': 0.0}, 'outcome': 'wrong_result',
               'turns': [{'invocation_id': 'root', 'sampling': _capture([{'kind': 'tokens', 'token_ids': [1, 2, 3]}])}]}
    (tmp_path / 'rollouts.jsonl').write_text(json.dumps(student) + '\n' + json.dumps(_teacher_rollout()) + '\n')
    c = SimpleNamespace(groups={0: {'dir': tmp_path, 'task_id': 't', 'versions': [0, 1]}}, generated=0, episodes=0,
                        args=SimpleNamespace(top_p=1.0, reward='binary'), states=None)
    stats = collections.Counter()
    _, episodes, _ = grpo.load_group(c, 0, stats)
    assert [e['role'] for e in episodes] == ['student', 'teacher']
    teacher_root = [r for r in episodes[1]['rows'] if r['invocation'] == 'root'][0]
    assert teacher_root['prompt'].tolist() == [1, 2, 3]       # the same prompt the student saw
    assert stats['teacher_calls_student_form'] == 1


def test_episode_advantages_without_teachers_are_plain_grpo():
    episodes = [{'reward': r, 'quality': r, 'role': 'student'} for r in (1., 0., 0., 1.)]
    assert grpo.episode_advantages(episodes) == pytest.approx(grpo.group_advantages([1., 0., 0., 1.]))
    assert grpo.episode_advantages([{'reward': 0., 'quality': 0.}] * 4) is None


def test_episode_advantages_train_only_teachers_with_signal():
    episodes = ([{'reward': 0., 'quality': 0., 'role': 'student'}] * 4 +
                [{'reward': 1., 'quality': 1., 'role': 'teacher'}, {'reward': 0., 'quality': 0., 'role': 'teacher'}])
    stats = collections.Counter()
    advantages = grpo.episode_advantages(episodes, stats)
    assert advantages[:4] == [None] * 4 and advantages[4] > 0 and advantages[5] is None
    assert stats['teacher_episodes'] == 2 and stats['teacher_successes'] == 1 and stats['teacher_episodes_trained'] == 1


class Custody:
    def check(self, case):
        return {'schema': 'natlang.world-custody/1', 'split': 'train', 'case_id': case['id']}


TASK = {'id': 't', 'family': 'f', 'split': 'train',
        'semantics': {'root': 'r.nl', 'files': {}, 'expected': 42},
        'curriculum': {'reference': {'root': [['eval', {'code': 'return compute()'}]]}}}


def _group(tmp_path, successes, size=4):
    directory = tmp_path / 'groups' / 'g000003'
    directory.mkdir(parents=True)
    (tmp_path / 'sampling-config.json').write_text(json.dumps({'temperature': 1}))
    (directory / 'packet.jsonl').write_text(json.dumps(TASK) + '\n')
    entries = [{'key': f's7-task-000-{s:02d}', 'index': 0, 'group': 0, 'sample': s, 'seed': 100 + s, 'count': 1,
                'task_id': 't', 'jobs': str(directory / f'row-000/sample-{s:02d}/jobs'), 'prompt_rendering': 'full',
                'rl_endpoint': 'http://actor', 'output': str(directory / f'row-000/sample-{s:02d}/results.jsonl')}
               for s in range(size)]
    (directory / 'queue.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries))
    rows = [{'task_id': 't', 'group': 0, 'sample': s, 'reward': {'quality': 1.0 if s < successes else 0.0}}
            for s in range(size)]
    (directory / 'rollouts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    c = grpo.Collector.__new__(grpo.Collector)
    c.args = SimpleNamespace(actor='http://actor', teacher_attempts=2, teacher_max_successes=0, seed_base=1000,
                             teacher_selection='crisp', teacher_fact_kinds=None, teacher_plan='crisp')
    c.work, c.lock, c.custody, c.trajectories = tmp_path, threading.Lock(), Custody(), None
    c.groups = {3: {'dir': directory, 'size': size, 'done': size, 'versions': [0] * size, 'task_id': 't'}}
    c.pending, c.ready = collections.deque(), queue.Queue()
    return c, directory


def test_an_all_fail_group_queues_teacher_attempts_first(tmp_path):
    c, directory = _group(tmp_path, successes=0)
    c.pending.append((9, {'key': 'other'}))
    assert c._teachers(3) == 2
    assert c.groups[3]['size'] == 6 and c.groups[3]['teachers'] == 2
    assert [item[0] for item in c.pending] == [3, 3, 9]
    entries = [json.loads(line) for line in (directory / 'queue.jsonl').read_text().splitlines()]
    teachers = [e for e in entries if e.get('jolt_role') == 'teacher']
    assert [e['sample'] for e in teachers] == [4, 5] and all(e['rl_endpoint'] == 'http://actor' for e in teachers)
    controls = json.loads(Path(teachers[0]['chat_request_config']).read_text())
    guidance = controls['x_natlang_guidance']
    assert controls['temperature'] == 1 and guidance['role'] == 'teacher'
    assert {fact['kind'] for fact in guidance['facts']} == {'reference_solution'}      # the answer (42) is not a fact
    assert '42' not in json.dumps(guidance['facts'])
    assert c._teachers(3) == 0                      # scored again after the teachers: the group is complete


def test_a_group_with_a_success_or_teachers_off_gets_none(tmp_path):
    c, _ = _group(tmp_path, successes=1)
    assert c._teachers(3) == 0 and c.groups[3]['size'] == 4
    c.args.teacher_attempts = 0
    c.groups[3].pop('teachers')
    assert c._teachers(3) == 0


def test_score_holds_a_group_back_while_its_teachers_run(tmp_path, monkeypatch):
    c, directory = _group(tmp_path, successes=0)
    (directory / 'workers').mkdir()
    c.own, c.rates_changed = {}, False
    from natlang_neuralese.rl import task_rollouts
    monkeypatch.setattr(task_rollouts, 'score', lambda *a, **k: None)
    c._score(3)
    assert c.ready.empty() and c.groups[3]['teachers'] == 2
    c._score(3)
    assert c.ready.get_nowait() == 3


def test_record_outcome_reads_students_only(tmp_path):
    c = grpo.Collector.__new__(grpo.Collector)
    c.own, c.rates_changed = {}, False
    (tmp_path / 'rewards.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in (
        {'task_id': 't', 'reward': 0.0}, {'task_id': 't', 'reward': 0.0},
        {'task_id': 't', 'reward': 1.0, 'jolt_role': 'teacher'})))
    c._record_outcome(tmp_path)
    assert c.own['t'] == (0, 2)


EPISODE = {'key': 'teacher:x:0', 'source': 'teacher', 'task_id': 't', 'family': 'f', 'task': TASK, 'turns': [
    {'invocation_id': 'root', 'completion_tokens': 30, 'messages': [
        {'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'Compute the answer.'},
        {'role': 'assistant', 'content': '', 'reasoning_content': 'read inputs',
         'tool_calls': [{'id': 'a', 'function': {'name': 'eval', 'arguments': json.dumps({'code': 'x = 1'})}}]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'x = 1'}],
     'target': {'role': 'assistant', 'content': '', 'tool_calls': [
         {'id': 'b', 'function': {'name': 'return_result', 'arguments': json.dumps({'value': 42})}}]}},
    {'invocation_id': 'child', 'completion_tokens': 5, 'messages': [{'role': 'user', 'content': 'child task'}],
     'target': {'role': 'assistant', 'content': 'done'}}]}


def test_trajectory_value_is_the_writer_input_shape():
    value = trajectory_value(EPISODE)
    assert value['role'] == 'root' and value['prompt'] == 'Compute the answer.'
    assert [s['tool'] for s in value['steps']] == ['eval', 'return_result']
    assert value['steps'][0]['code'] == 'x = 1' and value['steps'][0]['output'] == 'x = 1'
    assert value['steps'][0]['reasoning'] == 'read inputs'
    assert value['children'][0]['call'] == 'child' and value['children'][0]['steps'][0]['tool'] == 'reply'


def test_index_keeps_the_shortest_run_per_task_and_excludes_held(tmp_path):
    longer = {**EPISODE, 'key': 'long', 'turns': [{**EPISODE['turns'][0], 'completion_tokens': 900}]}
    other = {**EPISODE, 'task_id': 'held', 'key': 'h'}
    records = build_index([longer, EPISODE, other], excluded=frozenset({'held'}))
    assert set(records) == {'t'} and records['t']['key'] == 'teacher:x:0'
    path = tmp_path / 'index.jsonl'
    path.write_text(json.dumps(records['t']) + '\n')
    index = TrajectoryIndex(path)
    assert index.get('t')['prompt'] == 'Compute the answer.' and index.get('missing') is None and len(index) == 1


def test_a_task_with_a_verified_run_gets_a_hindsight_teacher(tmp_path):
    c, directory = _group(tmp_path, successes=0)
    c.trajectories = {'t': trajectory_value(EPISODE)}
    assert c._teachers(3) == 2
    teachers = [json.loads(line) for line in (directory / 'queue.jsonl').read_text().splitlines()][4:]
    assert all(e['hindsight'] for e in teachers)
    guidance = json.loads(Path(teachers[0]['chat_request_config']).read_text())['x_natlang_guidance']
    assert guidance['trajectory']['steps'][0]['tool'] == 'eval'


def test_engine_side_requests_accept_the_hint_writer():
    from natlang_neuralese.rl.privileged_context import _request_parts
    from natlang_neuralese.common.custody import CUSTODY_SCHEMA
    request = {'schema': 'natlang.guidance-request/1', 'invocation': {'name': 'run-1', 'role': 'root'}, 'mode': 'import',
               'custody': {'schema': CUSTODY_SCHEMA, 'split': 'train'},
               'facts': [{'id': 'reference_solution:0', 'kind': 'reference_solution', 'text': 'A verified reference solution'}],
               'selection': {'implementation': 'crisp', 'sha256': 'c'},
               'writer': {'function': 'writeHint', 'sha256': 'h'}, 'state': {'id': 'nzs1_x'}}
    assert _request_parts(request)[1] == 'import'
    with pytest.raises(ValueError):
        _request_parts({**request, 'writer': {'function': 'somethingElse', 'sha256': 'h'}})


def test_teacher_rho_summary_reads_the_student_form_ratio():
    import math
    assert grpo.teacher_rho_summary([], 2.) is None
    summary = grpo.teacher_rho_summary([np.array([math.log(.01), 0., math.log(3.)])], 2.)
    assert summary['tokens'] == 3 and summary['rho_below_0.1'] == pytest.approx(1 / 3, abs=1e-3)
    assert summary['rho_capped'] == pytest.approx(1 / 3, abs=1e-3) and summary['log_rho_p50'] == 0.
