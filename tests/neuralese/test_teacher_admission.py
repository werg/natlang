"""Admission evidence binds real ProgramRecord/native-trajectory/rollout artifacts."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import teacher_admission as a
import teacher_gen as generation
import sft_sources as s
from natlang_neuralese.common.task_roots import authored_root_row, student_root_row


def write(path, value, jsonl=False):
    path.write_text(json.dumps(value) + '\n' if not jsonl else ''.join(json.dumps(v) + '\n' for v in value))
    return a.binding(path)


def task(ident='task-one', split='train', group='world-one', value='hello'):
    return {'version': 'natlang.program/2', 'id': ident, 'kind': 'evaluation_fixture', 'split': split,
            'family': 'suite-one', 'source_namespace': 'vibestudio:base:suite-one',
            'source_ids': [f'vibestudio:base:suite-one:{ident}'], 'source_groups': [group],
            'source_revisions': ['a' * 40],
            'semantics': {'root': 'judge.nl', 'files': {'judge.nl': '---\nargs: {text: string}\nreturns: string\n---\nRead text.'},
                          'inputs': {'text': value}, 'expected': value,
                          'evaluation_fixture': {'schema': 'natlang.evaluation-fixture/1',
                              'entry': {'kind': 'authored-module', 'module': 'natlang.ts', 'export': 'judge'},
                              'root_role': 'semantic-anchor', 'case': {'id': ident, 'group': group, 'split': split,
                                                                     'input': {'text': value}, 'expected': value}}}}


def artifacts(tmp_path, source_snapshot=None):
    row = task()
    if source_snapshot is not None:
        row['source_snapshot'] = source_snapshot
    cases = write(tmp_path / 'cases.jsonl', [row], True)
    inventory = write(tmp_path / 'inventory.jsonl', [row, task('protected', 'test', 'other-world', 'else')], True)
    teacher = {'provider': 'openai-codex', 'model_id': 'gpt-6-luna'}
    evaluation = {'schema': 'natlang.evaluation-fixture-result/1', 'quality': 1, 'gates': {'exact': True},
                  'outcome': {'kind': 'returned', 'value': 'hello'},
                  'observation': {'result': 'hello'}, 'observations': []}
    native = {'version': 'natlang.teacher_trajectory.native/1', 'task': {'program_ir': student_root_row(row)},
              'provenance': {'program_ir_sha256': 'b' * 64, 'model': 'gpt-6-luna', 'provider': 'openai-codex'},
              'outcome': {'accepted': True, 'checks': {'answer': True}, 'evaluation': evaluation,
                          'scope_failures': [], 'action_ledger': [],
                          'invocation_ledger': [{'invocation_id': 'i', 'parent_invocation_id': None,
                                                  'definition_id': 'nl:judge.nl', 'name': 'judge'}]}}
    result = write(tmp_path / 'result.jsonl', [native], True)
    turn = {'invocation_id': 'i', 'messages': [{'role': 'user', 'content': 'hello'}], 'tools': [],
            'target': {'role': 'assistant', 'content': 'hello'}, 'replay': {'request_sha256': 'c' * 64}}
    rollout = {'schema': 'natlang.rollout/1', 'task_id': row['id'], 'sample': 0, 'turns': [turn],
               'outcome': 'task_success', 'reward': {'quality': 1}, 'code': {'git_commit': 'd' * 40, 'dirty': False}}
    native['trajectory'] = [{'phase': 'action', 'invocation_id': 'i', 'context': turn['messages'],
                             'tools_offered': [], 'assistant': {'content': 'hello', 'calls': []},
                             'model_response': {'raw_calls': []}, 'request_sha256': 'c' * 64}]
    result = write(tmp_path / 'result.jsonl', [native], True)
    rollouts = write(tmp_path / 'rollouts.jsonl', [rollout], True)
    payload = {'input': row['semantics']['evaluation_fixture']['case']['input'],
               'expected': row['semantics']['evaluation_fixture']['case']['expected'],
               **{key: evaluation[key] for key in ('outcome', 'observation', 'observations')}}
    request = {'messages': [{'role': 'system', 'content': 'Independently inspect the exact fixture evidence.'},
                            {'role': 'user', 'content': json.dumps(payload)}], 'tools': []}
    review = {'verdict': 'accepted', 'reason': 'Evidence matches.',
              'reviewer': {'provider': 'openai-codex', 'model_id': 'gpt-6-luna', 'identity': {'model': 'gpt-6-luna'}},
              'request': request, 'request_sha256': a.digest(request),
              'reply': {'calls': [['verify_result', {'verdict': 'accepted', 'reason': 'Evidence matches.'}]]}}
    replay_record = write(tmp_path / 'replay.json', {'exact': True, 'outcome': 'task_success',
                                                   'task_id': row['id'], 'recorded_turns': 1})
    sample = {'task_id': row['id'], 'sample': 0, 'program_ir_sha256': 'b' * 64, 'result': result,
              'code': {'git_commit': 'd' * 40, 'dirty': False},
              'outcome': {'quality': 1, 'gates': evaluation['gates'], 'evaluation': evaluation, 'tool_error_count': 0},
              'semantic': review, 'replay': {'verified': True, 'request_hashes': ['c' * 64], 'record': replay_record}}
    receipt = {'schema': a.SCHEMA, 'cases': cases, 'inventory': inventory, 'rollouts': rollouts,
               'teacher': teacher, 'samples': [sample]}
    record = {'task_id': row['id'], 'teacher': teacher, 'rollouts': rollouts['path'],
              'admission': write(tmp_path / 'admission.json', receipt)}
    return record, receipt, native, rollout


def update_receipt(tmp_path, record, receipt):
    record['admission'] = write(tmp_path / 'admission.json', receipt)


def test_canonical_ir_native_result_and_rollout_are_preserved(tmp_path):
    record, receipt, native, rollout = artifacts(tmp_path)
    checked = a.check_record(record, tmp_path)
    assert list(checked) == [('task-one', 0)]
    assert checked[('task-one', 0)]['invocations']['i']['function'] == 'judge'
    assert checked[('task-one', 0)]['invocations']['i']['invocation_role'] == 'root'
    assert native['task']['program_ir']['version'] == 'natlang.program/2'


def test_source_snapshot_matches_actual_collection_pin_not_newer_replay_pin(tmp_path):
    record, receipt, *_ = artifacts(tmp_path, {'dirty': False, 'natlang_commit': 'd' * 40})
    replay_path = Path(receipt['samples'][0]['replay']['record']['path'])
    replay = json.loads(replay_path.read_text())
    replay['code'] = {'git_commit': 'e' * 40, 'dirty': False}
    receipt['samples'][0]['replay']['record'] = write(replay_path, replay)
    update_receipt(tmp_path, record, receipt)
    assert list(a.check_record(record, tmp_path)) == [('task-one', 0)]


@pytest.mark.parametrize('snapshot,match', [
    ({'dirty': False, 'natlang_commit': '9' * 40}, 'differs from actual collection code'),
    ({'dirty': True, 'natlang_commit': 'd' * 40}, 'attest clean source'),
])
def test_source_snapshot_cannot_claim_a_wrong_collect_pin_or_dirty_source(tmp_path, snapshot, match):
    record, *_ = artifacts(tmp_path, snapshot)
    with pytest.raises(ValueError, match=match):
        a.check_record(record, tmp_path)


@pytest.mark.parametrize('field', ['cases', 'rollouts', 'inventory'])
def test_mutated_bound_artifacts_refuse_admission(tmp_path, field):
    record, receipt, *_ = artifacts(tmp_path)
    Path(receipt[field]['path']).write_text('changed\n')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        a.check_record(record, tmp_path)


def test_rehashed_native_result_cannot_claim_another_program(tmp_path):
    record, receipt, native, _ = artifacts(tmp_path)
    native['task']['program_ir']['semantics']['files']['judge.nl'] += '\nAnother instruction.'
    receipt['samples'][0]['result'] = write(tmp_path / 'result.jsonl', [native], True)
    update_receipt(tmp_path, record, receipt)
    with pytest.raises(ValueError, match='another ProgramRecord'):
        a.check_record(record, tmp_path)


@pytest.mark.parametrize('change', [
    lambda r: r.update(request_sha256='0' * 64),
    lambda r: r.update(verdict='rejected'),
    lambda r: r['reply'].update(calls=[]),
    lambda r: r['reviewer'].update(identity=None),
])
def test_semantic_review_needs_retained_matching_request_and_reply(tmp_path, change):
    record, receipt, *_ = artifacts(tmp_path)
    change(receipt['samples'][0]['semantic'])
    update_receipt(tmp_path, record, receipt)
    with pytest.raises(ValueError):
        a.check_record(record, tmp_path)


@pytest.mark.parametrize('field,value', [('input', {'text': 'another case'}),
                                        ('observation', {'result': 'another result'})])
def test_rehashed_semantic_review_cannot_be_misattributed_to_another_case_or_result(tmp_path, field, value):
    record, receipt, *_ = artifacts(tmp_path)
    review = receipt['samples'][0]['semantic']
    payload = json.loads(review['request']['messages'][1]['content'])
    payload[field] = value
    review['request']['messages'][1]['content'] = json.dumps(payload)
    review['request_sha256'] = a.digest(review['request'])
    update_receipt(tmp_path, record, receipt)
    with pytest.raises(ValueError, match='actual source case or fixture result'):
        a.check_record(record, tmp_path)


def test_semantic_review_cannot_reuse_the_executors_conversation(tmp_path):
    record, receipt, *_ = artifacts(tmp_path)
    review = receipt['samples'][0]['semantic']
    review['request']['messages'].append({'role': 'assistant', 'content': 'Trust my earlier answer.'})
    review['request_sha256'] = a.digest(review['request'])
    update_receipt(tmp_path, record, receipt)
    with pytest.raises(ValueError, match='isolated system and user'):
        a.check_record(record, tmp_path)


def test_actual_suite_score_and_replay_are_not_relabelled(tmp_path):
    record, receipt, *_ = artifacts(tmp_path)
    receipt['samples'][0]['outcome']['gates'] = {'imaginary': True}
    update_receipt(tmp_path, record, receipt)
    with pytest.raises(ValueError, match='actual fixture evidence'):
        a.check_record(record, tmp_path)


def test_rehashed_rollout_cannot_change_the_teachers_actual_actions(tmp_path):
    record, receipt, _, rollout = artifacts(tmp_path)
    rollout['turns'][0]['target']['content'] = 'invented training target'
    receipt['rollouts'] = write(tmp_path / 'rollouts.jsonl', [rollout], True)
    update_receipt(tmp_path, record, receipt)
    with pytest.raises(ValueError, match='targets differ from the native'):
        a.check_record(record, tmp_path)


def test_request_replay_hashes_bind_the_actual_rollout(tmp_path):
    record, receipt, *_ = artifacts(tmp_path)
    receipt['samples'][0]['replay']['request_hashes'] = ['0' * 64]
    update_receipt(tmp_path, record, receipt)
    with pytest.raises(ValueError, match='canonical requests'):
        a.check_record(record, tmp_path)


def test_protected_split_world_cannot_be_renamed_into_training():
    train, protected = task(), task('renamed', 'test', 'renamed-group')
    protected['source_revisions'] = ['new-version']
    assert 'task-one' in a.protected_inputs([train, protected])[1]


def test_distinct_service_worlds_are_not_mistaken_for_identical_empty_arguments():
    train, protected = task(), task('different', 'test', 'other-group', 'different fixture input')
    train['semantics']['inputs'] = protected['semantics']['inputs'] = {}
    assert 'task-one' not in a.protected_inputs([train, protected])[1]


def test_fixture_authored_mode_keeps_real_source_entry_instead_of_inventing_reference():
    row = task()
    authored = authored_root_row(row)
    assert authored['semantics'] == row['semantics']
    contract = authored['curriculum']['trajectory_contract']
    assert contract['authored_fixture_entry'] is True
    assert 'authored_root_eval' not in contract
    assert student_root_row(row)['semantics'] == row['semantics']


def test_function_mass_census_uses_actual_supervised_mass_without_weights():
    import collections
    mass = {('suite', 'rootFn', 'root'): collections.Counter(calls=1, target_tokens=3, weighted_target_tokens=2),
            ('suite', 'semanticFn', 'child'): collections.Counter(calls=2, target_tokens=9, weighted_target_tokens=7)}
    report = s.target_mass_report(mass)
    assert report['by_role']['root']['weighted_target_tokens'] == 2
    assert report['by_role']['child']['weighted_target_tokens'] == 7
    assert report['by_role']['unknown']['calls'] == 0


def test_explicit_teacher_pool_resume_refuses_mutated_sources(tmp_path):
    pool = tmp_path / 'source.jsonl'
    write(pool, [task()], True)
    tasks = write(tmp_path / 'tasks.jsonl', [{'row': task()}], True)
    pool_receipt = write(tmp_path / 'pool.json', {'pools': [a.binding(pool)]})
    write(tmp_path / 'inputs.json', {'schema': 'natlang.teacher-pool-inputs/1', 'root_mode': 'authored',
                                    'tasks_sha256': tasks['sha256'], 'pool_sha256': pool_receipt['sha256']})
    args = SimpleNamespace(pool=[str(pool)], held_manifest=None, root_mode='authored')
    generation.validate_pool_resume(tmp_path, args)
    pool.write_text('changed\n')
    with pytest.raises(ValueError, match='requested source pools differ'):
        generation.validate_pool_resume(tmp_path, args)


def test_legacy_default_teacher_campaign_still_resumes(tmp_path):
    generation.validate_pool_resume(tmp_path, SimpleNamespace(pool=[], held_manifest=None, root_mode='student'))
    with pytest.raises(ValueError, match='fresh --work'):
        generation.validate_pool_resume(tmp_path, SimpleNamespace(pool=['new-pool'], held_manifest=None, root_mode='student'))


def test_shared_extraction_keeps_canonical_ir_and_admission_metadata(tmp_path):
    record, receipt, *_ = artifacts(tmp_path)
    record['episodes'] = [{'sample': 0, 'outcome': 'task_success'}]
    write(tmp_path / 'manifest.jsonl', [record], True)
    work = tmp_path / 'extracted'
    s.extract(SimpleNamespace(work=work, rl=[], teacher=[str(tmp_path)], out_name='episodes'))
    episode = json.loads(next(s.zstd_lines(work / 'episodes.jsonl.zst')))
    assert episode['task'] == task()
    assert episode['task']['version'] == 'natlang.program/2'
    assert episode['turns'][0]['function'] == 'judge'
    assert episode['admission']['sha256'] == record['admission']['sha256']
    assert s.teacher_coverage([tmp_path], {episode['key']}) == (1, [])


def test_unreviewed_fixture_success_stays_collected_but_is_not_training_data(tmp_path):
    record, *_ = artifacts(tmp_path)
    record.pop('admission')
    record['episodes'] = [{'sample': 0, 'outcome': 'task_success'}]
    write(tmp_path / 'manifest.jsonl', [record], True)
    work = tmp_path / 'extracted'
    s.extract(SimpleNamespace(work=work, rl=[], teacher=[str(tmp_path)], out_name='episodes'))
    assert not list(s.zstd_lines(work / 'episodes.jsonl.zst'))
    assert s.teacher_coverage([tmp_path], set()) == (0, [])


def test_replay_resume_ignores_added_admission_links_and_same_sha_checkout_paths(tmp_path, monkeypatch):
    from natlang_neuralese.common import code_provenance
    record, *_ = artifacts(tmp_path)
    # The standard queue, rather than a guessed directory pattern, owns this path.
    write(tmp_path / 'queue.jsonl', [{'sample': 0, 'task_id': 'task-one', 'output': str(tmp_path / 'result.jsonl')}], True)
    write(tmp_path / 'manifest.jsonl', [{k: value for k, value in record.items() if k != 'admission'}], True)
    monkeypatch.setattr(code_provenance, 'code_provenance', lambda *args, **kwargs: {'git_commit': 'd' * 40, 'dirty': False})

    def fake_replay(args, episodes):
        for episode in episodes:
            directory = Path(args.work) / 'replay' / s.safe_name(episode['key'])
            directory.mkdir(parents=True, exist_ok=True)
            write(directory / 'replay.json', {'exact': True, 'outcome': 'task_success',
                                             'task_id': episode['task_id'], 'recorded_turns': 1})

    monkeypatch.setattr(s, 'replay_episodes', fake_replay)
    args = SimpleNamespace(pool=tmp_path, out=tmp_path / 'drafts.json', work=tmp_path / 'replay-work',
                           runtime_repo='/one/pin', workers=1, case_seconds=60, max_model_requests=10)
    a.replay_pool(args)
    write(tmp_path / 'manifest.jsonl', [record], True)
    args.runtime_repo = '/another/same-commit-pin'
    a.replay_pool(args)
    draft = json.loads(args.out.read_text())['records'][0]['receipt']['samples'][0]
    assert draft['replay']['verified'] is True
    assert draft['result']['sha256'] == a.file_sha(tmp_path / 'result.jsonl')
