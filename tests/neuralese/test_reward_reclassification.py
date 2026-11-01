import copy
import json

import pytest

from natlang_neuralese.rl.reward_reclassification import projected_rows


def fixture(tmp_path):
    result = tmp_path / 'result.jsonl'
    checks = dict(expected_status=True, honest_stop=True, stop_evidence=True, answer=True)
    result.write_text(json.dumps({'outcome': dict(status='quiesced', accepted=True,
        oracle={'accepted': True}, checks=checks)}) + '\n')
    (tmp_path / 'queue.jsonl').write_text(json.dumps(dict(group=0, sample=0, key='a',
        output=str(result))) + '\n')
    (tmp_path / 'journal.jsonl').write_text(json.dumps(dict(event='finish', key='a',
        status='complete')) + '\n')
    (tmp_path / 'collection-binding.json').write_text('{}')
    row = dict(group=0, sample=0, task_id='blocked', outcome='invalid_action',
        reward={'quality': 0}, turns=[{'sampling': {'samples': [1, 2]}}],
        policy={'checkpoint_sha256': 'a' * 64}, unscored=None)
    return row, result


def test_verified_stop_changes_only_derived_reward(tmp_path):
    row, _ = fixture(tmp_path)
    original = copy.deepcopy(row)
    rows, changes = projected_rows(tmp_path, [row])
    expected = {**row, 'outcome': 'task_success', 'reward': {'quality': 1.}}
    assert rows == [expected]
    assert row == original
    assert len(changes) == 1
    assert 'execution_efficiency' not in rows[0]['reward']


def test_unverified_classification_change_cannot_be_repaired(tmp_path):
    row, result = fixture(tmp_path)
    result.write_text(json.dumps({'outcome': {'status': 'done', 'accepted': True}}) + '\n')
    with pytest.raises(ValueError, match='outside the verified'):
        projected_rows(tmp_path, [row])


def test_duplicate_sample_identity_is_rejected(tmp_path):
    row, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='sample identities'):
        projected_rows(tmp_path, [row, row])


def babi_result():
    return {'task': {'program_ir': {'source': 'external-rl:babi', 'family': 'ext_babi_task01',
            'source_ids': ['facebook/babi_qa:train-1-1069:971fc517'],
            'semantics': {'oracle': 'normalized', 'expected': 'garden'}}},
            'outcome': {'status': 'done', 'accepted': False, 'value': 'the garden',
                        'rejection_reasons': ['answer'],
                        'checks': {'answer': False, 'expected_status': True, 'honest_stop': True, 'effects': True}}}


def test_babi_article_repair_changes_only_derived_reward(tmp_path):
    row, result = fixture(tmp_path)
    row['outcome'] = 'semantic_mismatch'
    raw = babi_result()
    result.write_text(json.dumps(raw) + '\n')
    rows, changes = projected_rows(tmp_path, [row])
    assert rows[0]['reward']['quality'] == 1
    assert rows[0]['outcome'] == 'task_success'
    assert len(changes) == 1
    assert json.loads(result.read_text()) == raw


@pytest.mark.parametrize('mutation', ['wrong_answer', 'effect', 'other_source', 'different_oracle', 'timeout', 'nonstring'])
def test_babi_article_repair_never_waives_other_contracts(mutation):
    from natlang_neuralese.eval.task_packet import accepted_babi_article_answer
    raw = babi_result()
    finish = {'status': 'complete'}
    if mutation == 'wrong_answer': raw['outcome']['value'] = 'the kitchen'
    if mutation == 'effect': raw['outcome']['checks']['effects'] = False
    if mutation == 'other_source': raw['task']['program_ir']['source'] = 'protected-other-task'
    if mutation == 'different_oracle': raw['task']['program_ir']['semantics']['oracle'] = 'exact'
    if mutation == 'timeout': finish['resource_limit_reason'] = 'case_wall_budget'
    if mutation == 'nonstring': raw['outcome']['value'] = ['garden']
    assert not accepted_babi_article_answer(raw, finish)


def test_reclassified_report_recomputes_quality_and_efficiency_routes():
    from natlang_neuralese.rl.reward_reclassification import projected_report
    original = {'groups': [{'task_id': 'babi'}], 'groups_with_spread': 1,
                'groups_with_quality_spread': 1, 'groups_with_efficiency_spread': 0}
    rows = [dict(task_id='babi', sample=i, outcome='task_success',
                 reward={'quality': 1, 'execution_efficiency': {'cost': cost}}, turns=[])
            for i, cost in enumerate([0.01, 0.02])]
    actual = projected_report(original, rows, '/corrected/rollouts.jsonl')
    assert actual['groups_with_quality_spread'] == 0
    assert actual['groups_with_efficiency_spread'] == 1
    assert actual['groups'][0]['route'] == 'grpo'
    assert original['groups_with_quality_spread'] == 1


def preparation_fixture(tmp_path):
    import hashlib
    out = tmp_path / 'round-000000'
    learner = out / 'learner'; learner.mkdir(parents=True)
    def write(path, data):
        path.write_text(json.dumps(data));raw=path.read_bytes()
        return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    parent = write(tmp_path / 'retained-parent.pt', {'full_parent_fixture': True})
    rollouts = write(out / 'rollouts.jsonl', {'capture': 'unchanged'})
    graph = write(learner / 'rl-native-graphs.jsonl', {'graph': 'whole'})
    admission = write(learner / 'rl-data-receipt.json', {'schema':'natlang.native-rl-admission/1',
        'policy':{'checkpoint_sha256':parent['sha256']},'objective_sha256':'d'*64,
        'native_graphs_sha256':graph['sha256'],'rollouts':{rollouts['path']:rollouts['sha256']}})
    capacity = write(learner / 'rl-policy-capacity-request.json', {'schema':'natlang.native-policy-capacity-request/1',
        'source':{'path':graph['path'],'sha256':graph['sha256']},'objective':{'sha256':'d'*64}})
    evidence = {'schema':'natlang.unearned-rl-preparation/1','out':str(out),'parent':parent,
                'admission':admission,'capacity_request':capacity}
    reference = write(out / 'unearned-preparation.json', evidence)
    return reference, {'out':out,'policy_id':parent['sha256']}, learner, write, evidence


def test_actual_unearned_preparation_keeps_exact_parent(tmp_path):
    from natlang_neuralese.rl.reward_reclassification import verify_unearned_preparation
    reference, request, learner, _, expected = preparation_fixture(tmp_path)
    assert verify_unearned_preparation(reference, request) == expected
    assert not (learner / 'report.json').exists()


@pytest.mark.parametrize('mutation', ['checkpoint', 'report', 'different_parent', 'changed_rollouts', 'different_graph', 'outside_admission'])
def test_unearned_preparation_rejects_other_state_and_bindings(tmp_path, mutation):
    from natlang_neuralese.rl.reward_reclassification import verify_unearned_preparation
    reference, request, learner, write, evidence = preparation_fixture(tmp_path)
    if mutation == 'checkpoint': write(learner / 'checkpoint.pt', {'earned':'preserve'})
    if mutation == 'report': write(learner / 'report.json', {'step':1})
    if mutation == 'different_parent': request['policy_id'] = 'b'*64
    if mutation == 'changed_rollouts': write(request['out'] / 'rollouts.jsonl', {'capture':'different'})
    if mutation == 'different_graph': write(learner / 'rl-native-graphs.jsonl', {'graph':'different'})
    if mutation == 'outside_admission':
        evidence['admission'] = write(tmp_path / 'foreign-admission.json', {})
        reference = write(request['out'] / 'unearned-preparation.json', evidence)
    with pytest.raises(ValueError): verify_unearned_preparation(reference, request)
