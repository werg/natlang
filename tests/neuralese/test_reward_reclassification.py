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
