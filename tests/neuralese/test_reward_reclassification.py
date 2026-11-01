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
