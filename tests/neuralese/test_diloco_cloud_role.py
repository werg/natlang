import json
import pytest
from natlang_neuralese.train.diloco import cloud_role
from natlang_neuralese.train.native_diloco import trainer_argv


def test_learner_argv_identity_is_common_and_resume_rebinds_only_parent(tmp_path):
    trainer = ['--native-graphs', 'a.jsonl', '--continue-from', '/p/parent.pt', '--out', '/work/learner']
    settings = json.dumps({'run_id': 'r', 'workers': 2, 'local_steps': 2})
    argv0 = cloud_role.learner_argv(settings, 0, 'https://c', trainer)
    argv1 = cloud_role.learner_argv(settings, 1, 'https://c', trainer)
    # The coordinator identity binds the trainer argv without --diloco: identical across workers.
    assert trainer_argv(argv0) == trainer_argv(argv1) == trainer
    block = json.loads(argv1[argv1.index('--diloco') + 1])
    assert block['worker_id'] == 1 and block['coordinator_url'] == 'https://c'
    resumed = cloud_role.resume_argv(argv1, '/work/learner/checkpoint.pt')
    assert [x for x in resumed if x != '/work/learner/checkpoint.pt'] == [x for x in argv1 if x != '/p/parent.pt']
    with pytest.raises(SystemExit):
        cloud_role.learner_argv(settings, 0, 'u', [*trainer, '--diloco', '{}'])


def test_out_link_points_into_stage_and_refuses_foreign_paths(tmp_path):
    link, target = tmp_path / 'work' / 'learner', tmp_path / 'stage' / 'w0'
    cloud_role.link_out(link, target)
    cloud_role.link_out(link, target)  # idempotent on restart
    assert link.resolve() == target.resolve()
    with pytest.raises(SystemExit):
        cloud_role.link_out(link, tmp_path / 'stage' / 'w1')
    (tmp_path / 'plain').mkdir()
    with pytest.raises(SystemExit):
        cloud_role.link_out(tmp_path / 'plain', target)
