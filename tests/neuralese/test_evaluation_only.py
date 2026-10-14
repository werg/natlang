"""Widen frozen-checkpoint evidence without relaxing training or input identity."""
from copy import deepcopy
import sys
import json

import pytest

from natlang_neuralese.train.trajectory_state import validate_continuation


def saved_run():
    identity = {'options': {'steps': 400, 'eval': 32, 'stream_held_chains': 4,
                            'latent_lr': 3e-6, 'seed': 20261011},
                'files': {'records': 'immutable-records', 'pieces': 'immutable-pieces'}}
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'step': 400, 'identity': identity}
    widened = deepcopy(identity)
    widened['options'].update(eval=64, stream_held_chains=12)
    return state, widened


def test_only_explicit_frozen_evaluation_can_widen_its_sample():
    state, widened = saved_run()
    before = deepcopy(widened)
    with pytest.raises(ValueError, match='undeclared'):
        validate_continuation(state, widened)
    validate_continuation(state, widened, evaluation_only=True)
    assert widened == before


@pytest.mark.parametrize('change', [{'latent_lr': 3e-5}, {'seed': 42}, {'steps': 401}, {'steps': 399}])
def test_evaluation_only_never_changes_training_controls_or_stage_end(change):
    state, widened = saved_run()
    widened['options'].update(change)
    with pytest.raises(ValueError):
        validate_continuation(state, widened, evaluation_only=True)


def test_evaluation_only_rejects_changed_inputs_and_curriculum_changes():
    state, widened = saved_run()
    with pytest.raises(ValueError, match='changed inputs'):
        validate_continuation(state, {**widened, 'files': {'records': 'other'}}, evaluation_only=True)
    with pytest.raises(ValueError, match='cannot declare curriculum'):
        validate_continuation(state, widened, allowed_changes=['latent_lr'], evaluation_only=True)


@pytest.mark.parametrize('source', [None, 'same-output'])
def test_eval_only_cli_requires_a_separate_parent_before_loading_inputs(tmp_path, monkeypatch, source):
    from natlang_neuralese.train import trajectories
    args = ['trajectories', '--eval-only', '--device', 'cpu', '--out', str(tmp_path / 'output'),
            '--records', str(tmp_path / 'absent-records'), '--pieces', str(tmp_path / 'absent-pieces')]
    if source:
        args += ['--continue-from', str(tmp_path / 'output' / 'checkpoint.pt')]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(ValueError, match='distinct output checkpoint path'):
        trajectories.main()
    assert not (tmp_path / 'output').exists()


def test_config_inspection_reports_direct_writer_normalization_without_loading_inputs(tmp_path, monkeypatch, capsys):
    from natlang_neuralese.train import trajectories
    monkeypatch.setattr(sys, 'argv', ['trajectories', '--inspect-training-config', '--device', 'cuda',
        '--records', str(tmp_path / 'absent-records'), '--pieces', str(tmp_path / 'absent-pieces'),
        '--out', str(tmp_path / 'output'), '--direct-artifact', str(tmp_path / 'absent-export'),
        '--direct-base-snapshot', str(tmp_path / 'absent-base'), '--write-depth', '1',
        '--backbone-training', 'latent', '--backward-policy', 'joint'])
    trajectories.main()
    config = json.loads(capsys.readouterr().out)
    assert config['content_transport'] == 'raw-identity'
    assert config['projection_anchor_weight'] == config['sketch_target_weight'] == 0.0
    assert not (tmp_path / 'output').exists()
