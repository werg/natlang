"""Widen frozen-checkpoint evidence without relaxing training or input identity."""
from copy import deepcopy

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
