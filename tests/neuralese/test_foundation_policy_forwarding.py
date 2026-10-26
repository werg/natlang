"""Bare local cycles must not revert to the trainer's synchronous default."""
import json

import pytest

from natlang_neuralese.rl import native_supervised as supervised
from natlang_neuralese.rl.native_raw_rounds import _seed_quantum
from natlang_neuralese.train.native_resident import binding
from natlang_neuralese.train.recipe import stage_parameter_args


@pytest.mark.parametrize('kind', ['native_state_sft', 'native_offline_preference'])
@pytest.mark.parametrize('stage_policy', [None, 'at-boundary'])
def test_actual_continuation_command_inherits_controller_policy(tmp_path, kind, stage_policy):
    inputs = {}
    for role in ('native_graphs', 'held_native_graphs', 'objective_spec'):
        path = tmp_path / role
        path.write_text('immutable source')
        if role == 'objective_spec':
            from natlang_neuralese.train.native_preference import SCHEMA, KIND
            path.write_text(json.dumps({'schema': SCHEMA, 'kind': KIND,
                                       'beta': .1, 'margin': 0., 'rows': {'pair': {}}}))
        inputs[role] = binding(path)
    parameters = {'epochs': 1}
    if stage_policy is not None:
        parameters['foundation_readiness_policy'] = stage_policy
    stage = {'id': 'actual-stage', 'kind': kind, 'requires': [],
             'inputs': {key: key for key in inputs if key != 'objective_spec'
                        or kind == 'native_offline_preference'}, 'parameters': parameters}
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps({'schema': 'natlang.neuralese-training-recipe/1',
        'id': 'policy-forwarding', 'input_bindings': inputs, 'stages': [stage]}))
    config = {'production_recipe': str(recipe), 'recipe_lineage_sha256': 'a' * 64,
              'foundation_readiness_policy': 'on-shared-request',
              'seed_command': ['python', '-m', 'natlang_neuralese.train.native_state_sft',
                               '{trainer_args}']}
    declared, _ = supervised.continuation_config(config, stage, checkpoint='earned-parent',
                                                adapter=None, out=tmp_path / 'operation')
    arguments = stage_parameter_args(declared['trainer_parameters'])
    index = arguments.index('--foundation-readiness-policy')
    assert arguments[index + 1] == (stage_policy or 'on-shared-request')
    assert declared['trainer_parameters']['recipe_binding']['lineage_sha256'] == 'a' * 64
    assert parameters.get('foundation_readiness_policy') == stage_policy


def test_raw_seed_retains_quantum_and_real_command_policy(tmp_path):
    config = {'production_stage': 'native_full_sft',
              'foundation_readiness_policy': 'on-shared-request',
              'supervised_quantum_updates': 50,
              'supervised_stage_quantum_updates': {'native_full_sft': 100},
              'seed_command': ['ordinary-seed', '{trainer_args}'],
              'trainer_parameters': {'epochs': 1}}
    _seed_quantum(config, {})
    selected = config['seed_trainer_parameters']
    assert selected['refresh_after_updates'] == 100
    assert selected['foundation_readiness_policy'] == 'on-shared-request'
    values = {}
    supervised.prepare_seed_command(config, tmp_path, values, {}, tmp_path / 'state.json',
                                    parameters=selected)
    arguments = values['seed_trainer_args']
    assert arguments[arguments.index('--foundation-readiness-policy') + 1] == 'on-shared-request'
    assert arguments[arguments.index('--refresh-after-updates') + 1] == '100'


def test_direct_fallback_inherits_policy_without_shared_role_plan(tmp_path):
    config = {'foundation_readiness_policy': 'on-shared-request',
              'seed_command': ['ordinary-seed', '{trainer_args}'], 'trainer_parameters': {}}
    values = {}
    supervised.prepare_seed_command(config, tmp_path, values, {}, tmp_path / 'state.json')
    assert values['trainer_args'] == ['--foundation-readiness-policy', 'on-shared-request']
    config['foundation_readiness_policy'] = 'invented'
    with pytest.raises(ValueError, match='unknown foundation'):
        supervised.prepare_seed_command(config, tmp_path, {}, {}, tmp_path / 'state.json')
