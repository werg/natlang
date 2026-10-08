import json
from pathlib import Path

import pytest

from natlang_neuralese.train.recipe import load_recipe, require_gate, stage_parameter_args


def declared():
    return json.loads((Path(__file__).parents[2] / 'training/neuralese/recipes/foundation-v1.json').read_text())


def test_shared_recipe_declares_warmup_and_identity_dependency(tmp_path):
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(declared()))
    recipe = load_recipe(path)
    assert recipe['stages'][1]['kind'] == 'causal_embedding_distillation'
    assert recipe['stages'][1]['requires'] == ['token_identity']


@pytest.mark.parametrize('change', ['missing_identity', 'forward_dependency', 'unknown_parameter', 'unsafe_id', 'unknown_kind'])
def test_invalid_recipe_is_rejected_before_launch(tmp_path, change):
    recipe = declared()
    if change == 'missing_identity':
        recipe['stages'][1]['requires'] = []
    elif change == 'forward_dependency':
        recipe['stages'][0]['requires'] = ['embedding_distillation']
    elif change == 'unknown_parameter':
        recipe['stages'][1]['parameters']['shell_command'] = 'unused'
    elif change == 'unsafe_id':
        recipe['stages'][0]['id'] = '../escape'
    else:
        recipe['stages'][0]['kind'] = 'arbitrary_module'
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError):
        load_recipe(path)


def test_finishing_steps_or_partial_agreement_does_not_pass_gate():
    with pytest.raises(ValueError, match='distillation gate failed'):
        require_gate({'step': 2048, 'agreement': .62, 'feedback_gate_passed': False}, 'causal_embedding_distillation')
    with pytest.raises(ValueError):
        require_gate({}, 'token_identity')
    require_gate({'feedback_gate_passed': True}, 'causal_embedding_distillation')


def test_runtime_stage_requires_the_embedding_gate(tmp_path):
    recipe = declared()
    assert recipe['stages'][2]['kind'] == 'raw_runtime_qualification'
    recipe['stages'][2]['requires'] = ['token_identity']
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='embedding foundation'):
        load_recipe(path)
    with pytest.raises(ValueError, match='runtime transport gate failed'):
        require_gate({}, 'raw_runtime_qualification')


def test_foundation_certificate_survives_verified_directory_relocation(tmp_path):
    from natlang_neuralese.train.recipe import require_foundation
    from natlang_neuralese.train.output_embedding_projection import sha
    heads, checkpoint = tmp_path / 'heads.pt', tmp_path / 'feedback.pt'
    heads.write_bytes(b'head')
    checkpoint.write_bytes(b'feedback')
    stages = []
    for name, kind, gate in [('identity', 'token_identity', {'token_aligned_reference_passed': True}),
                             ('embedding', 'causal_embedding_distillation', {'feedback_gate_passed': True})]:
        report = tmp_path / (name + '-report.json')
        report.write_text(json.dumps({'kind': kind, 'gate': gate}))
        stages.append({'kind': kind, 'report': '/old/machine/' + report.name, 'report_sha256': sha(report)})
    certificate = tmp_path / 'foundation-certificate.json'
    certificate.write_text(json.dumps({'schema': 'natlang.neuralese-foundation-certificate/1', 'qualified': True,
                                       'heads_sha256': sha(heads), 'feedback_checkpoint_sha256': sha(checkpoint),
                                       'stages': stages}))
    assert require_foundation(certificate, heads=heads, checkpoint=checkpoint)['qualified']
    (tmp_path / 'identity-report.json').write_text('{}')
    with pytest.raises(ValueError, match='report changed'):
        require_foundation(certificate, heads=heads, checkpoint=checkpoint)


def test_recipe_declares_runtime_without_hardcoded_order_guards_and_completion_is_not_admission(tmp_path):
    recipe = json.loads((Path(__file__).parents[2] / 'training/neuralese/recipes/raw-recurrence-v1.json').read_text())
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    assert load_recipe(path)['stages'][3]['requires'] == ['runtime_qualification']
    recipe['stages'][3]['requires'] = ['embedding_distillation']
    path.write_text(json.dumps(recipe))
    # The shared declared recipe carries the policy; loader checks graph shape,
    # not a second hardcoded implementation of the training order.
    assert load_recipe(path)['stages'][3]['requires'] == ['embedding_distillation']
    with pytest.raises(ValueError):
        require_gate({'training_stage_completed': True, 'errors': 1}, 'raw_recurrence_training')
    require_gate({'training_stage_completed': True, 'errors': 0, 'semantic_channel_qualified': False}, 'raw_recurrence_training')


def test_raw_recipe_declares_full_depth_output_and_native_value_sizing():
    recipe = load_recipe(Path(__file__).parents[2] / 'training/neuralese/recipes/raw-recurrence-v1.json')
    parameters = recipe['stages'][-1]['parameters']
    assert parameters['content_transport'] == 'top-state'
    assert parameters['writer_length_policy'] == 'native-value'
    assert parameters['max_write_vectors'] >= 163
    assert parameters['writer_supervision'] == 'native-value'
    assert parameters['stop_supervision'] == 'gold-native-boundary'


def test_shared_text_recipe_trains_both_projections_to_plateau_then_sequence_passes():
    recipe=load_recipe(Path(__file__).parents[2]/'training/neuralese/recipes/raw-recurrence-v1.json')
    p=next(stage['parameters'] for stage in recipe['stages'] if stage['kind']=='core_text_warmup')
    assert 'aligned_steps' not in p and 'ramp_steps' not in p
    assert p['projection_patience']==3 and p['projection_min_evals']==2
    assert p['sketch_weight']==p['embedding_weight']==1.
    assert p['sketch_lr']>p['lr']
    assert (p['neuralese_input'], p['input_map_kernel'], p['input_map_rank'], p['rollout_passes']) == ('map', 4, 64, 0)


def test_core_text_recipe_requires_and_propagates_explicit_map_mode(tmp_path):
    recipe = declared()
    recipe['stages'].append({
        'id': 'core_text_warmup', 'kind': 'core_text_warmup',
        'requires': ['runtime_qualification'],
        'parameters': {'neuralese_input': 'map', 'input_map_kernel': 4,
                       'input_map_rank': 64, 'rollout_passes': 0},
    })
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    assert load_recipe(path)['stages'][-1]['parameters']['neuralese_input'] == 'map'
    assert stage_parameter_args(recipe['stages'][-1]['parameters']) == [
        '--neuralese-input', 'map', '--input-map-kernel', '4',
        '--input-map-rank', '64', '--rollout-passes', '0',
    ]

    recipe['stages'][-1]['parameters'] = {'rollout_passes': 0}
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='must declare neuralese_input'):
        load_recipe(path)


def test_core_text_recipe_keeps_explicit_sketch_reproduction_and_rejects_map_rollout(tmp_path):
    recipe = declared()
    recipe['stages'].append({
        'id': 'core_text_warmup', 'kind': 'core_text_warmup',
        'requires': ['runtime_qualification'],
        'parameters': {'neuralese_input': 'sketch', 'rollout_passes': 4},
    })
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    assert load_recipe(path)['stages'][-1]['parameters']['neuralese_input'] == 'sketch'
    recipe['stages'][-1]['parameters'] = {'neuralese_input': 'map', 'rollout_passes': 4}
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='requires rollout_passes=0'):
        load_recipe(path)


def test_every_declared_shared_core_warmup_names_its_input_mode():
    recipes = Path(__file__).parents[2] / 'training/neuralese/recipes'
    declared_recipes = [json.loads(path.read_text()) for path in recipes.glob('*.json')]
    for recipe in declared_recipes:
        if recipe.get('schema') != 'natlang.neuralese-training-recipe/1':
            continue
        for stage in recipe['stages']:
            if stage['kind'] == 'core_text_warmup':
                assert stage['parameters']['neuralese_input'] in {'map', 'sketch'}
