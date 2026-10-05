import json
from pathlib import Path

import pytest

from natlang_neuralese.train.recipe import load_recipe, require_gate


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
