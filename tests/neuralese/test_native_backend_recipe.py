import copy

import pytest

from natlang_neuralese.train.recipe import native_stage_parameter_handoffs, _without_operational


def recipes():
    old = {'stages': [{'id': 'native', 'kind': 'native_state_sft', 'parameters': {
        'fp8_training': 'triton-experts', 'fp8_recipe': 'delayed',
        'forward_atol': 1e-4, 'gradient_atol': 1e-5}}]}
    new = copy.deepcopy(old)
    new['stages'][0]['parameters'].update(fp8_recipe='current-bf16-dx', qualify_training_backend=True)
    return old, new


def test_exact_expert_transition_preserves_recipe_and_requires_runtime_qualification(tmp_path):
    old, new = recipes()
    changes, before, after = native_stage_parameter_handoffs(old, new, tmp_path)
    assert changes == {'native': {'fp8_recipe': ['delayed', 'current-bf16-dx']}}
    assert _without_operational(before) == _without_operational(after)
    assert new['stages'][0]['parameters']['qualify_training_backend'] is True


@pytest.mark.parametrize('change', [
    {'qualify_training_backend': False}, {'gradient_atol': None},
    {'fp8_training': 'off'}, {'fp8_amax_history_len': 32}, {'fp8_recipe': 'delayed'},
])
def test_unqualified_or_nonstateless_expert_changes_do_not_bypass_guard(tmp_path, change):
    old, new = recipes()
    if change == {'fp8_recipe': 'delayed'}:
        old['stages'][0]['parameters']['fp8_recipe'] = 'current-bf16-dx'
    new['stages'][0]['parameters'].update(change)
    with pytest.raises(ValueError, match='exact stateless transition'):
        native_stage_parameter_handoffs(old, new, tmp_path)


def test_expert_transition_does_not_admit_backbone_dtype_change(tmp_path):
    old, new = recipes()
    new['stages'][0]['parameters']['dtype'] = 'float32'
    _, before, after = native_stage_parameter_handoffs(old, new, tmp_path)
    assert _without_operational(before) != _without_operational(after)


def test_finished_stage_cannot_change_expert_recipe(tmp_path):
    old, new = recipes()
    (tmp_path / 'native-report.json').write_text('{}')
    with pytest.raises(ValueError, match='finished native stage'):
        native_stage_parameter_handoffs(old, new, tmp_path)
