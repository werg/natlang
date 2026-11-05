import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    'prep', Path(__file__).parents[1] / 'scripts/neuralese_data/prepare_interleaved_rl.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_cloud_worker_learner_command_takes_the_session_join_wrapper():
    seed = ['/venv/python', '/s/bin/join.py', '--session', '/s/session.json', '--worker', '3', '--',
            '/venv/python', '-m', 'natlang_neuralese.train.native_state_sft', '--out', 'x']
    learner = ['/venv/python', '-m', 'natlang_neuralese.rl.native_stage', '--out', 'y']
    commands = prep.cloud_commands({'seed_command': seed, 'learner_command': learner})
    assert commands['seed_command'] == seed
    assert commands['learner_command'] == ['/venv/python', '/s/bin/join.py', '--session', '/s/session.json',
                                           '--worker', '3', '--', *learner]


def test_standalone_cloud_commands_stay_bare_and_dgx_wrappers_are_not_cloud():
    seed = ['/venv/python', '-m', 'natlang_neuralese.train.native_state_sft']
    learner = ['/venv/python', '-m', 'natlang_neuralese.rl.native_stage']
    assert prep.cloud_commands({'seed_command': seed, 'learner_command': learner}) == {
        'seed_command': seed, 'learner_command': learner}
    assert prep.cloud_commands({'seed_command': ['/p', '/ops/admit.py'], 'learner_command': ['/p', '/ops/admit.py']}) is None


def test_never_started_control_reads_its_prior_window_state(tmp_path):
    prior, own = tmp_path / 'w1', tmp_path / 'w2'
    prior.mkdir()
    (prior / 'state.json').write_text('{"phase": "awaiting_semantic_refresh"}')
    control = {'out': str(own), 'branch_origin': {'prepared_next_window': {'prior_out': str(prior)}}}
    assert prep.prior_state_path(control) == prior / 'state.json'
    own.mkdir()
    (own / 'state.json').write_text('{}')
    assert prep.prior_state_path(control) == own / 'state.json'
    with pytest.raises(SystemExit):
        prep.prior_state_path({'out': str(tmp_path / 'none'), 'branch_origin': {}})


def _recipe(**parameters):
    return {'stages': [{'id': 'sft', 'kind': 'native_state_sft', 'inputs': {'pending_focus': 'old_focus'},
                        'parameters': {'learning_rate': 2e-5, 'batch_tokens': 16384, 'checkpoint_input_offload': True,
                                       'checkpoint_input_offload_directory': '/scratch', 'epochs': 1.0, **parameters}}],
            'input_bindings': {'old_focus': {'path': '/f', 'sha256': 'a' * 64}}}


def test_window_parameters_follow_the_previous_windows_effective_controls():
    # Interleaved windows are rebuilt from the recipe stage, so an LR/memory decision made on the window config alone
    # would silently revert (bf95 low-LR restart, 2026-11-05).
    recipe = _recipe()
    previous = {'trainer_parameters': {'learning_rate': 4e-6, 'batch_tokens': 65536, 'checkpoint_input_offload': False,
                                       'epochs': 1.0, 'recipe_binding': {'sha256': 'x'}, 'stage_id': 'sft',
                                       'resident_data_directory': '/r'}}
    changes = prep.window_parameter_overrides(previous, recipe, 'sft')
    assert changes == {'learning_rate': 4e-6, 'batch_tokens': 65536, 'checkpoint_input_offload': False,
                       'checkpoint_input_offload_directory': None}
    prep.apply_stage_parameters(recipe, 'sft', changes)
    parameters = recipe['stages'][0]['parameters']
    assert parameters['learning_rate'] == 4e-6 and 'checkpoint_input_offload_directory' not in parameters
    assert prep.window_parameter_overrides(previous, recipe, 'sft') == {}


def test_pending_focus_rebinding_replaces_the_stage_input_and_drops_the_unused_binding():
    recipe = _recipe()
    prep.rebind_pending_focus(recipe, 'sft', {'path': '/new', 'sha256': 'b' * 64})
    assert recipe['stages'][0]['inputs']['pending_focus'] == 'pending_focus_' + 'b' * 12
    assert 'old_focus' not in recipe['input_bindings']
    assert recipe['input_bindings']['pending_focus_' + 'b' * 12] == {'path': '/new', 'sha256': 'b' * 64}
