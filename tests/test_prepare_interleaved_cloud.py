import importlib.util
from pathlib import Path

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
