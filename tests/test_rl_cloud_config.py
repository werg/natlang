import json

import pytest

from scripts.coreweave.rl_cloud import cloud_config


@pytest.mark.parametrize('name', ['runtime-9fbae2582', 'runtime-native-complete-resume-v36-1102'])
def test_cloud_config_rebinds_the_exact_runtime_in_all_execution_controls(tmp_path, name):
    runtime = tmp_path / name
    template = tmp_path / 'config.json'
    template.write_text(json.dumps({
        'out': '/local/learner',
        'runtime': str(runtime),
        'actor_environment': {'NATLANG_REPO': str(runtime),
                              'PYTHONPATH': str(runtime / 'training/neuralese') + ':' + str(runtime)},
        'actor_command': ['python', str(runtime / 'scripts/serve.py')],
        'learner_command': ['python', '-m', 'learner'],
        'recorded_other_runtime': str(runtime) + '-diagnostic/file.json',
    }))
    result = cloud_config({'template_runtime': str(runtime), 'root': '/cloud/run',
                           'lineages': {'production': {'template': str(template)}}}, 'production')
    assert result['runtime'] == '/cloud/run/runtime'
    assert result['actor_environment'] == {'NATLANG_REPO': '/cloud/run/runtime',
                                         'PYTHONPATH': '/cloud/run/runtime/training/neuralese:/cloud/run/runtime'}
    assert result['actor_command'][1] == '/cloud/run/runtime/scripts/serve.py'
    assert result['recorded_other_runtime'] == str(runtime) + '-diagnostic/file.json'
    assert result['out'] == '/cloud/run/production'
    assert result['checkpoint'] == '/cloud/run/production/parent/checkpoint.pt'
