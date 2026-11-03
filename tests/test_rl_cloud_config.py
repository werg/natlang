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


def test_actor_uses_sglang_follows_the_engine_adoption_default():
    """The sandbox installs the SGLang fork for actors production.adopt_actor_engine_defaults will switch."""
    from scripts.coreweave.rl_cloud import actor_uses_sglang
    native = {'model': 'lfm', 'actor_command': ['python', 'serve_native_checkpoint.py', '--engine', 'native']}
    assert actor_uses_sglang(native)
    assert actor_uses_sglang({**native, 'actor_command': ['python', 'serve_native_checkpoint.py']})
    assert actor_uses_sglang({**native, 'actor_command': ['python', 'x.py', '--engine', 'sglang'], 'model': 'mellum'})
    assert not actor_uses_sglang({**native, 'actor_engine_keep_native': True})
    assert not actor_uses_sglang({**native, 'coefficient_policy': '/p.json'})
    assert not actor_uses_sglang({**native, 'model': 'mellum'})
    assert not actor_uses_sglang({**native, 'actor_engine': 'llama-cpp'})
    assert not actor_uses_sglang({**native, 'mandatory_sft_only': True})


def test_prepared_control_relocates_window_and_refuses_a_started_control(tmp_path):
    from scripts.coreweave.rl_cloud import prepared_boundary, relocate_control
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'parent')
    out = tmp_path / 'dgx-window'
    control = tmp_path / 'ops' / 'w1' / 'config.json'
    control.parent.mkdir(parents=True)
    config = {'mandatory_sft_only': True, 'out': str(out), 'checkpoint': str(parent), 'runtime': '/rt/x/y',
              'trainer_parameters': {'checkpoint_input_offload_directory': str(out) + '/offload'},
              'rolling_sft_resume': {'schema': 'natlang.rolling-sft-resume/1'},
              'learner_command': ['python', '-m', 'natlang_neuralese.rl.native_stage', '{out}']}
    control.write_text(json.dumps(config))
    boundary = prepared_boundary({'controller': str(control)})
    assert boundary['mode'] == 'prepared-control' and boundary['parent'] == str(parent.resolve())
    moved = relocate_control(config, tmp_path / 'cloud' / 'w000000', tmp_path / 'cloud' / 'parent' / 'checkpoint.pt', 'id0')
    assert moved['out'] == str(tmp_path / 'cloud' / 'w000000')
    assert moved['trainer_parameters']['checkpoint_input_offload_directory'] == str(tmp_path / 'cloud' / 'w000000') + '/offload'
    assert moved['checkpoint'].endswith('cloud/parent/checkpoint.pt') and 'rolling_sft_resume' not in moved
    out.mkdir()
    (out / 'state.json').write_text('{}')
    with pytest.raises(SystemExit):
        prepared_boundary({'controller': str(control)})


def test_render_job_carries_shape_floor_to_the_sandbox(tmp_path):
    from scripts.coreweave.rl_cloud import render_job
    run = {'id': 'r', 'root': '/cloud/r', 'job': 'scripts/coreweave/jobs/lfm3/rl-lineage.job.json',
           'lineages': {'x': {'controller': '/c.json', 'resources': {'cpu': '8', 'memory': '64Gi'},
                              'minimum_resources': {'cpu': '4', 'memory': '32Gi'}}}}
    job = render_job(run, 'x', 0)
    assert job['minimum_resources'] == {'cpu': '4', 'memory': '32Gi'} and job['resources']['gpu'] == 1
    assert job['env']['NATLANG_CW_MIN_CPU'] == '4' and job['env']['NATLANG_CW_MIN_MEMORY'] == '32Gi'
    assert 'allow_default_shape' not in job
    check = render_job(run, 'x', 0, check=True)
    assert 'minimum_resources' not in check and 'NATLANG_CW_MIN_MEMORY' not in check['env']


def test_historical_checkpoints_ship_as_declared_bases_only(tmp_path, monkeypatch):
    import scripts.coreweave.rl_cloud as rc
    run = {'root': str(tmp_path), 'checkpoint_bases': [
        {'sha256': 'b' * 64, 'bytes': 10, 'encoding': 'raw', 'source': 'x'},
        {'sha256': 'c' * 64, 'bytes': 11, 'encoding': 'zip-delta', 'base': 'b' * 64, 'path': '/dgx/c.pt',
         'receipt': {'sha256': 'r' * 64, 'bytes': 1}, 'payload': {'sha256': 'p' * 64, 'bytes': 2}}]}
    monkeypatch.setattr(rc, 'locate', lambda run, digest: ('store', {'bytes': 1}))
    closure = {'historical_checkpoints': [
        {'path': '/dgx/parent.pt', 'sha256': 'c' * 64}, {'path': '/dgx/old.pt', 'sha256': 'b' * 64},
        {'path': '/dgx/unknown.pt', 'sha256': 'd' * 64}, {'path': '/cloud/parent/checkpoint.pt', 'sha256': 'c' * 64}]}
    blobs, skipped, links = rc.historical_checkpoint_blobs(run, closure, '/cloud/parent/checkpoint.pt')
    assert [(b['path'], b['encoding']) for b in blobs] == [('/dgx/parent.pt', 'zip-delta'), ('/dgx/old.pt', 'raw')]
    assert all(b['kind'] == 'checkpoint' and 'source' not in b for b in blobs)
    assert [s['path'] for s in skipped] == ['/dgx/unknown.pt'] and links == []


def test_historical_checkpoint_symlinks_ship_once_and_stay_links(tmp_path, monkeypatch):
    import scripts.coreweave.rl_cloud as rc
    real = tmp_path / 'external' / 'c.pt'
    real.parent.mkdir()
    real.write_bytes(b'x')
    alias = tmp_path / 'data' / 'c.pt'
    alias.parent.mkdir()
    alias.symlink_to(real)
    run = {'root': str(tmp_path), 'checkpoint_bases': [{'sha256': 'c' * 64, 'bytes': 1, 'encoding': 'raw', 'source': 'x'}]}
    monkeypatch.setattr(rc, 'locate', lambda run, digest: ('store', {'bytes': 1}))
    closure = {'historical_checkpoints': [{'path': str(alias), 'sha256': 'c' * 64}, {'path': str(real), 'sha256': 'c' * 64}]}
    blobs, skipped, links = rc.historical_checkpoint_blobs(run, closure, '/cloud/parent/checkpoint.pt')
    assert [b['path'] for b in blobs] == [str(real)] and not skipped
    assert links == [{'path': str(alias), 'symlink_to': str(real)}]
