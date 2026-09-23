import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import fcntl
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'scripts' / 'run_training_pipeline.py'


def config_file(tmp_path, stages, inputs=()):
    config = tmp_path / 'pipeline.json'
    config.write_text(json.dumps({
        'version': 'natlang.training_pipeline/1',
        'repository': str(tmp_path),
        'stages': stages,
    }))
    return config


def command(script):
    return [sys.executable, '-c', script]


def run(config, root, *, env=None):
    return subprocess.run([sys.executable, str(RUNNER), str(config), str(root)],
                          cwd=ROOT, text=True, capture_output=True, env=env, timeout=20)


def test_completed_stages_are_not_rerun_on_resume(tmp_path):
    run_dir = tmp_path / 'run'
    first, second = run_dir / 'first.json', run_dir / 'second.json'
    calls = run_dir / 'calls.json'
    stage1 = (
        f"from pathlib import Path; p=Path({str(calls)!r}); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text(p.read_text()+'1' if p.exists() else '1'); "
        f"Path({str(first)!r}).write_text('one')"
    )
    stage2 = (
        f"from pathlib import Path; p=Path({str(calls)!r}); "
        "p.write_text(p.read_text()+'2'); "
        f"Path({str(second)!r}).write_text(Path({str(first)!r}).read_text()+' two')"
    )
    config = config_file(tmp_path, [
        {'id': 'one', 'command': command(stage1), 'outputs': [str(first)]},
        {'id': 'two', 'command': command(stage2), 'inputs': [str(first)], 'outputs': [str(second)]},
    ])

    assert run(config, run_dir).returncode == 0
    assert run(config, run_dir).returncode == 0
    assert calls.read_text() == '12'
    assert json.loads((run_dir / 'pipeline-state.json').read_text())['status'] == 'complete'


def test_completed_stage_rejects_input_content_drift(tmp_path):
    run_dir = tmp_path / 'run'
    source, output = tmp_path / 'source.txt', run_dir / 'output.txt'
    source.write_text('first')
    script = f"from pathlib import Path; Path({str(output)!r}).write_text('done')"
    config = config_file(tmp_path, [
        {'id': 'build', 'command': command(script), 'inputs': [str(source)], 'outputs': [str(output)]},
    ])
    assert run(config, run_dir).returncode == 0

    source.write_text('changed')
    resumed = run(config, run_dir)

    assert resumed.returncode != 0
    assert 'input content changed' in resumed.stderr


def test_failed_stage_resumes_without_rerunning_prior_completed_stage(tmp_path):
    run_dir = tmp_path / 'run'
    first, second = run_dir / 'first', run_dir / 'second'
    calls = run_dir / 'calls'
    stage1 = (
        f"from pathlib import Path; p=Path({str(calls)!r}); "
        "p.write_text(p.read_text()+'1' if p.exists() else '1'); "
        f"Path({str(first)!r}).write_text('ok')"
    )
    stage2 = (
        f"from pathlib import Path; p=Path({str(calls)!r}); "
        "n=int(p.read_text().count('2')+1) if p.exists() else 1; "
        "p.write_text((p.read_text() if p.exists() else '')+'2'); "
        "sys.exit(7) if n == 1 else None; "
        f"Path({str(second)!r}).write_text('ok')"
    )
    config = config_file(tmp_path, [
        {'id': 'first', 'command': command(stage1), 'outputs': [str(first)]},
        {'id': 'second', 'command': [sys.executable, '-c', 'import sys; ' + stage2],
         'inputs': [str(first)], 'outputs': [str(second)]},
    ])

    failed = run(config, run_dir)
    resumed = run(config, run_dir)

    assert failed.returncode == 7
    assert resumed.returncode == 0
    assert calls.read_text() == '122'
    state = json.loads((run_dir / 'pipeline-state.json').read_text())
    assert state['stages']['first']['attempts'] == 1
    assert state['stages']['second']['attempts'] == 2


def test_child_checkpoint_and_zero_exit_after_signal_do_not_mark_stage_complete(tmp_path):
    run_dir = tmp_path / 'run'
    checkpoint, train_state, ready = (run_dir / name for name in
                                      ('checkpoint.json', 'training-state.json', 'child-ready'))
    script = (
        'import json, signal, time\nfrom pathlib import Path\n'
        f"ready=Path({str(ready)!r})\ncheckpoint=Path({str(checkpoint)!r})\n"
        f"training=Path({str(train_state)!r})\nready.parent.mkdir(parents=True, exist_ok=True)\n"
        'def stop(sig, frame):\n'
        ' checkpoint.write_text("consistent checkpoint")\n'
        ' training.write_text(json.dumps({"trained_examples": 4, "corpus": {"target_examples": 10}}))\n'
        ' raise SystemExit(0)\n'
        'signal.signal(signal.SIGTERM, stop)\n'
        'ready.write_text("ready")\n'
        'while True: time.sleep(0.02)\n'
    )
    config = config_file(tmp_path, [{
        'id': 'train', 'command': command(script), 'outputs': [str(checkpoint)],
        'training_state': str(train_state),
    }])
    process = subprocess.Popen([sys.executable, str(RUNNER), str(config), str(run_dir)],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.02)
        assert ready.exists(), 'stage child did not start'
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode == 130, (stdout, stderr)
    assert checkpoint.read_text() == 'consistent checkpoint'
    state = json.loads((run_dir / 'pipeline-state.json').read_text())
    assert state['status'] == 'stopped'
    assert state['stages']['train']['status'] == 'stopped'


def test_training_state_is_a_completion_gate_and_can_resume(tmp_path):
    run_dir = tmp_path / 'run'
    output, training = run_dir / 'weights', run_dir / 'training-state.json'
    script = (
        'import json; from pathlib import Path; '
        f"Path({str(output)!r}).write_text('checkpoint'); "
        f"p=Path({str(training)!r}); "
        'current=json.loads(p.read_text()) if p.exists() else {"trained_examples": 4, "corpus": {"target_examples": 10}}; '
        'current["trained_examples"] = 10 if p.exists() else 4; p.write_text(json.dumps(current))'
    )
    config = config_file(tmp_path, [{
        'id': 'train', 'command': command(script), 'outputs': [str(output)],
        'training_state': str(training),
    }])

    partial = run(config, run_dir)
    state = json.loads((run_dir / 'pipeline-state.json').read_text())
    assert partial.returncode == 75
    assert state['stages']['train']['status'] == 'stopped'
    assert state['status'] == 'stopped'

    resumed = run(config, run_dir)
    state = json.loads((run_dir / 'pipeline-state.json').read_text())
    assert resumed.returncode == 0
    assert state['stages']['train']['status'] == 'complete'
    assert state['status'] == 'complete'


@pytest.mark.parametrize('as_directory', [False, True])
def test_completed_output_corruption_is_rejected(tmp_path, as_directory):
    run_dir = tmp_path / 'run'
    output = run_dir / 'artifact'
    create = (f"p=Path({str(output)!r}); p.mkdir(parents=True); "
              "(p/'part').write_text('original')" if as_directory else
              f"Path({str(output)!r}).write_text('original')")
    config = config_file(tmp_path, [{'id': 'build', 'command': command(
        'from pathlib import Path; ' + create), 'outputs': [str(output)]}])
    assert run(config, run_dir).returncode == 0

    if as_directory:
        (output / 'part').write_text('tampered')
    else:
        output.write_text('tampered')
    resumed = run(config, run_dir)
    assert resumed.returncode != 0
    assert 'completed output changed or is missing' in resumed.stderr


def test_missing_completed_output_fails_closed_without_rerunning_or_advancing(tmp_path):
    run_dir = tmp_path / 'run'
    output, downstream = run_dir / 'artifact', run_dir / 'downstream-ran'
    first_script = (f"from pathlib import Path; Path({str(output)!r}).write_text('original')")
    second_script = (f"from pathlib import Path; p=Path({str(downstream)!r}); "
                     "p.write_text(p.read_text()+'x' if p.exists() else 'x')")
    config = config_file(tmp_path, [
        {'id': 'build', 'command': command(first_script), 'outputs': [str(output)]},
        {'id': 'next', 'command': command(second_script), 'outputs': [str(downstream)]},
    ])
    assert run(config, run_dir).returncode == 0
    output.unlink()

    resumed = run(config, run_dir)
    state = json.loads((run_dir / 'pipeline-state.json').read_text())
    assert resumed.returncode != 0
    assert not output.exists()
    assert downstream.read_text() == 'x'
    assert state['stages']['build']['attempts'] == 1


def test_config_drift_requires_a_new_run_directory(tmp_path):
    run_dir = tmp_path / 'run'
    output = run_dir / 'out'
    config = config_file(tmp_path, [{'id': 'build', 'command': command(
        f"from pathlib import Path; Path({str(output)!r}).write_text('ok')"),
        'outputs': [str(output)]}])
    assert run(config, run_dir).returncode == 0
    config_file(tmp_path, [{'id': 'build', 'command': command('pass'),
                            'outputs': [str(output)]}])
    resumed = run(config, run_dir)
    assert resumed.returncode != 0
    assert 'pipeline config changed' in resumed.stderr


def test_until_stops_after_requested_stage_and_can_resume(tmp_path):
    run_dir = tmp_path / 'run'
    calls = run_dir / 'calls'
    first, second = run_dir / 'first', run_dir / 'second'
    def write_stage(number, path):
        return command(f"from pathlib import Path; p=Path({str(calls)!r}); "
                       "p.write_text((p.read_text() if p.exists() else '')+" + repr(number) + "); "
                       f"Path({str(path)!r}).write_text('ok')")
    config = config_file(tmp_path, [
        {'id': 'first', 'command': write_stage('1', first), 'outputs': [str(first)]},
        {'id': 'second', 'command': write_stage('2', second), 'outputs': [str(second)]},
    ])
    until = subprocess.run([sys.executable, str(RUNNER), str(config), str(run_dir),
                            '--until', 'first'], cwd=ROOT, text=True,
                           capture_output=True, timeout=20)
    state = json.loads((run_dir / 'pipeline-state.json').read_text())
    assert until.returncode == 0
    assert state['status'] == 'ready'
    assert state['stages']['first']['status'] == 'complete'
    assert 'second' not in state['stages']

    assert run(config, run_dir).returncode == 0
    assert calls.read_text() == '12'


@pytest.mark.parametrize('lock_name, expected', [
    ('.pipeline.lock', 'another runner owns this pipeline'),
    ('.stage-build.lock', 'stage build is still running'),
])
def test_existing_lock_prevents_runner_from_starting(tmp_path, lock_name, expected):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    marker = run_dir / 'child-ran'
    config = config_file(tmp_path, [{'id': 'build', 'command': command(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"),
        'outputs': [str(run_dir / 'out')]}])
    lock_path = run_dir / lock_name
    with lock_path.open('a+') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run(config, run_dir)
    assert result.returncode != 0
    assert expected in result.stderr
    assert not marker.exists()


def test_gpu_resource_wait_does_not_start_stage_and_resumes_when_available(tmp_path):
    run_dir = tmp_path / 'run'
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    memory_file = tmp_path / 'free-memory'
    fake_smi = bin_dir / 'nvidia-smi'
    fake_smi.write_text(f"#!/bin/sh\ncat {str(memory_file)!r}\n")
    fake_smi.chmod(0o755)
    memory_file.write_text('100\n99999\n')
    calls, output = run_dir / 'calls', run_dir / 'output'
    script = (f"from pathlib import Path; p=Path({str(calls)!r}); "
              "p.parent.mkdir(parents=True, exist_ok=True); "
              "p.write_text('started'); "
              f"Path({str(output)!r}).write_text('ok')")
    config = config_file(tmp_path, [{'id': 'gpu', 'command': command(script),
                                     'outputs': [str(output)], 'min_free_vram_mib': 500}])
    env = os.environ.copy()
    env['PATH'] = str(bin_dir) + os.pathsep + env.get('PATH', '')

    waiting = run(config, run_dir, env=env)
    state = json.loads((run_dir / 'pipeline-state.json').read_text())
    assert waiting.returncode == 75
    assert not calls.exists()
    assert state['status'] == 'resource_wait'
    assert state['resource_wait'] == {'stage': 'gpu', 'free_mib': 100, 'required_mib': 500}

    memory_file.write_text('600\n99999\n')
    resumed = run(config, run_dir, env=env)
    assert resumed.returncode == 0
    assert calls.read_text() == 'started'
    assert json.loads((run_dir / 'pipeline-state.json').read_text())['status'] == 'complete'
