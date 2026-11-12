"""The SGLang actor wrapper (serve_native_checkpoint.supervise_sglang) exits when its engine dies, and not on SIGTERM."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_sglang_wrapper_exits_nonzero_when_its_engine_dies(tmp_path):
    """Real process, fake engine (a short sleep): the wrapper logs the reason and exits ENGINE_DIED_EXIT."""
    script = tmp_path / 'wrapper.py'
    script.write_text(textwrap.dedent(f'''
        import importlib.util, subprocess, sys, time
        spec = importlib.util.spec_from_file_location('actor_cli', {str(ROOT / 'scripts/neuralese_data/serve_native_checkpoint.py')!r})
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        engine = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1)'], start_new_session=True)
        cli.supervise_sglang(engine)
        time.sleep(60)   # serve_forever
    '''))
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=90)
    assert run.returncode == 70
    event = json.loads(next(line for line in run.stderr.splitlines() if line.startswith('{')))
    assert event['event'] == 'sglang_engine_exited' and event['returncode'] == 0


def test_the_sglang_wrapper_stops_its_engine_on_sigterm_without_reporting_a_crash(tmp_path):
    script = tmp_path / 'wrapper.py'
    script.write_text(textwrap.dedent(f'''
        import importlib.util, os, signal, subprocess, sys, time
        spec = importlib.util.spec_from_file_location('actor_cli', {str(ROOT / 'scripts/neuralese_data/serve_native_checkpoint.py')!r})
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        engine = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        cli.supervise_sglang(engine)
        print(engine.pid, flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(30)
    '''))
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=90)
    assert run.returncode == 143 and 'sglang_engine_exited' not in run.stderr
    import os
    try:
        os.kill(int(run.stdout.split()[0]), 0)
        raise AssertionError('engine still running')
    except ProcessLookupError:
        pass
