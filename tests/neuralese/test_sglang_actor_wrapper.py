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


def test_an_engine_whose_cuda_context_fails_to_create_is_started_again(tmp_path, monkeypatch):
    """GB10 fragmentation (common/cuda_context.py): an engine that exits with cuDevicePrimaryCtxRetain OOM before it is
    healthy is started again after a compaction nudge; any other early exit fails the launch."""
    import importlib.util
    import socket
    from natlang_neuralese.common import cuda_context
    spec = importlib.util.spec_from_file_location('actor_cli', ROOT / 'scripts/neuralese_data/serve_native_checkpoint.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    nudges = []
    monkeypatch.setattr(cuda_context, 'compact', lambda: nudges.append(1) or {})
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    marker = tmp_path / 'started'
    engine = tmp_path / 'engine.py'
    engine.write_text(textwrap.dedent(f'''
        import http.server, pathlib, sys
        marker = pathlib.Path({str(marker)!r})
        if not marker.exists():
            marker.write_text('1')
            print('Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuDevicePrimaryCtxRetain', flush=True)
            sys.exit(1)
        class Health(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers()
        http.server.HTTPServer(('127.0.0.1', {port}), Health).serve_forever()
    '''))
    process = cli.start_sglang([sys.executable, str(engine)], None, f'http://127.0.0.1:{port}', wait_seconds=0)
    try:
        assert process.poll() is None and nudges == [1]
    finally:
        process.kill()
    failing = tmp_path / 'failing.py'
    failing.write_text('import sys; print("ImportError: no module", flush=True); sys.exit(3)\n')
    import pytest
    with pytest.raises(RuntimeError, match='exited with 3'):
        cli.start_sglang([sys.executable, str(failing)], None, 'http://127.0.0.1:9', wait_seconds=0)
    assert nudges == [1]


def test_a_cuda_context_failure_reexecutes_the_process_after_a_nudge(monkeypatch):
    from natlang_neuralese.common import cuda_context
    import types
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    def zeros(*_args, **_kwargs):
        raise RuntimeError('CUDA error: out of memory')
    fake.zeros = zeros
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setattr(cuda_context, 'compact', lambda: {'nudged': True})
    monkeypatch.setattr(cuda_context.time, 'sleep', lambda _seconds: None)
    calls = []
    def execv(path, argv):
        calls.append(argv)
        raise SystemExit(0)
    monkeypatch.setattr(cuda_context.os, 'execv', execv)
    monkeypatch.delenv(cuda_context.ATTEMPT_ENV, raising=False)
    import pytest
    with pytest.raises(SystemExit):
        cuda_context.ensure_context()
    assert len(calls) == 1 and cuda_context.os.environ[cuda_context.ATTEMPT_ENV] == '1'
    monkeypatch.setenv(cuda_context.ATTEMPT_ENV, str(cuda_context.RETRIES))
    with pytest.raises(RuntimeError, match='out of memory'):
        cuda_context.ensure_context()
    assert cuda_context.context_failure('Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuDevicePrimaryCtxRetain')
    assert not cuda_context.context_failure('torch.OutOfMemoryError: tried to allocate 2 GiB')
