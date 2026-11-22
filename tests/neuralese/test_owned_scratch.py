"""Owned scratch directories (common/owned_scratch.py): swept when their owner is dead, removed at exit and on SIGTERM.
The GRPO weight slots on /dev/shm and the operator fits' spill stores use it."""
import os
import signal
import subprocess
import sys
import textwrap

from natlang_neuralese.common import owned_scratch


def test_sweep_removes_only_directories_of_dead_owners(tmp_path):
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    stale, live, legacy, other = (tmp_path / name for name in
                                  ('natlang-a-actor-weights', 'natlang-b-actor-weights', 'natlang-c-actor-weights', 'x'))
    for path in (stale, live, legacy, other):
        path.mkdir()
    (stale / owned_scratch.OWNER_FILE).write_text(f'{dead.pid}\n')
    (other / owned_scratch.OWNER_FILE).write_text(f'{dead.pid}\n')
    owned_scratch.claim(live)
    assert owned_scratch.sweep_dead(tmp_path, 'natlang-*-actor-weights') == [str(stale)]
    assert not stale.exists() and live.exists() and legacy.exists() and other.exists()


def _child(tmp_path, body):
    script = textwrap.dedent(f'''
        import sys, time
        from natlang_neuralese.common import owned_scratch
        path = owned_scratch.claim({str(tmp_path / 'slots')!r})
        owned_scratch.remove_on_exit(path)
        (path / 'model.safetensors').write_bytes(b'x' * 4096)
        print('ready', flush=True)
        {body}
    ''')
    return subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, text=True,
                            env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)})


def test_an_owned_directory_goes_on_sigterm_on_failure_and_at_exit(tmp_path):
    for body, signal_it, code in (('time.sleep(60)', True, -signal.SIGTERM), ('raise RuntimeError("boom")', False, 1),
                                  ('pass', False, 0)):
        child = _child(tmp_path, body)
        assert child.stdout.readline().strip() == 'ready'
        if signal_it:
            child.send_signal(signal.SIGTERM)
        assert child.wait(30) == code
        assert not (tmp_path / 'slots').exists(), body
