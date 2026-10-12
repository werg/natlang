import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from natlang_neuralese.common import io_quiet

MODULE = Path(io_quiet.__file__)
GUARD = Path(__file__).resolve().parents[2] / 'scripts' / 'disk_guard.sh'


@pytest.fixture
def qdir(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_IO_QUIET_DIR', str(tmp_path / 'q'))
    return tmp_path / 'q'


def test_marker_created_and_removed(qdir):
    with io_quiet.io_quiet('save'):
        files = list(qdir.glob('*.json'))
        assert len(files) == 1
        record = json.loads(files[0].read_text())
        assert record['pid'] == os.getpid() and record['reason'] == 'save' and 'started' in record
        assert io_quiet.live_markers()
    assert not list(qdir.glob('*.json')) and not io_quiet.live_markers()


def test_marker_removed_on_exception(qdir):
    with pytest.raises(RuntimeError):
        with io_quiet.io_quiet('save'):
            raise RuntimeError
    assert not list(qdir.glob('*.json'))


def test_atomic_checkpoint_holds_marker(qdir, tmp_path):
    torch = pytest.importorskip('torch')
    from natlang_neuralese.train import trajectory_state
    seen = []
    real = torch.save
    torch.save = lambda *a, **k: (seen.append(len(list(qdir.glob('*.json')))), real(*a, **k))
    try:
        trajectory_state.atomic_checkpoint(tmp_path / 'c.pt', {'x': torch.zeros(2)})
    finally:
        torch.save = real
    assert seen == [1] and not list(qdir.glob('*.json'))


def _marker(qdir, name, pid, started):
    qdir.mkdir(exist_ok=True)
    (qdir / name).write_text(json.dumps({'pid': pid, 'started': started, 'reason': 'test'}))


def test_stale_markers_ignored(qdir):
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    _marker(qdir, 'dead.json', dead.pid, time.time())
    _marker(qdir, 'old.json', os.getpid(), time.time() - 3 * 3600)
    assert io_quiet.live_markers() == []
    _marker(qdir, 'live.json', os.getpid(), time.time())
    assert [Path(m['file']).name for m in io_quiet.live_markers()] == ['live.json']


def test_shell_guard_sees_live_marker_and_ignores_dead(qdir, tmp_path):
    env = {**os.environ, 'NATLANG_IO_QUIET_DIR': str(qdir), 'NATLANG_IO_QUIET_POLL_S': '0.2'}
    args = ['sh', str(GUARD), str(tmp_path), '0']
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    _marker(qdir, 'dead.json', dead.pid, time.time())
    assert subprocess.run(args, env=env, timeout=30, capture_output=True).returncode == 0  # dead marker ignored
    holder = subprocess.Popen(['sleep', '60'])
    try:
        _marker(qdir, 'live.json', holder.pid, time.time())
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(args, env=env, timeout=3, capture_output=True)  # paused while the marker is live
        (qdir / 'live.json').unlink()
        assert subprocess.run(args, env=env, timeout=30, capture_output=True).returncode == 0
    finally:
        holder.kill()
