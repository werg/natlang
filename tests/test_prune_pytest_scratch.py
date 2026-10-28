"""Storage retention never treats live pytest runs as disposable siblings."""
import os
import time
from pathlib import Path

from scripts import prune_pytest_scratch as prune


def run(root: Path, number: int):
    path = root / f'pytest-{number}'
    path.mkdir(parents=True)
    (path / 'checkpoint.pt').write_bytes(b'fixture')
    for item in (path / 'checkpoint.pt', path):
        os.utime(item, (1, 1))
    return path


def test_preserves_live_locks_open_files_current_and_recent_siblings(tmp_path):
    runs = [run(tmp_path, n) for n in range(7)]
    (runs[0] / '.lock').write_text(str(os.getpid()))
    (tmp_path / 'pytest-current').symlink_to(runs[2])
    os.utime(runs[-1], None)
    held = {str(runs[1] / 'checkpoint.pt')}
    assert prune.candidates(tmp_path, held, age_seconds=60, keep_recent=1, now=time.time()) == runs[3:6]


def test_recent_nested_writes_and_foreign_symlinks_are_preserved(tmp_path):
    cold = run(tmp_path, 0)
    warm = run(tmp_path, 1)
    (warm / 'checkpoint.pt').touch()
    recent = run(tmp_path, 2)
    os.utime(recent, None)
    (tmp_path / 'pytest-3').symlink_to(cold, target_is_directory=True)
    assert prune.candidates(tmp_path, set(), age_seconds=60, keep_recent=1, now=time.time()) == [cold]


def test_dead_owner_is_prunable_but_partial_lock_is_protected(tmp_path, monkeypatch):
    cold = run(tmp_path, 0)
    partial = run(tmp_path, 1)
    recent = run(tmp_path, 2)
    (cold / '.lock').write_text('12345')
    (partial / '.lock').write_text('')
    os.utime(cold / '.lock', (1, 1)); os.utime(cold, (1, 1))
    os.utime(recent, None)
    def gone(pid, signal):
        raise ProcessLookupError
    monkeypatch.setattr(prune.os, 'kill', gone)
    assert prune.candidates(tmp_path, set(), age_seconds=60, keep_recent=1, now=time.time()) == [cold]
