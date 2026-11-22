import os
import subprocess
import sys

from natlang_neuralese.tune import build


def test_sweep_removes_only_stores_of_dead_owners(tmp_path):
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    stale, live, legacy = (tmp_path / f'natlang-fit-store-{n}' for n in ('stale', 'live', 'legacy'))
    for path in (stale, live, legacy):
        path.mkdir()
        (path / 'blob').write_bytes(b'x' * 1024)
    (stale / build.OWNER_FILE).write_text(f'{dead.pid}\n')
    (live / build.OWNER_FILE).write_text(f'{os.getppid()}\n')
    removed = build.sweep_dead_fit_stores(str(tmp_path))
    assert removed == [str(stale)] and not stale.exists() and live.exists() and legacy.exists()


def test_fit_store_records_its_owner_and_sweeps_first(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(build, 'sweep_dead_fit_stores', lambda scratch: calls.append(scratch) or [])
    store = build.fit_store({'store_scratch': str(tmp_path)})
    try:
        [path] = list(tmp_path.glob('natlang-fit-store-*'))
        assert (path / build.OWNER_FILE).read_text().strip() == str(os.getpid()) and calls == [str(tmp_path)]
    finally:
        store._fit_cleanup()
    assert not list(tmp_path.glob('natlang-fit-store-*'))
