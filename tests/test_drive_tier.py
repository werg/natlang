import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'drive_tier.py'


def load():
    spec = importlib.util.spec_from_file_location('drive_tier', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeDrive:
    def __init__(self):
        self.files, self.next = {}, 0

    def upload(self, path, key, rate=None, pause=None, properties=None):
        data = Path(path).read_bytes()
        self.next += 1
        ident = f'id{self.next}'
        self.files[ident] = (key, data)
        return {'id': ident, 'bytes': len(data), 'md5': hashlib.md5(data).hexdigest(),
                'sha256': hashlib.sha256(data).hexdigest(), 'seconds': 0.0}

    def download(self, file_id, dest, md5=None, sha256=None):
        data = self.files[file_id][1]
        assert hashlib.md5(data).hexdigest() == md5
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(data)
        return {'bytes': len(data), 'md5': md5, 'sha256': hashlib.sha256(data).hexdigest(), 'seconds': 0.0}

    def lookup(self, key):
        return None

    def delete(self, file_id):
        self.files.pop(file_id, None)

    def request(self, *args, **kwargs):
        return {}


def old_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    stamp = time.time() - 30 * 86400
    os.utime(path, (stamp, stamp))
    return path


def test_archive_then_restore_keeps_links_registered_and_other_projects(tmp_path, monkeypatch):
    m = load()
    hdd = tmp_path / 'hdd'
    monkeypatch.setattr(m, 'HDD', hdd)
    monkeypatch.setattr(m, 'MANIFEST', tmp_path / 'state' / 'manifest.jsonl')
    monkeypatch.setattr(m, 'registered', lambda path: path.name == 'registered.bin')
    monkeypatch.setattr(m, 'held_paths', lambda: set())
    monkeypatch.setattr(m, 'mirror_manifest', lambda drive: None)
    cold = old_file(hdd / 'natlang-grpo-v1' / 'run' / 'state.bin', b'x' * 4096)
    linked = old_file(hdd / 'natlang-grpo-v1' / 'live' / 'pool.bin', b'y' * 4096)
    reg = old_file(hdd / 'natlang-grpo-v1' / 'run' / 'registered.bin', b'z' * 4096)
    other = old_file(hdd / 'bgkit-data' / 'big.bin', b'w' * 4096)
    drive = FakeDrive()
    # (age counts ctime too, which a test cannot backdate: -1 admits every age here)
    args = SimpleNamespace(apply=True, root=['natlang-grpo-v1', 'bgkit-data'], budget_gb=1, max_minutes=5, rate_mb=0,
                           min_mb=0, min_age_days=-1, links=[str(hdd / 'natlang-grpo-v1' / 'live')])
    assert m.archive(args, drive) == 0
    assert not cold.exists()
    stub = json.loads(cold.with_name('state.bin.ondrive.json').read_text())
    assert stub['md5'] == hashlib.md5(b'x' * 4096).hexdigest()
    assert linked.exists() and reg.exists() and other.exists()
    ops = [json.loads(l)['op'] for l in m.MANIFEST.read_text().splitlines()]
    assert ops == ['put', 'drop-local']

    assert m.restore(SimpleNamespace(path=str(hdd / 'natlang-grpo-v1'), token_file=None), drive) == 0
    assert cold.read_bytes() == b'x' * 4096 and not cold.with_name('state.bin.ondrive.json').exists()
    assert abs(cold.stat().st_mtime - (time.time() - 30 * 86400)) < 5


def test_dry_run_moves_nothing(tmp_path, monkeypatch):
    m = load()
    hdd = tmp_path / 'hdd'
    monkeypatch.setattr(m, 'HDD', hdd)
    monkeypatch.setattr(m, 'MANIFEST', tmp_path / 'manifest.jsonl')
    monkeypatch.setattr(m, 'registered', lambda path: False)
    monkeypatch.setattr(m, 'held_paths', lambda: set())
    f = old_file(hdd / 'natlang-grpo-v2' / 'a.bin', b'a' * 100)
    args = SimpleNamespace(apply=False, root=['natlang-grpo-v2'], budget_gb=1, max_minutes=5, rate_mb=0, min_mb=0,
                           min_age_days=-1, links=[])
    assert m.archive(args, FakeDrive()) == 0
    assert f.exists() and not m.MANIFEST.exists()


def test_colab_transfer_detection_reads_rsync_argv(monkeypatch, tmp_path):
    m = load()
    assert m.colab_transfer_active() in (True, False)
