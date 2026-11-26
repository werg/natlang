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

    def upload(self, path, key, rate=None, pause=None, properties=None, deadline=None):
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
    other = old_file(hdd / 'bgkit-checkpoints' / 'big.bin', b'w' * 4096)
    drive = FakeDrive()
    # (age counts ctime too, which a test cannot backdate: -1 admits every age here)
    args = SimpleNamespace(apply=True, root=['natlang-grpo-v1', 'bgkit-checkpoints'], budget_gb=1, max_minutes=5, rate_mb=0,
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
    push = [b'rsync', b'-a', b'/data/x/', b'colab-g4:/content/x/']
    pull = [b'rsync', b'-a', b'--partial', b'colab-g4b:/content/run.snap/', b'/data/run.pull/']
    assert m._colab_transfer(push) and not m._colab_transfer(pull)      # a pull uses the downlink
    assert m._colab_transfer(pull, pulls=True)
    assert not m._colab_transfer([b'cp', b'colab-g4:/x', b'/y']) and not m._colab_transfer([b'rsync', b'a', b'b'])


def test_rearchiving_a_restored_file_reuses_its_drive_copy(tmp_path, monkeypatch):
    m = load()
    hdd = tmp_path / 'hdd'
    monkeypatch.setattr(m, 'HDD', hdd)
    monkeypatch.setattr(m, 'MANIFEST', tmp_path / 'manifest.jsonl')
    monkeypatch.setattr(m, 'registered', lambda path: False)
    monkeypatch.setattr(m, 'held_paths', lambda: set())
    monkeypatch.setattr(m, 'mirror_manifest', lambda drive: None)
    f = old_file(hdd / 'natlang-grpo-v3' / 'a.bin', b'q' * 2048)
    drive = FakeDrive()
    args = SimpleNamespace(apply=True, root=['natlang-grpo-v3'], budget_gb=1, max_minutes=5, rate_mb=0, min_mb=0,
                           min_age_days=-1, links=[])
    m.archive(args, drive)
    m.restore(SimpleNamespace(path=str(f), token_file=None), drive)
    assert f.exists() and drive.next == 1
    m.archive(args, drive)
    assert not f.exists() and drive.next == 1          # no second upload: the restored copy is still on Drive


def test_ei_archive_files_behind_links_are_eligible(tmp_path, monkeypatch):
    m = load()
    hdd = tmp_path / 'hdd'
    monkeypatch.setattr(m, 'HDD', hdd)
    monkeypatch.setattr(m, 'LINK_TOLERANT', {'dgx-moved-ei': -1})
    linked = old_file(hdd / 'dgx-moved-ei' / 'aei1' / 'collect' / 'v0001-x' / 'actor' / 'blocks' / 'b.bin', b'b' * 64)
    other = old_file(hdd / 'natlang-grpo-v1' / 'live' / 'a.bin', b'a' * 64)
    links = {str(linked.parents[2]), str(other.parent)}
    found = [str(p) for p, _ in m.candidates(['dgx-moved-ei', 'natlang-grpo-v1'], min_mb=0, min_age_days=-1,
                                              exclude=[], links=links, held=set())]
    assert found == [str(linked)]


def test_round_inflow_drains_first_and_native_pins_stay(tmp_path, monkeypatch):
    m = load()
    hdd = tmp_path / 'hdd'
    monkeypatch.setattr(m, 'HDD', hdd)
    for name in ('natlang-grpo-v1', 'dgx-moved-20261124', 'dgx-moved-ei', 'dgx-moved-20261123', 'bgkit-checkpoints'):
        (hdd / name).mkdir(parents=True)
    roots = m.default_roots()
    assert roots[:3] == ['dgx-moved-ei', 'dgx-moved-20261123', 'dgx-moved-20261124']
    assert 'bgkit-checkpoints' not in roots and roots.index('natlang-grpo-v1') > 2
    monkeypatch.setattr(m, 'LINK_TOLERANT', {'dgx-moved-2026112*': -1})
    pin = old_file(hdd / 'dgx-moved-20261123' / 'native-pins' / 'base.pt', b'p' * 64)
    blocks = old_file(hdd / 'dgx-moved-20261124' / 'aei1-collect' / 'v0000' / 'b.bin', b'b' * 64)
    found = [str(p) for p, _ in m.candidates(['dgx-moved-20261123', 'dgx-moved-20261124'], min_mb=0,
                                              min_age_days=99, exclude=[], links={str(blocks.parent)}, held=set())]
    assert found == [str(blocks)] and str(pin) not in found


def test_bgkit_data_released_except_live_reader_trees(tmp_path, monkeypatch):
    m = load()
    hdd = tmp_path / 'hdd'
    monkeypatch.setattr(m, 'HDD', hdd)
    repo = old_file(hdd / 'bgkit-data' / 'repos' / 'r1.git' / 'objects' / 'pack.pack', b'r' * 64)
    news = old_file(hdd / 'bgkit-data' / 'multi_news_v1' / 'train.parquet', b'n' * 64)
    traj = old_file(hdd / 'bgkit-data' / 'trajectories_ext' / 'a' / 't.jsonl.zst', b't' * 64)
    ckpt = old_file(hdd / 'bgkit-checkpoints' / 'c.pt', b'c' * 64)
    (hdd / 'dgx-moved-ei').mkdir()
    roots = m.default_roots()
    assert roots.index('dgx-moved-ei') < roots.index('bgkit-data') < roots.index('natlang-grpo-v1')
    found = [str(p) for p, _ in m.candidates(['bgkit-data', 'bgkit-checkpoints'], min_mb=0, min_age_days=-1,
                                              exclude=[], links=set(), held=set())]
    assert sorted(found) == sorted([str(repo), str(traj)]) and str(news) not in found and str(ckpt) not in found


def test_no_file_starts_that_cannot_finish_before_the_hard_stop(tmp_path, monkeypatch):
    m = load()
    hdd = tmp_path / 'hdd'
    monkeypatch.setattr(m, 'HDD', hdd)
    monkeypatch.setattr(m, 'MANIFEST', tmp_path / 'state' / 'manifest.jsonl')
    monkeypatch.setattr(m, 'registered', lambda path: False)
    monkeypatch.setattr(m, 'held_paths', lambda: set())
    monkeypatch.setattr(m, 'mirror_manifest', lambda drive: None)
    monkeypatch.setattr(m, 'EXPECT_RATE', 1000)  # bytes/s: 60 s of hard window fits 60,000 bytes
    monkeypatch.setattr(m, 'colab_transfer_active', lambda: False)    # hermetic: a live rsync would hold the start
    big = old_file(hdd / 'natlang-grpo-v1' / 'a' / 'big.bin', b'b' * 120_000)
    small = old_file(hdd / 'natlang-grpo-v1' / 'b' / 'small.bin', b's' * 4096)
    args = SimpleNamespace(apply=True, root=['natlang-grpo-v1'], budget_gb=1, max_minutes=0.5, hard_minutes=1,
                           rate_mb=0, min_mb=0, min_age_days=-1, links=[])
    assert m.archive(args, FakeDrive()) == 0
    assert big.exists() and not small.exists()


def test_an_upload_paused_by_colab_transfers_ends_the_pass_at_its_deadline(tmp_path, monkeypatch):
    """2026-11-26: remote collection pulls from Colab almost continuously; a paused upload ran into the unit's hard
    timeout 21 times in 26 h. Paused past the deadline, the pass ends cleanly and keeps the file."""
    m = load()
    hdd = tmp_path / 'hdd'
    monkeypatch.setattr(m, 'HDD', hdd)
    monkeypatch.setattr(m, 'MANIFEST', tmp_path / 'state' / 'manifest.jsonl')
    monkeypatch.setattr(m, 'registered', lambda path: False)
    monkeypatch.setattr(m, 'held_paths', lambda: set())
    monkeypatch.setattr(m, 'mirror_manifest', lambda drive: None)
    clock = [1000.0]
    monkeypatch.setattr(m.time, 'time', lambda: clock[0])
    monkeypatch.setattr(m.time, 'sleep', lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(m, 'colab_transfer_active', lambda: True)
    monkeypatch.setattr(m, 'PAUSED_START_SECONDS', 0)
    held = old_file(hdd / 'natlang-grpo-v1' / 'a' / 'f.bin', b'x' * 1000)

    class Paused(FakeDrive):
        def upload(self, path, key, rate=None, pause=None, properties=None, deadline=None):
            while pause and pause():
                if deadline is not None and m.time.time() > deadline:
                    raise m.PassDeadline(key)
                m.time.sleep(30)
            raise AssertionError('never resumes')

    args = SimpleNamespace(apply=True, root=['natlang-grpo-v1'], budget_gb=1, max_minutes=1, hard_minutes=2,
                           rate_mb=0, min_mb=0, min_age_days=-1, links=[])
    assert m.archive(args, Paused()) == 0
    assert held.exists() and clock[0] - 1000.0 <= 60 + 30
    monkeypatch.setattr(m, 'PAUSED_START_SECONDS', 600)        # paused now and near the deadline: no session opens
    assert m.archive(args, Paused()) == 0 and held.exists()


def test_the_candidate_walk_stops_at_the_pass_deadline(tmp_path, monkeypatch):
    m = load()
    hdd = tmp_path / 'hdd'
    monkeypatch.setattr(m, 'HDD', hdd)
    for i in range(3):
        old_file(hdd / 'natlang-grpo-v1' / f'd{i}' / 'f.bin', b'x' * 10)
    kw = dict(min_mb=0, min_age_days=-1, exclude=[], links=set(), held=set())
    assert len(list(m.candidates(['natlang-grpo-v1'], **kw))) == 3
    assert list(m.candidates(['natlang-grpo-v1'], until=time.time() - 1, **kw)) == []
