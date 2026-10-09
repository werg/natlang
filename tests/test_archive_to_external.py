import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('archive', Path(__file__).resolve().parents[1] / 'scripts/archive_to_external.py')
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def big(path: Path, size=4096, age_hours=48, read_hours=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(size))
    old = time.time() - age_hours * 3600
    read = old if read_hours is None else time.time() - read_hours * 3600
    os.utime(path, (read, old))
    return path


def run_candidates(root, excludes=(), held=()):
    return {p.name: reason for p, _, reason in archive.candidates(
        (root,), min_bytes=1024, min_age=24 * 3600, excludes=list(excludes), held=set(held))}


def test_candidates_are_large_old_closed_and_not_excluded(tmp_path):
    root = tmp_path / 'runs'
    big(root / 'a' / 'old.pt')
    big(root / 'a' / 'small.pt', size=10)
    big(root / 'a' / 'fresh.pt', age_hours=1)
    big(root / 'a' / 'hot-input.jsonl', read_hours=2)
    big(root / 'keep' / 'kept.pt')
    (root / 'keep' / archive.MARKER).write_text('')
    big(root / 'claimed' / 'out' / 'ckpt.pt')
    held = big(root / 'a' / 'open.pt')
    os.symlink(root / 'a' / 'old.pt', root / 'a' / 'link.pt')
    got = run_candidates(root, excludes=[root / 'claimed' / 'out', root], held=[str(held)])
    assert got['old.pt'] is None
    assert 'small.pt' not in got and 'link.pt' not in got  # too small; symlinks are never followed
    assert got['fresh.pt'].startswith('modified')
    assert got['hot-input.jsonl'].startswith('read')
    assert archive.MARKER in got['kept.pt']
    assert got['ckpt.pt'].startswith('excluded by')
    assert got['open.pt'] == 'open by a process'


def test_a_root_level_exclude_protects_nothing_by_itself(tmp_path):
    root = tmp_path / 'runs'
    big(root / 'x.pt')
    assert archive.excluded(root / 'x.pt', [root, tmp_path], roots=(root,)) is None


def test_archive_verifies_then_replaces_the_source_with_a_symlink(tmp_path):
    source = big(tmp_path / 'nvme' / 'run' / 'ckpt.pt', size=1 << 20)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    mtime = source.stat().st_mtime
    dest = archive.destination(source, (tmp_path / 'nvme',), tmp_path / 'hdd')
    record = archive.archive_file(source, dest, tmp_path / 'hdd' / 'manifest.jsonl')
    assert source.is_symlink() and Path(os.readlink(source)) == dest
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest == record['sha256']
    assert abs(dest.stat().st_mtime - mtime) < 1e-3
    line = json.loads((tmp_path / 'hdd' / 'manifest.jsonl').read_text())
    assert line['source'] == str(source) and line['bytes'] == 1 << 20


def test_a_file_that_changes_during_the_copy_stays(tmp_path, monkeypatch):
    source = big(tmp_path / 'nvme' / 'ckpt.pt', size=1 << 20)
    dest = tmp_path / 'hdd' / 'ckpt.pt'
    real_fsync = os.fsync

    def touch_then_fsync(fd):
        real_fsync(fd)
        os.utime(source, None)  # the writer touches the file mid-copy

    monkeypatch.setattr(archive.os, 'fsync', touch_then_fsync)
    with pytest.raises(RuntimeError, match='changed during the copy'):
        archive.archive_file(source, dest, tmp_path / 'manifest.jsonl')
    assert source.is_file() and not source.is_symlink()
    assert not dest.exists() and not (tmp_path / 'hdd' / 'ckpt.pt.archiving').exists()
    assert not (tmp_path / 'manifest.jsonl').exists()


def test_dry_run_moves_nothing(tmp_path, capsys):
    source = big(tmp_path / 'nvme' / 'ckpt.pt', size=2048)
    archive.main(['--root', str(tmp_path / 'nvme'), '--archive', str(tmp_path / 'hdd'), '--min-gb', '0.000001'])
    out = json.loads(capsys.readouterr().out)
    assert out['movable_files'] == 1 and out['moved_files'] == 0
    assert source.is_file() and not source.is_symlink() and not (tmp_path / 'hdd').exists()
