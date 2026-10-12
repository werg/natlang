"""Asynchronous checkpoint offload to the external disk (train/checkpoint_offload.py), with tiny files."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from natlang_neuralese.train import checkpoint_offload as co
from natlang_neuralese.train import checkpoint_policy as cp


@pytest.fixture
def layout(tmp_path, monkeypatch):
    repo, hdd = tmp_path / 'repo', tmp_path / 'hdd'
    out = repo / 'runs' / 'lineage' / 'run-v1' / 'stage'
    out.mkdir(parents=True)
    hdd.mkdir()
    monkeypatch.setenv('NATLANG_REPO', str(repo))
    monkeypatch.setenv('NATLANG_CHECKPOINT_OFFLOAD', '1')
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIRROR', str(hdd / 'natlang-active-lineage'))
    monkeypatch.setenv('NATLANG_CHECKPOINT_OFFLOAD_MIN_GB', '0')
    monkeypatch.setenv('NATLANG_CHECKPOINT_OFFLOAD_HDD_FLOOR_GB', '0')
    monkeypatch.setattr(co, '_mount_of', lambda path: hdd)  # the tmp "external disk"
    monkeypatch.setattr(co, '_same_filesystem', lambda a, b: False)
    monkeypatch.setattr(co, '_OFFLOADER', None)
    return out, hdd / 'natlang-active-lineage' / 'lineage' / 'run-v1' / 'stage'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data=b''):
    path.write_bytes(data or os.urandom(4096))
    return path


def test_the_offload_copy_verifies_and_an_archived_file_becomes_a_link(layout):
    out, mirror_dir = layout
    best = write(out / 'best-checkpoint.pt')
    digest = sha(best)
    event = co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.verified_copy)
    mirror = mirror_dir / 'best-checkpoint.pt'
    assert event['linked'] and event['sha256'] == digest and event['copied']
    assert best.is_symlink() and Path(os.readlink(best)) == mirror and sha(best) == digest
    assert co.read_record(mirror)['sha256'] == digest
    assert not list(mirror_dir.glob('*.offload.part')) and not list(out.glob('*.offload-link'))


def test_the_latest_slot_is_mirrored_but_stays_on_the_nvme(layout):
    out, mirror_dir = layout
    slot = write(out / 'checkpoint.pt')
    event = co.offload_one(slot, 'latest', co.OffloadSettings.from_env(), copier=co.verified_copy)
    assert not slot.is_symlink() and 'linked' not in event
    assert sha(mirror_dir / 'checkpoint.pt') == sha(slot)
    again = co.offload_one(slot, 'latest', co.OffloadSettings.from_env(), copier=co.verified_copy)
    assert not again['copied']  # the mirror is current: no second copy


def test_a_copy_that_fails_verification_leaves_the_file_and_no_partial(layout, monkeypatch):
    out, mirror_dir = layout
    best = write(out / 'best-checkpoint.pt')
    monkeypatch.setattr(co, 'file_sha256', lambda path, **_: '0' * 64)  # the copy reads back wrong
    with pytest.raises(RuntimeError, match='verification failed'):
        co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.verified_copy)
    assert best.is_file() and not best.is_symlink()
    assert not (mirror_dir / 'best-checkpoint.pt').exists() and not list(mirror_dir.glob('*.part'))


def test_the_link_swap_never_leaves_a_dangling_link_or_replaces_a_changed_file(tmp_path):
    path = write(tmp_path / 'best-checkpoint.pt')
    with pytest.raises(FileNotFoundError):
        co.replace_with_link(path, tmp_path / 'missing.pt')
    assert path.is_file() and not path.is_symlink()
    dest = write(tmp_path / 'mirror.pt')
    identity = co._identity(path.lstat())
    write(tmp_path / 'best-checkpoint.pt.new')
    os.replace(tmp_path / 'best-checkpoint.pt.new', path)  # a newer best was published meanwhile
    with pytest.raises(RuntimeError, match='changed before its link swap'):
        co.replace_with_link(path, dest, expect=identity)
    assert path.is_file() and not path.is_symlink() and not list(tmp_path.glob('*.offload-link'))
    co.replace_with_link(path, dest, expect=co._identity(path.lstat()))
    assert path.is_symlink() and path.read_bytes() == dest.read_bytes()


def test_a_failing_offload_never_fails_training_and_is_retried_at_the_next_write(layout):
    out, mirror_dir = layout
    calls = []

    def flaky(path, dest):
        calls.append(path)
        if len(calls) == 1:
            raise OSError('external disk went away')
        return co.verified_copy(path, dest)

    worker = co.CheckpointOffloader(co.OffloadSettings.from_env(), copier=flaky)
    best = write(out / 'best-checkpoint.pt')
    worker.submit(best, 'archive')  # returns at once; the failure is an event
    assert worker.wait(30)
    assert worker.events[-1]['event'] == 'checkpoint_offload_failed' and best.is_file() and not best.is_symlink()
    assert json.loads((out / 'checkpoint-offload.jsonl').read_text().splitlines()[-1])['event'] == \
        'checkpoint_offload_failed'
    worker.submit(write(out / 'checkpoint.pt'), 'latest')  # the next write retries the failed file
    assert worker.wait(30)
    assert best.is_symlink() and (mirror_dir / 'checkpoint.pt').is_file()


def test_policy_writes_are_offloaded_and_a_best_shares_the_mirror_by_a_hard_link(layout, monkeypatch):
    out, mirror_dir = layout
    worker = co.CheckpointOffloader(co.OffloadSettings.from_env(), copier=co.verified_copy)
    monkeypatch.setattr(co, '_OFFLOADER', worker)
    policy = cp.CheckpointPolicy(out)
    assert json.loads((out / 'checkpoint-policy.json').read_text())['offload']['mirror'] == str(mirror_dir)
    policy.save({'step': 1, 'w': torch.ones(3)}, metric=1.0)
    assert worker.wait(30)
    slot, best = out / 'checkpoint.pt', out / 'best-checkpoint.pt'
    assert not slot.is_symlink() and best.is_symlink()
    assert (mirror_dir / 'best-checkpoint.pt').stat().st_ino == (mirror_dir / 'checkpoint.pt').stat().st_ino
    policy.save({'step': 2, 'w': torch.zeros(3)}, metric=2.0)  # worse: the best keeps its step-1 mirror
    assert worker.wait(30)
    assert torch.load(best)['step'] == 1 and torch.load(mirror_dir / 'checkpoint.pt')['step'] == 2
    assert json.loads((out / 'checkpoint-policy.json').read_text())['best_metric'] == 1.0


def test_resume_reads_the_mirror_when_the_nvme_slot_is_missing_or_truncated(layout, monkeypatch):
    out, mirror_dir = layout
    monkeypatch.setattr(co, '_OFFLOADER', co.CheckpointOffloader(co.OffloadSettings.from_env(),
                                                                 copier=co.verified_copy))
    cp.CheckpointPolicy(out).save({'step': 5, 'w': torch.arange(4.)})
    assert co._OFFLOADER.wait(30)
    slot = out / 'checkpoint.pt'
    with open(slot, 'r+b') as stream:  # a cut-off write
        stream.truncate(slot.stat().st_size // 2)
    policy = cp.CheckpointPolicy(out)
    assert policy.recovered['local'] == 'truncated' and slot.is_symlink()
    assert cp.CheckpointPolicy.load(policy.slot)['step'] == 5
    slot.unlink()
    assert cp.CheckpointPolicy(out).recovered['local'] == 'missing'
    assert cp.CheckpointPolicy.load(slot)['step'] == 5
    fresh = layout[0].parent / 'never-mirrored'
    assert cp.CheckpointPolicy(fresh).recovered is None  # a run without a mirror record starts fresh


def test_stale_partials_are_cleaned_and_another_copiers_partial_is_left_alone(layout):
    out, mirror_dir = layout
    mirror_dir.mkdir(parents=True)
    ours = write(mirror_dir / 'best-checkpoint.pt.offload.part')
    os.utime(ours, (1, 1))
    theirs = write(mirror_dir / 'best-checkpoint.pt.part')
    best = write(out / 'best-checkpoint.pt')
    event = co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.verified_copy)
    assert event['event'] == 'checkpoint_offload_skipped' and 'another copy' in event['reason']
    assert theirs.exists() and best.is_file() and not best.is_symlink()
    theirs.unlink()
    co.offload_one(best, 'archive', co.OffloadSettings.from_env(), copier=co.verified_copy)
    assert not ours.exists() and best.is_symlink()


def test_no_mirror_on_the_source_filesystem_or_outside_the_runs(layout, monkeypatch, tmp_path):
    settings = co.OffloadSettings.from_env()
    assert co.mirror_unavailable(tmp_path / 'elsewhere' / 'checkpoint.pt', settings) == 'outside the mirrored trees'
    monkeypatch.setattr(co, '_same_filesystem', lambda a, b: True)
    assert 'source filesystem' in co.mirror_unavailable(layout[0] / 'checkpoint.pt', settings)
    monkeypatch.setattr(co, '_mount_of', lambda path: None)
    assert 'no mounted filesystem' in co.mirror_unavailable(layout[0] / 'checkpoint.pt', settings)


def test_the_default_copier_runs_in_an_idle_priority_child(layout):
    out, mirror_dir = layout
    slot = write(out / 'checkpoint.pt')
    record = co.child_copy(slot, mirror_dir / 'checkpoint.pt')
    assert record['sha256'] == sha(slot) == sha(mirror_dir / 'checkpoint.pt')
    with pytest.raises(RuntimeError, match='offload copy failed'):
        co.child_copy(out / 'missing.pt', mirror_dir / 'missing.pt')
