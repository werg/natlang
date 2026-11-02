import os
from pathlib import Path

import pytest
import torch

from natlang_neuralese.train.checkpoint_safety import (
    CheckpointReserveError, PersistentCheckpointReservation,
    optimizer_state_size_upper_bound,
)
from natlang_neuralese.train.trajectory_state import atomic_checkpoint


def reservation(path):
    return PersistentCheckpointReservation(path, transfer_bytes=0, minimum_free_bytes=0)


def test_empty_spare_retirement_preserves_committed_checkpoint_and_partial_payload(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    torch.save({'step': 7}, path)
    owner = reservation(path).acquire(128 * 1024)
    pending = owner.pending
    assert owner.discard_empty() is True
    assert not pending.exists() and not owner.receipt.exists()
    assert torch.load(path, weights_only=True)['step'] == 7
    owner = reservation(path).acquire(128 * 1024)
    os.pwrite(owner._fd, b'partial checkpoint', 64 * 1024)
    assert owner.discard_empty() is False
    owner.close()
    assert pending.read_bytes()[64 * 1024:][:18] == b'partial checkpoint'


def test_reserved_inode_is_committed_and_replenished(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    torch.save({'step': 1}, path)
    owner = reservation(path)
    owner.acquire(20 * 1024 * 1024)
    inode = owner.pending.stat().st_ino
    atomic_checkpoint(path, {'step': 2}, reservation=owner)
    assert path.stat().st_ino == inode
    assert torch.load(path, weights_only=True)['step'] == 2
    owner.acquire(20 * 1024 * 1024)
    assert owner.pending.stat().st_ino != inode
    owner.close()


def test_crash_reuse_and_concurrent_owner(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    owner = reservation(path).acquire(20 * 1024 * 1024)
    inode = owner.pending.stat().st_ino
    other = reservation(path)
    with pytest.raises(BlockingIOError):
        other.acquire(20 * 1024 * 1024)
    owner.close()
    other.acquire(20 * 1024 * 1024)
    assert other.pending.stat().st_ino == inode
    other.close()


@pytest.mark.parametrize('partial_payload',[False,True])
@pytest.mark.parametrize('same_object',[False,True])
def test_low_space_resume_reclaims_only_a_fully_zero_old_reserve(tmp_path,monkeypatch,partial_payload,same_object):
    import shutil
    from collections import namedtuple
    from natlang_neuralese.train.checkpoint_safety import CheckpointSpaceUnavailable
    path=tmp_path/'checkpoint.pt';path.write_bytes(b'last durable parent')
    bound=1024*1024
    owner=reservation(path).acquire(bound);inode=owner.pending.stat().st_ino
    if partial_payload:os.pwrite(owner._fd,b'uncommitted serialization',65536)
    owner.close()
    Usage=namedtuple('Usage','total used free')
    monkeypatch.setattr(shutil,'disk_usage',lambda _:Usage(2*bound,bound+owner.pending.stat().st_size,
        bound-owner.pending.stat().st_size))
    retry=owner if same_object else reservation(path)
    try:
        if partial_payload:
            with pytest.raises(CheckpointSpaceUnavailable):retry.acquire(bound)
            assert owner.pending.read_bytes()[65536:][:25]==b'uncommitted serialization'
        else:
            retry.acquire(bound)
            assert retry.pending.stat().st_ino==inode and retry._allocated_bound>=bound
        assert path.read_bytes()==b'last durable parent'
    finally:retry.close()


def test_write_failure_preserves_parent_and_reservation(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.pt'
    torch.save({'step': 1}, path)
    owner = reservation(path).acquire(20 * 1024 * 1024)
    inode = owner.pending.stat().st_ino
    def fail(*args, **kwargs):
        raise OSError(28, 'disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError):
        atomic_checkpoint(path, {'step': 2}, reservation=owner)
    assert torch.load(path, weights_only=True)['step'] == 1
    assert owner.pending.stat().st_ino == inode
    owner.close()


def test_receipt_inode_mismatch_rejected(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    owner = reservation(path).acquire(1024)
    owner.close()
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(b'changed')
    os.replace(replacement, owner.pending)
    with pytest.raises(CheckpointReserveError):
        reservation(path).acquire(1024)


def test_adamwsr_lazy_slots_counted():
    from natlang_neuralese.train.optim import AdamWSR
    param = torch.nn.Parameter(torch.ones(16))
    opt = AdamWSR([param])
    assert optimizer_state_size_upper_bound(opt, lazy=True) >= 2 * param.numel() * param.element_size() + 4
    assert optimizer_state_size_upper_bound(opt, lazy=False) == 0


def test_reserve_enospc_preserves_durable_parent(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.pt'
    torch.save({'step': 3}, path)
    owner = reservation(path)
    def fail(*args):
        raise OSError(28, 'disk full')
    monkeypatch.setattr(os, 'posix_fallocate', fail)
    with pytest.raises(OSError):
        owner.acquire(1024)
    assert torch.load(path, weights_only=True)['step'] == 3
    owner.close()


def test_reuse_after_crash_after_rename(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    owner = reservation(path).acquire(1024)
    inode = owner.pending.stat().st_ino
    os.replace(owner.pending, path)
    owner.close()
    successor = reservation(path).acquire(1024)
    assert path.stat().st_ino == inode
    assert successor.pending.stat().st_ino != inode
    successor.close()


def test_policy_reserves_before_update_and_replenishes(tmp_path, monkeypatch):
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    monkeypatch.setenv('NATLANG_CHECKPOINT_TRANSFER_HEADROOM_FACTOR', '0')
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES', '0')
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter])
    policy = CheckpointPolicy(tmp_path)
    state = {'step': 0, 'model': parameter, 'optimizer': optimizer.state_dict()}
    policy.reserve_next(state, optimizer)
    first_inode = policy._persistent_reservation.pending.stat().st_ino
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = {'step': 1, 'model': parameter, 'optimizer': optimizer.state_dict()}
    policy.save(state)
    assert policy.slot.stat().st_ino == first_inode
    policy.reserve_next(state, optimizer)
    assert policy._persistent_reservation.pending.stat().st_ino != first_inode
    policy._persistent_reservation.close()


def test_unreserved_writer_cannot_replace_owned_target(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    owner = reservation(path).acquire(20 * 1024 * 1024)
    with pytest.raises(BlockingIOError):
        atomic_checkpoint(path, {'step': 2})
    assert not path.exists()
    owner.close()


def test_unexpected_growth_extends_same_reserved_inode(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    owner = reservation(path).acquire(1024)
    inode = owner.pending.stat().st_ino
    atomic_checkpoint(path, {'large_metadata': 'x' * 10000}, reservation=owner)
    assert path.stat().st_ino == inode
    assert torch.load(path, weights_only=True)['large_metadata'] == 'x' * 10000
    owner.close()


def test_preallocated_metadata_growth_commits_after_free_space_falls_below_floor(tmp_path, monkeypatch):
    from collections import namedtuple
    from natlang_neuralese.train import checkpoint_safety

    # Scale the production threshold and headroom down for a small real
    # allocation, while keeping the real size estimator and atomic writer.
    monkeypatch.setattr(checkpoint_safety, '_PERSISTENT_HEADROOM_THRESHOLD', 32 * 1024)
    monkeypatch.setattr(checkpoint_safety, '_PERSISTENT_METADATA_HEADROOM', 16 * 1024)
    estimate = checkpoint_safety.checkpoint_size_upper_bound
    monkeypatch.setattr(checkpoint_safety, 'checkpoint_size_upper_bound',
                        lambda state: estimate(state, minimum_overhead=0))
    previous = {'step': 70, 'metadata': 'x' * 50000}
    current = {'step': 71, 'metadata': 'x' * 52627}
    old_bound = checkpoint_safety.checkpoint_size_upper_bound(previous)
    new_bound = checkpoint_safety.checkpoint_size_upper_bound(current)
    assert new_bound - old_bound == 3153
    usage = namedtuple('usage', 'total used free')
    floor = 40 * 1024**3
    monkeypatch.setattr(checkpoint_safety.shutil, 'disk_usage', lambda _: usage(floor * 2, 0, floor * 2))
    path = tmp_path / 'checkpoint.pt'
    owner = PersistentCheckpointReservation(path, transfer_bytes=0, minimum_free_bytes=floor).acquire(old_bound)
    inode = owner.pending.stat().st_ino
    allocated = owner._allocated_bound
    assert allocated >= old_bound + 16 * 1024
    assert owner.pending.stat().st_blocks * 512 >= allocated
    monkeypatch.setattr(checkpoint_safety.shutil, 'disk_usage', lambda _: usage(floor * 2, floor + 1, floor - 1))
    atomic_checkpoint(path, current, reservation=owner)
    assert path.stat().st_ino == inode
    assert path.stat().st_size <= new_bound
    assert torch.load(path, weights_only=True) == current
    owner.close()


def test_large_reservation_adds_bounded_aligned_headroom_but_small_reservation_does_not(tmp_path, monkeypatch):
    calls = []
    def record_allocation(fd, offset, length):
        calls.append((offset, length))
        os.ftruncate(fd, offset + length)
    monkeypatch.setattr(os, 'posix_fallocate', record_allocation)
    small = reservation(tmp_path / 'small.pt').acquire(1024)
    assert calls[-1] == (0, 1024)
    small.close()
    bound = 64 * 1024 * 1024 + 123
    large = reservation(tmp_path / 'large.pt').acquire(bound)
    capacity = calls[-1][1]
    assert calls[-1][0] == 0 and capacity % 4096 == 0
    assert 16 * 1024 * 1024 <= capacity - bound < 16 * 1024 * 1024 + 4096
    assert large.bound == bound and large._allocated_bound == capacity
    large.close()


def test_growth_past_preallocated_headroom_still_requires_admission(tmp_path, monkeypatch):
    from collections import namedtuple
    from natlang_neuralese.train import checkpoint_safety
    monkeypatch.setattr(checkpoint_safety, '_PERSISTENT_HEADROOM_THRESHOLD', 32 * 1024)
    monkeypatch.setattr(checkpoint_safety, '_PERSISTENT_METADATA_HEADROOM', 16 * 1024)
    floor = 40 * 1024**3
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(checkpoint_safety.shutil, 'disk_usage', lambda _: usage(floor * 2, 0, floor * 2))
    path = tmp_path / 'checkpoint.pt'
    torch.save({'step': 70}, path)
    owner = PersistentCheckpointReservation(path, transfer_bytes=0, minimum_free_bytes=floor).acquire(64 * 1024)
    pending_inode = owner.pending.stat().st_ino
    monkeypatch.setattr(checkpoint_safety.shutil, 'disk_usage', lambda _: usage(floor * 2, floor + 1, floor - 1))
    with pytest.raises(checkpoint_safety.CheckpointSpaceUnavailable):
        owner.acquire(owner._allocated_bound + 1)
    assert owner.pending.stat().st_ino == pending_inode
    assert torch.load(path, weights_only=True) == {'step': 70}
    owner.close()


def test_fresh_composite_native_optimizer_lazy_bound():
    from natlang_neuralese.train.optim import PortMuonAdamW
    ordinary = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    latent = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.bfloat16))
    optimizer = PortMuonAdamW([('norm', ordinary)], lr=1e-3, vocab_size=64,
        latent=[('latent', latent, 1e-3, 1)], latent_optimizer='muonsr',
        auxiliary_rounding='stochastic')
    bound = optimizer_state_size_upper_bound(optimizer, lazy=True)
    assert bound >= 2 * ordinary.numel() * 2 + 4 + latent.numel() * 2
    assert optimizer_state_size_upper_bound(optimizer, lazy=False) == 0


def test_recovered_hardlink_to_parent_cannot_overwrite_parent(tmp_path):
    import json
    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(b'last durable checkpoint')
    owner = reservation(path).acquire(1024)
    owner.close()
    owner.pending.unlink()
    os.link(path, owner.pending)
    st = path.stat()
    owner.receipt.write_text(json.dumps({'schema':'natlang.checkpoint-reservation/1',
                                        'device':st.st_dev, 'inode':st.st_ino}))
    with pytest.raises(CheckpointReserveError, match='link count'):
        reservation(path).acquire(1024)
    assert path.read_bytes() == b'last durable checkpoint'


def test_sparse_crash_recovery_rechecks_full_admission(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple
    from natlang_neuralese.train.checkpoint_safety import CheckpointSpaceUnavailable
    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(b'last durable checkpoint')
    owner = reservation(path).acquire(1024 * 1024)
    owner.close()
    with owner.pending.open('wb') as stream:
        stream.truncate(1024 * 1024)
    Usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: Usage(100, 100, 0))
    retry = reservation(path)
    with pytest.raises(CheckpointSpaceUnavailable):
        retry.acquire(1024 * 1024)
    assert path.read_bytes() == b'last durable checkpoint'
    retry.close()


def test_post_rename_receipt_enospc_retries_without_losing_parent(tmp_path, monkeypatch):
    import errno
    from natlang_neuralese.train.checkpoint_safety import save_with_space_retry
    path = tmp_path / 'checkpoint.pt'
    torch.save({'step': 1}, path)
    owner = reservation(path).acquire(20 * 1024 * 1024)
    real_unlink = Path.unlink
    attempts = []
    def unlink_once(target, *args, **kwargs):
        if target == owner.receipt and not attempts:
            attempts.append(True)
            raise OSError(errno.ENOSPC, 'injected receipt failure')
        return real_unlink(target, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink_once)
    save_with_space_retry(lambda: atomic_checkpoint(path, {'step': 2}, reservation=owner),
                          sleep=lambda _: None)
    assert attempts
    assert torch.load(path, weights_only=True)['step'] == 2
    owner.close()


def test_alternate_slot_crash_reopens_with_exact_inode(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    primary = reservation(path).acquire(1024 * 1024)
    alternate = PersistentCheckpointReservation(path, slot_suffix='.alternate', shared_lock=primary,
                                                 transfer_bytes=0, minimum_free_bytes=0).acquire(1024 * 1024)
    inode = alternate.pending.stat().st_ino
    primary.close()
    alternate.close()
    reopened = PersistentCheckpointReservation(path, slot_suffix='.alternate', transfer_bytes=0,
                                                minimum_free_bytes=0).acquire(1024 * 1024)
    assert reopened.pending.stat().st_ino == inode
    reopened.close()


def test_alternate_admission_accounts_for_its_own_space(tmp_path, monkeypatch):
    from collections import namedtuple
    from natlang_neuralese.train import checkpoint_safety
    path = tmp_path / 'checkpoint.pt'
    torch.save({'step': 1}, path)
    primary = reservation(path).acquire(1024 * 1024)
    inode = primary.pending.stat().st_ino
    alternate = PersistentCheckpointReservation(path, slot_suffix='.alternate', shared_lock=primary,
                                                 transfer_bytes=0, minimum_free_bytes=0)
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(checkpoint_safety.shutil, 'disk_usage', lambda _: usage(1000, 999, 1))
    with pytest.raises(checkpoint_safety.CheckpointSpaceUnavailable):
        alternate.acquire(1024 * 1024)
    assert primary.pending.stat().st_ino == inode and primary.bound == 1024 * 1024
    assert torch.load(path, weights_only=True)['step'] == 1
    alternate.close()
    primary.close()


def test_gb10_floor_applies_only_to_configured_hot_paths(tmp_path,monkeypatch):
    from natlang_neuralese.train.checkpoint_safety import checkpoint_minimum_free_bytes
    data=tmp_path/'data'; data.mkdir();repo=tmp_path/'repo';(repo/'runs').mkdir(parents=True)
    monkeypatch.setenv('NATLANG_DATA_NVME',str(data));monkeypatch.setenv('NATLANG_REPO',str(repo))
    monkeypatch.setenv('NATLANG_CHECKPOINT_DGX_NVME_MIN_FREE_BYTES',str(40*1024**3));monkeypatch.delenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES',raising=False)
    assert checkpoint_minimum_free_bytes(data/'training/checkpoint.pt')==40*1024**3
    assert checkpoint_minimum_free_bytes(repo/'runs/job/checkpoint.pt')==40*1024**3
    assert checkpoint_minimum_free_bytes(tmp_path/'elsewhere/checkpoint.pt')==512*1024**2
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES','123')
    assert checkpoint_minimum_free_bytes(data/'training/checkpoint.pt')==123
