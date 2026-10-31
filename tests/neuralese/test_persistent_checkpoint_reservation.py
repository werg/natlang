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
