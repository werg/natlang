import json
import errno
import os
import signal
from collections import namedtuple

import pytest
import torch

from natlang_neuralese.train import checkpoint_policy as cp


def test_checkpoints_are_due_at_declared_step_points_the_end_and_a_signal(tmp_path):
    policy = cp.CheckpointPolicy(tmp_path, every_steps=2048).install_signal_handlers((signal.SIGUSR1,))
    assert [s for s in range(0, 8193) if policy.due(s, end=s == 6000)] == [2048, 4096, 6000, 6144, 8192]
    policy.save({"step": 1, "w": torch.ones(3)})
    assert not policy.due(1)
    os.kill(os.getpid(), signal.SIGUSR1)
    assert policy.signaled and policy.due(1)
    assert torch.equal(cp.CheckpointPolicy.load(tmp_path / "checkpoint.pt")["w"], torch.ones(3))
    assert not cp.CheckpointPolicy(tmp_path / "end-only").due(4096)  # 0: the end and stops only


def test_best_is_a_link_to_a_full_state_write_never_a_write_of_its_own(tmp_path, monkeypatch):
    writes = []
    real = cp.atomic_checkpoint
    monkeypatch.setattr(cp, "atomic_checkpoint", lambda path, state: (writes.append(path.name), real(path, state)))
    policy = cp.CheckpointPolicy(tmp_path)
    assert policy.save({"step": 1, "w": torch.zeros(2)}, metric=2.0)["best"]
    assert "best" not in policy.save({"step": 2, "w": torch.ones(2)}, metric=2.5)  # worse: the slot only
    assert (tmp_path / "best-checkpoint.pt").stat().st_ino != (tmp_path / "checkpoint.pt").stat().st_ino
    assert torch.load(tmp_path / "best-checkpoint.pt")["step"] == 1  # the linked inode outlived the replacement
    assert policy.save({"step": 3, "w": torch.ones(2)}, metric=1.5)["best"]
    assert (tmp_path / "best-checkpoint.pt").stat().st_ino == (tmp_path / "checkpoint.pt").stat().st_ino
    assert "best" not in policy.save({"step": 4, "w": torch.ones(2)})  # a stop write: no metric, never best
    assert writes == ["checkpoint.pt"] * 4  # no extra write for best
    assert cp.CheckpointPolicy(tmp_path).best_metric == 1.5  # survives a resume
    assert json.loads((tmp_path / "best-checkpoint.json").read_text())["step"] == 3


def test_evaluation_points_are_declared_steps_and_include_every_write_and_the_end():
    from natlang_neuralese.train.loop import Cadence, check_declared_points
    evals = Cadence(512)
    assert [s for s in range(0, 8193) if evals.due(s)] == list(range(512, 8193, 512))  # 16 points, none at step 0
    check_declared_points(8192, 512, 2048)
    with pytest.raises(ValueError, match="multiple of eval_every"):
        check_declared_points(8000, 512, 2048)  # the end would not be an evaluation point
    with pytest.raises(ValueError, match="checkpoint_every 1000 is not a multiple"):
        check_declared_points(8192, 512, 1000)  # a write without its own evaluation


def test_a_short_disk_fails_the_save_and_keeps_the_only_complete_checkpoint(tmp_path, monkeypatch):
    policy = cp.CheckpointPolicy(tmp_path)
    policy.save({"w": torch.ones(1000), "optimizer": torch.ones(1000)})
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(cp.shutil, "disk_usage", lambda path: Usage(10, 10, 1))
    with pytest.raises(OSError) as refused:
        policy.save({"w": torch.zeros(1000), "optimizer": torch.zeros(1000)})
    assert refused.value.errno == errno.ENOSPC
    # The previous slot is intact and nothing half-written is left behind.
    assert torch.equal(torch.load(tmp_path / "checkpoint.pt")["optimizer"], torch.ones(1000))
    assert not list(tmp_path.glob("*.pending"))


def test_a_failed_write_never_removes_the_previous_slot(tmp_path, monkeypatch):
    policy = cp.CheckpointPolicy(tmp_path)
    policy.save({"w": torch.ones(4), "optimizer": torch.ones(4)})

    def broken_save(state, stream):
        stream.write(b"partial")
        raise RuntimeError("disk died mid-write")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(RuntimeError, match="disk died"):
        policy.save({"w": torch.zeros(4), "optimizer": torch.zeros(4)})
    monkeypatch.undo()
    assert torch.equal(torch.load(tmp_path / "checkpoint.pt")["optimizer"], torch.ones(4))
    assert not list(tmp_path.glob("*.pending"))


def test_finalize_keeps_the_full_state_continuation_parent_by_default(tmp_path):
    policy = cp.CheckpointPolicy(tmp_path)
    policy.save({"step": 9, "w": torch.ones(8), "optimizer": torch.ones(8)})
    done = policy.finalize({"w": torch.ones(8)})
    assert "dropped_resumable" not in done
    assert set(torch.load(tmp_path / "final-weights.pt")) == {"w"}
    assert set(torch.load(tmp_path / "checkpoint.pt")) == {"step", "w", "optimizer"}


def test_finalize_can_save_the_exact_final_full_state_beside_the_weights_only_export(tmp_path):
    policy = cp.CheckpointPolicy(tmp_path)
    policy.save({"step": 9, "w": torch.ones(8), "optimizer": torch.ones(8)})
    policy.finalize({"w": torch.zeros(8)}, resumable_state={"step": 12, "w": torch.zeros(8), "optimizer": torch.zeros(8)})
    assert torch.load(tmp_path / "checkpoint.pt")["step"] == 12
    assert set(torch.load(tmp_path / "final-weights.pt")) == {"w"}


def test_dropping_the_optimizer_slot_is_explicit_and_follows_a_successful_final_export(tmp_path, monkeypatch):
    policy = cp.CheckpointPolicy(tmp_path)
    policy.save({"w": torch.ones(1000), "optimizer": torch.ones(1000)})
    with pytest.raises(ValueError, match="requires a successful final weights export"):
        policy.finalize(None, drop_resumable=True)
    assert (tmp_path / "checkpoint.pt").exists()

    def failing_write(path, state):
        raise OSError(errno.EIO, "final export failed")

    monkeypatch.setattr(cp, "atomic_checkpoint", failing_write)
    with pytest.raises(OSError):
        policy.finalize({"w": torch.zeros(1000)}, drop_resumable=True)
    assert (tmp_path / "checkpoint.pt").exists()  # the export failed, so nothing was disposed of
    monkeypatch.undo()
    done = policy.finalize({"w": torch.zeros(1000)}, drop_resumable=True)
    assert not (tmp_path / "checkpoint.pt").exists() and done["freed_bytes"] > 0


def test_the_weights_only_snapshots_can_not_share_the_continuation_slot_name(tmp_path):
    with pytest.raises(ValueError, match="distinct file names"):
        cp.CheckpointPolicy(tmp_path, final_name="checkpoint.pt")
    with pytest.raises(ValueError, match="distinct file names"):
        cp.CheckpointPolicy(tmp_path, best_name="final-weights.pt")


def test_the_trainers_atomic_writer_leaves_the_previous_file_until_the_replacement_is_complete(tmp_path, monkeypatch):
    from natlang_neuralese.train.trajectory_state import atomic_checkpoint
    target = tmp_path / "checkpoint.pt"
    atomic_checkpoint(target, {"v": torch.ones(2)})
    monkeypatch.setattr(torch, "save", lambda state, stream: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")))
    with pytest.raises(OSError):
        atomic_checkpoint(target, {"v": torch.zeros(2)})
    monkeypatch.undo()
    assert torch.equal(torch.load(target)["v"], torch.ones(2))
    assert not list(tmp_path.glob("*.pending"))
