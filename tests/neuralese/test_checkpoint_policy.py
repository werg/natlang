import os
import signal
from collections import namedtuple

import torch

from natlang_neuralese.train import checkpoint_policy as cp


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_cadence_is_wall_clock_and_a_signal_makes_a_checkpoint_due(tmp_path):
    clock = Clock()
    policy = cp.CheckpointPolicy(tmp_path, every_minutes=45, clock=clock).install_signal_handlers((signal.SIGUSR1,))
    assert not policy.due()
    clock.now = 44 * 60
    assert not policy.due()
    clock.now = 45 * 60
    assert policy.due()
    policy.save({"step": 1, "w": torch.ones(3)})
    assert not policy.due()
    os.kill(os.getpid(), signal.SIGUSR1)
    assert policy.signaled and policy.due()
    assert torch.equal(cp.CheckpointPolicy.load(tmp_path / "checkpoint.pt")["w"], torch.ones(3))


def test_best_snapshot_only_on_improvement_and_survives_a_resume(tmp_path):
    policy = cp.CheckpointPolicy(tmp_path)
    assert policy.save_best({"w": torch.zeros(2)}, metric=2.0)
    assert not policy.save_best({"w": torch.ones(2)}, metric=2.5)
    assert policy.save_best({"w": torch.ones(2)}, metric=1.5)
    assert torch.equal(torch.load(tmp_path / "best-weights.pt")["w"], torch.ones(2))
    assert cp.CheckpointPolicy(tmp_path).best_metric == 1.5


def test_a_short_disk_drops_the_old_slot_before_writing_and_finalize_drops_optimizer_state(tmp_path, monkeypatch):
    policy = cp.CheckpointPolicy(tmp_path)
    policy.save({"w": torch.ones(1000), "optimizer": torch.ones(1000)})
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(cp.shutil, "disk_usage", lambda path: Usage(10, 10, 1))
    event = policy.save({"w": torch.zeros(1000), "optimizer": torch.zeros(1000)})
    assert event["dropped_first"] == str(tmp_path / "checkpoint.pt")
    assert torch.equal(torch.load(tmp_path / "checkpoint.pt")["w"], torch.zeros(1000))
    assert not (tmp_path / "checkpoint.pt.pending").exists()
    done = policy.finalize({"w": torch.zeros(1000)})
    assert not (tmp_path / "checkpoint.pt").exists() and done["freed_bytes"] > 0
    assert set(torch.load(tmp_path / "final-weights.pt")) == {"w"}
