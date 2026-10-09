"""Shared checkpoint policy (plans/STORAGE_POLICY.md, training/storage-policy.json "checkpoint_cadence").

One rolling resumable slot written on a wall-clock cadence (and when the process is asked to stop), a weights-only
"best" snapshot on eval improvement, and at the end of a run the resumable slot (optimizer state included) gives way
to a weights-only final file. Writes are atomic (pending file, then rename) and disk-aware: when the disk cannot hold
the new slot next to the old one, the old one goes first, so a full disk never kills a run.

    policy = CheckpointPolicy(out, every_minutes=45).install_signal_handlers()
    for step in ...:
        ...
        if policy.due():
            policy.save({"step": step, "weights": ..., "optimizer": ...})
            if policy.signaled:
                break
        if improved:
            policy.save_best({"step": step, "weights": ...}, metric=held_ce)
    policy.finalize({"step": step, "weights": ...})
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import time
from pathlib import Path

import torch


class CheckpointPolicy:
    def __init__(self, out_dir, *, every_minutes: float = 45.0, name: str = "checkpoint.pt",
                 best_name: str = "best-weights.pt", final_name: str = "final-weights.pt",
                 free_factor: float = 1.2, clock=time.monotonic):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.every = every_minutes * 60.0
        self.slot, self.best_path, self.final_path = self.out / name, self.out / best_name, self.out / final_name
        self.free_factor = free_factor
        self.clock = clock
        self.last = clock()
        self.signaled = False
        self.best_metric: float | None = None
        self.events: list[dict] = []
        meta = self.out / "checkpoint-policy.json"
        if meta.exists():  # a resumed run keeps its best metric
            try:
                self.best_metric = json.loads(meta.read_text()).get("best_metric")
            except ValueError:
                pass

    # Cadence ----------------------------------------------------------------------------------------------------
    def install_signal_handlers(self, signals=(signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)):
        """Ask for a checkpoint at the next ``due()`` check instead of dying mid-step."""
        def handler(signum, frame):
            self.signaled = True
        for sig in signals:
            signal.signal(sig, handler)
        return self

    def due(self) -> bool:
        return self.signaled or self.clock() - self.last >= self.every

    # Writes -----------------------------------------------------------------------------------------------------
    def _write(self, state: dict, target: Path, *, replaceable: Path | None = None) -> dict:
        """Atomic, disk-aware ``torch.save``. ``replaceable``: an existing file this write supersedes, removed first
        when the disk cannot hold both."""
        expected = (replaceable.stat().st_size if replaceable and replaceable.exists() else 0) or \
            (target.stat().st_size if target.exists() else 0)
        dropped = None
        if replaceable and replaceable.exists() and shutil.disk_usage(self.out).free < self.free_factor * expected:
            replaceable.unlink()
            dropped = str(replaceable)
        pending = target.with_name(target.name + ".pending")
        # GPU tensors are copied to host one storage at a time while writing; no second full copy in memory.
        torch.save(state, pending)
        os.replace(pending, target)
        event = {"time": time.time(), "path": str(target), "bytes": target.stat().st_size, "dropped_first": dropped}
        self.events.append(event)
        return event

    def save(self, state: dict) -> dict:
        """The rolling resumable slot (weights + optimizer + schedule + RNG, whatever the trainer passes)."""
        event = self._write(state, self.slot, replaceable=self.slot)
        self.last = self.clock()
        return event

    def save_best(self, weights_state: dict, metric: float, higher_is_better: bool = False) -> bool:
        """A weights-only snapshot when ``metric`` improves on the best so far. Returns whether it was written."""
        better = self.best_metric is None or (metric > self.best_metric if higher_is_better else metric < self.best_metric)
        if not better:
            return False
        self._write(dict(weights_state, metric=metric), self.best_path, replaceable=self.best_path)
        self.best_metric = metric
        (self.out / "checkpoint-policy.json").write_text(json.dumps({"best_metric": metric}) + "\n")
        return True

    def finalize(self, weights_state: dict | None = None, *, drop_resumable: bool = True) -> dict:
        """End of run: write the weights-only final file and drop the resumable slot (its optimizer state is only
        needed to continue this exact run; a continuation starts from weights with its own optimizer)."""
        event = self._write(weights_state, self.final_path) if weights_state is not None else {}
        if drop_resumable and self.slot.exists():
            freed = self.slot.stat().st_size
            self.slot.unlink()
            event = dict(event, dropped_resumable=str(self.slot), freed_bytes=freed)
        return event

    @staticmethod
    def load(path, map_location="cpu") -> dict:
        """Load paged from the file (mmap): no second full copy in (unified) memory."""
        return torch.load(path, map_location=map_location, mmap=True)
