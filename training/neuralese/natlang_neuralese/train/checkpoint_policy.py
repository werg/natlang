"""Shared checkpoint policy (plans/STORAGE_POLICY.md, training/storage-policy.json "checkpoint_cadence").

One rolling resumable slot written on a wall-clock cadence (default 3 h, owner 2026-10-10: "checkpoint every few
hours"; it only bounds crash loss) and when the process is asked to stop (SIGTERM: the full state, within the
stop grace the memory ledger gives every unit), a weights-only
"best" snapshot on eval improvement, and a weights-only final export alongside the
preserved resumable slot. The slot with its optimizer state is the continuation parent: a weights-only
final never replaces or deletes it, and nothing deletes the only complete checkpoint before a pending
replacement is durably in place. Writes are atomic and durable (pending, fsync, rename) through the one writer
every trainer uses (trajectory_state.atomic_checkpoint). Insufficient disk space fails the save without
deleting the previous checkpoint. Pruning is a separate, explicitly authorized operation.
Cadence and signals are the training-loop skeleton's (train/loop.py).

    policy = CheckpointPolicy(out, every_minutes=180).install_signal_handlers()
    for step in ...:
        ...
        if policy.due():
            policy.save({"step": step, "weights": ..., "optimizer": ...})
            if policy.signaled:
                break
        if improved:
            policy.save_best({"step": step, "weights": ...}, metric=held_ce)
    policy.finalize({"step": step, "weights": ...}, resumable_state=latest_full_state)
"""
from __future__ import annotations

import json
import errno
import os
import shutil
import signal
import time
from pathlib import Path

import torch

from .loop import Cadence, StopSignal
from .trajectory_state import atomic_checkpoint


class CheckpointPolicy:
    def __init__(self, out_dir, *, every_minutes: float = 180.0, name: str = "checkpoint.pt",
                 best_name: str = "best-weights.pt", final_name: str = "final-weights.pt",
                 free_factor: float = 1.2, clock=time.monotonic):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cadence = Cadence(every_minutes=every_minutes, clock=clock)
        self.stop = StopSignal()
        self.slot, self.best_path, self.final_path = self.out / name, self.out / best_name, self.out / final_name
        if len({name, best_name, final_name}) != 3:
            raise ValueError('the rolling slot, best and final snapshots need distinct file names: a weights-only '
                             'file must never share the optimizer-bearing continuation slot')
        self.free_factor = free_factor
        self.clock = clock
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
        self.stop.install(signals)
        return self

    @property
    def signaled(self) -> bool:
        return self.stop.requested

    def due(self) -> bool:
        return self.stop.requested or self.cadence.due(0)

    # Writes -----------------------------------------------------------------------------------------------------
    def _write(self, state: dict, target: Path) -> dict:
        """Preserve the previous slot until the replacement is completely written."""
        expected = target.stat().st_size if target.exists() else 0
        if expected and shutil.disk_usage(self.out).free < self.free_factor * expected:
            raise OSError(errno.ENOSPC, 'insufficient space for atomic checkpoint replacement; previous file preserved', str(target))
        # GPU tensors are copied to host one storage at a time while writing; no second full copy in memory.
        atomic_checkpoint(target, state)
        event = {"time": time.time(), "path": str(target), "bytes": target.stat().st_size,
                 "previous_preserved_until_replace": True}
        self.events.append(event)
        return event

    def save(self, state: dict) -> dict:
        """The rolling resumable slot (weights + optimizer + schedule + RNG, whatever the trainer passes)."""
        event = self._write(state, self.slot)
        self.cadence.mark()
        return event

    def save_best(self, weights_state: dict, metric: float, higher_is_better: bool = False) -> bool:
        """A weights-only snapshot when ``metric`` improves on the best so far. Returns whether it was written."""
        better = self.best_metric is None or (metric > self.best_metric if higher_is_better else metric < self.best_metric)
        if not better:
            return False
        self._write(dict(weights_state, metric=metric), self.best_path)
        self.best_metric = metric
        (self.out / "checkpoint-policy.json").write_text(json.dumps({"best_metric": metric}) + "\n")
        return True

    def finalize(self, weights_state: dict | None = None, *, resumable_state: dict | None = None,
                 drop_resumable: bool = False) -> dict:
        """Keep optimizer continuation by default; pass the final full state to save its exact final step.

        Without ``resumable_state``, the last rolling checkpoint is retained.
        Disposal must be explicitly chosen by the owner and follows a successful
        final weights export; it is never a disk-pressure fallback.
        """
        if drop_resumable and weights_state is None:
            raise ValueError('dropping resumable state requires a successful final weights export')
        if resumable_state is not None:
            self.save(resumable_state)
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
