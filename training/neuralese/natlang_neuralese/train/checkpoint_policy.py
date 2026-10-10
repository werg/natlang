"""Shared checkpoint policy (plans/STORAGE_POLICY.md, training/storage-policy.json "checkpoint_cadence").

Evaluations and full-state writes happen ONLY at declared step points (owner 2026-10-10): a stage declares
``eval_every`` (about 10-20 evenly spaced points per stage, the last at the stage end, where the stage gate runs) and
``checkpoint_every`` (a multiple of it, sized from the measured step time to about every 3 h). Step points are
reproducible and comparable across runs, resumes and machines; nothing runs on a wall clock. The rolling resumable slot
is written at the checkpoint points, at the stage end, and when the process is asked to stop (SIGTERM: the full state,
within the stop grace the memory ledger gives every unit; no evaluation). "Best" is the best AMONG THOSE WRITES: when the
evaluation of a write's point is the best so far, the written file is also published as ``best-checkpoint.pt`` by a hard
link (no second write, never an extra write or evaluation for best).
A weights-only final export may sit alongside the
preserved resumable slot. The slot with its optimizer state is the continuation parent: a weights-only
final never replaces or deletes it, and nothing deletes the only complete checkpoint before a pending
replacement is durably in place. Writes are atomic and durable (pending, fsync, rename) through the one writer
every trainer uses (trajectory_state.atomic_checkpoint). Insufficient disk space fails the save without
deleting the previous checkpoint. Pruning is a separate, explicitly authorized operation.
Points and signals are the training-loop skeleton's (train/loop.py).

    check_declared_points(steps, eval_every, checkpoint_every)
    policy = CheckpointPolicy(out, every_steps=checkpoint_every).install_signal_handlers()
    evals = Cadence(eval_every)
    for step in ...:
        ...
        metric = evaluate() if evals.due(step) else None
        if policy.due(step, end=step == steps):
            policy.save({"step": step, "weights": ..., "optimizer": ...}, metric=metric)
            if policy.signaled:
                break
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


def publish_best(slot: Path, best: Path, receipt: dict | None = None) -> None:
    """``best`` becomes the same file as ``slot`` (a hard link: no copy). A later atomic replacement of the slot leaves
    the linked inode, so the best survives the next rolling write. The receipt (if any) is written last."""
    pending = best.with_name(best.name + '.pending-link')
    pending.unlink(missing_ok=True)
    os.link(slot, pending)
    pending.replace(best)
    if receipt is not None:
        manifest = best.with_suffix('.json')
        temp = manifest.with_name(manifest.name + '.pending')
        temp.write_text(json.dumps(receipt, indent=2) + '\n')
        temp.replace(manifest)


class CheckpointPolicy:
    def __init__(self, out_dir, *, every_steps: int = 0, name: str = "checkpoint.pt",
                 best_name: str = "best-checkpoint.pt", final_name: str = "final-weights.pt",
                 free_factor: float = 1.2):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cadence = Cadence(every_steps)
        self.stop = StopSignal()
        self.slot, self.best_path, self.final_path = self.out / name, self.out / best_name, self.out / final_name
        if len({name, best_name, final_name}) != 3:
            raise ValueError('the rolling slot, best and final snapshots need distinct file names: a weights-only '
                             'file must never share the optimizer-bearing continuation slot')
        self.free_factor = free_factor
        self.best_metric: float | None = None
        self.events: list[dict] = []
        meta = self.out / "checkpoint-policy.json"
        if meta.exists():  # a resumed run keeps its best metric
            try:
                self.best_metric = json.loads(meta.read_text()).get("best_metric")
            except ValueError:
                pass

    # Points -----------------------------------------------------------------------------------------------------
    def install_signal_handlers(self, signals=(signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)):
        """Ask for a checkpoint at the next ``due()`` check instead of dying mid-step."""
        self.stop.install(signals)
        return self

    @property
    def signaled(self) -> bool:
        return self.stop.requested

    def due(self, step: int, *, end: bool = False) -> bool:
        """A declared checkpoint point, the stage end, or a stop request."""
        return self.stop.requested or self.cadence.due(step, force=end)

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

    def save(self, state: dict, *, metric: float | None = None, higher_is_better: bool = False) -> dict:
        """The rolling resumable slot (weights + optimizer + schedule + RNG, whatever the trainer passes). With
        ``metric`` (the evaluation of this write's step point) better than every earlier write's, the same file is
        also published as the best (``publish_best``); a stop write passes no metric."""
        event = self._write(state, self.slot)
        better = metric is not None and (self.best_metric is None or (
            metric > self.best_metric if higher_is_better else metric < self.best_metric))
        if better:
            publish_best(self.slot, self.best_path, {"step": state.get("step"), "metric": metric,
                                                     "file": self.slot.name})
            self.best_metric = metric
            (self.out / "checkpoint-policy.json").write_text(json.dumps({"best_metric": metric}) + "\n")
            event = dict(event, best=True)
        return event

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
