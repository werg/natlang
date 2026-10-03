"""The resumable port trainer (S3 §4).

Trains the port modules and control rows through the phases; the backbone stays frozen
(phases A–E). Checkpoints are written atomically at optimizer-step boundaries and on
SIGTERM, and a run resumes from its output directory. Metrics go to `metrics.jsonl`
(one line per step) for whatever reporting the run uses.

Optimiser: AdamW over the port modules. The repository's Muon path
(`scripts/training_optimizers.py`) applies to the 2-D matrices once the shared backbone is
released in phase F.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from ..model.heads import PortHeads
from ..model.lfm2_port import PortBackbone
from .losses import consumer_loss, distill_loss, span_loss
from .phases import Phase


@contextmanager
def deltas_off(model):
    """Disable trainable adapters for the self-distillation teacher (phase F; a no-op before)."""
    disable = getattr(model, "disable_adapter", None)
    if disable is None:
        yield
    else:
        with disable():
            yield


def trainable_parameters(backbone: PortBackbone, heads: PortHeads):
    return [backbone.control_rows, *(p for p in heads.parameters() if p.requires_grad)]


class Trainer:
    def __init__(self, backbone: PortBackbone, heads: PortHeads, phases: list[Phase], out_dir: str | Path,
                 span_train=None, records_train=None, seed: int = 0, grad_clip: float = 1.0,
                 checkpoint_every: int = 100, eval_fn=None, eval_every: int | None = None, log=print):
        self.backbone, self.heads, self.phases = backbone, heads, phases
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.span_train = list(span_train or [])
        self.records_train = list(records_train or [])
        self.seed, self.grad_clip = seed, grad_clip
        self.checkpoint_every = checkpoint_every
        self.eval_fn, self.eval_every = eval_fn, eval_every
        self.log = log
        self.params = trainable_parameters(backbone, heads)
        self.optimizer = torch.optim.AdamW(self.params, lr=phases[0].lr if phases else 1e-3, weight_decay=0.0)
        self.generator = torch.Generator().manual_seed(seed)
        self.global_step, self.phase_index, self.phase_step = 0, 0, 0
        self._stop_requested = False
        self._resume()

    # Checkpoints -------------------------------------------------------------------------
    @property
    def checkpoint_path(self) -> Path:
        return self.out / "checkpoint.pt"

    def save(self):
        state = {
            "heads": self.heads.state_dict(),
            "control_rows": self.backbone.control_rows.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step, "phase_index": self.phase_index, "phase_step": self.phase_step,
            "generator": self.generator.get_state(),
            "phases": [p.to_dict() for p in self.phases],
        }
        pending = self.checkpoint_path.with_suffix(".pending")
        torch.save(state, pending)
        os.replace(pending, self.checkpoint_path)

    def _resume(self):
        if not self.checkpoint_path.exists():
            return
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.heads.load_state_dict(state["heads"])
        with torch.no_grad():
            self.backbone.control_rows.copy_(state["control_rows"].to(self.backbone.control_rows))
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step, self.phase_index, self.phase_step = state["global_step"], state["phase_index"], state["phase_step"]
        self.generator.set_state(state["generator"])
        self.log(f"resumed at step {self.global_step} (phase {self.phase_index}, step {self.phase_step})")

    # Data ----------------------------------------------------------------------------------
    def _batches(self, items, size: int, offset: int):
        if not items:
            raise ValueError("phase has no training data")
        cycle = itertools.cycle(items)
        for _ in range((offset * size) % len(items)):
            next(cycle)
        while True:
            yield [next(cycle) for _ in range(size)]

    # Training ------------------------------------------------------------------------------
    def _step_loss(self, phase: Phase, batch) -> tuple[torch.Tensor, dict]:
        if phase.name == "A":
            return span_loss(self.backbone, self.heads, batch, generated_fraction=0.0,
                             entry_weight=phase.entry_weight, stop_weight=phase.stop_weight,
                             kl_weight=phase.kl_weight, temperature=phase.temperature(self.phase_step),
                             payload_kl_weight=phase.payload_kl_weight, generator=self.generator)
        if phase.name == "B":
            return distill_loss(self.backbone, self.heads, batch)
        if phase.name == "C":
            loss, metrics = span_loss(
                self.backbone, self.heads, batch, generated_fraction=phase.generated_fraction(self.phase_step),
                passes=phase.passes, unroll=phase.unrolled(self.phase_step), entry_weight=phase.entry_weight,
                stop_weight=phase.stop_weight, kl_weight=phase.kl_weight, generator=self.generator,
                temperature=phase.temperature(self.phase_step), payload_kl_weight=phase.payload_kl_weight)
            if phase.replay_weight > 0:
                replay, replay_metrics = distill_loss(self.backbone, self.heads, batch)
                loss = loss + phase.replay_weight * replay
                metrics.update({f"replay_{k}": v for k, v in replay_metrics.items()})
                metrics["loss"] = loss.item()
            return loss, metrics
        if phase.name == "D":
            total, merged = 0.0, {}
            for rendered in batch:
                loss, metrics = consumer_loss(self.backbone, self.heads, rendered, kl_weight=phase.kl_weight,
                                              temperature=phase.temperature(self.phase_step),
                                              payload_kl_weight=phase.payload_kl_weight, generator=self.generator)
                total = total + loss / len(batch)
                for key, value in metrics.items():
                    merged[key] = merged.get(key, 0.0) + value / len(batch)
            return total, merged
        raise ValueError(f"unknown phase {phase.name!r}")

    def run(self) -> Path:
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop_requested", True))
        metrics_file = open(self.out / "metrics.jsonl", "a")
        try:
            while self.phase_index < len(self.phases):
                phase = self.phases[self.phase_index]
                items = self.records_train if phase.name == "D" else self.span_train
                batches = self._batches(items, phase.batch_size, self.phase_step)
                for group in self.optimizer.param_groups:
                    group["lr"] = phase.lr
                while self.phase_step < phase.steps:
                    started = time.perf_counter()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, metrics = self._step_loss(phase, next(batches))
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
                    self.optimizer.step()
                    self.phase_step += 1
                    self.global_step += 1
                    record = {"step": self.global_step, "phase": phase.name, "phase_step": self.phase_step,
                              "lr": phase.lr, "grad_norm": float(norm), "seconds": time.perf_counter() - started, **metrics}
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    if self.global_step % self.checkpoint_every == 0:
                        self.save()
                    if self.eval_fn and self.eval_every and self.global_step % self.eval_every == 0:
                        self.eval_fn(self, f"step{self.global_step}")
                    if self._stop_requested:
                        self.save()
                        self.log("SIGTERM: checkpoint saved at an optimizer-step boundary")
                        return self.checkpoint_path
                self.phase_index += 1
                self.phase_step = 0
                self.save()
                self.log(f"phase {phase.name} finished at step {self.global_step}")
            return self.checkpoint_path
        finally:
            metrics_file.close()
            signal.signal(signal.SIGTERM, previous)
