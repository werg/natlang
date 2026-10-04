"""The resumable port trainer (S3 §4).

Trains the port modules and control rows through the phases; the backbone stays frozen
(phases A–E). Checkpoints are written atomically at optimizer-step boundaries and on
SIGTERM, and a run resumes from its output directory. Metrics go to `metrics.jsonl`
(one line per step) for whatever reporting the run uses.

Optimiser: AdamW over the port modules and phase-F adapter deltas. Muon integration
for a new full port run remains a prerequisite in the full-run handoff plan.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import time
from pathlib import Path

import torch

from ..model.heads import PortHeads
from ..model.lfm2_port import PortBackbone
from .adapters import adapter_layers, deltas_off, inject_lora, lora_state
from .losses import consumer_batch_loss, distill_loss, replay_loss, span_loss
from .phases import Phase


def trainable_parameters(backbone: PortBackbone, heads: PortHeads):
    return [backbone.control_rows, *(p for p in heads.parameters() if p.requires_grad)]


class Trainer:
    def __init__(self, backbone: PortBackbone, heads: PortHeads, phases: list[Phase], out_dir: str | Path,
                 span_train=None, records_train=None, seed: int = 0, grad_clip: float = 1.0,
                 checkpoint_every: int = 100, eval_fn=None, eval_every: int | None = None, log=print,
                 stop_after_phase: str | None = None):
        self.backbone, self.heads, self.phases = backbone, heads, phases
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.span_train = list(span_train or [])
        self.records_train = list(records_train or [])
        self.seed, self.grad_clip = seed, grad_clip
        self.checkpoint_every = checkpoint_every
        self.eval_fn, self.eval_every = eval_fn, eval_every
        self.log = log
        if stop_after_phase is None:
            self.phase_limit = len(phases)
        else:
            self.phase_limit = next((i + 1 for i, phase in enumerate(phases) if phase.name == stop_after_phase), None)
            if self.phase_limit is None:
                raise ValueError(f'Unknown phase boundary: {stop_after_phase}')
        self.params = trainable_parameters(backbone, heads)
        self.optimizer = torch.optim.AdamW(self.params, lr=phases[0].lr if phases else 1e-3, weight_decay=0.0)
        self.generator = torch.Generator().manual_seed(seed)
        self.global_step, self.phase_index, self.phase_step = 0, 0, 0
        self._stop_requested = False
        self._lora_groups: set[int] = set()
        self._lora_group_order: list[int] = []
        self._resume()

    # Checkpoints -------------------------------------------------------------------------
    @property
    def checkpoint_path(self) -> Path:
        return self.out / "checkpoint.pt"

    def save(self):
        rank = next((p.lora_rank for p in self.phases if p.lora_layers), 16)
        state = {
            "port_config": {"cutoff": self.heads.cutoff, "max_length": self.heads.max_length,
                            "lora_alpha": 2 * rank},
            "heads": self.heads.state_dict(),
            "control_rows": self.backbone.control_rows.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step, "phase_index": self.phase_index, "phase_step": self.phase_step,
            "generator": self.generator.get_state(),
            "phases": [p.to_dict() for p in self.phases],
            "lora": lora_state(self.backbone),
            "lora_layers": adapter_layers(self.backbone),
            "optimizer_lora_layers": self._lora_group_order,
            "optimizer_lora_parameter_names": self._lora_parameter_names(),
            "lora_rank": rank,
        }
        pending = self.checkpoint_path.with_suffix(".pending")
        torch.save(state, pending)
        os.replace(pending, self.checkpoint_path)

    def _resume(self):
        if not self.checkpoint_path.exists():
            return
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        metadata = state['port_config']
        if (metadata['cutoff'] != self.heads.cutoff or
                metadata['max_length'] != self.heads.max_length):
            raise ValueError('Checkpoint port configuration differs from this run')
        saved_phases = state['phases']
        if not saved_phases:
            raise ValueError('Checkpoint has no phase schedule')
        index, step = state['phase_index'], state['phase_step']
        if (not isinstance(index, int) or not isinstance(step, int) or
                not 0 <= index <= len(saved_phases) or step < 0 or
                (index == len(saved_phases) and step != 0) or
                (index < len(saved_phases) and step > saved_phases[index]['steps'])):
            raise ValueError('Checkpoint phase position is invalid')
        if saved_phases != [p.to_dict() for p in self.phases[:len(saved_phases)]]:
            raise ValueError('Checkpoint phase schedule differs; preserve this run before changing its schedule')
        if state.get("lora_layers"):
            phase = next(p for p in self.phases if p.lora_layers)
            order = state.get("optimizer_lora_layers")
            if (order is None or len(order) != len(set(order)) or
                    set(order) != set(state["lora_layers"])):
                raise ValueError('LoRA checkpoint lacks an unambiguous optimizer group order; preserve it for explicit repair')
            self._add_lora(phase, order, state["lora_rank"])
            if state.get("optimizer_lora_parameter_names") != self._lora_parameter_names():
                raise ValueError('Checkpoint LoRA optimizer parameter identities differ')
            params = dict(self.backbone.hf.named_parameters())
            with torch.no_grad():
                for name, value in state["lora"].items():
                    params[name].copy_(value.to(params[name]))
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

    # Phase F deltas --------------------------------------------------------------------------
    def _add_lora(self, phase: Phase, layers, rank: int):
        """Inject LoRA into `layers` and add one optimizer group per layer (upper layers higher lr)."""
        layers = [int(i) for i in layers]
        grouped = inject_lora(self.backbone, layers, rank=rank, alpha=2 * rank)
        order = sorted(set(phase.lora_layers) | set(layers), reverse=True)
        for layer in layers:
            params = grouped[layer]
            if layer in self._lora_groups:
                continue
            depth_rank = order.index(layer) if layer in order else len(order)
            scale = phase.lora_lr_scale * phase.lora_layer_decay ** depth_rank
            self.optimizer.add_param_group({"params": params, "lr": phase.lr * scale, "lr_scale": scale,
                                            "weight_decay": 0.0})
            self.params.extend(params)
            self._lora_groups.add(layer)
            self._lora_group_order.append(layer)

    def _lora_parameter_names(self) -> list[list[str]]:
        names = {id(param): name for name, param in self.backbone.hf.named_parameters()}
        return [[names[id(param)] for param in group["params"]]
                for group in self.optimizer.param_groups[1:]]

    def _release_layers(self, phase: Phase):
        """Release `phase.lora_layers` gradually: one more layer every steps/len(layers) steps."""
        if not phase.lora_layers:
            return
        per = max(1, phase.steps // len(phase.lora_layers))
        wanted = list(phase.lora_layers)[: 1 + self.phase_step // per]
        missing = [i for i in wanted if i not in self._lora_groups]
        if missing:
            self._add_lora(phase, missing, phase.lora_rank)
            self.log(f"phase {phase.name}: released layers {missing} at step {self.global_step}")

    def _teacher_context(self):
        return deltas_off(self.backbone)

    # Training ------------------------------------------------------------------------------
    def _consumer_step(self, phase: Phase, batch) -> tuple[torch.Tensor, dict]:
        loss, metrics = consumer_batch_loss(
            self.backbone, self.heads, batch, kl_weight=phase.kl_weight,
            contrastive_weight=phase.contrastive_weight, margin=phase.margin,
            diversity_weight=phase.diversity_weight, max_length=phase.max_length,
            temperature=phase.temperature(self.phase_step), payload_kl_weight=phase.payload_kl_weight,
            stop_policy_weight=phase.stop_policy_weight, length_cost=phase.length_cost,
            policy_samples=phase.policy_samples, generator=self.generator,
            teacher_context=self._teacher_context, stop_exploration=phase.stop_exploration,
            stop_temperature=phase.stop_temperature, stop_ratio_clip=phase.stop_ratio_clip)
        if phase.text_replay_weight > 0 and self.span_train:
            spans = [self.span_train[(self.global_step * phase.batch_size + i) % len(self.span_train)]
                     for i in range(phase.batch_size)]
            replay, replay_metrics = replay_loss(self.backbone, spans, self._teacher_context)
            loss = loss + phase.text_replay_weight * replay
            metrics.update(replay_metrics)
            metrics["loss"] = loss.item()
        return loss, metrics

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
        if phase.name in ("D", "E", "F"):
            return self._consumer_step(phase, batch)
        raise ValueError(f"unknown phase {phase.name!r}")

    def run(self) -> Path:
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop_requested", True))
        metrics_file = open(self.out / "metrics.jsonl", "a")
        try:
            while self.phase_index < self.phase_limit:
                phase = self.phases[self.phase_index]
                items = self.records_train if phase.name in ("D", "E", "F") else self.span_train
                batches = self._batches(items, phase.batch_size, self.phase_step)
                while self.phase_step < phase.steps:
                    self._release_layers(phase)
                    for group in self.optimizer.param_groups:
                        group["lr"] = phase.lr * group.get("lr_scale", 1.0)
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
