"""The resumable port trainer (S3 §4).

Trains the port modules and control rows through the phases; the backbone stays frozen
(phases A–E). Checkpoints are written atomically at optimizer-step boundaries and on
SIGTERM, and a run resumes from its output directory. Metrics go to `metrics.jsonl`
(one line per step) for whatever reporting the run uses.

Optimiser: the `optimizer` policy (train/optim.py). `adamw` is the A–F pilot lineage; `muon` puts hidden
matrices under Muon and the rest, including phase-F adapter deltas, under AdamW. The policy is saved with
the checkpoint and a run never resumes under another one.
"""

from __future__ import annotations

import itertools
import math
import json
import os
import signal
import time
from pathlib import Path

import torch

from ..model.heads import PortHeads
from ..model.lfm2_port import PortBackbone
from .adapters import adapter_layers, deltas_off, inject_lora, lora_state
from .optim import make_port_optimizer
from .losses import consumer_batch_loss, distill_loss, replay_loss, span_loss
from .phases import Phase
from ..data.shortcuts import audit_lengths, count_baseline, count_hazard


def trainable_parameters(backbone: PortBackbone, heads: PortHeads):
    return [backbone.control_rows, *(p for p in heads.parameters() if p.requires_grad)]


class Trainer:
    def __init__(self, backbone: PortBackbone, heads: PortHeads, phases: list[Phase], out_dir: str | Path,
                 span_train=None, records_train=None, seed: int = 0, grad_clip: float = 1.0,
                 checkpoint_every: int = 100, eval_fn=None, eval_every: int | None = None, log=print,
                 stop_after_phase: str | None = None, optimizer: str = "adamw", fail_on_shortcut: bool = False,
                 backbone_identity: dict | None = None):
        self.backbone, self.heads, self.phases = backbone, heads, phases
        # The frozen base the port trains on (base model, merged student LoRA and its hash): saved with every
        # checkpoint, so servers rebuild the same backbone and a resume cannot silently change it.
        self.backbone_identity = backbone_identity or {}
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.span_train = list(span_train or [])
        # A record stream (data/stream.py) instead of a list: batches come from it and its position is checkpointed.
        self.record_stream = records_train if hasattr(records_train, "state_dict") and hasattr(records_train, "take") else None
        self.records_train = [] if self.record_stream else list(records_train or [])
        self.seed, self.grad_clip = seed, grad_clip
        self.checkpoint_every = checkpoint_every
        self.eval_fn, self.eval_every = eval_fn, eval_every
        self.log = log
        self.fail_on_shortcut = fail_on_shortcut
        self._count_hazard: dict[int, float] | None = None
        if stop_after_phase is None:
            self.phase_limit = len(phases)
        else:
            self.phase_limit = next((i + 1 for i, phase in enumerate(phases) if phase.name == stop_after_phase), None)
            if self.phase_limit is None:
                raise ValueError(f'Unknown phase boundary: {stop_after_phase}')
        self.params = trainable_parameters(backbone, heads)
        self.optimizer_policy = optimizer
        self.optimizer = make_port_optimizer(optimizer, backbone, heads, lr=phases[0].lr if phases else 1e-3)
        self._base_groups = len(self.optimizer.param_groups)
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
                            "lora_alpha": 2 * rank, **self.heads.port_config()},
            "heads": self.heads.state_dict(),
            "control_rows": self.backbone.control_rows.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_policy": self.optimizer_policy,
            "global_step": self.global_step, "phase_index": self.phase_index, "phase_step": self.phase_step,
            "generator": self.generator.get_state(),
            "phases": [p.to_dict() for p in self.phases],
            "lora": lora_state(self.backbone),
            "lora_layers": adapter_layers(self.backbone),
            "optimizer_lora_layers": self._lora_group_order,
            "optimizer_lora_parameter_names": self._lora_parameter_names(),
            "lora_rank": rank,
            "backbone": self.backbone_identity,
        }
        if self.record_stream is not None:
            state["record_stream"] = self.record_stream.state_dict()
        pending = self.checkpoint_path.with_suffix(".pending")
        torch.save(state, pending)
        os.replace(pending, self.checkpoint_path)

    def _resume(self):
        if not self.checkpoint_path.exists():
            return
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        metadata = state['port_config']
        if (metadata['cutoff'] != self.heads.cutoff or
                metadata['max_length'] != self.heads.max_length or
                metadata.get('stop_source', 'shallow') != self.heads.stop_source or
                metadata.get('stop_position', True) != self.heads.stop.use_position):
            raise ValueError('Checkpoint port configuration differs from this run')
        # Checkpoints from before policies were recorded are AdamW; never reinterpret one as another policy.
        if state.get('optimizer_policy', 'adamw') != self.optimizer_policy:
            raise ValueError(f"Checkpoint optimiser policy {state.get('optimizer_policy', 'adamw')!r} differs from "
                             f"{self.optimizer_policy!r}; start a new run directory instead")
        if state.get("backbone", {}) != self.backbone_identity:
            raise ValueError(f"Checkpoint backbone {state.get('backbone', {})} differs from this run's "
                             f"{self.backbone_identity}; start a new run directory instead")
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
        if (self.record_stream is not None) != ("record_stream" in state):
            raise ValueError("Checkpoint and run disagree on whether training records come from a stream")
        if self.record_stream is not None:
            self.record_stream.load_state_dict(state["record_stream"])
        self.global_step, self.phase_index, self.phase_step = state["global_step"], state["phase_index"], state["phase_step"]
        self.generator.set_state(state["generator"])
        self.log(f"resumed at step {self.global_step} (phase {self.phase_index}, step {self.phase_step})")

    # Data ----------------------------------------------------------------------------------
    def _batches(self, items, size: int, offset: int):
        """Batches of equal span length (one bucket when lengths do not vary: plain cycling, as before).

        Buckets take turns in proportion to their size; batch b is determined by b alone, so a resumed phase
        continues exactly where it stopped.
        """
        if not items:
            raise ValueError("phase has no training data")
        buckets: dict = {}
        for item in items:
            buckets.setdefault(len(item.span) if hasattr(item, "span") else None, []).append(item)
        keys = sorted(buckets, key=lambda k: (k is None, k))
        if len(keys) == 1:
            cycle = itertools.cycle(items)
            for _ in range((offset * size) % len(items)):
                next(cycle)
            while True:
                yield [next(cycle) for _ in range(size)]
        # Smooth weighted round-robin over buckets by size; positions within a bucket advance per use.
        weights = [len(buckets[k]) for k in keys]
        credits, used = [0] * len(keys), [0] * len(keys)
        b = 0
        while True:
            for i, w in enumerate(weights):
                credits[i] += w
            chosen = max(range(len(keys)), key=lambda i: (credits[i], -i))
            credits[chosen] -= sum(weights)
            bucket = buckets[keys[chosen]]
            start = used[chosen] * size
            used[chosen] += 1
            if b >= offset:
                yield [bucket[(start + j) % len(bucket)] for j in range(size)]
            b += 1

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
                for group in self.optimizer.param_groups[self._base_groups:]]

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
            stop_temperature=phase.stop_temperature, stop_ratio_clip=phase.stop_ratio_clip,
            target_lengths=self._target_lengths(phase, batch), stop_weight=phase.stop_weight)
        if phase.text_replay_weight > 0 and self.span_train:
            spans = [self.span_train[(self.global_step * phase.batch_size + i) % len(self.span_train)]
                     for i in range(phase.batch_size)]
            replay, replay_metrics = replay_loss(self.backbone, spans, self._teacher_context)
            loss = loss + phase.text_replay_weight * replay
            metrics.update(replay_metrics)
            metrics["loss"] = loss.item()
        return loss, metrics

    def _target_lengths(self, phase: Phase, batch) -> list[int] | None:
        """Supervised block lengths from each record's source size (phase `tokens_per_vector`), or None."""
        if phase.tokens_per_vector <= 0 or phase.stop_policy_weight > 0:
            return None
        cap = min(phase.max_length or self.heads.max_length, self.heads.max_length)
        return [max(phase.min_length, min(cap, math.ceil(r.source_tokens / phase.tokens_per_vector))) for r in batch]

    # Shortcuts -----------------------------------------------------------------------------
    def _phase_lengths(self, phase: Phase, sample: int = 2000) -> list[int] | None:
        """The block lengths the phase supervises, from a sample of its data (None: lengths are not supervised)."""
        if phase.name in ("A", "C"):
            return [len(example.span) for example in self.span_train[:sample]] or None
        if phase.name in ("D", "E", "F") and self.records_train:
            return self._target_lengths(phase, self.records_train[:sample])
        return None

    def _audit_phase(self, phase: Phase):
        """Profile the phase's supervised lengths and the count-only stop baseline (data/shortcuts.py); record them in
        shortcuts.json, warn on flags, and keep the count hazard for the per-step baseline metric."""
        lengths = self._phase_lengths(phase)
        self._count_hazard = count_hazard(lengths) if lengths else None
        if not lengths:
            return
        report = audit_lengths(lengths, f"phase {phase.name}")
        path = self.out / "shortcuts.json"
        reports = json.loads(path.read_text()) if path.exists() else {}
        reports[phase.name] = report
        path.write_text(json.dumps(reports, indent=2) + "\n")
        for flag in report["flags"]:
            self.log(f"shortcut: {flag}")
        if report["flags"] and self.fail_on_shortcut:
            raise RuntimeError("shortcut in supervised lengths: " + "; ".join(report["flags"]))

    def _count_metric(self, phase: Phase, batch, metrics: dict):
        """Next to the head's stop BCE, the count-only predictor's BCE on the same batch's lengths."""
        if self._count_hazard is None or "stop_bce" not in metrics:
            return
        lengths = ([len(example.span) for example in batch] if phase.name in ("A", "C")
                   else self._target_lengths(phase, batch))
        if lengths:
            metrics["stop_bce_count_baseline"] = count_baseline(lengths, self._count_hazard)["bce"]

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
                if phase.name in ("D", "E", "F") and self.record_stream is not None:
                    batches = iter(lambda: self.record_stream.take(phase.batch_size), None)
                else:
                    items = self.records_train if phase.name in ("D", "E", "F") else self.span_train
                    batches = self._batches(items, phase.batch_size, self.phase_step)
                self._audit_phase(phase)
                while self.phase_step < phase.steps:
                    self._release_layers(phase)
                    for group in self.optimizer.param_groups:
                        group["lr"] = phase.lr * group.get("lr_scale", 1.0)
                    started = time.perf_counter()
                    self.optimizer.zero_grad(set_to_none=True)
                    batch = next(batches)
                    loss, metrics = self._step_loss(phase, batch)
                    self._count_metric(phase, batch, metrics)
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
