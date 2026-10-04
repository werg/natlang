"""Phase configurations and schedules (S3 §5.1, phases A–F).

A phase names its loss, its length in optimizer steps, and its schedule. Phases run in
order within one run lineage; the trainer records which phase every step belongs to.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Phase:
    name: str                     # "A" bootstrap, "B" shallow distillation, "C" transition, "D" consumer,
                                  # "E" stop policy, "F" gradual unfreezing
    steps: int
    lr: float = 1e-3
    batch_size: int = 4
    # Phase C: parallel scheduled sampling ramps the generated fraction, then switches to full unroll.
    fraction_start: float = 0.0
    fraction_end: float = 0.0
    ramp_steps: int = 0
    passes: int = 2
    unroll_after: int | None = None
    # Loss weights.
    entry_weight: float = 0.1
    stop_weight: float = 1.0
    kl_weight: float = 0.0
    replay_weight: float = 0.0    # phase B distillation kept as replay from C on
    # Payload temperature (tau) schedule and the beta-weighted payload KL (VAE-style robustness).
    temperature_start: float = 0.0
    temperature_end: float = 0.0
    temperature_ramp_steps: int = 0
    payload_kl_weight: float = 0.0
    # Phases D–F: anti-collapse terms (S3 §5.1, §12).
    contrastive_weight: float = 0.0
    margin: float = 0.5
    diversity_weight: float = 0.0
    max_length: int | None = None
    # Phase E: stop policy against consumer quality and a per-vector length cost.
    stop_policy_weight: float = 0.0
    length_cost: float = 0.0
    policy_samples: int = 1
    # Phase E exploration (see consumer_batch_loss): behaviour mixture weight, stop-head temperature for
    # the behaviour, and the importance-ratio truncation. Omitted from `to_dict` at their defaults, so
    # schedules written before these existed still compare equal on resume.
    stop_exploration: float = 0.0
    stop_temperature: float = 1.0
    stop_ratio_clip: float = 5.0
    # Phase D (and F) supervised lengths: each record writes ceil(source tokens / tokens_per_vector) vectors
    # (clamped to [min_length, max_length]), teacher-forced, with the stop boundary trained at that length.
    # 0 leaves the length to the stop head.
    tokens_per_vector: float = 0.0
    min_length: int = 1
    # Phase F: LoRA deltas on `lora_layers` (released in order, upper layers first), with
    # learning rate lr * lora_lr_scale * lora_layer_decay^(rank from the top), and replay.
    lora_layers: tuple[int, ...] = ()
    lora_rank: int = 16
    lora_lr_scale: float = 0.1
    lora_layer_decay: float = 0.7
    text_replay_weight: float = 0.0
    extra: dict = field(default_factory=dict)

    def generated_fraction(self, step: int) -> float:
        if self.ramp_steps <= 0:
            return self.fraction_end
        t = min(1.0, step / self.ramp_steps)
        return self.fraction_start + (self.fraction_end - self.fraction_start) * t

    def temperature(self, step: int) -> float:
        if self.temperature_ramp_steps <= 0:
            return self.temperature_end
        t = min(1.0, step / self.temperature_ramp_steps)
        return self.temperature_start + (self.temperature_end - self.temperature_start) * t

    def unrolled(self, step: int) -> bool:
        return self.unroll_after is not None and step >= self.unroll_after

    def to_dict(self) -> dict:
        data = asdict(self)
        for name, default in _LATER_DEFAULTS.items():
            if data[name] == default:
                del data[name]
        return data


_LATER_DEFAULTS = {"stop_exploration": 0.0, "stop_temperature": 1.0, "stop_ratio_clip": 5.0, "tokens_per_vector": 0.0,
                   "min_length": 1}


def smoke_phases(scale: float = 1.0) -> list[Phase]:
    """A tiny A→B→C→D schedule for smoke runs."""
    n = lambda steps: max(1, int(steps * scale))
    return [
        Phase("A", n(60), lr=1e-3, batch_size=4),
        Phase("B", n(60), lr=1e-3, batch_size=4),
        Phase("C", n(100), lr=5e-4, batch_size=4, fraction_start=0.0, fraction_end=1.0, ramp_steps=n(60),
              unroll_after=n(70), kl_weight=0.5, replay_weight=0.2,
              temperature_start=0.0, temperature_end=0.5, temperature_ramp_steps=n(100), payload_kl_weight=0.01),
        Phase("D", n(40), lr=3e-4, batch_size=2, kl_weight=1.0, temperature_end=0.5, payload_kl_weight=0.01),
    ]


def pilot_phases(scale: float = 1.0, max_length: int = 32) -> list[Phase]:
    """The S3 pilot schedule on real port records (A→F, short)."""
    n = lambda steps: max(1, int(steps * scale))
    anti = dict(contrastive_weight=0.5, margin=0.5, diversity_weight=1.0, max_length=max_length)
    return [
        Phase("A", n(150), lr=1e-3, batch_size=8),
        Phase("B", n(150), lr=1e-3, batch_size=8),
        Phase("C", n(200), lr=5e-4, batch_size=8, fraction_start=0.0, fraction_end=1.0, ramp_steps=n(120),
              unroll_after=n(140), kl_weight=0.5, replay_weight=0.2,
              temperature_start=0.0, temperature_end=0.3, temperature_ramp_steps=n(200), payload_kl_weight=0.01),
        Phase("D", n(900), lr=3e-4, batch_size=8, kl_weight=1.0, temperature_end=0.3, payload_kl_weight=0.01, **anti),
        Phase("E", n(200), lr=1e-4, batch_size=4, kl_weight=1.0, temperature_end=0.3, payload_kl_weight=0.01,
              stop_policy_weight=1.0, length_cost=0.01, policy_samples=2, **anti),
        Phase("F", n(300), lr=1e-4, batch_size=8, kl_weight=1.0, temperature_end=0.3, payload_kl_weight=0.01,
              lora_layers=(15, 14, 13, 12), lora_rank=16, lora_lr_scale=0.5, text_replay_weight=0.5, **anti),
    ]
