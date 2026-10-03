"""Phase configurations and schedules (S3 §5.1, phases A–D).

A phase names its loss, its length in optimizer steps, and its schedule. Phases run in
order within one run lineage; the trainer records which phase every step belongs to.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Phase:
    name: str                     # "A" bootstrap, "B" shallow distillation, "C" transition, "D" consumer
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
        return asdict(self)


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
