"""Four-Tier Scoped Experience KB Bridge connecting ~/hindsight-kb retrieval to Natlang Compiler (§7, W4.1).

Supports:
- Four compositional scopes: `natlang:global`, `group:<domain>`, `repo:<sha>`, `fn:<site_id>`
- Strict held-closure admission guard (`held_eval.exclusion_ids()`)
- Pre-top-k `exclude_tasks` filtering matching ~/hindsight-kb's canonical retrieval semantics
- Stage A (retrieved demos in context), Stage B (compacted spans), and Stage C (unified
  `[1, M_kb, 256]` semantic slots ready to concatenate with `z_site` in `UnifiedCompilerHead`)
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ExperienceEntry:
    record_id: str
    scope: str
    key: torch.Tensor                  # [D_k] normalized FP32
    value: torch.Tensor                # [D_v] FP32 (256-d semantic slot)
    text: str                          # Stage A demonstration / repair lesson
    task_ids: tuple[str, ...]
    strength: float = 1.0
    kind: str = "success"              # "success" | "repair" | "doc" | "guidance"
    compacted_state_id: str | None = None


class ScopedExperienceKB:
    """In-process scoped experience bridge (backed by tensors; drop-in mirror of SQLite/PackedKBStore)."""

    def __init__(
        self,
        *,
        key_dim: int = 128,
        value_dim: int = 256,
        excluded_held_ids: set[str] | frozenset[str] = frozenset(),
    ):
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.excluded_held_ids = frozenset(excluded_held_ids)
        self._entries: dict[str, ExperienceEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: ExperienceEntry) -> bool:
        if entry.strength <= 0.0:
            return False
        for tid in entry.task_ids:
            if tid in self.excluded_held_ids:
                raise ValueError(f"Refusing to admit held-excluded task {tid!r} into ScopedExperienceKB")
        if entry.key.shape != (self.key_dim,) or entry.value.shape != (self.value_dim,):
            raise ValueError(
                f"Expected key ({self.key_dim},) and value ({self.value_dim},), "
                f"got {tuple(entry.key.shape)} and {tuple(entry.value.shape)}"
            )
        norm_entry = ExperienceEntry(
            record_id=entry.record_id,
            scope=entry.scope,
            key=F.normalize(entry.key.detach().float(), dim=-1).cpu(),
            value=entry.value.detach().float().cpu(),
            text=entry.text,
            task_ids=tuple(entry.task_ids),
            strength=float(entry.strength),
            kind=entry.kind,
            compacted_state_id=entry.compacted_state_id,
        )
        self._entries[norm_entry.record_id] = norm_entry
        return True

    def retrieve(
        self,
        query_key: torch.Tensor,
        *,
        scopes: tuple[str, ...] | list[str],
        exclude_tasks: set[str] | frozenset[str] = frozenset(),
        top_k: int = 16,
        threshold: float = 0.2,
        null_mass: float = 0.1,
    ) -> tuple[list[tuple[ExperienceEntry, float, float]], float]:
        """Pre-top-k scope & exclude_tasks filtered quadratic retrieval with explicit null mass."""
        q = F.normalize(query_key.detach().float().cpu(), dim=-1)
        scope_set = set(scopes)
        scored: list[tuple[ExperienceEntry, float, float]] = []
        for rec in self._entries.values():
            if rec.scope not in scope_set or rec.strength <= 0.0:
                continue
            if exclude_tasks and any(tid in exclude_tasks for tid in rec.task_ids):
                continue
            sim = float(torch.dot(q, rec.key).item())
            if sim <= threshold:
                continue
            raw_w = rec.strength * ((sim - threshold) ** 2)
            scored.append((rec, sim, raw_w))

        scored.sort(key=lambda item: item[2], reverse=True)
        kept = scored[: max(0, int(top_k))]
        total_w = sum(item[2] for item in kept)
        denom = float(null_mass) + total_w
        evidence_gate = (total_w / denom) if denom > 0.0 else 0.0
        normalized = [(rec, sim, raw_w / denom) for rec, sim, raw_w in kept]
        return normalized, evidence_gate

    def retrieve_unified_slots(
        self,
        query_key: torch.Tensor,
        *,
        scopes: tuple[str, ...] | list[str],
        exclude_tasks: set[str] | frozenset[str] = frozenset(),
        max_slots: int = 16,
        threshold: float = 0.2,
        null_mass: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage-C output: padded (z_kb [1, M, Dv], k_kb [1, M, Dk], weights [1, M], valid [1, M])."""
        hits, _ = self.retrieve(
            query_key,
            scopes=scopes,
            exclude_tasks=exclude_tasks,
            top_k=max_slots,
            threshold=threshold,
            null_mass=null_mass,
        )
        z = torch.zeros((1, max_slots, self.value_dim), dtype=torch.float32, device=device)
        k = torch.zeros((1, max_slots, self.key_dim), dtype=torch.float32, device=device)
        w = torch.zeros((1, max_slots), dtype=torch.float32, device=device)
        valid = torch.zeros((1, max_slots), dtype=torch.bool, device=device)
        for idx, (rec, _sim, weight) in enumerate(hits):
            z[0, idx] = rec.value.to(device=device)
            k[0, idx] = rec.key.to(device=device)
            w[0, idx] = float(weight)
            valid[0, idx] = True
        return z, k, w, valid
