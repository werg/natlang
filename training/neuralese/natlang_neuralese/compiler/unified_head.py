"""UnifiedCompilerHead: single dual head (MemoryKVBank + AdapterReader V2) over unified Z = [z_site, z_kb].

Bridges natlang's online LFM3 HeaderEncoder and ~/hindsight-kb's sparse Experience KB through
one 256-dim semantic slot interface. Supports:
- Mode P (positioned pre-RoPE .nzs-exact memory) and Mode W (window-exempt RoPE-free memory)
- Linear weight-space composition of C cores inside the shared (B, A) basis
- Independent reader dropout (kv_drop, adapter_drop) and straight-through memory_gate warmup floor
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .hkb_dependency import require as _require_hkb

_require_hkb()
from hindsight_kb.models.readers import AdapterReader, MemoryKVBank  # noqa: E402
from hindsight_kb.retrieval import BlockMemoryRecords  # noqa: E402


@dataclass
class CompiledSiteOutput:
    """Outputs emitted from the unified dual head for one site / scope / episode block."""

    c_cores: torch.Tensor              # [B, M_modules, r, r] middle matrices in shared (B, A) basis
    mem_kv: torch.Tensor               # [B, M_slots, L_attn, 2, H_kv, D_head]
    mem_valid: torch.Tensor            # [B, M_slots] bool
    mem_gate: torch.Tensor             # [L_attn] effective gate
    z_slots: torch.Tensor              # [B, M_slots, value_dim] unified semantic slots
    k_slots: torch.Tensor              # [B, M_slots, key_dim] unified slot keys
    weights: torch.Tensor              # [B, M_slots] slot evidence weights


def compose_cores(*cores: torch.Tensor, scales: tuple[float, ...] | None = None) -> torch.Tensor:
    """Exact linear weight-space algebra in the shared (B, A) basis: C_call = sum_i s_i * C_i.

    Works across lexical scopes (family + repo + folder + site + partial) with zero basis mismatch.
    """
    if not cores:
        raise ValueError("compose_cores requires at least one core tensor")
    if scales is not None and len(scales) != len(cores):
        raise ValueError("scales length must match number of cores")
    total = None
    for idx, core in enumerate(cores):
        term = core if scales is None else core * float(scales[idx])
        total = term if total is None else (total + term)
    return total


class UnifiedCompilerHead(nn.Module):
    """Maps unified semantic slots Z = [z_site, z_kb] -> (C_cores, MemoryKVBank tensors).

    All trainable parameters remain FP32 masters. During training, optional independent reader
    dropout prevents either MemoryKVBank or AdapterReader V2 from collapsing, and an optional
    straight-through gate floor warms MemoryKVBank without breaking step-0 eval identity.
    """

    def __init__(
        self,
        hkb_config,
        module_count: int,
        *,
        module_families: list[int] | tuple[int, ...] | None = None,
        null_mass: float = 0.1,
        kv_drop: float = 0.1,
        adapter_drop: float = 0.1,
    ):
        super().__init__()
        self.hkb_config = hkb_config
        self.module_count = int(module_count)
        self.kv_drop = float(kv_drop)
        self.adapter_drop = float(adapter_drop)
        self.memory_bank = MemoryKVBank(hkb_config, init_gate=0.0, null_mass=null_mass)
        self.adapter_reader = AdapterReader(
            hkb_config,
            self.module_count,
            module_families=list(module_families) if module_families is not None else None,
        )
        # Bridge from arbitrary-length slot list Z [B, M, Dv] + K [B, M, Dk] into AdapterReader's
        # expected [B, T, slots * (value_dim + 1) + key_dim] feature layout without averaging away slots.
        adapter_inp = hkb_config.slots * (hkb_config.value_dim + 1) + hkb_config.key_dim
        self.slot_to_adapter = nn.Sequential(
            nn.Linear(hkb_config.value_dim + hkb_config.key_dim + 1, adapter_inp),
            nn.SiLU(),
        )

    def _effective_gate(self, gate_floor: float = 0.0) -> torch.Tensor:
        raw = self.memory_bank.effective_gate()
        if not self.training or gate_floor <= 0.0:
            return raw
        # Straight-through warmup floor: forward sees at least gate_floor; backward flows directly.
        floored = torch.clamp(raw, min=float(gate_floor))
        return raw + (floored - raw).detach()

    def forward(
        self,
        z_slots: torch.Tensor,
        k_slots: torch.Tensor,
        weights: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
        query_summary: torch.Tensor | None = None,
        gate_floor: float = 0.0,
        kv_enabled: bool = True,
        adapter_enabled: bool = True,
    ) -> CompiledSiteOutput:
        """Synthesize (C_cores, mem_kv, mem_valid, mem_gate) from unified slots."""
        b, m, _ = z_slots.shape
        if valid is None:
            valid = torch.ones((b, m), dtype=torch.bool, device=z_slots.device)
        if query_summary is None:
            w = (weights * valid.to(weights.dtype)).unsqueeze(-1)
            denom = w.sum(dim=1).clamp_min(1e-6)
            query_summary = F.normalize((k_slots.float() * w).sum(dim=1) / denom, dim=-1)

        slot_ids = torch.arange(m, device=z_slots.device, dtype=torch.long)[None].expand(b, -1)
        records = BlockMemoryRecords(
            keys=k_slots.float(),
            values=z_slots.float(),
            weights=weights.float(),
            ids=slot_ids,
            valid=valid,
        )
        mem_kv, mem_valid, _ = self.memory_bank(records, query_summary.float())
        mem_gate = self._effective_gate(gate_floor=gate_floor)

        # AdapterReader V2 multi-head slot routing over each slot feature vector
        slot_feat_in = torch.cat(
            (z_slots.float(), k_slots.float(), (weights.float() * valid.float()).unsqueeze(-1)),
            dim=-1,
        )
        adapter_features = self.slot_to_adapter(slot_feat_in)  # [B, M, adapter_inp]
        gates = (weights.float() / (0.1 + weights.float().clamp_min(0.0))) * valid.float()
        zero_prev = adapter_features.new_zeros(
            b, self.module_count, self.hkb_config.rank, self.hkb_config.rank
        )
        # Use AdapterReader.next_control with decay=0 equivalent by scaling out (1 - decay)
        c_raw = self.adapter_reader.next_control(adapter_features, gates, zero_prev)
        decay_norm = max(1e-6, 1.0 - float(self.hkb_config.adapter_decay))
        c_cores = c_raw / decay_norm

        # Independent reader dropout during training (never drop both simultaneously)
        if self.training and kv_enabled and adapter_enabled:
            if self.kv_drop > 0.0 and torch.rand((), device=z_slots.device).item() < self.kv_drop:
                kv_enabled = False
            elif self.adapter_drop > 0.0 and torch.rand((), device=z_slots.device).item() < self.adapter_drop:
                adapter_enabled = False

        if not kv_enabled:
            mem_gate = mem_gate * 0.0
        if not adapter_enabled:
            c_cores = c_cores * 0.0

        return CompiledSiteOutput(
            c_cores=c_cores,
            mem_kv=mem_kv,
            mem_valid=mem_valid,
            mem_gate=mem_gate,
            z_slots=z_slots,
            k_slots=k_slots,
            weights=weights,
        )
