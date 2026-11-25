"""HeaderEncoder: 2-layer cross-attention pooler turning LFM3 hidden states into 256-d semantic slots.

Reads a call site's rendered header (instruction, signature/types, captures, partial args,
optional Neuralese compacted context and memetic guidance spans) through frozen LFM3-1.7B
and emits M_site continuous semantic slots z_site [B, M_site, value_dim] and lookup keys
k_site [B, M_site, key_dim] in the same space as ~/hindsight-kb records.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


def default_lfm3_tap_layers(num_layers: int) -> tuple[int, ...]:
    """Select ~25%, ~60%, and ~85% depth layers (plus final norm handled separately)."""
    if num_layers < 1:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if num_layers <= 3:
        return tuple(range(num_layers))
    raw = (
        max(0, round(0.25 * (num_layers - 1))),
        max(0, round(0.60 * (num_layers - 1))),
        max(0, round(0.85 * (num_layers - 1))),
    )
    dedup = tuple(dict.fromkeys(raw))
    return dedup


@dataclass(frozen=True)
class HeaderEncoderConfig:
    hidden_dim: int
    num_tap_layers: int = 3
    num_slots: int = 16
    value_dim: int = 256
    key_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        if self.hidden_dim <= 0 or self.num_tap_layers <= 0 or self.num_slots <= 0:
            raise ValueError("HeaderEncoderConfig dimensions must be positive")
        if self.value_dim <= 0 or self.key_dim <= 0 or self.num_heads <= 0:
            raise ValueError("HeaderEncoderConfig slot/key/head dims must be positive")
        if self.value_dim % self.num_heads != 0:
            raise ValueError(f"value_dim ({self.value_dim}) must divide num_heads ({self.num_heads})")


class _CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        q = self.q_norm(queries)
        kv = self.kv_norm(context)
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        x = queries + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HeaderEncoder(nn.Module):
    """Pool LFM3 tapped hidden states into unified (z_site, k_site, weights) semantic slots.

    All parameters live in FP32 (AGENTS.md LFM3 precision rule). Forward accepts BF16 or FP32
    tapped tensors from LFM3 and computes in FP32 for exact Tier-1.5 and T2 gradients.
    """

    def __init__(self, config: HeaderEncoderConfig):
        super().__init__()
        self.config = config
        in_dim = config.hidden_dim * (config.num_tap_layers + 1)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, config.value_dim),
            nn.LayerNorm(config.value_dim),
        )
        self.slot_queries = nn.Parameter(torch.randn(1, config.num_slots, config.value_dim) * 0.02)
        self.blocks = nn.ModuleList([
            _CrossAttentionBlock(config.value_dim, config.num_heads, config.dropout),
            _CrossAttentionBlock(config.value_dim, config.num_heads, config.dropout),
        ])
        self.out_norm = nn.LayerNorm(config.value_dim)
        self.value_head = nn.Linear(config.value_dim, config.value_dim)
        self.key_head = nn.Linear(config.value_dim, config.key_dim)
        self.weight_head = nn.Linear(config.value_dim, 1)
        # Initialize value head softly so step-0 compiled slots stay well-conditioned.
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)
        nn.init.zeros_(self.weight_head.bias)

    def forward(
        self,
        tapped_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        final_hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        extra_context_slots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode tapped LFM3 hidden states -> (z_site [B, M, Dv], k_site [B, M, Dk], weights [B, M]).

        Args:
            tapped_states: tuple of num_tap_layers tensors [B, T, D_hidden]
            final_hidden: final-norm LFM3 hidden state [B, T, D_hidden]
            attention_mask: optional bool/0-1 mask [B, T] (True/1 = valid token)
            extra_context_slots: optional pre-encoded semantic slots [B, M_extra, Dv]
                (e.g., memetic guidance states or Neuralese compacted spans) concatenated into
                the cross-attention key/value context.
        """
        if len(tapped_states) != self.config.num_tap_layers:
            raise ValueError(
                f"Expected {self.config.num_tap_layers} tapped layers, got {len(tapped_states)}"
            )
        cat = torch.cat([*(t.float() for t in tapped_states), final_hidden.float()], dim=-1)
        ctx = self.input_proj(cat)  # [B, T, Dv]
        pad_mask = None
        if attention_mask is not None:
            pad_mask = ~attention_mask.to(torch.bool)  # True = ignored by MultiheadAttention
        if extra_context_slots is not None and extra_context_slots.shape[1] > 0:
            extra = extra_context_slots.float()
            ctx = torch.cat((ctx, extra), dim=1)
            if pad_mask is not None:
                extra_valid = torch.zeros(
                    (pad_mask.shape[0], extra.shape[1]), dtype=torch.bool, device=pad_mask.device
                )
                pad_mask = torch.cat((pad_mask, extra_valid), dim=1)

        queries = self.slot_queries.expand(ctx.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, ctx, key_padding_mask=pad_mask)
        h = self.out_norm(queries)
        z_site = self.value_head(h)
        k_site = F.normalize(self.key_head(h), dim=-1)
        weights = F.softplus(self.weight_head(h)).squeeze(-1)
        return z_site, k_site, weights
