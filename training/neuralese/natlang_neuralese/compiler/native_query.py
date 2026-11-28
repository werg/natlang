"""NativeQueryProjector: zero-extra-backbone 128-d query projector tapping LFM3-1.7B hidden states.

Projects LFM3's tapped hidden states (~25%, ~60%, ~85% depth + final norm) directly into 000000002's normalized
128-d Hindsight-KB query space so online call-time HKB retrieval needs ZERO separate 0.6B PPLX query model in VRAM.
Distilled directly from 000000002's 45k-step ``query_Q.pt`` (``distill_native_query_projector``).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class NativeQueryProjector(nn.Module):
    """3-tap FP32 SwiGLU projector mapping pooled LFM3 taps -> normalized 128-d HKB query vector."""

    def __init__(
        self,
        tap_dims: tuple[int, ...],
        *,
        key_dim: int = 128,
        hidden_dim: int = 512,
        tap_source: str = 'state',
        tap_names: tuple[str, ...] = (),
    ):
        super().__init__()
        if not tap_dims or any(int(d) <= 0 for d in tap_dims):
            raise ValueError(f'tap_dims must be non-empty positive ints, got {tap_dims!r}')
        if tap_source not in ('state', 'hidden'):
            raise ValueError(f"tap_source must be 'state' or 'hidden', got {tap_source!r}")
        self.tap_dims = tuple(int(d) for d in tap_dims)
        self.tap_source = str(tap_source)
        self.tap_names = tuple(str(n) for n in tap_names)
        self.key_dim = int(key_dim)
        self.hidden_dim = int(hidden_dim)
        self.tap_norms = nn.ModuleList([nn.LayerNorm(d) for d in self.tap_dims])
        self.tap_projs = nn.ModuleList([nn.Linear(d, self.hidden_dim) for d in self.tap_dims])
        self.in_norm = nn.LayerNorm(self.hidden_dim)
        # 2-layer SwiGLU FFN (~3M params for hidden_dim=512, tap_dims=(2048,)*4)
        self.w_gate = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.w_up = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.w_down = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.key_head = nn.Linear(self.hidden_dim, self.key_dim)
        self.float()

    def forward(self, taps: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Accepts ``taps`` where each tensor is ``[B, d_i]`` (pooled) or ``[B, T, d_i]`` (mean-pooled).
        Returns L2-normalized FP32 query keys ``[B, key_dim]``.
        """
        if len(taps) != len(self.tap_projs):
            raise ValueError(f'expected {len(self.tap_projs)} taps, got {len(taps)}')
        accum = None
        for t, norm, proj in zip(taps, self.tap_norms, self.tap_projs):
            x = t.float()
            if x.ndim == 3:
                x = x.mean(dim=1)
            term = proj(norm(x))
            accum = term if accum is None else (accum + term)
        h = self.in_norm(accum)
        h = h + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return F.normalize(self.key_head(self.out_norm(h)), dim=-1)


def distill_native_query_projector(
    projector: NativeQueryProjector,
    optimizer: torch.optim.Optimizer,
    taps_batch: list[torch.Tensor] | tuple[torch.Tensor, ...],
    teacher_queries: torch.Tensor,
    *,
    clip_norm: float = 1.0,
) -> dict[str, float]:
    """One distillation step aligning ``NativeQueryProjector(lfm3_taps)`` to 000000002's ``query_Q`` targets."""
    optimizer.zero_grad(set_to_none=True)
    pred = projector(taps_batch)
    target = F.normalize(teacher_queries.detach().float().to(pred.device), dim=-1)
    cos = (pred * target).sum(dim=-1)
    loss = (1.0 - cos).mean()
    loss.backward()
    if clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(projector.parameters(), clip_norm)
    optimizer.step()
    return {
        'loss': float(loss.detach().item()),
        'mean_cosine': float(cos.detach().mean().item()),
    }
