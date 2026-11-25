"""Hidden-State Process Step-Verifier Head & Prefix-Shared Branch Scorer (Recipe T0, §6.2 & §9.2).

Attaches a lightweight 2-layer FP32 MLP (~2M params at hidden_dim=2048) to LFM3-1.7B's final
hidden state at tool-call and `return_result` boundaries. Trained on existing pass/fail
student rollout archives to score candidate branches in prefix-shared `branch@4` search.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


class StepVerifierHead(nn.Module):
    """Predicts [episode_success_logit, immediate_error_logit] from LFM3 final hidden state."""

    def __init__(self, hidden_dim: int, mlp_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        if hidden_dim <= 0 or mlp_dim <= 0:
            raise ValueError("hidden_dim and mlp_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.mlp_dim = int(mlp_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 2),
        )
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [..., hidden_dim] -> logits [..., 2] in FP32."""
        return self.net(self.norm(hidden.float()))

    @torch.no_grad()
    def success_probability(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.forward(hidden)
        return torch.sigmoid(logits[..., 0])


def compute_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Exact Mann-Whitney / Wilcoxon rank AUROC with tie handling (no sklearn dependency)."""
    s = scores.detach().float().flatten()
    y = labels.detach().float().flatten()
    pos = y > 0.5
    n_pos = int(pos.sum().item())
    n_neg = int((~pos).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    s_sorted = s[order]
    y_sorted = pos[order]
    # Assign average ranks for ties (1-indexed)
    n = s_sorted.numel()
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and float(s_sorted[j].item()) == float(s_sorted[i].item()):
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_rank
        i = j
    pos_rank_sum = float(ranks[y_sorted].sum().item())
    return (pos_rank_sum - 0.5 * n_pos * (n_pos + 1)) / float(n_pos * n_neg)


@dataclass(frozen=True)
class BranchCandidate:
    branch_id: str
    ensures_ok: bool
    type_ok: bool
    tool_exit_ok: bool
    verifier_score: float
    mean_logprob: float = 0.0


def rank_branches(candidates: list[BranchCandidate]) -> list[BranchCandidate]:
    """Rank prefix-shared branch@N candidates: crisp checks strictly first, then Step-Verifier V(h_t)."""
    return sorted(
        candidates,
        key=lambda c: (
            1 if (c.ensures_ok and c.type_ok and c.tool_exit_ok) else 0,
            1 if c.type_ok else 0,
            1 if c.tool_exit_ok else 0,
            float(c.verifier_score),
            float(c.mean_logprob),
        ),
        reverse=True,
    )


@dataclass(frozen=True)
class VerifierTrainConfig:
    epochs: int = 2
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    error_loss_weight: float = 0.3
    clip_norm: float = 1.0


def train_step_verifier(
    head: StepVerifierHead,
    *,
    train_hidden: torch.Tensor,
    train_success: torch.Tensor,
    train_error: torch.Tensor | None = None,
    val_hidden: torch.Tensor | None = None,
    val_success: torch.Tensor | None = None,
    cfg: VerifierTrainConfig = VerifierTrainConfig(),
) -> dict[str, float]:
    """Train StepVerifierHead in FP32 with cosine LR schedule and return final metrics."""
    device = next(head.parameters()).device
    n = int(train_hidden.shape[0])
    if n == 0:
        raise ValueError("train_hidden is empty")
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(n / cfg.batch_size))
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr * 0.1)

    head.train()
    last_loss = 0.0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb = train_hidden[idx].to(device=device, dtype=torch.float32)
            yb = train_success[idx].to(device=device, dtype=torch.float32)
            logits = head(xb)
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], yb)
            if train_error is not None and cfg.error_loss_weight > 0.0:
                eb = train_error[idx].to(device=device, dtype=torch.float32)
                loss = loss + cfg.error_loss_weight * F.binary_cross_entropy_with_logits(logits[:, 1], eb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_norm)
            optimizer.step()
            scheduler.step()
            last_loss = float(loss.detach().item())

    head.eval()
    with torch.no_grad():
        tr_probs = head.success_probability(train_hidden.to(device=device, dtype=torch.float32)).cpu()
        tr_auroc = compute_auroc(tr_probs, train_success.cpu())
        val_auroc = float("nan")
        if val_hidden is not None and val_success is not None and len(val_hidden) > 0:
            v_probs = head.success_probability(val_hidden.to(device=device, dtype=torch.float32)).cpu()
            val_auroc = compute_auroc(v_probs, val_success.cpu())
    return {"train_loss": last_loss, "train_auroc": tr_auroc, "val_auroc": val_auroc}
