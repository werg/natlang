"""Tier-1.5 Latent-Manifold TTT: sample-efficient per-site refinement over z_site (4,096 floats).

Freezes LFM3-1.7B and UnifiedCompilerHead and optimizes ONLY z_site [1, M_site, value_dim]
(plus optional per-family scalar gains) in FP32, regularized by:
1. Meta-initialized starting point z_site^(0) = HeaderEncoder(header)
2. Proximal L2 pull (lambda_prox / 2) * ||z - z^(0)||^2
3. Mandatory held-out query split (group-aware, or LOO-CV when n < 10) for step selection;
   rejects any refinement that fails to beat Tier 1 (step 0) on the query split.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentTTTConfig:
    steps: int = 32
    lr: float = 0.05
    prox_lambda: float = 0.05
    clip_norm: float = 1.0
    eval_every: int = 4
    min_query_gain: float = 1e-4
    memory_mode: str = "positioned"


@dataclass
class LatentTTTResult:
    accepted: bool
    best_step: int
    tier1_query_loss: float
    best_query_loss: float
    final_support_loss: float
    z_best: torch.Tensor            # [1, M_site, value_dim] FP32
    history: list[dict[str, float]]


def split_support_and_query(
    examples: list[dict],
    *,
    query_fraction: float = 0.25,
) -> tuple[list[dict], list[dict]]:
    """Deterministic group-aware support/query split; ensures >=1 query example whenever len >= 2."""
    if len(examples) < 2:
        raise ValueError("Tier-1.5 latent TTT requires at least 2 examples to hold out a query split")
    groups: dict[str, list[dict]] = {}
    for idx, ex in enumerate(examples):
        g = str(ex.get("group") or f"item-{idx}")
        groups.setdefault(g, []).append(ex)
    ordered_groups = sorted(
        groups.keys(),
        key=lambda g: hashlib.sha256(g.encode("utf-8")).hexdigest(),
    )
    if len(ordered_groups) == 1:
        # Single group: fall back to deterministic item-level split so query evaluation still runs.
        n_q = max(1, round(len(examples) * query_fraction))
        return examples[:-n_q], examples[-n_q:]
    n_q_groups = max(1, min(len(ordered_groups) - 1, round(len(ordered_groups) * query_fraction)))
    q_set = set(ordered_groups[:n_q_groups])
    support = [ex for g in ordered_groups if g not in q_set for ex in groups[g]]
    query = [ex for g in ordered_groups if g in q_set for ex in groups[g]]
    return support, query


def refine_site_latents(
    z_init: torch.Tensor,
    k_slots: torch.Tensor,
    weights: torch.Tensor,
    *,
    support_loss_fn,
    query_loss_fn,
    cfg: LatentTTTConfig = LatentTTTConfig(),
) -> LatentTTTResult:
    """Optimize only z_site in FP32 against support_loss_fn and select strictly by query_loss_fn.

    Both loss functions have signature ``(z_site, k_slots, weights) -> scalar_loss_tensor``.
    Returns z_init unchanged (accepted=False) if no step beats step-0 Tier-1 query loss by min_query_gain.
    """
    z_ref = z_init.detach().float().clone()
    k_fixed = k_slots.detach().float()
    w_fixed = weights.detach().float()

    z = z_ref.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([z], lr=cfg.lr, weight_decay=0.0)

    with torch.no_grad():
        tier1_q = float(query_loss_fn(z_ref, k_fixed, w_fixed).item())

    best_q = tier1_q
    best_step = 0
    z_best = z_ref.clone()
    history: list[dict[str, float]] = [{"step": 0, "query_loss": tier1_q, "prox_dist": 0.0}]
    last_support = float("nan")

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        sup_loss = support_loss_fn(z, k_fixed, w_fixed)
        prox_penalty = 0.5 * float(cfg.prox_lambda) * (z - z_ref).pow(2).mean()
        total = sup_loss + prox_penalty
        total.backward()
        if cfg.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([z], cfg.clip_norm)
        optimizer.step()
        last_support = float(sup_loss.detach().item())

        if step % cfg.eval_every == 0 or step == cfg.steps:
            with torch.no_grad():
                q_loss = float(query_loss_fn(z, k_fixed, w_fixed).item())
                dist = float((z - z_ref).norm().item())
            history.append({
                "step": float(step),
                "support_loss": last_support,
                "query_loss": q_loss,
                "prox_dist": dist,
            })
            if q_loss < best_q - cfg.min_query_gain:
                best_q = q_loss
                best_step = step
                z_best = z.detach().clone()

    accepted = best_step > 0 and (tier1_q - best_q) >= cfg.min_query_gain
    return LatentTTTResult(
        accepted=accepted,
        best_step=best_step if accepted else 0,
        tier1_query_loss=tier1_q,
        best_query_loss=best_q if accepted else tier1_q,
        final_support_loss=last_support,
        z_best=z_best if accepted else z_ref,
        history=history,
    )
