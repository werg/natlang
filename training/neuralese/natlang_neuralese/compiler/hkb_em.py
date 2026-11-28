"""Two-mode Hindsight-KB training: Hindsight Encoding + Alternating Direct Entry Optimization <-> Reproduction.

Mode 1 — Hindsight Encoding (``hindsight_encode_step``):
  Trains the hindsight encoder (``LiveExperienceWriter`` / ``PplxWarmstartWriter``) directly from static sources and
  trajectories using either:
  - ``local_masked``: encode a hindsight source span of a trajectory into ``(z_kb, k_kb)`` to predict masked targets
    in the same trajectory without exposing the raw source text; or
  - ``cross_peer``: encode source doc/trajectory A into ``(z_kb, k_kb)`` and train it to help a *distinct* similar
    peer target B (strictly refusing ``source_task_id == target_task_id`` to prevent self-leakage), plus query-key
    alignment.

Mode 2 — Alternating Direct Entry Training <-> Reproduction (EM / wake-sleep):
  - E-step (``optimize_kb_entry``): directly optimize an on-disk KB entry's ``(value, key)`` (and optional ``.nzs``
    state tensors) using KL-anchored execution-grounded advantages + contrastive key pull/push + proximal L2 anchor
    + mandatory held-out query split gate. Accepted drifted entries are written straight back to ``ScopedExperienceKB``.
  - M-step (``reproduction_step``): train the hindsight encoder to reproduce the drifted, execution-learned on-disk
    encodings (``strength``-weighted value MSE + cosine key loss), then mark converged entries reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..tune.latent_ttt import split_support_and_query
from ..tune.low_dof import _group_advantages


@dataclass(frozen=True)
class HindsightEncodeConfig:
    mode: str = 'cross_peer'      # 'local_masked' | 'cross_peer'
    key_align_weight: float = 0.25
    clip_norm: float = 1.0


def hindsight_encode_step(
    writer,
    engine,
    optimizer: torch.optim.Optimizer,
    pairs: list[dict],
    decoder_loss_fn,
    *,
    cfg: HindsightEncodeConfig = HindsightEncodeConfig(),
    max_tokens: int = 128,
) -> dict[str, float]:
    """One Mode-1 hindsight encoding step over ``pairs``.

    Each item in ``pairs`` carries:
      - ``source_text``: text encoded by ``writer`` into ``(z_kb, k_kb)``
      - ``target``: target payload passed to ``decoder_loss_fn(z_kb, k_kb, targets)``
      - ``query_text`` (optional): context text whose query key ``q`` aligns with ``k_kb``
      - ``source_task_id`` / ``target_task_id`` (required in ``cross_peer`` mode; must differ to prevent leakage)
    """
    if cfg.mode not in ('local_masked', 'cross_peer'):
        raise ValueError(f"mode must be 'local_masked' or 'cross_peer', got {cfg.mode!r}")
    if not pairs:
        raise ValueError('hindsight_encode_step requires at least one pair')
    if cfg.mode == 'cross_peer':
        for item in pairs:
            src_id = item.get('source_task_id')
            tgt_id = item.get('target_task_id')
            if not src_id or not tgt_id:
                raise ValueError('cross_peer mode requires source_task_id and target_task_id on every pair')
            if str(src_id) == str(tgt_id):
                raise ValueError(f'cross_peer leakage: source_task_id == target_task_id ({src_id!r})')

    optimizer.zero_grad(set_to_none=True)
    source_texts = [str(p['source_text']) for p in pairs]
    z_kb, k_kb = writer.encode_experience_texts(engine, source_texts, max_tokens=max_tokens)
    dec_loss = decoder_loss_fn(z_kb, k_kb, [p['target'] for p in pairs])

    key_loss = torch.zeros((), dtype=torch.float32, device=z_kb.device)
    if cfg.key_align_weight > 0.0 and all(p.get('query_text') for p in pairs):
        q = writer.encode_query_texts(engine, [str(p['query_text']) for p in pairs], max_tokens=max_tokens)
        key_loss = (1.0 - (F.normalize(q, dim=-1) * F.normalize(k_kb, dim=-1)).sum(dim=-1)).mean()

    total = dec_loss + float(cfg.key_align_weight) * key_loss
    total.backward()
    if cfg.clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg.clip_norm)
    optimizer.step()
    return {
        'mode': cfg.mode,
        'total_loss': float(total.detach().item()),
        'decoder_loss': float(dec_loss.detach().item()),
        'key_align_loss': float(key_loss.detach().item()),
        'pairs': float(len(pairs)),
    }


@dataclass(frozen=True)
class EntryOptConfig:
    steps: int = 24
    lr: float = 0.05
    key_lr: float = 0.02
    kl_beta: float = 0.01
    prox_lambda: float = 0.05
    key_contrast_weight: float = 0.2
    clip_norm: float = 1.0
    eval_every: int = 4
    min_query_gain: float = 1e-4
    query_fraction: float = 0.25


@dataclass
class EntryOptResult:
    accepted: bool
    best_step: int
    base_query_score: float
    best_query_score: float
    value_best: torch.Tensor
    key_best: torch.Tensor
    drift_norm: float
    history: list[dict[str, float]]


def optimize_kb_entry(
    init_value: torch.Tensor,
    init_key: torch.Tensor,
    score_fn,
    examples: list[dict],
    *,
    cfg: EntryOptConfig = EntryOptConfig(),
    kb=None,
    scope: str | None = None,
    record_id: str | None = None,
) -> EntryOptResult:
    """Mode-2 E-step: directly optimize one KB entry's ``(value, key)`` on execution outcomes with held-query gate.

    ``score_fn(value, key, batch) -> (policy_logps, ref_logps)`` scores the batch using the live candidate entry.
    Examples carry ``reward`` (float), optional ``group`` (task id), and optional ``query_key`` (``[D_k]``) for
    contrastive key alignment (pulling ``key`` toward positive-advantage queries, pushing away from negative ones).
    When ``kb``, ``scope``, and ``record_id`` are supplied and the update passes the held-query gate, writes the
    drifted entry straight back to ``ScopedExperienceKB.update_entry_payload(..., mark_drifted=True)``.
    """
    support, query = split_support_and_query(examples, query_fraction=cfg.query_fraction)
    v0 = init_value.detach().float().clone()
    k0 = F.normalize(init_key.detach().float().clone(), dim=-1)
    v = v0.clone().requires_grad_(True)
    k_raw = k0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([
        {'params': [v], 'lr': cfg.lr, 'weight_decay': 0.0},
        {'params': [k_raw], 'lr': cfg.key_lr, 'weight_decay': 0.0},
    ])

    def _objective(v_cur: torch.Tensor, k_cur: torch.Tensor, batch: list[dict]) -> tuple[torch.Tensor, float]:
        k_unit = F.normalize(k_cur, dim=-1)
        pol_lp, ref_lp = score_fn(v_cur, k_unit, batch)
        adv = _group_advantages(batch, pol_lp.device)
        dlogp = pol_lp - ref_lp.detach()
        kl_k2 = 0.5 * dlogp.pow(2).mean()
        score = (adv * dlogp).mean() - float(cfg.kl_beta) * kl_k2
        if cfg.key_contrast_weight > 0.0 and all(ex.get('query_key') is not None for ex in batch):
            q_mat = F.normalize(
                torch.stack([ex['query_key'].detach().float().to(k_unit.device) for ex in batch], dim=0),
                dim=-1,
            )
            sims = q_mat @ k_unit
            score = score + float(cfg.key_contrast_weight) * (adv * sims).mean()
        return score, float(kl_k2.detach().item())

    with torch.no_grad():
        base_q_t, base_kl = _objective(v0, k0, query)
        base_q = float(base_q_t.item())

    best_q = base_q
    best_step = 0
    best_v = v0.clone()
    best_k = k0.clone()
    history: list[dict[str, float]] = [{'step': 0.0, 'query_score': base_q, 'query_kl': base_kl, 'drift_norm': 0.0}]

    for step in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        sup_score, sup_kl = _objective(v, k_raw, support)
        k_unit = F.normalize(k_raw, dim=-1)
        prox = 0.5 * float(cfg.prox_lambda) * ((v - v0).pow(2).sum() + (k_unit - k0).pow(2).sum())
        loss = -sup_score + prox
        loss.backward()
        if cfg.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([v, k_raw], cfg.clip_norm)
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            with torch.no_grad():
                k_eval = F.normalize(k_raw, dim=-1)
                q_t, q_kl = _objective(v, k_eval, query)
                q_score = float(q_t.item())
                dnorm = float(torch.linalg.vector_norm(v - v0).item() + torch.linalg.vector_norm(k_eval - k0).item())
            history.append({
                'step': float(step),
                'support_loss': float(loss.detach().item()),
                'support_kl': sup_kl,
                'query_score': q_score,
                'query_kl': q_kl,
                'drift_norm': dnorm,
            })
            if q_score > best_q + cfg.min_query_gain:
                best_q = q_score
                best_step = step
                best_v = v.detach().clone()
                best_k = k_eval.detach().clone()

    accepted = best_step > 0 and best_q > base_q + cfg.min_query_gain
    final_v = best_v if accepted else v0.clone()
    final_k = best_k if accepted else k0.clone()
    drift_norm = float(torch.linalg.vector_norm(final_v - v0).item() + torch.linalg.vector_norm(final_k - k0).item())

    if accepted and kb is not None and scope is not None and record_id is not None:
        kb.update_entry_payload(scope, record_id, new_value=final_v, new_key=final_k, mark_drifted=True)

    return EntryOptResult(
        accepted=accepted,
        best_step=best_step,
        base_query_score=base_q,
        best_query_score=best_q,
        value_best=final_v,
        key_best=final_k,
        drift_norm=drift_norm,
        history=history,
    )


@dataclass(frozen=True)
class ReproductionConfig:
    steps: int = 20
    key_weight: float = 0.5
    clip_norm: float = 1.0
    max_tokens: int = 128
    converge_tol: float | None = None
    mark_after: bool = True


def reproduction_loss(
    writer,
    engine,
    kb,
    *,
    scope: str | None = None,
    only_unreproduced: bool = True,
    key_weight: float = 0.5,
    max_tokens: int = 128,
) -> tuple[torch.Tensor | None, list[tuple[str, str]], dict[str, float]]:
    """Differentiable M-step reproduction loss over unreproduced drifted KB entries.

    Returns ``(loss_tensor, pairs, stats)`` WITHOUT stepping any optimizer so callers like ``run_t2.py`` can
    backpropagate reproduction directly into their single ``MasterAdamW`` (avoiding dual FP32-master-copy conflicts).
    """
    entries = [e for e in kb.iter_drifted_entries(scope=scope, only_unreproduced=only_unreproduced) if e.get('text')]
    if not entries:
        return None, [], {'entries': 0.0, 'loss': 0.0}
    texts = [e['text'] for e in entries]
    device = next(writer.parameters()).device
    target_z = torch.stack([e['value'].to(device=device, dtype=torch.float32) for e in entries], dim=0)
    target_k = F.normalize(torch.stack([e['key'].to(device=device, dtype=torch.float32) for e in entries], dim=-1 if False else 0), dim=-1)
    weights = torch.tensor([max(1e-3, float(e.get('strength', 1.0))) for e in entries], dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    pred_z, pred_k = writer.encode_experience_texts(engine, texts, max_tokens=max_tokens)
    val_err = (pred_z - target_z).pow(2).mean(dim=-1)
    key_err = 1.0 - (F.normalize(pred_k, dim=-1) * target_k).sum(dim=-1)
    loss = (weights * (val_err + float(key_weight) * key_err)).sum()
    pairs = [(e['scope'], e['record_id']) for e in entries]
    return loss, pairs, {'entries': float(len(entries)), 'loss': float(loss.detach().item())}


def reproduction_step(
    writer,
    engine,
    optimizer: torch.optim.Optimizer,
    kb,
    *,
    scope: str | None = None,
    only_unreproduced: bool = True,
    cfg: ReproductionConfig = ReproductionConfig(),
) -> dict[str, float]:
    """Mode-2 M-step: train the hindsight encoder to reproduce drifted, execution-learned on-disk KB entries."""
    entries = [e for e in kb.iter_drifted_entries(scope=scope, only_unreproduced=only_unreproduced) if e.get('text')]
    if not entries:
        return {'entries': 0.0, 'steps': 0.0, 'initial_loss': 0.0, 'final_loss': 0.0, 'marked': 0.0}

    texts = [e['text'] for e in entries]
    device = next(writer.parameters()).device
    target_z = torch.stack([e['value'].to(device=device, dtype=torch.float32) for e in entries], dim=0)
    target_k = F.normalize(torch.stack([e['key'].to(device=device, dtype=torch.float32) for e in entries], dim=0), dim=-1)
    weights = torch.tensor([max(1e-3, float(e.get('strength', 1.0))) for e in entries], dtype=torch.float32, device=device)
    weights = weights / weights.sum()

    init_loss = final_loss = 0.0
    per_item = torch.zeros((len(entries),), dtype=torch.float32, device=device)
    for step in range(1, int(cfg.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        pred_z, pred_k = writer.encode_experience_texts(engine, texts, max_tokens=cfg.max_tokens)
        val_err = (pred_z - target_z).pow(2).mean(dim=-1)
        key_err = 1.0 - (F.normalize(pred_k, dim=-1) * target_k).sum(dim=-1)
        per_item = val_err + float(cfg.key_weight) * key_err
        loss = (weights * per_item).sum()
        if step == 1:
            init_loss = float(loss.detach().item())
        loss.backward()
        if cfg.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(writer.parameters(), cfg.clip_norm)
        optimizer.step()
        final_loss = float(loss.detach().item())

    marked = 0
    if cfg.mark_after:
        if cfg.converge_tol is None:
            done_pairs = [(e['scope'], e['record_id']) for e in entries]
        else:
            done_pairs = [
                (e['scope'], e['record_id'])
                for e, err in zip(entries, per_item.detach().cpu().tolist())
                if err <= float(cfg.converge_tol)
            ]
        if done_pairs:
            marked = kb.mark_reproduced(done_pairs)

    return {
        'entries': float(len(entries)),
        'steps': float(cfg.steps),
        'initial_loss': init_loss,
        'final_loss': final_loss,
        'marked': float(marked),
    }
