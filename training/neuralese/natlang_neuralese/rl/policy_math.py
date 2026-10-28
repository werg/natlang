"""Shared token-level policy objective used by native recorded-policy training."""
from __future__ import annotations

import torch


def clipped_token_loss(logps, old_logps, advantage, *, clip_low=0.2, clip_high=0.28):
    """DAPO-style asymmetric clipping, with stable float32 ratios and signed gradients."""
    if not 0 <= clip_low < 1 or not 0 <= clip_high:
        raise ValueError('invalid RL clipping range')
    if logps.shape != old_logps.shape or logps.numel() == 0:
        raise ValueError('behavior/current token positions differ or are empty')
    if not torch.isfinite(logps).all() or not torch.isfinite(old_logps).all():
        raise ValueError('nonfinite RL log probabilities')
    ratios = (logps.float() - old_logps.float()).clamp(-20, 20).exp()
    a = torch.as_tensor(advantage, dtype=torch.float32, device=logps.device)
    if not torch.isfinite(a).all():
        raise ValueError('nonfinite advantage')
    clipped = ratios.clamp(1 - clip_low, 1 + clip_high)
    return -torch.minimum(ratios * a, clipped * a)


# Score-centered policy gradients (Marek & Ryabinin, arXiv 2609.20807).
SCORE_CENTERED = 'score-centered/1'
IMPORTANCE_MODES = ('none', 'tis', 'mis')
DEFAULT_SCORE_CENTERED = {'schema': SCORE_CENTERED, 'k': 128, 'importance': 'none', 'eps': 1e-6}


def policy_loss_declaration(value):
    """Canonical declared RL policy loss; None keeps the clipped GRPO loss bit-identical.

    ``{"schema": "score-centered/1", "k": 128, "importance": "none"|"tis"|"mis", "eps": 1e-6}``. Omitted fields
    take these defaults; the canonical full form is what objectives, checkpoints and round journals bind.
    """
    import math
    if value is None:
        return None
    if isinstance(value, str):
        import json
        value = json.loads(value)
    if (not isinstance(value, dict) or value.get('schema') != SCORE_CENTERED
            or set(value) - set(DEFAULT_SCORE_CENTERED)):
        raise ValueError('RL policy loss must be a score-centered/1 declaration with only k, importance and eps')
    result = {**DEFAULT_SCORE_CENTERED, **value}
    if type(result['k']) is not int or not 1 <= result['k'] <= 4096:
        raise ValueError('score-centered k must be an integer in [1, 4096]')
    if result['importance'] not in IMPORTANCE_MODES:
        raise ValueError('score-centered importance must be none, tis or mis')
    if (type(result['eps']) not in (int, float) or not math.isfinite(result['eps'])
            or not 0 < result['eps'] < 1):
        raise ValueError('score-centered eps must be in (0, 1)')
    result['eps'] = float(result['eps'])
    return result


def importance_weight(ratio, mode):
    """f(r): none -> 1; TIS -> min(r, 2); MIS -> r on [0.5, 5], else 0."""
    if mode == 'none':
        return torch.ones_like(ratio)
    if mode == 'tis':
        return ratio.clamp(max=2.)
    if mode == 'mis':
        return torch.where((ratio >= .5) & (ratio <= 5.), ratio, torch.zeros_like(ratio))
    raise ValueError('unsupported importance mode')


def score_centered_surrogate(log_probs, labels, behavior_logp, top_ids, top_logq, *, importance='none', eps=1e-6):
    """Per-position surrogate U whose gradient is the score-centered estimator; the loss is -R U.

    ``log_probs`` [n, V]: the trainer's current normalized log distribution under the sampler's exact
    temperature/mask law, differentiable. ``labels`` [n] sampled tokens y; ``behavior_logp`` [n] the sampler's
    exact log q_y. ``top_ids`` [n, k] the sampler's logged top-k H (pad -1); ``top_logq`` [n, k] its log q
    (pad -inf).

        grad U = w_y s_y - sum_{v in H} (q_v w_v - alpha p_v) s_v,     s_v = grad log p_v,
        rho = (1 - sum_H q) / (1 - sum_H p) (tails floored at eps),     alpha = rho f(1/rho),   w = f(p/q).

    Tail identity: sum_V p_v s_v = 0, so alpha sum_H p_v s_v = -alpha sum_{v not in H} p_v s_v. The alpha term is
    evaluated in that exactly equivalent tail form and the trainer tail mass is summed over the tail directly,
    which avoids cancellation when the trainer's top-k holds nearly all mass. Entries with q_v = 0 (masked by the
    sampler) carry no weight. y may lie inside or outside H; its term is the same either way.
    Returns (U [n], detached per-position diagnostics).
    """
    if (log_probs.ndim != 2 or labels.shape != (log_probs.shape[0],)
            or behavior_logp.shape != labels.shape):
        raise ValueError('score-centered positions/labels/behavior shapes differ')
    if top_ids.shape != top_logq.shape or top_ids.ndim != 2 or top_ids.shape[0] != log_probs.shape[0]:
        raise ValueError('score-centered behavior top-k shapes differ from positions')
    if (top_ids >= log_probs.shape[1]).any():
        raise ValueError('behavior top-k names a token outside the current vocabulary')
    log_probs = log_probs.float()
    top_logq = top_logq.float()
    valid = (top_ids >= 0) & (top_logq > float('-inf'))
    safe_ids = top_ids.clamp(min=0)
    logp_h = log_probs.gather(1, safe_ids)
    if (valid & ~torch.isfinite(logp_h)).any():
        raise ValueError('sampler top-k token has zero current probability: sampler and trainer masks differ')
    zero = torch.zeros_like(logp_h)
    with torch.no_grad():
        p_all = log_probs.detach().exp()
        p_h = torch.where(valid, logp_h.detach().exp(), zero)
        q_h = torch.where(valid, top_logq.exp(), zero)
        ratio_h = p_h / q_h.clamp(min=torch.finfo(torch.float32).tiny)
        w_h = torch.where(valid, importance_weight(ratio_h, importance), zero)
        in_h = torch.zeros_like(p_all).scatter_add(1, safe_ids, valid.float()) > 0
        p_tail = torch.where(in_h, torch.zeros_like(p_all), p_all)
        tail_p = p_tail.sum(1)
        tail_q = 1 - q_h.sum(1)
        rho = tail_q.clamp(min=eps) / tail_p.clamp(min=eps)
        alpha = rho * importance_weight(1 / rho, importance)
        coefficient = q_h * w_h
    logp_y = log_probs.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        w_y = importance_weight((logp_y.detach() - behavior_logp.float()).clamp(-20, 20).exp(), importance)
    head = (coefficient * torch.where(valid, logp_h, zero)).sum(1)
    tail = (p_tail * torch.where(p_tail > 0, log_probs, torch.zeros_like(log_probs))).sum(1)
    surrogate = w_y * logp_y - head - alpha * tail
    diagnostics = {'rho': rho, 'alpha': alpha, 'tail_q': tail_q, 'tail_p': tail_p, 'w_y': w_y,
                   'y_in_top_k': (top_ids == labels[:, None]).any(1)}
    return surrogate, diagnostics


def score_centered_token_loss(log_probs, labels, behavior_logp, top_ids, top_logq, advantage, *,
                              importance='none', eps=1e-6):
    """L_t = -R U_t, the declared score-centered/1 loss of each sampled position."""
    a = torch.as_tensor(advantage, dtype=torch.float32, device=log_probs.device)
    if not torch.isfinite(a).all():
        raise ValueError('nonfinite advantage')
    surrogate, diagnostics = score_centered_surrogate(log_probs, labels, behavior_logp, top_ids, top_logq,
                                                      importance=importance, eps=eps)
    return -a * surrogate, diagnostics
