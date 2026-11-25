"""Single-Pass Calibrated `readout: decision` Compilation (`plans/NATLANG_COMPILER.md` §6.1, W2.1).

Compiles finite-union `readout: decision` functions into single-forward-pass calibrated
option scorers over `POST /v1/natlang/score`:
- Hard-negative margin + soft-KL training objective over option log-probabilities
- Scalar temperature scaling T_f fit on the development split minimizing NLL / ECE
- Margin-gated fast-path settlement: settle in 1 prefill (0 decode tokens) when calibrated
  probability >= fast_threshold (and clears any asymmetric risk floor), else escalate/CoT.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, *, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) over top-1 predicted confidences."""
    p = probs.detach().float()
    y = targets.detach().long()
    conf, pred = p.max(dim=-1)
    correct = (pred == y).float()
    ece = 0.0
    n = max(1, y.numel())
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        cnt = int(in_bin.sum().item())
        if cnt == 0:
            continue
        acc = float(correct[in_bin].mean().item())
        avg_conf = float(conf[in_bin].mean().item())
        ece += (cnt / n) * abs(acc - avg_conf)
    return float(ece)


def fit_temperature(
    option_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    min_temp: float = 0.1,
    max_temp: float = 10.0,
    steps: int = 80,
) -> CalibrationResult:
    """Fit scalar temperature T_f > 0 on development option logits minimizing NLL."""
    logits = option_logits.detach().float()
    y = targets.detach().long()
    if logits.ndim != 2 or logits.shape[0] != y.shape[0] or logits.shape[0] == 0:
        raise ValueError("option_logits must be [N, K] matching targets [N]")

    with torch.no_grad():
        p0 = F.softmax(logits, dim=-1)
        nll0 = float(F.cross_entropy(logits, y).item())
        ece0 = compute_ece(p0, y)

    log_t = torch.zeros((), dtype=torch.float32, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.25, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        t = torch.clamp(log_t.exp(), min=min_temp, max=max_temp)
        loss = F.cross_entropy(logits / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    t_final = float(torch.clamp(log_t.detach().exp(), min=min_temp, max=max_temp).item())
    with torch.no_grad():
        scaled = logits / t_final
        nll1 = float(F.cross_entropy(scaled, y).item())
        ece1 = compute_ece(F.softmax(scaled, dim=-1), y)

    if nll1 > nll0:
        return CalibrationResult(temperature=1.0, nll_before=nll0, nll_after=nll0, ece_before=ece0, ece_after=ece0)
    return CalibrationResult(temperature=t_final, nll_before=nll0, nll_after=nll1, ece_before=ece0, ece_after=ece1)


def decision_margin_loss(
    option_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    soft_teacher_probs: torch.Tensor | None = None,
    margin: float = 1.0,
    hard_neg_weight: float = 0.5,
    soft_kl_weight: float = 0.5,
    rule: str = 'logLoss',
) -> torch.Tensor:
    """Proper decision score (the shared ``common.decision_objective``) + hardest-negative margin.

    The proper-score part is the one finite-decision equation every readout learner uses: against the verified label,
    plus ``soft_kl_weight`` times the same score against the teacher's option distribution when one is given (with
    logLoss that is a KL to the teacher up to a constant). The hinge adds ``max(0, margin - (s_y - max_{j != y} s_j))``.
    """
    from ..common.decision_objective import decision_loss_to_goal
    logits = option_logits.float()
    y = targets.long()
    b, k = logits.shape
    onehot = F.one_hot(y, num_classes=k).float()
    proper = torch.stack([decision_loss_to_goal(logits[i], onehot[i], rule=rule) for i in range(b)]).mean()
    total = proper
    if soft_teacher_probs is not None and soft_kl_weight > 0.0:
        goal = soft_teacher_probs.detach().float().clamp_min(0.0)
        total = total + float(soft_kl_weight) * torch.stack(
            [decision_loss_to_goal(logits[i], goal[i], rule=rule) for i in range(b)]).mean()
    gold = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    hardest_neg = logits.masked_fill(onehot.bool(), float('-inf')).max(dim=-1).values
    hinge = F.relu(float(margin) - (gold - hardest_neg)).mean()
    return total + float(hard_neg_weight) * hinge


def should_settle_fast(
    calibrated_probs: list[float] | torch.Tensor,
    labels: list[str],
    *,
    fast_threshold: float = 0.85,
    risk_thresholds: dict[str, float] | None = None,
) -> tuple[bool, str, float]:
    """Decide whether single-pass calibrated decision settles locally or falls back to CoT/escalate.

    For asymmetric `risk` thresholds (e.g. `{block: 0.25}`), if any risky label has probability
    >= its risk threshold and is not the confident top-1 label, force fallback instead of settling.
    """
    probs = calibrated_probs.tolist() if torch.is_tensor(calibrated_probs) else list(calibrated_probs)
    if len(probs) != len(labels) or not labels:
        raise ValueError("calibrated_probs and labels must be nonempty and equal length")
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    best_label = labels[best_idx]
    best_prob = float(probs[best_idx])

    if risk_thresholds:
        for label, prob in zip(labels, probs):
            floor = risk_thresholds.get(label)
            if floor is not None and float(prob) >= float(floor) and label != best_label:
                return False, best_label, best_prob
    return best_prob >= float(fast_threshold), best_label, best_prob
