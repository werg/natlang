"""On-Policy Privileged Self-Distillation (OPSD) for LFM3-1.7B (`plans/NATLANG_COMPILER.md` §8.2, §9.3).

Generates dense, voice-matched token-level supervision over BOTH reasoning and action tokens
from LFM3's own privileged-context forward pass on failed on-policy student rollouts—with
strict anti-shortcut safeguards:
1. Literal expected answers are NEVER placed in the privileged hindsight block (only verifier
   diffs, violated spec clauses, and structural failure diagnoses).
2. Tokens whose privileged-to-student log-ratio exceeds `max_log_ratio` (default 6.0 nats)
   are masked out of the JSD loss unless marked as structural syntax tokens, preventing
   hindsight-only answer jumps from corrupting unprivileged policy gradients.
3. Held-v3 / held-v4 closures are rejected before any privileged context is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import torch
import torch.nn.functional as F

OPSD_PRIVILEGED_HEADER = (
    "[HINDSIGHT DIAGNOSTIC — INTERNAL GUIDANCE ONLY]\n"
    "The previous attempt failed verification. Use the structural diagnosis below to reason "
    "carefully from the visible specification and inputs:"
)


@dataclass(frozen=True)
class VerifierDiagnosis:
    task_id: str
    failure_kind: str                  # e.g. "misread_spec", "tool_idiom", "wrong_files", "contract_violation"
    verifier_diff: str                 # structural diff / violated contract description
    violated_clauses: tuple[str, ...] = ()
    diverging_turn: int | None = None
    forbidden_literals: tuple[str, ...] = ()  # exact expected answer strings that MUST NOT leak


def _scrub_forbidden_literals(text: str, forbidden: tuple[str, ...]) -> str:
    """Reject or scrub any accidental literal expected-answer echo in verifier diagnostics."""
    cleaned = text
    for lit in forbidden:
        s = str(lit).strip()
        if len(s) >= 4 and s in cleaned:
            cleaned = cleaned.replace(s, "[REDACTED_EXPECTED]")
    return cleaned


def build_opsd_privileged_block(
    diagnosis: VerifierDiagnosis,
    *,
    excluded_task_ids: set[str] | frozenset[str] = frozenset(),
) -> str:
    """Build a shortcut-safe privileged hindsight system block for OPSD.

    Raises ValueError if task_id belongs to held exclusions or if diagnosis carries no actionable info.
    """
    if diagnosis.task_id in excluded_task_ids:
        raise ValueError(f"Refusing to build OPSD privileged block for held-excluded task {diagnosis.task_id!r}")
    diff_clean = _scrub_forbidden_literals(diagnosis.verifier_diff.strip(), diagnosis.forbidden_literals)
    clauses_clean = [
        _scrub_forbidden_literals(c.strip(), diagnosis.forbidden_literals)
        for c in diagnosis.violated_clauses
        if c and c.strip()
    ]
    if not diff_clean and not clauses_clean:
        raise ValueError("VerifierDiagnosis must include a non-empty verifier_diff or violated_clauses")
    # Also collapse repeated whitespace to keep the privileged block compact.
    diff_clean = re.sub(r"\n{3,}", "\n\n", diff_clean)
    lines = [OPSD_PRIVILEGED_HEADER, f"- Failure kind: {diagnosis.failure_kind}"]
    if diagnosis.diverging_turn is not None:
        lines.append(f"- First diverging turn: {diagnosis.diverging_turn}")
    if clauses_clean:
        lines.append("- Violated specification clauses:")
        for clause in clauses_clean:
            lines.append(f"  * {clause}")
    if diff_clean:
        lines.append(f"- Verifier diagnosis: {diff_clean}")
    return "\n".join(lines)


def opsd_jsd_loss(
    student_logits: torch.Tensor,
    privileged_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    syntax_mask: torch.Tensor | None = None,
    topk: int = 64,
    beta: float = 0.5,
    max_log_ratio: float = 6.0,
    ce_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Anti-shortcut-masked top-k JSD(p_student || p_privileged) + optional verified CE.

    Args:
        student_logits: [B, T, V] unprivileged student logits (requires grad)
        privileged_logits: [B, T, V] privileged-context logits (detached)
        labels: [B, T] target token ids (-100 for prompt/tool-output context)
        syntax_mask: optional [B, T] bool marking structural/tool-syntax positions exempt from shortcut clip
        topk: number of privileged nucleus/top logits used for JSD (memory-safe on 128k vocab)
        beta: interpolation weight in JSD_beta (0.5 = symmetric JSD)
        max_log_ratio: mask non-syntax tokens where log(p_priv(y_t) / p_student(y_t)) > max_log_ratio
        ce_weight: weight of verified-target cross-entropy alongside JSD
    """
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    valid = labels != -100
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        zero = student_logits.sum() * 0.0
        return zero, {"n_valid": 0.0, "n_kept": 0.0, "shortcut_masked_frac": 0.0, "jsd": 0.0, "ce": 0.0}

    s_sel = student_logits[valid].float()
    p_sel = privileged_logits[valid].detach().float()
    y_sel = labels[valid]

    s_logp_full = F.log_softmax(s_sel, dim=-1)
    p_logp_full = F.log_softmax(p_sel, dim=-1)

    # Anti-shortcut log-ratio check on target tokens (detached mask so gradients don't game the mask)
    with torch.no_grad():
        s_y = s_logp_full.gather(-1, y_sel.unsqueeze(-1)).squeeze(-1)
        p_y = p_logp_full.gather(-1, y_sel.unsqueeze(-1)).squeeze(-1)
        log_ratio = p_y - s_y
        shortcut = log_ratio > float(max_log_ratio)
        if syntax_mask is not None:
            syn_sel = syntax_mask[valid].to(torch.bool)
            shortcut = shortcut & (~syn_sel)
        kept = ~shortcut

    n_kept = int(kept.sum().item())
    masked_frac = 1.0 - (float(n_kept) / float(n_valid))
    if n_kept == 0:
        zero = student_logits.sum() * 0.0
        return zero, {
            "n_valid": float(n_valid),
            "n_kept": 0.0,
            "shortcut_masked_frac": masked_frac,
            "jsd": 0.0,
            "ce": 0.0,
        }

    s_k = s_sel[kept]
    p_k = p_sel[kept]
    y_k = y_sel[kept]
    s_logp_k = s_logp_full[kept]

    k_eff = min(int(topk), s_k.shape[-1])
    p_top_vals, top_idx = torch.topk(p_k, k_eff, dim=-1)
    p_top_logp = F.log_softmax(p_top_vals, dim=-1)
    # Renormalize student log-probs onto the same top-k support for well-posed JSD
    s_top_logp = F.log_softmax(s_k.gather(-1, top_idx), dim=-1)
    p_prob = p_top_logp.exp()
    s_prob = s_top_logp.exp()

    m_prob = (beta * s_prob + (1.0 - beta) * p_prob).clamp_min(1e-12)
    m_logp = m_prob.log()
    kl_s_m = (s_prob * (s_top_logp - m_logp)).sum(dim=-1)
    kl_p_m = (p_prob * (p_top_logp - m_logp)).sum(dim=-1)
    jsd = (beta * kl_s_m + (1.0 - beta) * kl_p_m).mean()

    ce = F.nll_loss(s_logp_k, y_k, reduction="mean") if ce_weight > 0.0 else (jsd * 0.0)
    total = (1.0 - float(ce_weight)) * jsd + float(ce_weight) * ce
    metrics = {
        "n_valid": float(n_valid),
        "n_kept": float(n_kept),
        "shortcut_masked_frac": masked_frac,
        "jsd": float(jsd.detach().item()),
        "ce": float(ce.detach().item()),
        "mean_log_ratio": float(log_ratio[kept].mean().item()),
    }
    return total, metrics
