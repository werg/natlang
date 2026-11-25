"""Recipe T2 trainer: unified LFM3-1.7B Compiler & HKB Live-KB joint training (`plans/NATLANG_COMPILER.md` §9.4).

Implements:
- Dual LR optimizer groups (2e-4 fresh modules + re-fit value head; 2e-5 warm-started 000000002 H/Q backbone)
- 4-part step objectives:
  1. Context distillation (1st Futamura Projection, full or top-k+tail KL + CE, additive -> header-free curriculum)
  2. Live-KB cross-task experience learning (differentiable positives + hard distractors + null-memory restraint)
  3. Judge / decision site calibration
  4. Tier-1.5 FO-MAML / Reptile inner loop on z_site [B, M_site, 256]
- Mandatory shuffled-memory ablation & reader attribution check (kv_only vs adapter_only vs joint vs base)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CompilerTrainConfig:
    total_steps: int = 20_000
    fresh_lr: float = 2e-4
    warm_lr: float = 2e-5
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    warmup_steps: int = 500
    gate_floor_init: float = 0.1
    gate_floor_steps: int = 1000
    kv_drop: float = 0.1
    adapter_drop: float = 0.1
    topk_kl: int | None = None         # None: full-vocabulary KL; k: top-k + tail bucket
    ce_weight: float = 0.25
    meta_inner_steps: int = 4
    meta_inner_lr: float = 0.05
    meta_prox_lambda: float = 0.01
    reptile_weight: float = 0.1
    memory_mode: str = "positioned"    # "positioned" (Mode P) or "window_exempt" (Mode W)


def gate_floor_for_step(step: int, cfg: CompilerTrainConfig) -> float:
    """Linearly anneal straight-through MemoryKVBank gate floor 0.1 -> 0.0 over gate_floor_steps."""
    if step >= cfg.gate_floor_steps or cfg.gate_floor_init <= 0.0:
        return 0.0
    frac = max(0.0, 1.0 - float(step) / float(max(1, cfg.gate_floor_steps)))
    return float(cfg.gate_floor_init) * frac


def header_dropout_prob(step: int, total_steps: int) -> float:
    """Additive for first 30% of steps, then ramp static-header dropout 0.5 -> 1.0 for headerFree mode."""
    frac = float(step) / float(max(1, total_steps))
    if frac < 0.30:
        return 0.0
    ramp = min(1.0, (frac - 0.30) / 0.70)
    return 0.5 + 0.5 * ramp


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    topk: int | None = None,
) -> torch.Tensor:
    """Mean KL(p_teacher || p_student) over supervised positions (mask True/1).

    ``topk`` None (default): the full distributions (``common.stable_softmax.forward_kl``, chunked, so it fits for the
    LFM3 vocabulary). ``topk`` k: the teacher's top-k plus one tail bucket holding the rest of its mass, against the
    student's mass at the same k ids and outside them (``stable_softmax.topk_tail_divergence``). Neither side is
    renormalized over the k, so the student pays for mass it puts outside the teacher's head (the top-k-only form
    this replaces renormalized the teacher and let that mass go unpenalized)."""
    from ..common import stable_softmax as S
    valid = mask.to(torch.bool)
    if not bool(valid.any()):
        return student_logits.sum() * 0.0
    s_sel = student_logits[valid]
    t_sel = teacher_logits[valid].detach()
    if topk is None:
        return S.forward_kl(s_sel, t_sel, reduction='mean')
    t_logp = torch.log_softmax(t_sel.float(), dim=-1)
    k_eff = min(int(topk), t_logp.shape[-1])
    head, ids = torch.topk(t_logp, k_eff, dim=-1)
    tail = torch.logsumexp(t_logp.scatter(-1, ids, float('-inf')), dim=-1)
    return S.topk_tail_divergence(s_sel, ids, head, 'forward_kl', teacher_log_tail=tail, reduction='mean')


def context_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    topk: int | None = None,
    ce_weight: float = 0.25,
) -> torch.Tensor:
    """1st Futamura Projection objective: KL to the full-context teacher + output-only CE (prompt labels are -100).

    ``labels`` are the verified continuation's tokens (the teacher answer or a verified own success); never pass a
    failed sample as labels."""
    mask = labels != -100
    kl = distillation_kl_loss(student_logits, teacher_logits, mask, topk=topk)
    if ce_weight <= 0.0 or not bool(mask.any()):
        return kl
    ce = F.cross_entropy(student_logits[mask].float(), labels[mask], reduction="mean")
    return kl + float(ce_weight) * ce


def build_dual_lr_optimizer(
    *,
    fresh_modules: list[torch.nn.Module],
    warm_modules: list[torch.nn.Module],
    cfg: CompilerTrainConfig,
) -> torch.optim.AdamW:
    """FP32 AdamW with 2e-4 for fresh compiler/readers/re-fit value head and 2e-5 for warm 000000002 backbone."""
    fresh_params: list[torch.nn.Parameter] = []
    seen = set()
    for mod in fresh_modules:
        for p in mod.parameters():
            if p.requires_grad and id(p) not in seen:
                if p.dtype != torch.float32:
                    raise TypeError(f"LFM3 compiler trainable param must be FP32 master, got {p.dtype}")
                seen.add(id(p))
                fresh_params.append(p)
    warm_params: list[torch.nn.Parameter] = []
    for mod in warm_modules:
        for p in mod.parameters():
            if p.requires_grad and id(p) not in seen:
                if p.dtype != torch.float32:
                    raise TypeError(f"Warm writer trainable param must be FP32 master, got {p.dtype}")
                seen.add(id(p))
                warm_params.append(p)
    groups = []
    if fresh_params:
        groups.append({"params": fresh_params, "lr": cfg.fresh_lr, "weight_decay": cfg.weight_decay})
    if warm_params:
        groups.append({"params": warm_params, "lr": cfg.warm_lr, "weight_decay": cfg.weight_decay})
    if not groups:
        raise ValueError("No trainable FP32 parameters found for Compiler optimizer")
    return torch.optim.AdamW(groups)


def fo_maml_inner_adapt_z(
    z_init: torch.Tensor,
    k_slots: torch.Tensor,
    weights: torch.Tensor,
    loss_fn,
    *,
    steps: int = 4,
    inner_lr: float = 0.05,
    prox_lambda: float = 0.01,
) -> torch.Tensor:
    """First-order MAML inner loop: K SGD steps on z_site only (proximal pull to z_init), returning z_K for the query loss.

    The inner updates are taken without second-order terms, and the returned tensor is
    ``z_init + sg(z_K - z_init)``: its value is the adapted z_K, and its Jacobian with respect to ``z_init`` is the
    identity (the first-order approximation). The query loss computed at the returned z therefore reaches
    ``HeaderEncoder`` through ``z_init`` and the head through its own use of z, instead of stopping at a detached leaf.
    """
    z_ref = z_init.detach()
    z = z_ref.clone().requires_grad_(True)
    for _ in range(int(steps)):
        support_loss = loss_fn(z, k_slots.detach(), weights.detach())
        if prox_lambda > 0.0:
            support_loss = support_loss + 0.5 * float(prox_lambda) * (z - z_ref).pow(2).mean()
        (grad_z,) = torch.autograd.grad(support_loss, z, create_graph=False)
        z = (z - float(inner_lr) * grad_z).detach().requires_grad_(True)
    return z_init + (z.detach() - z_ref)


@torch.no_grad()
def evaluate_reader_and_shuffle_ablations(
    decoder,
    head,
    *,
    tokens: torch.Tensor,
    blocks: torch.Tensor,
    labels: torch.Tensor,
    z_slots: torch.Tensor,
    k_slots: torch.Tensor,
    weights: torch.Tensor,
    memory_mode: str = "positioned",
) -> dict[str, float]:
    """Mandatory AGENTS.md shuffled-memory + reader-attribution diagnostic on a paired batch.

    Returns mean output-token NLL for: base, joint, shuffled, kv_only, adapter_only, and
    the key deltas (base_minus_joint, shuffled_minus_joint). A shuffled_minus_joint <= 0
    indicates the consumer is ignoring slot content (a defect).
    """
    was_training = head.training
    head.eval()
    mask = labels != -100

    def _nll(logits: torch.Tensor) -> float:
        if not bool(mask.any()):
            return float("nan")
        return float(F.cross_entropy(logits[mask].float(), labels[mask], reduction="mean").item())

    try:
        base_nll = _nll(decoder(tokens, blocks).logits)

        def _run(z_in, k_in, w_in, *, kv_en: bool, ad_en: bool) -> float:
            compiled = head(z_in, k_in, w_in, kv_enabled=kv_en, adapter_enabled=ad_en)
            out = decoder(
                tokens,
                blocks,
                controls=compiled.c_cores if ad_en else None,
                memory_kv=compiled.mem_kv if kv_en else None,
                memory_valid=compiled.mem_valid if kv_en else None,
                memory_gate=compiled.mem_gate if kv_en else None,
                memory_mode=memory_mode,
            )
            return _nll(out.logits)

        joint_nll = _run(z_slots, k_slots, weights, kv_en=True, ad_en=True)
        kv_only_nll = _run(z_slots, k_slots, weights, kv_en=True, ad_en=False)
        adapter_only_nll = _run(z_slots, k_slots, weights, kv_en=False, ad_en=True)

        # Deterministic cyclic roll across batch (or across slots if batch==1) for shuffled control
        if z_slots.shape[0] > 1:
            z_shuf = torch.roll(z_slots, shifts=1, dims=0)
            k_shuf = torch.roll(k_slots, shifts=1, dims=0)
            w_shuf = torch.roll(weights, shifts=1, dims=0)
        else:
            z_shuf = torch.roll(z_slots, shifts=1, dims=1) * -1.0
            k_shuf = torch.roll(k_slots, shifts=1, dims=1)
            w_shuf = torch.roll(weights, shifts=1, dims=1)
        shuffled_nll = _run(z_shuf, k_shuf, w_shuf, kv_en=True, ad_en=True)

        return {
            "base_nll": base_nll,
            "joint_nll": joint_nll,
            "shuffled_nll": shuffled_nll,
            "kv_only_nll": kv_only_nll,
            "adapter_only_nll": adapter_only_nll,
            "base_minus_joint": base_nll - joint_nll,
            "shuffled_minus_joint": shuffled_nll - joint_nll,
        }
    finally:
        head.train(was_training)
