"""Supervised objectives of the learned optimizer (plan "Behavior-first supervision", "Supervised matching of native
states", "Adapter and structural supervision", "Downstream supervised behavior").

``L_sup = l_D L_distill + l_P L_instruction-state + l_S L_skill-state + l_M L_mixture-tokens + l_C L_core/weight
         + l_H L_head + l_A L_structure + l_B L_downstream``

Distillation and downstream behavior carry the primary signal; the state, mixture-token, core and head terms are
auxiliaries whose gradient shares are reported (`gradient_shares`) and kept a minority. Mixture-token CE covers only
the free sign/digit tokens of available slots, a mask disjoint from the structural token terms.
"""
from __future__ import annotations

import math

import torch

from . import LearnedOptimizerError
from .numbers import magnitude_logprobs, sign_logprobs

TERMS = ('distill', 'instruction_state', 'skill_state', 'mixture_tokens', 'core', 'head', 'structure', 'downstream')


# -- executor behavior ---------------------------------------------------------------------------------------------
def supervised_logprobs(model, example, *, external_blocks=None, external_identity=None):
    """Full next-token log distributions at the example's positive-weight positions and their weights. Run inside
    the bundle's active native execution (`run_native` with its composed adapters)."""
    result = model(example.plan, external_blocks=external_blocks, external_identity=external_identity, retain_states=False)
    hidden = result['h_final'][0]
    queries, weights = [], []
    for node, offset in zip(example.plan.invocations, example.plan.offsets):
        for local in range(1, len(node.token_ids)):
            weight = example.target_weights[offset + local]
            if weight > 0:
                queries.append(offset + local - 1)
                weights.append(weight)
    if not queries:
        raise LearnedOptimizerError('the example has no supervised reply/action positions')
    logits = model.logits(hidden[torch.tensor(queries, device=hidden.device)])
    return torch.log_softmax(logits.float(), -1), torch.tensor(weights, device=hidden.device, dtype=torch.float32)


def distill_loss(target_logp, student_logp, weights):
    """Weighted mean over supervised positions of ``KL(p_target || p_student)``; ``target_logp`` is detached."""
    if target_logp.shape != student_logp.shape:
        raise LearnedOptimizerError('target and student distributions cover different positions')
    target_logp = target_logp.detach()
    kl = (target_logp.exp() * (target_logp - student_logp)).sum(-1)
    return (kl * weights).sum() / weights.sum()


def gold_action_loss(student_logp, labels, weights):
    """Cross-entropy on verified actions beside the distillation term, so the learner can exceed a wrong target."""
    picked = student_logp.gather(-1, labels[:, None])[:, 0]
    return -(picked * weights).sum() / weights.sum()


# -- native states -------------------------------------------------------------------------------------------------
STATE_FIELDS = ('raw_k', 'v', 'gated_inputs', 'mixed_qkv', 'z', 'b', 'a')


def field_scales(blocks, eps=1e-6):
    """Per (layer, field) scale ``sigma`` from training targets only."""
    sums = {}
    for block in blocks:
        for index, layer in enumerate(block.layers):
            for name in STATE_FIELDS:
                if name in layer:
                    value = layer[name].detach().float()
                    total, square, count = sums.get((index, name), (0., 0., 0))
                    sums[(index, name)] = (total + float(value.sum()), square + float((value ** 2).sum()), count + value.numel())
    return {key: math.sqrt(max(square / count - (total / count) ** 2, 0.)) + eps for key, (total, square, count) in sums.items()}


def state_loss(captured, target, scales, *, robust=None):
    """Normalized state matching: mean over layers/fields of the mean ``rho((T_hat - T*) / sigma)`` over entries.
    Alignment is exact or rejected: the same token IDs and slot count, never padding or truncation."""
    if not torch.equal(captured.token_ids.cpu(), target.token_ids.cpu()):
        raise LearnedOptimizerError('state target and captured span differ in token alignment; use behavior-only supervision')
    terms, report = [], {}
    for index, (ours, theirs) in enumerate(zip(captured.layers, target.layers)):
        if ours['kind'] != theirs['kind']:
            raise LearnedOptimizerError('state target and captured span differ in layer layout')
        for name in STATE_FIELDS:
            if name in ours:
                scale = scales[(index, name)] if (index, name) in scales else scales[f'{index}.{name}']
                error = (ours[name].float() - theirs[name].detach().float().to(ours[name].device)) / scale
                value = (error ** 2 if robust is None else robust(error)).mean()
                terms.append(value)
                report[f'{index}.{name}'] = float(value.detach())
    if not terms:
        raise LearnedOptimizerError('no comparable state fields')
    return torch.stack(terms).mean(), report


# -- weight arrays -------------------------------------------------------------------------------------------------
def mixture_token_loss(grid, fmt, sign_logits, digit_logits, targets, available, *, zero_share=.5):
    """Grammar-restricted CE of the target grid integers at available slots (realized branch teacher-forced).

    Zero and nonzero targets are balanced: zeros carry ``zero_share`` of the weight whenever both kinds occur, so
    abundant masked zeros do not dominate useful coefficient choices."""
    free = [slot for slot, flag in enumerate(available) if flag]
    if not free:
        return sign_logits.new_zeros((), dtype=torch.float32), {}
    losses, kinds = [], []
    for slot in free:
        n = int(targets[slot])
        signs = sign_logprobs(grid, sign_logits[slot])
        magnitude = magnitude_logprobs(grid, digit_logits[slot], negative=n < 0, max_int=fmt.max_int)
        losses.append(-(signs[1 if n < 0 else 0] + magnitude[abs(n)]))
        kinds.append(n == 0)
    zeros, nonzeros = sum(kinds), len(kinds) - sum(kinds)
    if zeros and nonzeros:
        weights = [zero_share / zeros if kind else (1 - zero_share) / nonzeros for kind in kinds]
    else:
        weights = [1. / len(kinds)] * len(kinds)
    loss = sum(w * l for w, l in zip(weights, losses))
    return loss, {'slots': len(free), 'zero_targets': zeros}


def decoded_weight_loss(bank, cores, targets, *, residuals=None, eps=1e-12):
    """``mean_m ||U C_hat V^T - Delta W*||^2 / (||Delta W*||^2 + eps)`` with targets as ``{m: (B, A)}`` factors,
    computed in basis coordinates without dense expansion (orthonormal ``U, V``):
    ``||C_hat||^2 - 2 <C_hat, U^T B A V> + ||B A||^2``. ``residuals`` adds nothing differentiable; it is reported."""
    from ..gates import factor_sq_norm
    terms = []
    for name, (B, A) in targets.items():
        energy = factor_sq_norm(B, A)
        if name not in cores:
            terms.append(torch.ones((), device=B.device))
            continue
        core = cores[name]
        U, V = bank.basis.left[name].to(core), bank.basis.right[name].to(core)
        projected = (U.transpose(-2, -1) @ B.to(core)) @ (A.to(core) @ V)
        error = (core ** 2).sum() - 2 * (core * projected).sum() + energy.to(core)
        terms.append(error / (energy.to(core) + eps))
    return torch.stack(terms).mean()


def head_regression(rho, delta, target_rho, target_delta, available):
    """Normalized regression of the head's refinement (grid steps, bound 1/2) and modulation (bound beta)."""
    mask = torch.tensor(available, dtype=torch.bool, device=rho.device)
    if not bool(mask.any()):
        return rho.new_zeros(())
    rho_term = ((rho - target_rho.to(rho)) ** 2)[mask].mean() / .25
    delta_term = ((delta - target_delta.to(delta)) ** 2)[mask].mean()
    return rho_term + delta_term


# -- composition of the terms --------------------------------------------------------------------------------------
def combine(terms, weights):
    unknown = set(terms) - set(TERMS)
    if unknown:
        raise LearnedOptimizerError(f'unknown objective terms {sorted(unknown)}')
    total = None
    for name, value in terms.items():
        weight = float(weights.get(name, 0.))
        if weight == 0:
            continue
        total = weight * value if total is None else total + weight * value
    if total is None:
        raise LearnedOptimizerError('every objective term has zero weight')
    return total


def gradient_shares(terms, weights, groups):
    """Per trainable group, each weighted term's gradient norm and share of the group's total. Reporting only:
    auxiliaries above their declared share are a recipe signal (plan: keep them a minority)."""
    report = {}
    for group, params in groups.items():
        params = [param for param in params if param.requires_grad]
        norms = {}
        for name, value in terms.items():
            weight = float(weights.get(name, 0.))
            if weight == 0 or not value.requires_grad:
                continue
            grads = torch.autograd.grad(weight * value, params, retain_graph=True, allow_unused=True)
            norms[name] = math.sqrt(sum(float(g.float().pow(2).sum()) for g in grads if g is not None))
        total = sum(norms.values())
        report[group] = {name: {'norm': norm, 'share': norm / total if total else 0.} for name, norm in norms.items()}
    return report
