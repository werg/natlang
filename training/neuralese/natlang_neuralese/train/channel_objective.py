"""Shared same-generated-history channel objective and bounded readout math.

The ordinary consumer is a live, stop-gradient target. Only the projected
consumer receives gradients; unrelated gold tails are not training targets.
Training and evaluation use the same distribution comparison.
"""
import torch
from torch.utils.checkpoint import checkpoint

from ..maple.model import eager_rms_norm


def through_first_close(tokens, close_id):
    """Include the first emitted close decision and exclude later positions."""
    closes = tokens.eq(close_id).to(torch.int64)
    return (closes.cumsum(dim=1) - closes).eq(0)


def chunked_distribution_comparison(backbone, plain_states, projected_states, *,
                                    chunk_size=128, gradients=True):
    """Exact KL(plain || projected) and argmax agreement in bounded vocab blocks."""
    if plain_states.shape != projected_states.shape or plain_states.ndim != 3:
        raise ValueError('identical [batch,time,width] consumer states required')
    if chunk_size < 1 or projected_states.shape[1] < 1:
        raise ValueError('positive chunk size and nonempty sequence required')
    needs_grad = gradients and torch.is_grad_enabled() and (
        projected_states.requires_grad or any(p.requires_grad for p in backbone.parameters()))
    divergences, agreements = [], []
    def compare(plain, projected):
        with torch.no_grad(), eager_rms_norm():
            plain_logits = backbone.logits(plain).float()
            plain_logp = torch.log_softmax(plain_logits, dim=-1)
        with eager_rms_norm():
            projected_logits = backbone.logits(projected).float()
        projected_logp = torch.log_softmax(projected_logits, dim=-1)
        kl = (plain_logp.exp() * (plain_logp - projected_logp)).sum(dim=-1)
        with torch.no_grad():
            agreement = plain_logits.argmax(-1).eq(projected_logits.argmax(-1))
        return kl, agreement
    for start in range(0, projected_states.shape[1], chunk_size):
        plain = plain_states[:, start:start + chunk_size].detach()
        projected = projected_states[:, start:start + chunk_size]
        if needs_grad:
            kl, agreement = checkpoint(compare, plain, projected, use_reentrant=False)
        elif gradients:
            kl, agreement = compare(plain, projected)
        else:
            with torch.no_grad():
                kl, agreement = compare(plain, projected)
        divergences.append(kl)
        agreements.append(agreement.detach())
    return torch.cat(divergences, dim=1), torch.cat(agreements, dim=1)


def generated_history_channel_loss(backbone, plain_states, projected_states,
                                   generated_tokens, close_id, *, chunk_size=128,
                                   gradients=True):
    """Train channel faithfulness through close, including past gold divergence."""
    if generated_tokens.shape != projected_states.shape[:2]:
        raise ValueError('generated decisions must align with consumer states')
    kl, agreement = chunked_distribution_comparison(
        backbone, plain_states, projected_states, chunk_size=chunk_size, gradients=gradients)
    mask = through_first_close(generated_tokens, close_id)
    count = mask.sum()
    loss = (kl * mask).sum() / count.clamp_min(1)
    tail_mask = mask[:, -256:]
    tail_count = tail_mask.sum()
    return loss, {'channel_consistency_kl': loss.detach(),
                  'channel_consistency_agreement': (agreement.float() * mask).sum().detach() / count.clamp_min(1),
                  'channel_consistency_tokens': count.detach(),
                  'channel_consistency_last256_kl': (kl[:, -256:] * tail_mask).sum().detach() / tail_count.clamp_min(1),
                  'channel_consistency_last256_agreement': (agreement[:, -256:].float() * tail_mask).sum().detach() / tail_count.clamp_min(1),
                  'channel_consistency_last256_tokens': tail_count.detach()}


def generated_tail_projection_loss(backbone, heads, ordinary_states, generated_tokens,
                                   gold_prefix_mask, close_id, *, chunk_size=128,
                                   group_ids=None, role_ids=None, role_names=()):
    """Distill the full projector after gold targets stop belonging to this history.

    The live ordinary consumer already ran on these same generated decisions.
    Its detached next-token choice supplies raw embedding targets, requiring no
    additional backbone pass. The added gradient reaches only the full projector;
    gold supervision, including the first difficult decision, remains separate.
    """
    from .projection_anchor import relative_mse_positions
    if (ordinary_states.shape[:2] != generated_tokens.shape or
            gold_prefix_mask.shape != generated_tokens.shape or
            gold_prefix_mask.dtype != torch.bool or chunk_size < 1):
        raise ValueError('aligned generated history and boolean gold prefix required')
    if group_ids is not None and len(group_ids) != ordinary_states.shape[0]:
        raise ValueError('one diagnostic group is required per batch row')
    if role_ids is not None and role_ids.shape != generated_tokens.shape:
        raise ValueError('aligned role IDs are required for role diagnostics')
    mask = through_first_close(generated_tokens, close_id) & ~gold_prefix_mask
    errors = []
    for start in range(0, ordinary_states.shape[1], chunk_size):
        states = ordinary_states[:, start:start + chunk_size].detach()
        with torch.no_grad(), eager_rms_norm():
            next_ids = backbone.logits(states).argmax(-1)
            targets = backbone.embed(next_ids).detach()
        projected = heads.content(torch.zeros_like(states), states)
        errors.append(relative_mse_positions(projected, targets))
    positions = torch.cat(errors, dim=1)
    count = mask.sum()
    loss = (positions * mask).sum() / count.clamp_min(1)
    metrics = {'generated_tail_projection_mse': loss.detach(),
               'generated_tail_projection_tokens': count.detach()}
    # Keep only scalar sums/counts for diagnostics. These preserve the exact
    # selected-tail weighting when held windows are aggregated across batches.
    if group_ids is not None:
        grouped = {'by_cohort': {}, 'by_cohort_role': {}}
        for row, group in enumerate(group_ids):
            selected = mask[row]
            error_sum = (positions[row] * selected).sum().detach()
            selected_count = selected.sum().detach()
            if group not in grouped['by_cohort']:
                grouped['by_cohort'][group] = {'error_sum': error_sum,
                                               'tokens': selected_count}
            else:
                grouped['by_cohort'][group]['error_sum'] = (
                    grouped['by_cohort'][group]['error_sum'] + error_sum)
                grouped['by_cohort'][group]['tokens'] = (
                    grouped['by_cohort'][group]['tokens'] + selected_count)
            if role_ids is None:
                continue
            for role_code, role_name in enumerate(role_names):
                role_selected = selected & role_ids[row].eq(role_code)
                role_error_sum = (positions[row] * role_selected).sum().detach()
                role_count = role_selected.sum().detach()
                by_role = grouped['by_cohort_role'].setdefault(group, {})
                if role_name not in by_role:
                    by_role[role_name] = {'error_sum': role_error_sum,
                                          'tokens': role_count}
                else:
                    by_role[role_name]['error_sum'] += role_error_sum
                    by_role[role_name]['tokens'] += role_count
        metrics['generated_tail_projection_groups'] = grouped
    return loss, metrics
