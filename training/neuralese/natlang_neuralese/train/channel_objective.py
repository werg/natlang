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
    loss = (kl * mask).sum() / count
    return loss, {'channel_consistency_kl': loss.detach(),
                  'channel_consistency_agreement': (agreement.float() * mask).sum().detach() / count,
                  'channel_consistency_tokens': count.detach()}


def generated_tail_projection_loss(backbone, heads, ordinary_states, generated_tokens,
                                   gold_prefix_mask, close_id, *, chunk_size=128):
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
    return loss, {'generated_tail_projection_mse': loss.detach(),
                  'generated_tail_projection_tokens': count.detach()}
