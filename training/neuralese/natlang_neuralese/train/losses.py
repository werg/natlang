"""Losses for curriculum phases A–D (S3 §5.1).

All losses return a scalar plus a dict of detached metrics. Causal alignment is kept
throughout: a decision at position i sees h_k[i] and never the input it predicts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model.heads import PortHeads
from ..model.lfm2_port import PortBackbone
from .execution import (Prefilled, Written, consumer_forward, consumer_forward_batch, parallel_write, prefill,
                        prefill_batch, read_continue, stop_log_prob, supplied_inputs, teacher_logits_batch,
                        teacher_target_logits, unroll_write)


def _ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))


def _kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) per position, averaged."""
    t = F.log_softmax(teacher_logits.float(), -1)
    s = F.log_softmax(student_logits.float(), -1)
    return (t.exp() * (t - s)).sum(-1).mean()


def stop_boundary_loss(written: Written) -> torch.Tensor:
    """BCE on closure: stop after exactly `lengths` vectors, continue before."""
    length = written.stop_logits.shape[1]
    counts = torch.arange(1, length + 1, device=written.stop_logits.device)[None]
    target = (counts == written.lengths[:, None]).float()
    valid = counts <= written.lengths[:, None]
    loss = F.binary_cross_entropy_with_logits(written.stop_logits.float(), target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def span_batch(batch, device):
    tensor = lambda rows: torch.tensor(rows, dtype=torch.long, device=device)
    return tensor([e.prefix for e in batch]), tensor([e.span for e in batch]), tensor([e.continuation for e in batch])


def span_loss(backbone: PortBackbone, heads: PortHeads, batch, *, generated_fraction: float = 0.0,
              passes: int = 2, unroll: bool = False, entry_weight: float = 0.1, stop_weight: float = 1.0,
              kl_weight: float = 0.0, generator: torch.Generator | None = None, temperature: float = 0.0,
              payload_kl_weight: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Phases A and C on ordinary text with a designated span.

    Writer: the prefix ends with the open marker; the span's known-text embeddings fill the
    sketch positions (fraction 0, phase A), are partly replaced by generated sketches
    (phase C, parallel scheduled sampling), or are replaced entirely by the sequential
    writer (`unroll`). Reader: read back the payload and continue with the text after the
    span. Losses: continuation CE (+ KL to the plain-text teacher), entry (predict the open
    marker at the span start; ordinary CE elsewhere in the prefix), stop boundary. With
    `temperature` > 0 the payload is sampled around its mean; `payload_kl_weight` adds the
    beta-weighted KL of the payload distribution to N(0, I) in normalised space.
    """
    device = backbone.embedding_weight.device
    prefix, span, continuation = span_batch(batch, device)
    open_col = torch.full((prefix.shape[0], 1), backbone.controls.open_id, device=device)
    pre = prefill(backbone, heads, torch.cat([prefix, open_col], 1))
    if unroll:
        written = unroll_write(backbone, heads, pre, length=span.shape[1], generator=generator,
                               temperature=temperature)
    else:
        written = parallel_write(backbone, heads, pre, supplied_inputs(backbone, heads, span),
                                 generated_fraction=generated_fraction, passes=passes, generator=generator,
                                 temperature=temperature)
    logits = read_continue(backbone, heads, pre.cache, written.payload, continuation)
    continuation_loss = _ce(logits, continuation)
    # Entry: logits at prefix positions predict the next prefix token, then the open marker.
    entry_targets = torch.cat([prefix[:, 1:], open_col], 1)
    entry_loss = _ce(pre.logits[:, :-1], entry_targets)
    stop_loss = stop_boundary_loss(written)
    loss = continuation_loss + entry_weight * entry_loss + stop_weight * stop_loss
    metrics = {"continuation_ce": continuation_loss.item(), "entry_ce": entry_loss.item(), "stop_bce": stop_loss.item()}
    if kl_weight > 0:
        with torch.no_grad():
            plain = torch.cat([prefix, span, continuation[:, :-1]], 1)
            teacher = backbone.forward_ids(plain)["logits"][:, -continuation.shape[1]:]
        kl = _kl(logits, teacher)
        loss = loss + kl_weight * kl
        metrics["continuation_kl"] = kl.item()
    loss = _payload_terms(written, loss, metrics, temperature, payload_kl_weight)
    metrics["generated_share"] = written.generated.float().mean().item()
    metrics["loss"] = loss.item()
    return loss, metrics


def _payload_terms(written: Written, loss: torch.Tensor, metrics: dict, temperature: float,
                   payload_kl_weight: float) -> torch.Tensor:
    """Payload-distribution metrics and the optional beta-weighted KL term."""
    sample = written.sample
    valid = written.valid()
    metrics["payload_sigma"] = (sample.log_sigma.exp().mean(-1) * valid).sum().item() / max(1, int(valid.sum()))
    metrics["temperature"] = temperature
    if payload_kl_weight > 0:
        kl = written.kl()
        loss = loss + payload_kl_weight * kl
        metrics["payload_kl"] = kl.item()
    return loss


def policy_gradient_surrogate(written: Written, advantage: torch.Tensor) -> torch.Tensor:
    """REINFORCE surrogate for sampled payloads: -A * log N(z; mu, tau^2 sigma^2), batch mean.

    Gradients reach mu (hence the writer) and sigma with the sampled payload held fixed; use
    with temperature > 0. `advantage` is [B], already baseline-subtracted and detached.
    """
    return -(advantage.detach() * written.log_prob()).mean()


def distill_loss(backbone: PortBackbone, heads: PortHeads, batch, *, kl_weight: float = 1.0,
                 cosine_weight: float = 1.0, mse_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Phase B: the shallow state h_k[i] predicts the next input.

    R_k(h_k[i]) is distilled from the frozen full model's next-token distribution at i, and
    F(h_k[i]) regresses onto the (interface-normed) embedding of token i+1.
    """
    device = backbone.embedding_weight.device
    prefix, span, continuation = span_batch(batch, device)
    ids = torch.cat([prefix, span, continuation], 1)
    with torch.no_grad():
        out = backbone.forward_ids(ids, cutoff=heads.cutoff)
        h = out["h_cut"][:, :-1]
        teacher = out["logits"][:, :-1]
        target = heads.interface(backbone.embed(ids[:, 1:]))
    kl = _kl(heads.feedback.readout_logits(h), teacher)
    predicted = heads.feedback(h)
    cosine = 1 - F.cosine_similarity(predicted.float(), target.float(), dim=-1).mean()
    rms = lambda x: x.float().pow(2).mean(-1, keepdim=True).sqrt()
    mse = F.mse_loss(predicted.float() / rms(predicted), target.float() / rms(target))
    loss = kl_weight * kl + cosine_weight * cosine + mse_weight * mse
    with torch.no_grad():
        agree = (heads.feedback.readout_logits(h).argmax(-1) == teacher.argmax(-1)).float().mean()
    return loss, {"readout_kl": kl.item(), "sketch_cosine_loss": cosine.item(), "sketch_mse": mse.item(),
                  "readout_agreement": agree.item(), "loss": loss.item()}


def consumer_loss(backbone: PortBackbone, heads: PortHeads, rendered, *, kl_weight: float = 1.0,
                  stop_weight: float = 0.0, max_length: int | None = None, temperature: float = 0.0,
                  payload_kl_weight: float = 0.0, generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """Phase D on one rendered port record: producer writes, consumer reads with the source withheld.

    Loss: CE on the consumer's target plus KL to the teacher (the crisp base given the full
    source instead of the block). The write length comes from the stop head (detached
    decision); gradients flow through the payload, P, the completion, the sketch recurrence
    and F. Reinforcing the chosen boundary (`stop_weight`) is off by default: it would only
    entrench the current choice; phase E trains the stop policy against consumer quality.
    """
    device = backbone.embedding_weight.device
    pre = prefill(backbone, heads, torch.tensor([rendered.producer], device=device))
    written = unroll_write(backbone, heads, pre, max_length=max_length, temperature=temperature, generator=generator)
    length = int(written.lengths[0])
    payload = written.payload[:, :length]
    logits = consumer_forward(backbone, heads, rendered.consumer_before, payload, rendered.consumer_after, rendered.target)
    target = torch.tensor([rendered.target], device=device)
    ce = _ce(logits, target)
    teacher = teacher_target_logits(backbone, rendered.teacher_prefix, rendered.target)
    kl = _kl(logits, teacher)
    stop = stop_boundary_loss(written)
    loss = ce + kl_weight * kl + stop_weight * stop
    metrics = {"consumer_ce": ce.item(), "consumer_kl": kl.item(), "block_length": length,
               "truncated": float(written.truncated[0])}
    loss = _payload_terms(written, loss, metrics, temperature, payload_kl_weight)
    metrics["loss"] = loss.item()
    return loss, metrics


def _row_nll(logits: torch.Tensor, target: list[int]) -> torch.Tensor:
    """Mean token NLL of one row's target (a differentiable scalar)."""
    t = torch.tensor(target, device=logits.device)
    return F.cross_entropy(logits.float(), t)


def _row_kl(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    t = F.log_softmax(teacher.float(), -1)
    return (t.exp() * (t - F.log_softmax(student.float(), -1))).sum(-1).mean()


def pooled_payloads(payload: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """[B, d]: each row's mean over its valid vectors, in normalised (unit-RMS) space."""
    valid = (torch.arange(payload.shape[1], device=payload.device)[None] < lengths[:, None]).float()
    u = payload.float() / payload.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    return (u * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)


def diversity_loss(pooled: torch.Tensor, target_std: float = 0.5) -> torch.Tensor:
    """VICReg-style variance hinge across the batch: different sources must not share one block."""
    if pooled.shape[0] < 2:
        return pooled.new_zeros(())
    std = pooled.std(0)
    return F.relu(target_std - std).mean()


def consumer_batch_loss(backbone: PortBackbone, heads: PortHeads, rendered: list, *, kl_weight: float = 1.0,
                        contrastive_weight: float = 0.0, margin: float = 0.5, diversity_weight: float = 0.0,
                        max_length: int | None = None, temperature: float = 0.0, payload_kl_weight: float = 0.0,
                        stop_policy_weight: float = 0.0, length_cost: float = 0.0, policy_samples: int = 1,
                        sample_stop: bool = False, generator: torch.Generator | None = None,
                        teacher_context=None) -> tuple[torch.Tensor, dict]:
    """Phases D and E on a batch of rendered port records (ragged lengths).

    The producer writes (left-padded prefill, lockstep unroll, learned stopping); the
    consumer reads with the source withheld. Terms:

    - CE on the target plus KL to the self-distillation teacher (full source, deltas off):
      per-example content pressure.
    - Contrastive (`contrastive_weight`): a hinge asking the correct payload to beat another
      row's payload (matched length) by `margin` nats per token, so a block that works for
      every example (a shared prompt) is not a solution.
    - Diversity (`diversity_weight`): a variance hinge on pooled payloads across the batch.
    - Phase E (`stop_policy_weight` > 0): stopping is sampled `policy_samples` times per record
      and trained by REINFORCE on reward = -consumer NLL - `length_cost` x length, with the
      mean reward of the record's samples as baseline.
    """
    if policy_samples > 1:
        rendered = [r for r in rendered for _ in range(policy_samples)]
    pre = prefill_batch(backbone, heads, [r.producer for r in rendered])
    written = unroll_write(backbone, heads, pre, max_length=max_length, sample=sample_stop or stop_policy_weight > 0,
                           generator=generator, temperature=temperature)
    lengths = written.lengths.clamp(min=1)
    logits = consumer_forward_batch(backbone, heads, rendered, written.payload, lengths)
    ctx = teacher_context() if teacher_context else _nullcontext()
    with ctx:
        teacher = teacher_logits_batch(backbone, rendered)
    nll = torch.stack([_row_nll(lg, r.target) for lg, r in zip(logits, rendered)])
    kl = torch.stack([_row_kl(lg, t) for lg, t in zip(logits, teacher)])
    loss = nll.mean() + kl_weight * kl.mean()
    metrics = {"consumer_ce": nll.mean().item(), "consumer_kl": kl.mean().item(),
               "block_length": lengths.float().mean().item(), "truncated": written.truncated.float().mean().item()}
    batch = len(rendered)
    if contrastive_weight > 0 and batch > 1:
        step = policy_samples if policy_samples > 1 else 1
        perm = [(b + step) % batch for b in range(batch)]
        other = written.payload[perm]
        # Matched length: the other row's vectors, cycled to this row's length.
        reps = -(-int(lengths.max()) // max(1, int(lengths.min())))
        cycled = torch.stack([other[b, : int(lengths[perm[b]])].repeat(reps + 1, 1)[: other.shape[1]] for b in range(batch)])
        neg_logits = consumer_forward_batch(backbone, heads, rendered, cycled, lengths)
        neg = torch.stack([_row_nll(lg, r.target) for lg, r in zip(neg_logits, rendered)])
        hinge = F.relu(margin - (neg - nll)).mean()
        loss = loss + contrastive_weight * hinge
        metrics.update({"contrastive_hinge": hinge.item(), "shuffled_ce": neg.mean().item(),
                        "correct_minus_shuffled": (nll - neg).mean().item()})
    pooled = pooled_payloads(written.payload, lengths)
    with torch.no_grad():
        normed = F.normalize(pooled, dim=-1)
        sim = normed @ normed.t()
        off = sim[~torch.eye(batch, dtype=torch.bool, device=sim.device)]
        metrics["batch_cross_similarity"] = off.mean().item() if off.numel() else 0.0
    if diversity_weight > 0:
        div = diversity_loss(pooled)
        loss = loss + diversity_weight * div
        metrics["diversity_hinge"] = div.item()
    if stop_policy_weight > 0:
        with torch.no_grad():
            reward = -nll.detach() - length_cost * lengths.float()
            if policy_samples > 1:
                grouped = reward.view(-1, policy_samples)
                advantage = (grouped - grouped.mean(1, keepdim=True)).view(-1)
            else:
                advantage = reward - reward.mean()
        policy = -(advantage * stop_log_prob(heads, written)).mean()
        loss = loss + stop_policy_weight * policy
        metrics.update({"stop_policy": policy.item(), "reward": reward.mean().item()})
    loss = _payload_terms(written, loss, metrics, temperature, payload_kl_weight)
    metrics["loss"] = loss.item()
    return loss, metrics


def replay_loss(backbone: PortBackbone, batch, teacher_context) -> tuple[torch.Tensor, dict]:
    """Ordinary-text replay (phase F): KL from the frozen base (deltas off) on plain text."""
    device = backbone.embedding_weight.device
    prefix, span, continuation = span_batch(batch, device)
    ids = torch.cat([prefix, span, continuation], 1)
    with torch.no_grad(), teacher_context():
        teacher = backbone.forward_ids(ids)["logits"]
    student = backbone.forward_ids(ids)["logits"]
    kl = _kl(student, teacher)
    return kl, {"replay_kl": kl.item()}


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
