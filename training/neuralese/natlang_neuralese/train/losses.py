"""Losses for curriculum phases A–D (S3 §5.1).

All losses return a scalar plus a dict of detached metrics. Causal alignment is kept
throughout: a decision at position i sees h_k[i] and never the input it predicts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model.heads import PortHeads
from ..model.lfm2_port import PortBackbone
from .execution import (Prefilled, Written, consumer_forward, parallel_write, prefill, read_continue,
                        supplied_inputs, teacher_target_logits, unroll_write)


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
