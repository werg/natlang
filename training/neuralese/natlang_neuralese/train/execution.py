"""Training-time execution of the ports (S3 §4.2).

Everything here is differentiable with respect to the port modules (and the control rows);
the backbone stays frozen in phases A–E, but gradients still flow *through* it.

- `parallel_write`: teacher-forced and parallel-scheduled-sampling writes. Supplied inputs
  (interface-normed embeddings of known text) fill the sketch positions; a schedule replaces
  a fraction of them with generated sketches `F(h_k)` from a previous pass. With fraction 0
  this is the bootstrap (phase A); several passes approximate longer rollouts (phase C).
- `unroll_write`: the real write procedure, sequential through the k shallow layers, with
  backpropagation through time over the whole sketch recurrence (phase C onward). Lengths come
  from the stop head (detached decisions) or are forced.
- `read_continue` / `consumer_forward`: the read port with the completed payload, from the
  block-start cache (readback) or in one pass for the consumer.

No activation checkpointing anywhere (S3 §4.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..model.heads import PayloadSample, PortHeads, payload_kl, payload_log_prob, sample_payload
from ..model.lfm2_port import PortBackbone, PortCache


@dataclass
class Prefilled:
    cache: PortCache        # block-start snapshot: every layer at the open marker
    state: torch.Tensor     # [B, d] shallow residual at the open marker
    h_cut: torch.Tensor     # [B, T, d]
    logits: torch.Tensor    # [B, T, V]


@dataclass
class Written:
    payload: torch.Tensor      # [B, L, d]
    inputs: torch.Tensor       # [B, L, d] sketch inputs actually used
    shallow: torch.Tensor      # [B, L, d] residual after k layers
    final: torch.Tensor        # [B, L, d] residual after all layers
    stop_logits: torch.Tensor  # [B, L]: logit after j+1 vectors
    lengths: torch.Tensor      # [B]
    truncated: torch.Tensor    # [B] bool
    generated: torch.Tensor    # [B, L] bool: input came from the generator
    sample: PayloadSample | None = None  # mean, log-sigma, noise and temperature of the payload
    behavior_log_prob: torch.Tensor | None = None  # [B] log-probability of the sampled stop decisions under the behaviour policy

    def log_prob(self) -> torch.Tensor:
        """[B]: log N(z; mu, tau^2 sigma^2) over each row's valid vectors (temperature > 0)."""
        return (payload_log_prob(self.sample) * self.valid()).sum(-1)

    def kl(self) -> torch.Tensor:
        """Mean KL(N(u_mu, sigma^2) || N(0, I)) over valid vectors."""
        valid = self.valid()
        return (payload_kl(self.sample) * valid).sum() / valid.sum().clamp(min=1)

    def valid(self) -> torch.Tensor:
        return torch.arange(self.payload.shape[1], device=self.payload.device)[None] < self.lengths[:, None]


def prefill(backbone: PortBackbone, heads: PortHeads, ids: torch.Tensor) -> Prefilled:
    if not bool((ids[:, -1] == backbone.controls.open_id).all()):
        raise ValueError("prefix must end with the open marker")
    out = backbone.forward_ids(ids, cutoff=heads.cutoff)
    return Prefilled(out["cache"], out["h_cut"][:, -1], out["h_cut"], out["logits"])


def prefill_batch(backbone: PortBackbone, heads: PortHeads, producers: list[list[int]]) -> Prefilled:
    """Left-padded prefill of producers of different lengths, all ending at the open marker.

    Rows write in lockstep afterwards: every row's open marker is the last cache position,
    and padded keys stay masked (the cache carries `pad`).
    """
    device = backbone.embedding_weight.device
    width = max(len(p) for p in producers)
    pad = torch.tensor([width - len(p) for p in producers], device=device, dtype=torch.long)
    ids = torch.tensor([[0] * (width - len(p)) + p for p in producers], device=device, dtype=torch.long)
    if not bool((ids[:, -1] == backbone.controls.open_id).all()):
        raise ValueError("every producer must end with the open marker")
    cache = _context_cache(backbone, ids[:, :-1], pad)
    # Only the open marker carries a trainable row; it is the one prefix position run with autograd.
    out = backbone.forward_ids(ids[:, -1:], cache=cache, cutoff=heads.cutoff, logits=False)
    return Prefilled(out["cache"], out["h_cut"][:, -1], out["h_cut"], None)


def _context_cache(backbone: PortBackbone, ids: torch.Tensor, pad: torch.Tensor) -> PortCache:
    """Left-padded context run without autograd (nothing trainable precedes the block).

    Gradients from later positions still flow through attention to the *values* computed
    here only as constants, which is exact: these positions depend on no trainable input.
    """
    with torch.no_grad():
        out = backbone.forward_ids(ids, left_pad=pad if bool((pad > 0).any()) else None, logits=False)
    cache = out["cache"]
    if cache.pad is None and bool((pad > 0).any()):
        raise AssertionError("left padding lost")
    return cache


def supplied_inputs(backbone: PortBackbone, heads: PortHeads, span_ids: torch.Tensor) -> torch.Tensor:
    """Known-text embeddings at sketch positions, at the scale sketches have."""
    return heads.interface(backbone.embed(span_ids))


def _complete(backbone: PortBackbone, heads: PortHeads, block_start: PortCache, inputs, shallow,
              temperature: float = 0.0, generator: torch.Generator | None = None):
    final, _ = backbone.run_layers(shallow, range(heads.cutoff, backbone.num_layers), block_start)
    mu, log_sigma = heads.content.distribution(inputs, final)
    sample = sample_payload(mu, log_sigma, temperature, generator)
    return final, sample


def _stop_logits(heads: PortHeads, shallow: torch.Tensor) -> torch.Tensor:
    batch, length, _ = shallow.shape
    counts = torch.arange(1, length + 1, device=shallow.device).expand(batch, -1)
    return heads.stop(shallow, counts)


def parallel_write(backbone: PortBackbone, heads: PortHeads, pre: Prefilled, supplied: torch.Tensor,
                   generated_fraction: float = 0.0, passes: int = 2,
                   generator: torch.Generator | None = None, temperature: float = 0.0) -> Written:
    """Parallel scheduled sampling over a block of the supplied length."""
    k = heads.cutoff
    batch, length, _ = supplied.shape
    if generated_fraction > 0:
        draw = torch.rand(batch, length, generator=generator, device="cpu").to(supplied.device)
        mask = draw < generated_fraction
    else:
        mask = torch.zeros(batch, length, dtype=torch.bool, device=supplied.device)
    inputs = supplied
    if bool(mask.any()):
        for _ in range(max(1, passes - 1)):
            shallow, _ = backbone.run_layers(inputs, range(0, k), pre.cache)
            states = torch.cat([pre.state[:, None], shallow[:, :-1]], 1)
            generated = heads.feedback(states)
            inputs = torch.where(mask[..., None], generated.to(supplied.dtype), supplied)
    shallow, _ = backbone.run_layers(inputs, range(0, k), pre.cache)
    final, sample = _complete(backbone, heads, pre.cache, inputs, shallow, temperature, generator)
    lengths = torch.full((batch,), length, dtype=torch.long, device=supplied.device)
    return Written(sample.payload, inputs, shallow, final, _stop_logits(heads, shallow), lengths,
                   torch.zeros(batch, dtype=torch.bool, device=supplied.device), mask, sample)


def unroll_write(backbone: PortBackbone, heads: PortHeads, pre: Prefilled, length: int | None = None,
                 max_length: int | None = None, sample: bool = False,
                 generator: torch.Generator | None = None, temperature: float = 0.0,
                 stop_exploration: float = 0.0, stop_temperature: float = 1.0) -> Written:
    """The write procedure with gradients through the whole sketch recurrence.

    With `length`, every row writes exactly that many vectors (used when a target length is
    known). Otherwise each row stops by its stop head (greedy or sampled, detached decision,
    masked before the first vector) or at the hard maximum; the returned tensors have the
    longest row's length and `lengths` says where each row stopped.

    Sampled stopping may explore (phase E): the behaviour probability of stopping after c vectors is
    (1 - e) * sigmoid(logit / t) + e / (limit - c + 1), where the second term alone makes every length
    1..limit equally likely. `behavior_log_prob` records each row's decisions under that behaviour so
    the policy gradient can be importance-weighted to the stop head itself. With e = 0 and t = 1 the
    behaviour is the stop head and the random draws are the same as without exploration.
    """
    k = heads.cutoff
    limit = length if length is not None else min(max_length or heads.max_length, heads.max_length)
    state, cache = pre.state, pre.cache
    batch, device = state.shape[0], state.device
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    lengths = torch.full((batch,), limit, dtype=torch.long, device=device)
    inputs, shallow = [], []
    behavior = torch.zeros(batch, dtype=torch.float32, device=device) if sample and length is None else None
    for count in range(limit):
        if count > 0 and length is None:
            logit = heads.stop(state, torch.full((batch,), count, device=device, dtype=torch.long))
            with torch.no_grad():
                if sample:
                    p_stop = torch.sigmoid(logit.float() / stop_temperature)
                    if stop_exploration > 0:
                        p_stop = (1 - stop_exploration) * p_stop + stop_exploration / (limit - count + 1)
                    stop = torch.rand(batch, generator=generator).to(device) < p_stop
                    chosen = torch.where(stop, p_stop, 1 - p_stop).clamp(min=1e-12).log()
                    behavior = behavior + torch.where(done, torch.zeros_like(chosen), chosen)
                else:
                    stop = logit > 0
                newly = stop & ~done
                lengths = torch.where(newly, torch.full_like(lengths, count), lengths)
                done = done | stop
            if bool(done.all()):
                break
        sketch = heads.feedback(state)
        h, cache = backbone.run_layers(sketch[:, None], range(0, k), cache)
        state = h[:, 0]
        inputs.append(sketch)
        shallow.append(state)
    inputs_t, shallow_t = torch.stack(inputs, 1), torch.stack(shallow, 1)
    truncated = ~done if length is None else torch.zeros(batch, dtype=torch.bool, device=device)
    final, payload_sample = _complete(backbone, heads, pre.cache, inputs_t, shallow_t, temperature, generator)
    return Written(payload_sample.payload, inputs_t, shallow_t, final, _stop_logits(heads, shallow_t), lengths,
                   truncated, torch.ones(batch, inputs_t.shape[1], dtype=torch.bool, device=device), payload_sample,
                   behavior)


def read_continue(backbone: PortBackbone, heads: PortHeads, block_start: PortCache, payload: torch.Tensor,
                  continuation: torch.Tensor) -> torch.Tensor:
    """Readback of the payload plus close marker, then the continuation.

    Returns logits that predict `continuation` ([B, C, V]): the logit at the close marker
    predicts its first token.
    """
    batch = payload.shape[0]
    close = torch.full((batch, 1), backbone.controls.close_id, device=payload.device, dtype=torch.long)
    embeds = torch.cat([heads.interface(payload), backbone.embed(close), backbone.embed(continuation[:, :-1])], 1)
    logits = backbone.forward_embeds(embeds, cache=block_start)["logits"]
    return logits[:, payload.shape[1]:]


def consumer_forward(backbone: PortBackbone, heads: PortHeads, before: list[int], payload: torch.Tensor | None,
                     after: list[int], target: list[int]) -> torch.Tensor:
    """One consumer sequence (batch 1): logits predicting `target` ([1, len(target), V]).

    `before` ends with the open marker and `after` starts with the close marker. With
    `payload=None` the block is left out entirely (the no-block control): the open marker is
    dropped from `before` and the close marker from `after`.
    """
    device = backbone.embedding_weight.device
    tensor = lambda ids: torch.tensor([ids], device=device, dtype=torch.long)
    if payload is None:
        pieces = [backbone.embed(tensor(before[:-1]))]
        rest = after[1:]
    else:
        pieces = [backbone.embed(tensor(before)), heads.interface(payload.reshape(1, -1, payload.shape[-1]))]
        rest = after
    tail = rest + target[:-1]
    pieces.append(backbone.embed(tensor(tail)))
    logits = backbone.forward_embeds(torch.cat(pieces, 1))["logits"]
    return logits[:, -len(target):]


def consumer_forward_batch(backbone: PortBackbone, heads: PortHeads, rendered: list, payload: torch.Tensor | None,
                           lengths: torch.Tensor | None) -> list[torch.Tensor]:
    """Batched consumer pass over ragged records: one right-padded forward.

    `payload` is [B, L, d] with row b valid up to `lengths[b]`; `None` leaves the block out
    (the no-block control). Returns per-row logits predicting each row's target.
    """
    device = backbone.embedding_weight.device
    tensor = lambda ids: torch.tensor(ids, device=device, dtype=torch.long)
    # The context before the open marker has nothing trainable: run it without autograd,
    # left-padded so every row's block starts at the same cache position.
    contexts = [r.consumer_before[:-1] for r in rendered]
    width = max(len(c) for c in contexts)
    pad = torch.tensor([width - len(c) for c in contexts], device=device, dtype=torch.long)
    cache = _context_cache(backbone, tensor([[0] * (width - len(c)) + c for c in contexts]), pad)
    rows, spans = [], []
    for b, r in enumerate(rendered):
        if payload is None:
            pieces = [backbone.embed(tensor(r.consumer_after[1:] + r.target[:-1]))]
        else:
            block = heads.interface(payload[b, : int(lengths[b])])
            pieces = [backbone.embed(tensor(r.consumer_before[-1:])), block,
                      backbone.embed(tensor(r.consumer_after + r.target[:-1]))]
        row = torch.cat(pieces, 0)
        rows.append(row)
        spans.append((row.shape[0] - len(r.target), row.shape[0]))
    # Right padding at the end of the chunk needs no mask: causal attention and the causal
    # convolution never let a real position see a later (padded) one.
    width = max(row.shape[0] for row in rows)
    embeds = rows[0].new_zeros(len(rows), width, rows[0].shape[-1])
    for b, row in enumerate(rows):
        embeds[b, : row.shape[0]] = row
    h = backbone.forward_embeds(embeds, cache=cache, logits=False)["h_final"]
    return [backbone.logits(h[b, start:end]) for b, (start, end) in enumerate(spans)]


@torch.no_grad()
def teacher_logits_batch(backbone: PortBackbone, rendered: list) -> list[torch.Tensor]:
    """The self-distillation teacher for a batch (full source instead of the block), right-padded.

    Callers wrap this in `deltas_off` once reader or writer adapters exist (phase F).
    """
    device = backbone.embedding_weight.device
    seqs = [r.teacher_prefix + r.target[:-1] for r in rendered]
    width = max(len(x) for x in seqs)
    ids = torch.zeros(len(seqs), width, device=device, dtype=torch.long)
    padding = torch.zeros(len(seqs), width, device=device, dtype=torch.long)
    for b, x in enumerate(seqs):
        ids[b, : len(x)] = torch.tensor(x, device=device)
        padding[b, : len(x)] = 1
    full = bool(padding.all())
    h = backbone.forward_ids(ids, padding=None if full else padding, logits=False)["h_final"]
    # Deltas off includes the control rows: the base model's own output columns.
    rows = []
    for b, (x, r) in enumerate(zip(seqs, rendered)):
        normed = backbone.final_norm(h[b, len(x) - len(r.target): len(x)])
        rows.append(normed @ backbone.embedding_weight.t().to(normed.dtype))
    return rows


def stop_log_prob(heads: PortHeads, written: Written) -> torch.Tensor:
    """[B] log-probability of each row's stopping decisions under the stop head (phase E).

    Decision c (after c vectors, c >= 1) uses the stop logit for count c. A row of length L
    that stopped continued at counts 1..L-1 and stopped at L; a truncated row continued at
    every decision it made (1..L-1). The shallow states are detached: the policy term trains
    the stop head, not the writer's states.
    """
    shallow = written.shallow.detach()
    batch, length, _ = shallow.shape
    counts = torch.arange(1, length + 1, device=shallow.device).expand(batch, -1)
    logits = heads.stop(shallow, counts).float()
    c = counts
    L = written.lengths[:, None]
    cont = (c < L).float()
    stop_here = ((c == L) & ~written.truncated[:, None]).float()
    return (F.logsigmoid(-logits) * cont + F.logsigmoid(logits) * stop_here).sum(-1)


@torch.no_grad()
def teacher_target_logits(backbone: PortBackbone, prefix: list[int], target: list[int]) -> torch.Tensor:
    """The self-distillation teacher: the frozen crisp base on the full-source view, deltas off.

    In phases A–E the only deltas are the port modules and control rows, which the plain
    full-source view never touches. When reader or writer LoRA is added (phase F) the caller
    wraps this in the adapter-disabling context (`deltas_off`).
    """
    device = backbone.embedding_weight.device
    ids = torch.tensor([prefix + target[:-1]], device=device, dtype=torch.long)
    logits = backbone.hf(input_ids=ids).logits
    return logits[:, -len(target):]
