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
- `one_step_write`: the same write generated greedily without gradients, then re-run once in parallel with
  gradients (one step of the sketch recurrence per position) plus a self-target loss: the sketch written from position
  i predicts the completed top-layer payload at i, which it then feeds as input i + 1 (owner 2026-10-06: no
  backpropagation through the recurrent rollout).
- `read_continue` / `consumer_forward`: the read port with the completed payload, from the
  block-start cache (readback) or in one pass for the consumer.
- `local_stage_write`: fixed-input history replay plus isolated full-stack completions;
  each sketch receives credit only through its own completion, never a later cache.

Layer checkpointing follows the backbone's declared execution policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..model.heads import PayloadSample, PortHeads, payload_kl, payload_log_prob, sample_payload
from ..model.lfm2_port import PortBackbone, PortCache


def causal_gold_prefix_mask(generated: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Select context-valid gold decisions, including the first mismatch.

    Target t has a gold token history only when every generated token before
    t matches gold. The mismatching decision is useful supervision; later gold
    targets belong to a different history. This token-history condition alone
    does not establish equivalence of continuous Neuralese payloads.
    """
    if generated.ndim != 2 or generated.shape != gold.shape or not gold.shape[1]:
        raise ValueError('nonempty aligned rank-two generated and gold tokens required')
    matches = generated.eq(gold)
    return torch.cat((torch.ones_like(matches[:, :1]), matches[:, :-1]), dim=1).cumprod(dim=1).bool()


@dataclass
class Prefilled:
    cache: PortCache        # block-start snapshot: every layer at the open marker
    state: torch.Tensor     # [B, d] shallow residual at the open marker
    h_cut: torch.Tensor     # [B, T, d]
    logits: torch.Tensor    # [B, T, V]
    top: torch.Tensor | None = None  # [B, d] top-layer state of the last prefix position (autoregressive payload 0)


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
    sketch_target_loss: torch.Tensor | None = None  # scalar: one-step writes' sketch inputs vs their completed payloads
    sketch_target_loss_by_row: torch.Tensor | None = None  # [B]: same auxiliary, normalized over each row's valid positions
    local_replay_max_abs_error: torch.Tensor | None = None  # scalar: local-stage vs detached rollout state discrepancy

    def log_prob(self) -> torch.Tensor:
        """[B]: log N(z; mu, tau^2 sigma^2) over each row's valid vectors (temperature > 0)."""
        return (payload_log_prob(self.sample) * self.valid()).sum(-1)

    def kl(self) -> torch.Tensor:
        """Mean KL(N(u_mu, sigma^2) || N(0, I)) over valid vectors."""
        valid = self.valid()
        return (payload_kl(self.sample) * valid).sum() / valid.sum().clamp(min=1)

    def valid(self) -> torch.Tensor:
        return torch.arange(self.payload.shape[1], device=self.payload.device)[None] < self.lengths[:, None]


def prefill(backbone: PortBackbone, heads: PortHeads, ids: torch.Tensor, *, logits: bool = True) -> Prefilled:
    if not bool((ids[:, -1] == backbone.controls.open_id).all()):
        raise ValueError("prefix must end with the open marker")
    if not heads.read_markers:
        ids = ids[:, :-1]
        if ids.shape[1] == 0:
            raise ValueError("a raw write needs a nonempty causal prefix")
    out = backbone.forward_ids(ids, cutoff=heads.cutoff, logits=logits)
    return Prefilled(out["cache"], out["h_cut"][:, -1], out["h_cut"], out.get("logits"), out["h_final"][:, -1])


def prefill_write_context(backbone: PortBackbone, heads: PortHeads, context: torch.Tensor) -> Prefilled:
    """Profile-aware write boundary for already embedded, differentiable scope.

    The caller supplies ordinary prompt/prefix embeddings without a host marker.
    Legacy ports consume that marker; raw ports keep the causal sequence intact.
    """
    if heads.read_markers:
        opened = backbone.embed(torch.full((context.shape[0], 1), backbone.controls.open_id,
                                          device=context.device, dtype=torch.long))
        context = torch.cat([context, opened], 1)
    if context.shape[1] == 0:
        raise ValueError('write requires a nonempty causal context')
    out = backbone.forward_embeds(context, cutoff=heads.cutoff, logits=False)
    return Prefilled(out['cache'], out['h_cut'][:, -1], out['h_cut'], None, out['h_final'][:, -1])


def full_depth_projected_feedback_step(backbone: PortBackbone, heads: PortHeads,
                                       top: torch.Tensor, cache: PortCache):
    """Project the current top state and consume that payload at the next full-depth position.

    This is the shared transition used by autoregressive evaluation and the
    detached producer pass of text-warmup feedback exposure. The returned
    payload predicts the next gold token; the returned state is the next
    position's full-depth state after reading that payload. Callers choose the
    gradient context explicitly: evaluation/producer rollouts use no-grad,
    while training retains gradients for the consumer path as required.
    """
    if top.ndim not in (2, 3) or (top.ndim == 3 and top.shape[1] != 1):
        raise ValueError('full-depth feedback expects a [batch,width] or [batch,1,width] top state')
    if heads.read_markers:
        raise ValueError('token-aligned full-depth feedback requires the raw read profile')
    payload = heads.content(torch.zeros_like(top), top)
    history = heads.read_embeddings(backbone, payload[:, None] if payload.ndim == 2 else payload)
    states, updated_cache = backbone.run_layers(history, range(backbone.num_layers), cache)
    return payload, states[:, 0], updated_cache


def prefill_write_contexts(backbone: PortBackbone, heads: PortHeads,
                           contexts: list[torch.Tensor]) -> Prefilled:
    """Tensor-batched write boundaries with gradients through every scope row.

    Each context is [T,d], without a host marker. Unlike the constant-token
    prefill_batch path this must retain both parameter and child-result adjoints.
    Left padding aligns the causal boundary; the cache masks padded keys during
    all later recurrent calls. Row order is an execution choice the scheduler
    must pin for both the primal and replay.
    """
    if not contexts or any(c.ndim != 2 or c.shape[0] == 0 for c in contexts):
        raise ValueError('batched write contexts must be nonempty [tokens,dim] rows')
    reference = contexts[0]
    if any(c.shape[1] != reference.shape[1] or c.device != reference.device or c.dtype != reference.dtype
           for c in contexts):
        raise ValueError('batched write contexts must share dimension, device and dtype')
    if heads.read_markers:
        marker = backbone.embed(torch.tensor([backbone.controls.open_id], device=reference.device))
        contexts = [torch.cat([c, marker], 0) for c in contexts]
    width = max(c.shape[0] for c in contexts)
    offsets = [width - c.shape[0] for c in contexts]
    embeds = torch.stack([F.pad(c, (0, 0, offset, 0)) for c, offset in zip(contexts, offsets)])
    pad = torch.tensor(offsets, device=reference.device) if any(offsets) else None
    cache = PortCache.empty(backbone.num_layers)
    if pad is not None:
        from dataclasses import replace
        cache = replace(cache, pad_offsets=tuple(offsets))
    out = backbone.forward_embeds(embeds, cache=cache, left_pad=pad, cutoff=heads.cutoff, logits=False)
    return Prefilled(out['cache'], out['h_cut'][:, -1], out['h_cut'], None, out['h_final'][:, -1])


def prefill_batch(backbone: PortBackbone, heads: PortHeads, producers: list[list[int]]) -> Prefilled:
    """Left-padded prefill of producers of different lengths, all ending at the open marker.

    Rows write in lockstep afterwards: every row's open marker is the last cache position,
    and padded keys stay masked (the cache carries `pad`).
    """
    device = backbone.embedding_weight.device
    width = max(len(p) for p in producers)
    offsets = [width - len(p) for p in producers]
    pad = torch.tensor(offsets, device=device, dtype=torch.long) if any(offsets) else None
    if any(not p or p[-1] != backbone.controls.open_id for p in producers):
        raise ValueError("every producer must end with the open marker")
    ids = torch.tensor([[0] * (width - len(p)) + p for p in producers], device=device, dtype=torch.long)
    if not heads.read_markers:
        with torch.no_grad():
            out = backbone.forward_ids(ids[:, :-1], left_pad=pad,
                                       cutoff=heads.cutoff, logits=False)
        return Prefilled(out['cache'], out['h_cut'][:, -1], out['h_cut'], None, out['h_final'][:, -1])
    cache = _context_cache(backbone, ids[:, :-1], pad)
    # Only the open marker carries a trainable row; it is the one prefix position run with autograd.
    out = backbone.forward_ids(ids[:, -1:], cache=cache, cutoff=heads.cutoff, logits=False)
    return Prefilled(out["cache"], out["h_cut"][:, -1], out["h_cut"], None, out["h_final"][:, -1])


def _context_cache(backbone: PortBackbone, ids: torch.Tensor, pad: torch.Tensor | None) -> PortCache:
    """Left-padded context run without autograd (nothing trainable precedes the block).

    Gradients from later positions still flow through attention to the *values* computed
    here only as constants, which is exact: these positions depend on no trainable input.
    """
    with torch.no_grad():
        out = backbone.forward_ids(ids, left_pad=pad, logits=False)
    cache = out["cache"]
    if cache.pad is None and pad is not None:
        raise AssertionError("left padding lost")
    return cache


def supplied_inputs(backbone: PortBackbone, heads: PortHeads, span_ids: torch.Tensor) -> torch.Tensor:
    """Known-text embeddings at sketch positions, at the scale sketches have."""
    return heads.interface(backbone.embed(span_ids))


def _complete(backbone: PortBackbone, heads: PortHeads, block_start: PortCache, inputs, shallow,
              temperature: float = 0.0, generator: torch.Generator | None = None, top: torch.Tensor | None = None):
    """Upper layers and the payload. `top` is the top-layer state before the block (the autoregressive layout projects
    payload j from the top state at j - 1)."""
    final, _ = backbone.run_layers(shallow, range(heads.cutoff, backbone.num_layers), block_start)
    mu, log_sigma = heads.content.distribution(inputs, heads.payload_states(final, top))
    sample = sample_payload(mu, log_sigma, temperature, generator)
    return final, sample


def _stop_logits(heads: PortHeads, shallow: torch.Tensor, final: torch.Tensor | None = None) -> torch.Tensor:
    """[B, L]: the stop logit after j+1 vectors, from the states the heads' stop source names."""
    batch, length, _ = shallow.shape
    counts = torch.arange(1, length + 1, device=shallow.device).expand(batch, -1)
    return heads.stop(heads.stop_states(shallow, final), counts)


def _sketch_self_target_terms(heads: PortHeads, written: Written, guess: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return legacy global loss and exact per-row losses for detached targets."""
    guess = guess.float()
    if heads.autoregressive:
        valid, target = written.valid().float(), written.sample.mean.detach().float()
    else:
        valid, target, guess = written.valid()[:, 1:].float(), written.sample.mean[:, :-1].detach().float(), guess[:, 1:]
    error = (guess - target).square().mean(-1) / target.square().mean(-1).clamp(min=1e-6)
    weighted = error * valid
    row_counts = valid.sum(-1)
    row_losses = weighted.sum(-1) / row_counts.clamp(min=1)
    # Preserve the scalar API's historical token-weighted reduction. Training
    # a frontier of independent producers consumes row_losses instead, so each
    # producer receives the same per-producer normalization as a singleton.
    global_loss = weighted.sum() / row_counts.sum().clamp(min=1)
    return global_loss, row_losses


def _set_sketch_self_target(heads: PortHeads, written: Written, guess: torch.Tensor) -> None:
    written.sketch_target_loss, written.sketch_target_loss_by_row = _sketch_self_target_terms(heads, written, guess)


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
    final, sample = _complete(backbone, heads, pre.cache, inputs, shallow, temperature, generator, pre.top)
    lengths = torch.full((batch,), length, dtype=torch.long, device=supplied.device)
    return Written(sample.payload, inputs, shallow, final, _stop_logits(heads, shallow, final), lengths,
                   torch.zeros(batch, dtype=torch.bool, device=supplied.device), mask, sample)


def blockwise_sketch(backbone: PortBackbone, heads: PortHeads, state: torch.Tensor, cache, length: int,
                     passes: int | None = None, tol: float = 0.0):
    """The sketch recurrence of a block of known length, block-wise: every position runs through layers [0, k) at once
    and the inputs are refined by fixed-point (Jacobi) iteration, inputs[j] = feedback(shallow[j - 1]) with
    inputs[0] = feedback(state). Causality makes the first p positions exact after p passes, so `passes` = `length`
    (the default) reproduces the sequential write exactly; fewer passes trade exactness for fewer sequential steps, and
    iteration stops early once the inputs change by at most `tol`. Returns (inputs, shallow, passes run), the inputs
    being the sketches the shallow residuals were computed from."""
    k = heads.cutoff
    first = heads.feedback(state)[:, None]
    inputs = first.expand(-1, length, -1)
    passes = length if passes is None else max(1, min(passes, length))
    for done in range(1, passes + 1):
        shallow, _ = backbone.run_layers(inputs, range(0, k), cache)
        if done == passes:
            break
        refined = torch.cat([first, heads.feedback(shallow[:, :-1])], 1).to(inputs.dtype)
        if float((refined - inputs).abs().max()) <= tol:
            break
        inputs = refined
    return inputs, shallow, done


def rollout_sketch_inputs(backbone, heads, pre, limit, *, reference_inputs=None,
                          fraction=1., before_step=None):
    """Shared causal shallow rollout; complete upper layers in parallel later.

    Full-depth states do not drive F, so computing them on every iteration only
    serializes work. A callback supports the serving/training stop policy without
    changing the input recurrence. Text scheduling supplies raw gold inputs.
    """
    state,cache=pre.state,pre.cache
    inputs,shallow,predictions=[],[],[]
    for count in range(limit):
        if before_step is not None and not before_step(state,count):break
        prediction=heads.feedback(state)
        value=prediction if reference_inputs is None else (
            (1-fraction)*reference_inputs[:,count]+fraction*prediction.to(reference_inputs.dtype))
        h,cache=backbone.run_layers(value[:,None],range(heads.cutoff),cache)
        state=h[:,0]
        inputs.append(value);shallow.append(state);predictions.append(prediction)
    if not inputs:raise ValueError('sketch rollout must contain at least one input')
    return torch.stack(inputs,1),torch.stack(shallow,1),torch.stack(predictions,1),state


def unroll_write(backbone: PortBackbone, heads: PortHeads, pre: Prefilled, length: int | None = None,
                 max_length: int | None = None, sample: bool = False,
                 generator: torch.Generator | None = None, temperature: float = 0.0,
                 stop_exploration: float = 0.0, stop_temperature: float = 1.0,
                 lengths: torch.Tensor | None = None) -> Written:
    """The write procedure with gradients through the whole sketch recurrence.

    Supervised lengths: with `length` every row writes exactly that many vectors; with `lengths` ([B]) row b writes
    `lengths[b]` (the tensors have the longest row's length; `lengths` says where each row ends). No stop decision is
    taken, and the stop head is trained on the boundary by the caller. Otherwise each row stops by its stop head
    (greedy or sampled, detached decision, masked before the first vector) or at the hard maximum.

    The stop source decides when: "shallow" decides online from the sketch state after each vector; "final" writes
    to the maximum, completes, and decides along the completed states. Completion is causal, so the truncated block
    is exactly the block that would have stopped there.

    Sampled stopping may explore (phase E): the behaviour probability of stopping after c vectors is
    (1 - e) * sigmoid(logit / t) + e / (limit - c + 1), where the second term alone makes every length
    1..limit equally likely. `behavior_log_prob` records each row's decisions under that behaviour so
    the policy gradient can be importance-weighted to the stop head itself. With e = 0 and t = 1 the
    behaviour is the stop head and the random draws are the same as without exploration.
    """
    k = heads.cutoff
    fixed = length is not None or lengths is not None
    if lengths is not None:
        limit = int(lengths.max())
        if limit > heads.max_length or int(lengths.min()) < 1:
            raise ValueError(f"supervised lengths must be within 1..{heads.max_length}")
    elif length is not None:
        limit = length
    else:
        limit = min(max_length or heads.max_length, heads.max_length)
    state, cache = pre.state, pre.cache
    batch, device = state.shape[0], state.device
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    chosen_lengths = torch.full((batch,), limit, dtype=torch.long, device=device)
    behavior = torch.zeros(batch, dtype=torch.float32, device=device) if sample and not fixed else None

    def decide(logit: torch.Tensor, count: int):
        nonlocal done, chosen_lengths, behavior
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
            chosen_lengths = torch.where(newly, torch.full_like(chosen_lengths, count), chosen_lengths)
            done = done | stop

    online = not fixed and heads.stop_source == "shallow"
    def before_step(state,count):
        if count > 0 and online:
            decide(heads.stop(state, torch.full((batch,), count, device=device, dtype=torch.long)),count)
            return not bool(done.all())
        return True
    inputs_t,shallow_t,_,_=rollout_sketch_inputs(backbone,heads,pre,limit,
                                               before_step=before_step if online else None)
    final, payload_sample = _complete(backbone, heads, pre.cache, inputs_t, shallow_t, temperature, generator,
                                    pre.top)
    stop_logits = _stop_logits(heads, shallow_t, final)
    if not fixed and heads.stop_source == "final":
        for count in range(1, limit):  # decision after `count` vectors reads h_D[count - 1]
            decide(stop_logits[:, count - 1], count)
            if bool(done.all()):
                break
    if fixed:
        out_lengths = lengths.to(device) if lengths is not None else torch.full((batch,), limit, dtype=torch.long, device=device)
        truncated = torch.zeros(batch, dtype=torch.bool, device=device)
    else:
        out_lengths, truncated = chosen_lengths, ~done
    return Written(payload_sample.payload, inputs_t, shallow_t, final, stop_logits, out_lengths,
                   truncated, torch.ones(batch, inputs_t.shape[1], dtype=torch.bool, device=device), payload_sample,
                   behavior)


def one_step_write(backbone: PortBackbone, heads: PortHeads, pre: Prefilled, temperature: float = 0.0,
                   generator: torch.Generator | None = None, sketch_target_backbone_scale: float = 0.05, **unroll) -> Written:
    """The write procedure without backpropagation through the sketch recurrence (owner 2026-10-06).

    1. Generate: `unroll_write` under no_grad (greedy payload; stop decisions as configured), which fixes the inputs
       and the lengths, at the cost of inference.
    2. Re-run once, in parallel, with gradients: the shallow layers over the generated (detached) inputs give each
       position's state; the feedback projection recomputes every input from the previous position's state, so each
       sketch step receives gradient without a recursive chain of feedback projections. The recomputed inputs are run
       through the shallow layers again and completed through the upper layers to the top-layer payload. This replay
       retains ordinary causal attention/convolution gradients from later positions to earlier sketch inputs; it does
       not enforce a strict one-stage gradient horizon through those cache paths.
    3. Self-target: the sketch the shallow stack writes from position i (the input at i + 1) is pulled toward the
       top-layer payload mean its own stack completes at position i (detached), as text feeds the top-layer output at
       i to the input at i + 1. The sketch has no
       next-token fidelity target and no quality gate of its own; this term only stands in for the dropped
       backpropagation through time.

    Same lengths, truncation and behaviour log-probabilities as the generated write; the payload is resampled at
    `temperature` in the re-run.
    """
    with torch.no_grad():
        generated = unroll_write(backbone, heads, pre, generator=generator, temperature=0.0, **unroll)
    if not 0 <= sketch_target_backbone_scale <= 1:
        raise ValueError("sketch target backbone gradient scale must be between 0 and 1")
    k = heads.cutoff
    fixed = generated.inputs.detach()
    states, _ = backbone.run_layers(fixed, range(0, k), pre.cache)
    source = torch.cat([pre.state[:, None], states[:, :-1]], 1)
    sketch = heads.feedback(source).to(fixed.dtype)
    # Auxiliary distillation gives F its full signal, and only a small signal
    # to its source states. Forward values and consumer gradients are unchanged.
    target_source = source.detach() + sketch_target_backbone_scale * (source - source.detach())
    target_sketch = heads.feedback(target_source).to(fixed.dtype)
    shallow, _ = backbone.run_layers(sketch, range(0, k), pre.cache)
    final, sample = _complete(backbone, heads, pre.cache, sketch, shallow, temperature, generator, pre.top)
    written = Written(sample.payload, sketch, shallow, final, _stop_logits(heads, shallow, final), generated.lengths,
                      generated.truncated, generated.generated, sample, generated.behavior_log_prob)
    # As with text, the top-layer output at position i is the input at i + 1, and the sketch written from h_k[i] stands
    # in for it. Autoregressive layout: that output is payload i + 1, which sits in the same slot as the sketch input
    # i + 1, so every input has a target (input 0, from the position before the block, predicts payload 0). Earlier
    # profiles project payload i from position i itself: input i + 1 predicts payload i and input 0 has none.
    _set_sketch_self_target(heads, written, target_sketch)
    return written


def write_generated(backbone: PortBackbone, heads: PortHeads, pre: Prefilled, sketch_gradient: str = "local_stage",
                    sketch_target_backbone_scale: float = 0.05, local_stage_batch_size: int = 1, **kwargs) -> Written:
    """Generated write with full recurrence, parallel feedback replay, or isolated
    local-stage adjoints. Local-stage batch size is an execution resource choice.
    """
    if sketch_gradient == "unroll":
        return unroll_write(backbone, heads, pre, **kwargs)
    if sketch_gradient == "one_step":
        return one_step_write(backbone, heads, pre, sketch_target_backbone_scale=sketch_target_backbone_scale, **kwargs)
    if sketch_gradient == "local_stage":
        return local_stage_write(backbone, heads, pre, sketch_target_backbone_scale=sketch_target_backbone_scale,
                                 stage_batch_size=local_stage_batch_size, **kwargs)
    raise ValueError(f"unknown sketch_gradient {sketch_gradient!r}")


def replay_local_stages(backbone, heads, pre, fixed, *, group_size=1,
                        reference_inputs=None, fraction=1., auxiliary_scale=.05,
                        terminal_guess=False):
    """Shared isolated replay for gold-text training and Natlang writers.

    History inputs are constants with respect to earlier sketch outputs; backbone
    and original scope gradients remain live. Each independent branch replaces
    just its own input and completes it through the whole stack.
    """
    if group_size < 1:
        raise ValueError('positive replay group size required')
    k = heads.cutoff
    history, _ = backbone.run_layers(fixed, range(k), pre.cache)
    sources = torch.cat((pre.state[:, None], history[:, :-1]), 1)
    sketches = heads.feedback(sources).to(fixed.dtype)
    auxiliary_source = sources.detach() + auxiliary_scale * (sources - sources.detach())
    guesses = heads.feedback(auxiliary_source).to(fixed.dtype)
    replacements = sketches if reference_inputs is None else (1-fraction)*reference_inputs + fraction*sketches
    result = backbone.isolated_sequence(fixed, replacements, pre.cache, cutoff=k)
    if terminal_guess:
        last = history[:, -1]
        auxiliary = last.detach() + auxiliary_scale * (last - last.detach())
        guesses = torch.cat((guesses, heads.feedback(auxiliary)[:, None].to(fixed.dtype)), 1)
    return sketches, guesses, result['shallow'], result['final']


def local_stage_write(backbone: PortBackbone, heads: PortHeads, pre: Prefilled,
                      temperature: float = 0.0, generator: torch.Generator | None = None,
                      sketch_target_backbone_scale: float = 0.05, stage_batch_size: int = 1, **unroll) -> Written:
    """One full-stack completion of credit per sketch; sequential reference.

    A detached greedy rollout fixes the history and lengths. Replay that history
    using its detached *inputs*, retaining parameter and original scope/child
    adjoints. At each position branch from this history, recompute F, and complete
    just that position. Discard the branch's cache: only the fixed-input history
    cache is allowed into later positions. Thus no later completion or auxiliary
    target can reach an earlier F output, through attention, convolution, or the
    shallow recurrence. Shared parameter gradients still sum across positions.

    Under v2, completion at j supplies payload j+1 and stop j. Payload 0 comes
    directly from the context. Self-target s[j] <- stop_gradient(p[j]) remains,
    with attenuated source-state credit and full projection credit. This is a
    deterministic path; stage_batch_size=1 is the sequential reference, larger
    groups batch independent branches while retaining that gradient support.
    The forward uses exact rollout states with local replay adjoints: different
    BF16 sequential/parallel GEMM layouts must not change discrete payload choices.
    Report replay state discrepancy rather than treating it as exact Jacobian parity.
    """
    if not 0 <= sketch_target_backbone_scale <= 1:
        raise ValueError("sketch target backbone gradient scale must be between 0 and 1")
    if stage_batch_size < 1:
        raise ValueError('local stage batch size must be positive')
    # Independent replays require a deterministic primal. Active dropout makes
    # the fixed history differ from the local branches even with identical inputs.
    for module in backbone.hf.modules():
        if isinstance(module, torch.nn.Dropout) and module.training and module.p:
            raise ValueError("local_stage requires deterministic execution (active dropout)")
    if backbone.hf.training and getattr(backbone.hf.config, 'attention_dropout', 0):
        raise ValueError("local_stage requires deterministic attention")
    with torch.no_grad():
        generated = unroll_write(backbone, heads, pre, generator=generator, temperature=0.0, **unroll)
    if not torch.is_grad_enabled():
        # Staged primal collection and held probes need no surrogate adjoints.
        # Exact rollout forward means the entire branch replay is redundant.
        generated.sample = sample_payload(generated.sample.mean, generated.sample.log_sigma, temperature, generator)
        generated.payload = generated.sample.payload
        _set_sketch_self_target(heads, generated, generated.inputs)
        return generated
    fixed = generated.inputs.detach()
    inputs, guess, shallow, final = replay_local_stages(
        backbone, heads, pre, fixed, group_size=stage_batch_size,
        auxiliary_scale=sketch_target_backbone_scale)
    discrepancy = torch.stack([(replayed.detach().float() - primal.float()).abs().max()
                               for replayed, primal in ((inputs, generated.inputs),
                                                       (shallow, generated.shallow), (final, generated.final))]).max()
    inputs = generated.inputs.detach() + (inputs - inputs.detach())
    shallow = generated.shallow.detach() + (shallow - shallow.detach())
    final = generated.final.detach() + (final - final.detach())
    mu, log_sigma = heads.content.distribution(inputs, heads.payload_states(final, pre.top))
    sample = sample_payload(mu, log_sigma, temperature, generator)
    written = Written(sample.payload, inputs, shallow, final, _stop_logits(heads, shallow, final),
                      generated.lengths, generated.truncated, generated.generated,
                      sample, generated.behavior_log_prob)
    written.local_replay_max_abs_error = discrepancy
    guess = generated.inputs.detach() + (guess - guess.detach())
    _set_sketch_self_target(heads, written, guess)
    return written


def read_continue(backbone: PortBackbone, heads: PortHeads, block_start: PortCache, payload: torch.Tensor,
                  continuation: torch.Tensor) -> torch.Tensor:
    """Readback of the payload plus close marker, then the continuation.

    Returns logits that predict `continuation` ([B, C, V]): the logit at the close marker
    predicts its first token.
    """
    read = heads.read_embeddings(backbone, payload, close_only=True)
    embeds = torch.cat([read, backbone.embed(continuation[:, :-1])], 1)
    logits = backbone.forward_embeds(embeds, cache=block_start)["logits"]
    return logits[:, read.shape[1] - 1:]


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
        pieces = [backbone.embed(tensor(before if heads.read_markers else before[:-1])), heads.read_in(payload.reshape(1, -1, payload.shape[-1]))]
        rest = after if heads.read_markers else after[1:]
    tail = rest + target[:-1]
    pieces.append(backbone.embed(tensor(tail)))
    logits = backbone.forward_embeds(torch.cat(pieces, 1))["logits"]
    return logits[:, -len(target):]


def consumer_context_cache(backbone: PortBackbone, rendered: list) -> PortCache:
    """Compute the constant reader prefix once for positive and shuffled reads."""
    device = backbone.embedding_weight.device
    contexts = [r.consumer_before[:-1] for r in rendered]
    width = max(len(c) for c in contexts)
    offsets = [width - len(c) for c in contexts]
    pad = torch.tensor(offsets, device=device, dtype=torch.long) if any(offsets) else None
    ids = torch.tensor([[0] * (width - len(c)) + c for c in contexts], device=device, dtype=torch.long)
    return _context_cache(backbone, ids, pad)


def consumer_forward_batch(backbone: PortBackbone, heads: PortHeads, rendered: list, payload: torch.Tensor | None,
                           lengths: torch.Tensor | None, *, context_cache: PortCache | None = None) -> list[torch.Tensor]:
    """Batched consumer pass over ragged records: one right-padded forward.

    `payload` is [B, L, d] with row b valid up to `lengths[b]`; `None` leaves the block out
    (the no-block control). Returns per-row logits predicting each row's target.
    """
    device = backbone.embedding_weight.device
    tensor = lambda ids: torch.tensor(ids, device=device, dtype=torch.long)
    # The context before the open marker has nothing trainable: run it without autograd,
    # left-padded so every row's block starts at the same cache position.
    cache = context_cache if context_cache is not None else consumer_context_cache(backbone, rendered)
    rows, spans = [], []
    for b, r in enumerate(rendered):
        if payload is None:
            pieces = [backbone.embed(tensor(r.consumer_after[1:] + r.target[:-1]))]
        else:
            block = heads.interface(payload[b, : int(lengths[b])])
            if heads.read_markers:
                pieces = [backbone.embed(tensor(r.consumer_before[-1:])), block,
                          backbone.embed(tensor(r.consumer_after + r.target[:-1]))]
            else:
                pieces = [block, backbone.embed(tensor(r.consumer_after[1:] + r.target[:-1]))]
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
    full = all(len(x) == width for x in seqs)
    h = backbone.forward_ids(ids, padding=None if full else padding, logits=False)["h_final"]
    # Deltas off includes the control rows: the base model's own output columns.
    rows = []
    for b, (x, r) in enumerate(zip(seqs, rendered)):
        normed = backbone.final_norm(h[b, len(x) - len(r.target): len(x)])
        rows.append(normed @ backbone.output_weight.t().to(normed.dtype))
    return rows


def stop_log_prob(heads: PortHeads, written: Written) -> torch.Tensor:
    """[B] log-probability of each row's stopping decisions under the stop head (phase E).

    Decision c (after c vectors, c >= 1) uses the stop logit for count c. A row of length L
    that stopped continued at counts 1..L-1 and stopped at L; a truncated row continued at
    every decision it made (1..L-1). The shallow states are detached: the policy term trains
    the stop head, not the writer's states.
    """
    shallow = heads.stop_states(written.shallow, written.final).detach()
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
