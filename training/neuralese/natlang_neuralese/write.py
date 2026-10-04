"""The single write procedure (S3 §3.1): sketch, complete, read back.

1. Open: the prefix, ending with the open marker, has been run through every layer.
   That cache is the block-start snapshot.
2. Sketch: layers 0..k-1 run position by position. Before each new vector the stop head
   decides whether to close (masked before the first vector); on continue, the
   feedback projection turns the current shallow state into the next sketch input.
   The runtime's hard maximum forces closure and records truncation.
3. Complete: layers k..D-1 run once, causally, over the collected shallow residuals,
   from the upper layers' block-start cache. The content projection gives the payload.
4. Read back: every layer is restored to the block-start snapshot and the payload plus
   the close marker are prefilled through the full model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .model.heads import PortHeads, payload_log_prob, sample_payload
from .model.lfm2_port import PortBackbone, PortCache


@dataclass
class Opened:
    """State at the open marker: block-start cache and the shallow residual at the marker."""

    cache: PortCache
    h_cut: torch.Tensor  # [B, d]
    logits: torch.Tensor  # [B, V] at the marker (unused by the writer; useful for checks)


@dataclass
class WriteResult:
    payload: torch.Tensor      # [B, Lmax_written, d], rows valid up to lengths[b]
    sketches: torch.Tensor     # [B, Lmax_written, d]
    shallow: torch.Tensor      # [B, Lmax_written, d] residuals after k layers
    final: torch.Tensor        # [B, Lmax_written, d] residuals after all layers
    lengths: torch.Tensor      # [B]
    truncated: torch.Tensor    # [B] bool
    stop_logits: torch.Tensor  # [B, Lmax_written] stop logit evaluated after j+1 vectors
    block_start: PortCache
    mean: torch.Tensor | None = None       # [B, L, d] mu (equals payload at temperature 0)
    log_sigma: torch.Tensor | None = None  # [B, L, d]
    log_prob: torch.Tensor | None = None   # [B] log N(z; mu, tau^2 sigma^2) over each row's valid vectors
    temperature: float = 0.0

    def row(self, b: int) -> torch.Tensor:
        return self.payload[b, : int(self.lengths[b])]


def open_block(backbone: PortBackbone, heads: PortHeads, prefix_ids: torch.Tensor) -> Opened:
    """Prefill a prefix whose last token is the open marker."""
    if not bool((prefix_ids[:, -1] == backbone.controls.open_id).all()):
        raise ValueError("prefix must end with the open marker")
    out = backbone.forward_ids(prefix_ids, cutoff=heads.cutoff)
    return Opened(cache=out["cache"], h_cut=out["h_cut"][:, -1], logits=out["logits"][:, -1])


def write_block(
    backbone: PortBackbone,
    heads: PortHeads,
    opened: Opened,
    max_length: int | None = None,
    sample: bool = False,
    generator: torch.Generator | None = None,
    allow_empty: bool = False,
    timings: dict | None = None,
    temperature: float = 0.0,
    lookahead: int = 4,
) -> WriteResult:
    """Write one block per batch row. Rows stop independently; the hard maximum truncates.

    With `timings`, adds seconds for `shallow_generation`, `completion` and `projection`.
    `temperature` > 0 samples the payload around its mean (S3 §2, payload distribution);
    the default 0 delivers the mean.

    With the "final" stop source the writer sketches `lookahead` positions at a time, completes them through the upper
    layers, and stops each row at the first completed position whose stop logit says so. Completion is causal, so the
    block (completed again over its kept positions below) is exactly the block that stopped there.
    """
    if heads.stop_source == "final":
        return _write_block_lookahead(backbone, heads, opened, max_length, sample, generator, timings, temperature,
                                      lookahead, allow_empty)
    clock = _Clock(state_device(opened)) if timings is not None else None
    k, depth = heads.cutoff, backbone.num_layers
    max_length = min(max_length or heads.max_length, heads.max_length)
    cache = opened.cache
    state = opened.h_cut
    batch = state.shape[0]
    device = state.device
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    lengths = torch.zeros(batch, dtype=torch.long, device=device)
    truncated = torch.zeros(batch, dtype=torch.bool, device=device)
    sketches, shallow, stop_logits = [], [], []
    count = 0
    if clock:
        clock.start()
    while True:
        if count > 0 or allow_empty:
            logit = heads.stop(state, torch.full((batch,), count, device=device, dtype=torch.long))
            if count > 0:
                stop_logits.append(logit)
            if sample:
                stop = torch.rand(batch, generator=generator, device=device) < torch.sigmoid(logit.float())
            else:
                stop = logit > 0
            newly = stop & ~done
            lengths = torch.where(newly, torch.full_like(lengths, count), lengths)
            done = done | stop
        if bool(done.all()):
            break
        if count == max_length:
            lengths = torch.where(done, lengths, torch.full_like(lengths, count))
            truncated = ~done
            break
        sketch = heads.feedback(state)  # [B, d]
        h, cache = backbone.run_layers(sketch[:, None], range(0, k), cache)
        sketches.append(sketch)
        shallow.append(h[:, 0])
        state = h[:, 0]
        count += 1

    if count == 0:
        empty = opened.h_cut.new_zeros(batch, 0, opened.h_cut.shape[-1])
        return WriteResult(empty, empty, empty, empty, lengths, truncated,
                           opened.h_cut.new_zeros(batch, 0), opened.cache)
    sketches_t = torch.stack(sketches, 1)
    shallow_t = torch.stack(shallow, 1)
    if clock:
        timings["shallow_generation"] = clock.lap()
    # Upper layers continue from the block start; the shallow layers' sketch cache is discarded.
    final, _ = backbone.run_layers(shallow_t, range(k, depth), opened.cache)
    if clock:
        timings["completion"] = clock.lap()
    mu, log_sigma = heads.content.distribution(sketches_t, final)
    sampled = sample_payload(mu, log_sigma, temperature, generator)
    payload = sampled.payload
    log_prob = None
    if temperature > 0:
        valid = torch.arange(payload.shape[1], device=device)[None] < lengths[:, None]
        log_prob = (payload_log_prob(sampled) * valid).sum(-1)
    if clock:
        timings["projection"] = clock.lap()
    stops = torch.stack(stop_logits, 1) if stop_logits else payload.new_zeros(batch, 0)
    return WriteResult(payload, sketches_t, shallow_t, final, lengths, truncated, stops, opened.cache,
                       mu, log_sigma, log_prob, temperature)


def _write_block_lookahead(backbone, heads, opened, max_length, sample, generator, timings, temperature, lookahead,
                           allow_empty) -> WriteResult:
    if allow_empty:
        raise ValueError("the final stop source decides on completed positions; it cannot write an empty block")
    clock = _Clock(state_device(opened)) if timings is not None else None
    k, depth = heads.cutoff, backbone.num_layers
    max_length = min(max_length or heads.max_length, heads.max_length)
    cache, upper, state = opened.cache, opened.cache, opened.h_cut
    batch, device = state.shape[0], state.device
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    lengths = torch.full((batch,), max_length, dtype=torch.long, device=device)
    sketches, shallow, stop_logits = [], [], []
    if clock:
        clock.start()
    count = 0
    while count < max_length and not bool(done.all()):
        steps = min(max(1, lookahead), max_length - count)
        for _ in range(steps):
            sketch = heads.feedback(state)
            h, cache = backbone.run_layers(sketch[:, None], range(0, k), cache)
            sketches.append(sketch)
            shallow.append(h[:, 0])
            state = h[:, 0]
        chunk, upper = backbone.run_layers(torch.stack(shallow[count:count + steps], 1), range(k, depth), upper)
        counts = torch.arange(count + 1, count + steps + 1, device=device).expand(batch, -1)
        logits = heads.stop(chunk, counts)
        for j in range(steps):
            c = count + j + 1
            if c == max_length:  # the hard maximum truncates; no decision there
                break
            stop_logits.append(logits[:, j])
            if sample:
                stop = torch.rand(batch, generator=generator, device=device) < torch.sigmoid(logits[:, j].float())
            else:
                stop = logits[:, j] > 0
            newly = stop & ~done
            lengths = torch.where(newly, torch.full_like(lengths, c), lengths)
            done = done | stop
        count += steps
    truncated = ~done
    written = int(lengths.max())
    sketches_t = torch.stack(sketches[:written], 1)
    shallow_t = torch.stack(shallow[:written], 1)
    if clock:
        timings["shallow_generation"] = clock.lap()
    final, _ = backbone.run_layers(shallow_t, range(k, depth), opened.cache)
    if clock:
        timings["completion"] = clock.lap()
    mu, log_sigma = heads.content.distribution(sketches_t, final)
    sampled = sample_payload(mu, log_sigma, temperature, generator)
    payload = sampled.payload
    log_prob = None
    if temperature > 0:
        valid = torch.arange(payload.shape[1], device=device)[None] < lengths[:, None]
        log_prob = (payload_log_prob(sampled) * valid).sum(-1)
    if clock:
        timings["projection"] = clock.lap()
    stops = torch.stack(stop_logits, 1)[:, :written] if stop_logits else payload.new_zeros(batch, 0)
    return WriteResult(payload, sketches_t, shallow_t, final, lengths, truncated, stops, opened.cache,
                       mu, log_sigma, log_prob, temperature)


def state_device(opened: Opened) -> torch.device:
    return opened.h_cut.device


class _Clock:
    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.t = 0.0

    def _now(self) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def start(self):
        self.t = self._now()

    def lap(self) -> float:
        now = self._now()
        elapsed, self.t = now - self.t, now
        return elapsed


def read_back(backbone: PortBackbone, heads: PortHeads, block_start: PortCache, payload: torch.Tensor) -> dict:
    """Restore the block-start cache and prefill payload + close marker through the full model.

    `payload` is [B, L, d] with the same L for all rows (use a single-row cache via
    `PortCache.select` for ragged blocks). Returns the cache after the close marker and the
    logits at the close marker, from which ordinary decoding resumes.
    """
    batch = payload.shape[0]
    close = torch.full((batch, 1), backbone.controls.close_id, device=payload.device, dtype=torch.long)
    embeds = torch.cat([heads.interface(payload), backbone.embed(close)], dim=1)
    out = backbone.forward_embeds(embeds, cache=block_start)
    return {"cache": out["cache"], "logits": out["logits"][:, -1], "all_logits": out["logits"]}


def greedy_continue(backbone: PortBackbone, cache: PortCache, logits: torch.Tensor, steps: int) -> tuple[list[int], list[torch.Tensor]]:
    """Greedy text decoding for one row from a cache and the logits at its last position."""
    tokens, step_logits = [], [logits]
    current = logits
    for _ in range(steps):
        token = int(current[0].argmax())
        tokens.append(token)
        out = backbone.forward_ids(torch.tensor([[token]], device=logits.device), cache=cache)
        cache, current = out["cache"], out["logits"][:, -1]
        step_logits.append(current)
    return tokens, step_logits
