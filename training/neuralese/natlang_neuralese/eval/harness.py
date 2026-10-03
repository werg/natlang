"""The S3 evaluation harness (S3 §6), producing a JSON report.

Every consumer evaluation compares, at matched length: the **correct** payload, a
**shuffled** payload (another example's block, same length), a **zeroed** payload, **no
block**, and the **full-text** reference. Blocks are written free-running (learned
stopping, hard maximum), never from supplied inputs. Aggregate improvement alone is not
evidence: the correct-versus-shuffled margin is reported per family.

Sections: payload ablations (spans and port records), stopping, representation monitors,
cached-versus-recomputed agreement, and latency per phase.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from ..model.heads import PortHeads
from ..model.lfm2_port import PortBackbone
from ..write import greedy_continue, open_block, read_back, write_block
from ..train.execution import consumer_forward, read_continue, teacher_target_logits


def _nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)))


def _matched(block: torch.Tensor, length: int) -> torch.Tensor:
    """`block` cut or cycled to `length` rows."""
    if block.shape[0] >= length:
        return block[:length]
    reps = math.ceil(length / max(1, block.shape[0]))
    return block.repeat(reps, 1)[:length]


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rank = lambda v: torch.tensor(v, dtype=torch.float).argsort().argsort().float()
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


def representation_monitors(blocks: list[torch.Tensor]) -> dict:
    """RMS, effective rank, collapse towards the mean direction, cross-source similarity."""
    if not blocks:
        return {}
    vectors = torch.cat([b.float() for b in blocks], 0)
    rms = vectors.pow(2).mean(-1).sqrt()
    centered = vectors - vectors.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    p = singular / singular.sum().clamp(min=1e-12)
    effective_rank = float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()))
    mean_direction = F.normalize(vectors.mean(0), dim=0)
    cos_to_mean = F.cosine_similarity(vectors, mean_direction[None], dim=-1)
    means = torch.stack([F.normalize(b.float().mean(0), dim=0) for b in blocks])
    sim = means @ means.t()
    off = sim[~torch.eye(len(blocks), dtype=torch.bool)]
    return {
        "vectors": vectors.shape[0],
        "rms_mean": float(rms.mean()), "rms_std": float(rms.std()) if rms.numel() > 1 else 0.0,
        "effective_rank": effective_rank,
        "cosine_to_mean": float(cos_to_mean.mean()),
        "cross_source_similarity": float(off.mean()) if off.numel() else None,
    }


@torch.no_grad()
def evaluate_spans(backbone: PortBackbone, heads: PortHeads, examples, max_length: int | None = None) -> dict:
    """Ablations on text spans: the block stands for the span; the consumer continues the text."""
    device = backbone.embedding_weight.device
    rows, blocks, lengths, truncated, span_sizes = [], [], [], [], []
    for example in examples:
        prefix = torch.tensor([example.prefix + [backbone.controls.open_id]], device=device)
        opened = open_block(backbone, heads, prefix)
        written = write_block(backbone, heads, opened, max_length=max_length)
        blocks.append(written.row(0))
        lengths.append(int(written.lengths[0]))
        truncated.append(bool(written.truncated[0]))
        span_sizes.append(len(example.span))
        rows.append((example, opened))
    results = defaultdict(list)
    for i, (example, opened) in enumerate(rows):
        cont = torch.tensor([example.continuation], device=device)
        block = blocks[i]
        other = blocks[(i + 1) % len(blocks)]
        for name, payload in (("correct", block), ("shuffled", _matched(other, block.shape[0])),
                              ("zeroed", torch.zeros_like(block))):
            if payload.shape[0] == 0:
                continue
            logits = read_continue(backbone, heads, opened.cache, payload[None], cont)
            results[name].append(_nll(logits, cont))
        plain_no_block = torch.tensor([example.prefix + example.continuation[:-1]], device=device)
        logits = backbone.forward_ids(plain_no_block)["logits"][:, -len(example.continuation):]
        results["no_block"].append(_nll(logits, cont))
        plain_full = torch.tensor([example.prefix + example.span + example.continuation[:-1]], device=device)
        logits = backbone.forward_ids(plain_full)["logits"][:, -len(example.continuation):]
        results["full_text"].append(_nll(logits, cont))
    summary = _ablation_summary(results)
    summary["stopping"] = _stopping(lengths, truncated, span_sizes)
    summary["representation"] = representation_monitors(blocks)
    return summary


@torch.no_grad()
def evaluate_records(backbone: PortBackbone, heads: PortHeads, rendered_records, max_length: int | None = None) -> dict:
    """Ablations on port records, per family: the consumer answers with the source withheld."""
    device = backbone.embedding_weight.device
    written_rows = []
    for rendered in rendered_records:
        opened = open_block(backbone, heads, torch.tensor([rendered.producer], device=device))
        written = write_block(backbone, heads, opened, max_length=max_length)
        written_rows.append((rendered, written.row(0), int(written.lengths[0]), bool(written.truncated[0])))
    by_family = defaultdict(lambda: defaultdict(list))
    for i, (rendered, block, _, _) in enumerate(written_rows):
        target = torch.tensor([rendered.target], device=device)
        others = [w for w in written_rows if w[0].family == rendered.family and w[0].record_id != rendered.record_id]
        other = (others[i % len(others)] if others else written_rows[(i + 1) % len(written_rows)])[1]
        variants = {"correct": block, "shuffled": _matched(other, block.shape[0]), "zeroed": torch.zeros_like(block),
                    "no_block": None}
        for name, payload in variants.items():
            if payload is not None and payload.shape[0] == 0:
                continue
            logits = consumer_forward(backbone, heads, rendered.consumer_before, payload, rendered.consumer_after,
                                      rendered.target)
            by_family[rendered.family][name].append(_nll(logits, target))
        by_family[rendered.family]["full_text"].append(
            _nll(teacher_target_logits(backbone, rendered.teacher_prefix, rendered.target), target))
    report = {"families": {family: _ablation_summary(results) for family, results in by_family.items()}}
    report["stopping"] = _stopping([w[2] for w in written_rows], [w[3] for w in written_rows],
                                   [w[0].source_tokens for w in written_rows])
    report["representation"] = representation_monitors([w[1] for w in written_rows])
    return report


def _ablation_summary(results: dict) -> dict:
    mean = {name: sum(values) / len(values) for name, values in results.items() if values}
    out = {"n": len(results.get("correct", [])), "nll": mean}
    if "correct" in mean and "shuffled" in mean:
        pairs = list(zip(results["correct"], results["shuffled"]))
        out["correct_minus_shuffled"] = mean["correct"] - mean["shuffled"]
        out["correct_better_than_shuffled"] = sum(c < s for c, s in pairs) / len(pairs)
    if {"correct", "no_block", "full_text"} <= mean.keys():
        gap = mean["no_block"] - mean["full_text"]
        out["gap_recovered"] = (mean["no_block"] - mean["correct"]) / gap if abs(gap) > 1e-6 else None
    return out


def _stopping(lengths: list[int], truncated: list[bool], source_sizes: list[int]) -> dict:
    if not lengths:
        return {}
    histogram = defaultdict(int)
    for length in lengths:
        histogram[length] += 1
    return {
        "mean_length": sum(lengths) / len(lengths),
        "truncation_rate": sum(truncated) / len(truncated),
        "length_one_rate": sum(1 for x in lengths if x == 1) / len(lengths),
        "length_histogram": dict(sorted(histogram.items())),
        "length_source_spearman": _spearman([float(x) for x in lengths], [float(x) for x in source_sizes]),
        "mean_source_tokens_per_vector": sum(source_sizes) / max(1, sum(lengths)),
    }


@torch.no_grad()
def cache_agreement(backbone: PortBackbone, heads: PortHeads, prefixes: list[list[int]], steps: int = 8,
                    max_length: int | None = None) -> dict:
    """Incremental write + readback + decoding versus one from-scratch forward of the committed sequence."""
    device = backbone.embedding_weight.device
    same, max_diff = 0, 0.0
    for prefix in prefixes:
        ids = torch.tensor([prefix + [backbone.controls.open_id]], device=device)
        opened = open_block(backbone, heads, ids)
        written = write_block(backbone, heads, opened, max_length=max_length)
        payload = written.row(0)[None]
        back = read_back(backbone, heads, written.block_start, payload)
        tokens, cached = greedy_continue(backbone, back["cache"], back["logits"], steps=steps)
        close = torch.tensor([[backbone.controls.close_id]], device=device)
        sequence = torch.cat([backbone.embed(ids), heads.interface(payload), backbone.embed(close),
                              backbone.embed(torch.tensor([tokens], device=device))], 1)
        full = backbone.forward_embeds(sequence)["logits"][0]
        start = ids.shape[1] + payload.shape[1]
        recomputed = full[start: start + len(tokens) + 1]
        same += int(recomputed[:-1].argmax(-1).tolist() == tokens)
        max_diff = max(max_diff, float((torch.cat(cached, 0) - recomputed).abs().max()))
    return {"cases": len(prefixes), "identical_greedy_rate": same / max(1, len(prefixes)), "max_logit_diff": max_diff}


@torch.no_grad()
def latency(backbone: PortBackbone, heads: PortHeads, prefix: list[int], max_length: int | None = None,
            repeats: int = 3) -> dict:
    """Seconds per phase: prefix, shallow generation, completion + projection, readback; vs text decoding."""
    device = backbone.embedding_weight.device
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    totals = defaultdict(float)
    for _ in range(repeats):
        timings: dict = {}
        sync(); t0 = time.perf_counter()
        opened = open_block(backbone, heads, torch.tensor([prefix + [backbone.controls.open_id]], device=device))
        sync(); timings["prefix"] = time.perf_counter() - t0
        written = write_block(backbone, heads, opened, max_length=max_length, timings=timings)
        sync(); t0 = time.perf_counter()
        back = read_back(backbone, heads, written.block_start, written.row(0)[None])
        sync(); timings["readback"] = time.perf_counter() - t0
        length = int(written.lengths[0])
        t0 = time.perf_counter()
        greedy_continue(backbone, opened.cache, opened.logits, steps=max(1, length))
        sync(); timings["text_decode_same_length"] = time.perf_counter() - t0
        timings["block_length"] = length
        for key, value in timings.items():
            totals[key] += value / repeats
    return dict(totals)


def run_harness(backbone: PortBackbone, heads: PortHeads, *, span_examples=(), rendered_records=(),
                cache_prefixes=(), latency_prefix=None, max_length: int | None = None,
                out_path: str | Path | None = None, label: str = "") -> dict:
    was_training = heads.training
    heads.eval()
    report = {"label": label, "cutoff": heads.cutoff, "max_length": max_length or heads.max_length}
    if span_examples:
        report["spans"] = evaluate_spans(backbone, heads, list(span_examples), max_length)
    if rendered_records:
        report["records"] = evaluate_records(backbone, heads, list(rendered_records), max_length)
    if cache_prefixes:
        report["cache_agreement"] = cache_agreement(backbone, heads, list(cache_prefixes), max_length=max_length)
    if latency_prefix:
        report["latency_seconds"] = latency(backbone, heads, latency_prefix, max_length)
    if was_training:
        heads.train()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    return report
