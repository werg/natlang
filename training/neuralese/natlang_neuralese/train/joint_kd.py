"""Joint training with exact distillation between two models that share a tokenizer (Maple and Qwen3-0.6B).

The vocabulary is large (151,936), so logits are never materialised for a whole batch: ``chunked_ce_kl`` walks the
supervised positions in chunks, computes the student's cross-entropy and the full-vocabulary KL to the teacher there,
and computes the gradients with respect to the student's hidden states and output matrix in the same pass (no
recomputation; nothing of size tokens × vocabulary outlives a chunk). The teacher is represented by its final-norm
hidden states and output matrix, both detached. See plans/neuralese/MAPLE_QWEN_JOINT.md §2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

IGNORE = -100


def tokenizer_fingerprint(model_dir: str | Path) -> str:
    """Hash of what decides token IDs: vocabulary, merges, pre-tokenizer, normalizer and added tokens."""
    data = json.loads((Path(model_dir) / "tokenizer.json").read_text())
    model = data["model"]
    payload = {
        "vocab": model.get("vocab"),
        "merges": model.get("merges"),
        "pre_tokenizer": data.get("pre_tokenizer"),
        "normalizer": data.get("normalizer"),
        "added": sorted((t["id"], t["content"]) for t in data.get("added_tokens", [])),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def require_shared_tokenizer(teacher_dir: str | Path, student_dir: str | Path) -> str:
    a, b = tokenizer_fingerprint(teacher_dir), tokenizer_fingerprint(student_dir)
    if a != b:
        raise ValueError(f"teacher and student tokenizers differ ({a[:12]} vs {b[:12]}): exact distillation needs one")
    return a


@dataclass
class LossParts:
    ce: float
    kl: float
    tokens: int


class _ChunkedCEKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, teacher_hidden, teacher_weight, labels, ce_weight, kl_weight, temperature,
                normaliser, chunk, parts):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return _ChunkedCEKL._forward(ctx, hidden, weight, teacher_hidden, teacher_weight, labels, ce_weight,
                                         kl_weight, temperature, normaliser, chunk, parts)

    @staticmethod
    def _forward(ctx, hidden, weight, teacher_hidden, teacher_weight, labels, ce_weight, kl_weight, temperature,
                 normaliser, chunk, parts):
        flat = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        positions = (flat_labels != IGNORE).nonzero().squeeze(-1)
        grad_hidden = torch.zeros_like(flat, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        teacher_flat = None if teacher_hidden is None else teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
        total = ce_sum = kl_sum = 0.0
        w = weight.float()
        tw = None if teacher_weight is None else teacher_weight.float()
        for start in range(0, positions.numel(), chunk):
            index = positions[start:start + chunk]
            h = flat[index].float()
            logits = h @ w.T
            target = flat_labels[index]
            log_p = torch.log_softmax(logits, dim=-1)
            p = log_p.exp()
            ce = -log_p.gather(1, target[:, None]).squeeze(1)
            grad_logits = ce_weight * p
            grad_logits[torch.arange(index.numel(), device=h.device), target] -= ce_weight
            loss = ce_weight * ce.sum()
            ce_sum += ce.sum().item()
            if teacher_flat is not None and kl_weight:
                teacher_logits = teacher_flat[index].float() @ tw.T
                log_q = torch.log_softmax(teacher_logits / temperature, dim=-1)
                log_p_t = torch.log_softmax(logits / temperature, dim=-1) if temperature != 1 else log_p
                q = log_q.exp()
                kl = (q * (log_q - log_p_t)).sum(-1)
                scale = kl_weight * temperature ** 2
                loss = loss + scale * kl.sum()
                kl_sum += kl.sum().item()
                grad_logits += (kl_weight * temperature) * (log_p_t.exp() - q)
            total += loss.item()
            grad_logits /= normaliser
            grad_hidden[index] = grad_logits @ w
            grad_weight += grad_logits.T @ h
        parts.append(LossParts(ce_sum, kl_sum, positions.numel()))
        ctx.save_for_backward(grad_hidden.reshape(hidden.shape), grad_weight)
        ctx.dtypes = (hidden.dtype, weight.dtype)
        return hidden.new_tensor(total / normaliser, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        return ((grad_hidden * grad_output).to(ctx.dtypes[0]), (grad_weight * grad_output).to(ctx.dtypes[1]),
                None, None, None, None, None, None, None, None, None)


def chunked_ce_kl(hidden, weight, labels, teacher_hidden=None, teacher_weight=None, *, ce_weight=1.0,
                  kl_weight=0.0, temperature=1.0, normaliser=None, chunk=1024):
    """``(ce_weight * CE + kl_weight * T² * KL(teacher ‖ student)) / normaliser`` over positions whose label is not
    ``IGNORE``. ``hidden`` (…, d) are the student's final-norm states (labels already shifted to align),
    ``weight`` (vocab, d) its output matrix; the teacher's are used detached. Returns ``(loss, LossParts)``."""
    if normaliser is None:
        normaliser = max(int((labels != IGNORE).sum()), 1)
    parts: list[LossParts] = []
    teacher_hidden = None if teacher_hidden is None else teacher_hidden.detach()
    teacher_weight = None if teacher_weight is None else teacher_weight.detach()
    loss = _ChunkedCEKL.apply(hidden, weight, teacher_hidden, teacher_weight, labels, ce_weight, kl_weight,
                              temperature, float(normaliser), chunk, parts)
    return loss, parts[0]


def kl_ramp(step: int, ramp_steps: int, weight: float) -> float:
    """Distillation weight: 0 at step 0, rising linearly to ``weight`` (the teacher is training too)."""
    if ramp_steps <= 0:
        return weight
    return weight * min(1.0, step / ramp_steps)
