"""The nested family inside any Neuralese stage (plans/neuralese/MAPLE_NESTED.md §4a, §9 item 5).

Owner goal (2026-10-08): every member ``LxE`` of a Maple-family student (Maple, Mellum) is a genuine objective in
every stage that moves shared weights, not only in ``nested_train``. Stages call these helpers so the family term and
its evaluation are one implementation:

- ``family_members``/``private_parameters``: the members of a loaded student and their private parts (router rows,
  norm gain corrections, private attention LoRA).
- ``member_loss``: one member's joint-phase objective on a token window, CE on the supervised targets plus
  KL(full ‖ member) to the current full model (detached teacher), exactly ``nested_train.member_step``'s terms.
- ``evaluate_members``: held CE and KL for the full model (KL to adapters-off original) and every member.
"""
from __future__ import annotations

import torch

from ..train.joint_kd import IGNORE, chunked_ce_kl
from .nested_train import run
from .ternary import adapters_disabled


def family_members(backbone) -> list:
    """Members of the student under a port backbone (empty for a non-family backbone)."""
    return list(getattr(backbone.hf, "members", None) or [])


def private_parameters(backbone) -> list[tuple[str, torch.nn.Parameter]]:
    """Every member's private parts, named as in ``backbone.hf``."""
    return [(name, parameter) for name, parameter in backbone.hf.named_parameters()
            if ".private." in name or "private_gate." in name]


def window_labels(ids: torch.Tensor, prefix: int) -> torch.Tensor:
    """Next-token labels of a [B, T] window whose first ``prefix`` tokens are context only (IGNORE)."""
    labels = torch.full_like(ids, IGNORE)
    labels[:, prefix - 1:-1] = ids[:, prefix:]
    return labels


def member_loss(backbone, member, ids: torch.Tensor, labels: torch.Tensor, *, kl_weight: float = 1.0,
                chunk: int = 256):
    """``CE_member + kl_weight · KL(full ‖ member)`` over the labelled positions; the full model is the detached
    teacher (it is trained by the stage's own objective). Returns (loss, LossParts)."""
    head = backbone.hf.get_output_embeddings().weight
    with torch.no_grad():
        full = run(backbone.hf, None, ids).last_hidden_state
    hidden = run(backbone.hf, member, ids).last_hidden_state
    return chunked_ce_kl(hidden, head, labels, full, head, kl_weight=kl_weight, chunk=chunk)


@torch.no_grad()
def evaluate_members(backbone, windows, *, chunk: int = 256) -> dict:
    """``windows``: iterable of (ids [1, T], labels [1, T]). Full model: CE and KL to the adapters-off original;
    members: CE and KL to the current full model. Token-weighted means."""
    head = backbone.hf.get_output_embeddings().weight
    totals: dict[str, list[float]] = {}

    def add(key, parts):
        row = totals.setdefault(key, [0.0, 0.0, 0])
        row[0] += parts.ce
        row[1] += parts.kl
        row[2] += parts.tokens

    for ids, labels in windows:
        with adapters_disabled():
            original = run(backbone.hf, None, ids).last_hidden_state
        full = run(backbone.hf, None, ids).last_hidden_state
        add("full", chunked_ce_kl(full, head, labels, original, head, kl_weight=1.0, chunk=chunk)[1])
        for member in family_members(backbone):
            hidden = run(backbone.hf, member, ids).last_hidden_state
            add(member.key, chunked_ce_kl(hidden, head, labels, full, head, kl_weight=1.0, chunk=chunk)[1])
    return {key: {"ce": c / max(n, 1), "kl": kl / max(n, 1), "tokens": n} for key, (c, kl, n) in totals.items()}
