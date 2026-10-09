"""Shared inference context for Neuralese evaluation runners."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def evaluation_inference(*modules, ffn_owner=None, ffn_chunk_tokens: int | None = None) -> Iterator[None]:
    """Run evaluation without autograd and restore module/runtime settings on exit.

    ``ffn_owner`` is normally the backbone. Setting a positive chunk size keeps
    long evaluation prefixes on the same bounded FFN path used by warmup evals.
    """
    if ffn_chunk_tokens is not None and ffn_chunk_tokens < 1:
        raise ValueError("ffn_chunk_tokens must be positive")
    modes = [(module, bool(module.training)) for module in modules if hasattr(module, "training")]
    old_chunk = getattr(ffn_owner, "ffn_chunk_tokens", None) if ffn_owner is not None else None
    try:
        for module, _ in modes:
            module.eval()
        if ffn_owner is not None and ffn_chunk_tokens is not None:
            ffn_owner.ffn_chunk_tokens = ffn_chunk_tokens
        with torch.inference_mode():
            yield
    finally:
        if ffn_owner is not None and ffn_chunk_tokens is not None:
            ffn_owner.ffn_chunk_tokens = old_chunk
        for module, was_training in modes:
            module.train(was_training)
