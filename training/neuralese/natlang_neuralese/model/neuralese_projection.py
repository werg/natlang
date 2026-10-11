"""The two projections of a Neuralese direct stage (plans/neuralese/NEURALESE_DIRECT_V1.md), kept separate:

- ``NeuraleseProjection``: the Neuralese output, the value passed on and fed back. It lives in the input
  token-embedding space: the base model's own soft-token mixture softmax(lm_head(norm(h)))·E plus a learned causal
  correction (zero at initialization), so at step 0 the value is exactly that mixture.
- ``TokenChoiceProjection``: the last hidden state into the hidden space the frozen final norm and lm_head read, for
  token prediction; identity at initialization. A recipe-declared stage component, absent until the stage that trains
  the backbone end to end on Neuralese.

Both are built from one ``CausalConvStack``: residual blocks of RMS-norm, a depthwise causal convolution over positions
and a pointwise MLP whose output projection starts at zero. A ``segments`` id per position (``-1``: no segment) keeps
every convolution tap inside its own segment (a turn in training, a call span at serving time).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def segment_shifted(values: torch.Tensor, segments: torch.Tensor | None, shift: int) -> torch.Tensor:
    """``values`` ([B, T, D]) delayed by ``shift`` positions, zero where the delayed position is outside the window or in
    another segment than the current one (or the current position has no segment)."""
    if shift == 0:
        if segments is None:
            return values
        return values * (segments >= 0).unsqueeze(-1).to(values.dtype)
    delayed = F.pad(values, (0, 0, shift, 0))[:, :values.shape[1]]
    if segments is None:
        return delayed
    earlier = F.pad(segments, (shift, 0), value=-1)[:, :segments.shape[1]]
    keep = (earlier == segments) & (segments >= 0)
    return delayed * keep.unsqueeze(-1).to(values.dtype)


class CausalConvBlock(nn.Module):
    def __init__(self, width: int, hidden: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.norm = nn.RMSNorm(width)
        self.taps = nn.Parameter(torch.zeros(kernel, width))
        with torch.no_grad():
            self.taps[0].fill_(1.0)  # the current position; earlier taps start at zero
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor, segments: torch.Tensor | None = None) -> torch.Tensor:
        y = self.norm(x)
        mixed = sum(segment_shifted(y, segments, k) * self.taps[k] for k in range(self.kernel))
        return x + self.down(F.silu(self.up(mixed)))


class CausalConvStack(nn.Module):
    """``blocks`` residual causal-conv blocks; the identity map at initialization."""

    def __init__(self, width: int, *, blocks: int = 2, hidden: int | None = None, kernel: int = 4):
        super().__init__()
        if blocks < 1 or kernel < 1:
            raise ValueError('a causal conv stack needs at least one block and a positive kernel')
        hidden = hidden or max(1, width // 4)
        self.config = {'width': width, 'blocks': blocks, 'hidden': hidden, 'kernel': kernel}
        self.blocks = nn.ModuleList(CausalConvBlock(width, hidden, kernel) for _ in range(blocks))

    def forward(self, x: torch.Tensor, segments: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, segments)
        return x


class NeuraleseProjection(nn.Module):
    """The Neuralese value at each position: ``mixture + correction``, both in input token-embedding space.

    ``mixture`` is the soft-token mixture of the base readout at that position (computed by the caller from the frozen
    final norm, lm_head and embedding, see ``soft_token_mixture``); ``correction`` is a ``CausalConvStack`` over the
    last hidden states, minus its input, so it is exactly zero at initialization."""

    def __init__(self, width: int, *, blocks: int = 2, hidden: int | None = None, kernel: int = 4):
        super().__init__()
        self.stack = CausalConvStack(width, blocks=blocks, hidden=hidden, kernel=kernel)

    def forward(self, hidden: torch.Tensor, mixture: torch.Tensor, segments: torch.Tensor | None = None,
                *, last_only: bool = False):
        correction = self.stack(hidden, segments) - hidden
        if last_only:
            # Incremental serving needs the receptive-field history but only the
            # newest position's mixture. The projection equation is shared.
            correction = correction[:, -1:]
        return mixture + correction.to(mixture.dtype)


class TokenChoiceProjection(nn.Module):
    """Last hidden state -> the hidden space the frozen final norm and lm_head read; the identity at initialization."""

    def __init__(self, width: int, *, blocks: int = 2, hidden: int | None = None, kernel: int = 4):
        super().__init__()
        self.stack = CausalConvStack(width, blocks=blocks, hidden=hidden, kernel=kernel)

    def forward(self, hidden: torch.Tensor, segments: torch.Tensor | None = None) -> torch.Tensor:
        return self.stack(hidden, segments)


@torch.no_grad()
def soft_token_mixture(logits_fn, embedding: torch.Tensor, hidden: torch.Tensor, *, chunk: int = 128) -> torch.Tensor:
    """softmax(logits_fn(hidden))·embedding per position, in position chunks (never all positions' full-vocabulary
    probabilities at once). ``logits_fn`` is the frozen readout (final norm + lm_head)."""
    pieces = []
    for start in range(0, hidden.shape[1], chunk):
        probabilities = torch.softmax(logits_fn(hidden[:, start:start + chunk]).float(), -1)
        pieces.append((probabilities.to(embedding.dtype) @ embedding))
    return torch.cat(pieces, 1)
