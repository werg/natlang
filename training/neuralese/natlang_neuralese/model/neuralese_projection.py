"""The two projections of a Neuralese direct lineage (plans/neuralese/NEURALESE_DIRECT_V1.md), kept separate:

- ``NeuraleseProjection`` (P): the Neuralese output, the value passed on and fed back, in the input token-embedding
  space: value(h) = E[chosen(h)] + C(h). chosen(h) is the token the line's readout chooses at that position (argmax
  of lm_head(norm(TCP(h))), the token that drives generation) and E[chosen] its ordinary input embedding, so the value
  starts exactly on the token-embedding manifold; C is a learned per-position correction (RMS normalisation, a linear
  map and a residual MLP), zero at its start, through which the consumer's gradient reaches the producer. There is no
  soft-token mixture softmax(lm_head(norm h))·E anywhere (owner 2026-10-13: the model reads a proper token embedding
  well and mixtures poorly; the embedding space is not a VAE space). A generic learned map h -> embedding cannot get
  there: the chosen token is an argmax over the whole vocabulary, and least squares or an MLP from h reached only
  cosine 0.72-0.82 / nearest 53-80% on LFM held positions (NEURALESE_DIRECT_V1.md "Projection alignment fit"), so the
  vocabulary-scale choice is the readout's and P learns only the correction. After entry no alignment loss or anchor
  applies; C moves freely under the consumer's gradient. Projections saved by the earlier mixture design (a
  ``stack.*`` causal-conv correction added to the mixture) are refused with ``ProjectionNeedsAlignment``; they are
  never silently treated as aligned.
- ``TokenChoiceProjection``: the last hidden state into the hidden space the frozen final norm and lm_head read, for
  token prediction; identity at initialization. A recipe-declared stage component, absent until the stage that trains
  the backbone end to end on Neuralese. Built from ``CausalConvStack``: residual blocks of RMS-norm, a depthwise causal
  convolution over positions and a pointwise MLP whose output projection starts at zero. A ``segments`` id per position
  (``-1``: no segment) keeps every convolution tap inside its own segment (a turn in training, a call span at serving).
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


class ProjectionNeedsAlignment(ValueError):
    """A NeuraleseProjection state that is not an aligned P: a pre-2026-10-13 mixture+stack correction, or none."""


ALIGNMENT_SCHEMA = 'natlang.neuralese-projection-alignment/1'


def is_legacy_projection_state(state) -> bool:
    """Whether a saved ``neuralese_projection`` state is the removed mixture design (a causal-conv correction added to
    the soft-token mixture: ``stack.*`` tensors)."""
    return isinstance(state, dict) and any(str(k).startswith('stack.') for k in state)


def needs_alignment_message(where: str) -> str:
    return (f'{where}: its NeuraleseProjection is the removed soft-token-mixture design (or absent), which needs an '
            'alignment fit before any use: run `python -m natlang_neuralese.train.projection_align fit` on this '
            'export and pass its aligned projection (aligned_projection / --aligned-projection)')


def rms_normalized(hidden: torch.Tensor) -> torch.Tensor:
    hidden = hidden.float()
    return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)


class NeuraleseProjection(nn.Module):
    """value(h) = anchor + C(h): ``anchor`` is E[chosen] at each position (``chosen_token_embedding``, given by the
    caller, who holds the readout), C(h) = W·rms(h) + b + MLP(rms(h)) with W, b and the MLP's output zero at
    initialization, so the value starts as the chosen token's embedding exactly. Per position: a later position never
    changes an earlier value (``segments`` and ``last_only`` keep the writer's calling convention)."""

    def __init__(self, width: int, *, embedding_width: int | None = None, hidden: int | None = None):
        super().__init__()
        embedding_width = embedding_width or width
        hidden = width * 2 if hidden is None else int(hidden)
        self.config = {'kind': 'token-anchored/1', 'width': width, 'embedding_width': embedding_width,
                       'hidden': hidden}
        self.linear = nn.Linear(width, embedding_width)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        if hidden:
            self.norm = nn.RMSNorm(width)
            self.up = nn.Linear(width, hidden)
            self.down = nn.Linear(hidden, embedding_width)
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    def correction(self, hidden: torch.Tensor) -> torch.Tensor:
        x = rms_normalized(hidden).to(self.linear.weight.dtype)
        value = self.linear(x)
        if self.config['hidden']:
            value = value + self.down(F.gelu(self.up(self.norm(x))))
        return value

    def forward(self, hidden: torch.Tensor, segments: torch.Tensor | None = None, *, anchor: torch.Tensor,
                last_only: bool = False):
        if last_only:
            hidden = hidden[:, -1:]
        if anchor.shape[:-1] != hidden.shape[:-1]:
            raise ValueError('the anchor is E[chosen] at exactly the projected positions')
        return anchor.detach() + self.correction(hidden).to(anchor.dtype)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if any(k.startswith(prefix + 'stack.') for k in state_dict):
            raise ProjectionNeedsAlignment(needs_alignment_message('NeuraleseProjection state'))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


ALIGNED_KINDS = ('token-anchored/1',)


def projection_from_config(config: dict) -> NeuraleseProjection:
    """The NeuraleseProjection a saved ``config`` describes; a legacy (causal-conv) config needs an alignment fit."""
    if not isinstance(config, dict) or config.get('kind') not in ALIGNED_KINDS:
        raise ProjectionNeedsAlignment(needs_alignment_message('NeuraleseProjection config'))
    return NeuraleseProjection(config['width'], embedding_width=config['embedding_width'], hidden=config['hidden'])


def projection_config_from_state(state: dict) -> dict:
    """The projection's config from its tensors (exports and checkpoints carry the state)."""
    if is_legacy_projection_state(state) or 'linear.weight' not in state:
        raise ProjectionNeedsAlignment(needs_alignment_message('NeuraleseProjection state'))
    embedding_width, width = state['linear.weight'].shape
    hidden = int(state['up.weight'].shape[0]) if 'up.weight' in state else 0
    return {'kind': 'token-anchored/1', 'width': int(width), 'embedding_width': int(embedding_width),
            'hidden': hidden}


def save_aligned_projection(path, projection: NeuraleseProjection, *, artifact_manifest_sha256: str, report: dict):
    """The P that enters the end-to-end stage from one exact direct export (bound by its manifest SHA-256), with its
    manifold report."""
    from pathlib import Path
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    torch.save({'schema': ALIGNMENT_SCHEMA, 'artifact_manifest_sha256': artifact_manifest_sha256,
                'config': dict(projection.config),
                'state': {k: v.detach().cpu().contiguous() for k, v in projection.state_dict().items()},
                'report': report}, pending)
    pending.replace(path)


def load_aligned_projection(path, artifact_manifest_sha256: str):
    """(P, alignment) from ``save_aligned_projection``'s file; refused unless it was made on this exact export."""
    from pathlib import Path
    from ..common.hashing import sha256_file_hex
    path = Path(path).resolve(strict=True)
    item = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(item, dict) or item.get('schema') != ALIGNMENT_SCHEMA:
        raise ValueError(f'{path} is not an aligned NeuraleseProjection ({ALIGNMENT_SCHEMA})')
    if item.get('artifact_manifest_sha256') != artifact_manifest_sha256:
        raise ValueError(f'{path} was fitted on another direct export (manifest '
                         f'{item.get("artifact_manifest_sha256")}, this export {artifact_manifest_sha256})')
    projection = projection_from_config(item['config'])
    projection.load_state_dict(item['state'], strict=True)
    return projection, {'source': 'aligned_projection', 'path': str(path), 'sha256': sha256_file_hex(path),
                        'report': item.get('report')}


@torch.no_grad()
def chosen_token_ids(backbone, hidden: torch.Tensor, token_choice_projection=None,
                     segments: torch.Tensor | None = None, *, chunk: int = 512) -> torch.Tensor:
    """The token the line's readout chooses at each position: argmax of lm_head(norm(TCP(h))), [B, T]."""
    states = project_token_states(hidden, token_choice_projection, segments)
    return torch.cat([backbone.logits(states[:, s:s + chunk]).argmax(-1)
                      for s in range(0, states.shape[1], chunk)], 1)


def chosen_token_embedding(backbone, ids: torch.Tensor) -> torch.Tensor:
    """E[chosen]: the ordinary input embedding of the chosen tokens ([B, T] ids), exactly what a consumer reads for
    that token, detached (the anchor carries no gradient)."""
    return backbone.embed(ids).detach()


def neg_squared_distance_logits(value: torch.Tensor, embedding: torch.Tensor,
                                embedding_sq: torch.Tensor | None = None) -> torch.Tensor:
    """-||value - E_v||^2 up to a per-row constant: 2 value·E_v - ||E_v||^2. Its argmax is the nearest token."""
    embedding = embedding.float()
    if embedding_sq is None:
        embedding_sq = embedding.square().sum(-1)
    return 2 * value.float() @ embedding.t() - embedding_sq


@torch.no_grad()
def alignment_metrics(value: torch.Tensor, chosen: torch.Tensor, embedding: torch.Tensor,
                      gold: torch.Tensor | None = None) -> dict:
    """Per-position tensors: cosine and norm ratio to E[chosen], nearest-token hit for chosen (and gold), the rank of
    chosen among all tokens by distance, relative squared error to E[chosen]."""
    embedding = embedding.float()
    value = value.float()
    target = embedding[chosen]
    logits = neg_squared_distance_logits(value, embedding)
    nearest = logits.argmax(-1)
    out = {'cos_chosen': F.cosine_similarity(value, target, -1),
           'norm_ratio': value.norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-12),
           'nearest_chosen': (nearest == chosen).float(),
           'rank_chosen': (logits > logits.gather(1, chosen[:, None])).sum(-1).float(),
           'relative_error_chosen': (value - target).square().sum(-1) / target.square().sum(-1).clamp_min(1e-12)}
    if gold is not None:
        out['cos_gold'] = F.cosine_similarity(value, embedding[gold], -1)
        out['nearest_gold'] = (nearest == gold).float()
    return out


class TokenChoiceProjection(nn.Module):
    """Last hidden state -> the hidden space the frozen final norm and lm_head read; the identity at initialization."""

    def __init__(self, width: int, *, blocks: int = 2, hidden: int | None = None, kernel: int = 4):
        super().__init__()
        self.stack = CausalConvStack(width, blocks=blocks, hidden=hidden, kernel=kernel)

    def forward(self, hidden: torch.Tensor, segments: torch.Tensor | None = None) -> torch.Tensor:
        return self.stack(hidden, segments)


def causal_conv_receptive_field(stack_or_projection) -> int:
    """Number of current and preceding positions read by a causal projection stack.

    This is derived from the declared stack geometry so incremental callers can
    retain exactly the history needed by the projection.
    """
    if isinstance(stack_or_projection, NeuraleseProjection):
        return 1  # per position
    stack = getattr(stack_or_projection, 'stack', stack_or_projection)
    config = getattr(stack, 'config', None)
    if not isinstance(config, dict):
        raise TypeError('expected a causal convolution stack or projection')
    blocks, kernel = config.get('blocks'), config.get('kernel')
    if type(blocks) is not int or blocks < 1 or type(kernel) is not int or kernel < 1:
        raise ValueError('causal convolution stack has invalid blocks/kernel configuration')
    return 1 + blocks * (kernel - 1)


def project_token_states(hidden: torch.Tensor, projection: TokenChoiceProjection | None = None,
                         segments: torch.Tensor | None = None,
                         detach_output_positions: torch.Tensor | None = None) -> torch.Tensor:
    """Apply the optional causal token-choice projection to hidden states."""
    read = hidden if projection is None else projection(hidden, segments)
    if detach_output_positions is None or not bool(detach_output_positions.any()):
        return read
    if projection is None:
        raise ValueError('detached output positions require a token-choice projection')
    if detach_output_positions.shape != hidden.shape[:-1]:
        raise ValueError('detached output mask must match hidden positions')
    # Detach the entire causal history for these outputs. Detaching only their
    # own input would leak backbone gradients through earlier convolution taps.
    detached_read = projection(hidden.detach(), segments)
    return torch.where(detach_output_positions[..., None], detached_read, read)


def token_choice_logits(backbone, hidden: torch.Tensor, projection: TokenChoiceProjection | None = None,
                        segments: torch.Tensor | None = None, last_only: bool = False) -> torch.Tensor:
    """Apply the optional causal token-choice projection, then the base vocabulary readout.

    ``last_only`` slices after projection, so the final position still sees its
    full supplied causal history. Callers must provide the complete current
    segment (or its receptive-field suffix) when using it incrementally.
    """
    if hidden.ndim != 3 or hidden.shape[1] < 1:
        raise ValueError('token readout expects nonempty [batch,time,width] hidden states')
    hidden = project_token_states(hidden, projection, segments)
    if last_only:
        hidden = hidden[:, -1:]
    return backbone.logits(hidden)
