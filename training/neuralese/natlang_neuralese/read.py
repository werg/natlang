"""The read port (S3 §3.2): splice payload vectors into the embedding sequence.

A rendered sequence holds the open marker, one placeholder token per payload vector and
the close marker. The model input is built in embedding space: token embeddings for
tokens, the interface-normed payload at placeholder positions. Positions are contiguous
across text and payload, masks are causal, and batch padding is masked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model.heads import PortHeads
from .model.lfm2_port import PortBackbone


@dataclass
class ReadInputs:
    ids: torch.Tensor          # [B, T] with placeholder IDs at payload positions
    payload_mask: torch.Tensor  # [B, T] bool, True at payload positions
    padding: torch.Tensor      # [B, T] 1 for real positions, 0 for right padding
    payloads: list[list[torch.Tensor]]  # per row, the blocks in order of appearance
    markers: bool = True


def render_segments(backbone: PortBackbone, segments: list[list[int] | torch.Tensor], placeholder_id: int,
                    markers=True) -> tuple[list[int], list[bool], list[torch.Tensor]]:
    """Render a row from segments: token-ID lists, or payload tensors [L, d] wrapped in markers."""
    ids, mask, blocks = [], [], []
    for segment in segments:
        if isinstance(segment, torch.Tensor):
            if markers:
                ids.append(backbone.controls.open_id)
                mask.append(False)
            ids.extend([placeholder_id] * segment.shape[0])
            mask.extend([True] * segment.shape[0])
            if markers:
                ids.append(backbone.controls.close_id)
                mask.append(False)
            blocks.append(segment)
        else:
            ids.extend(int(t) for t in segment)
            mask.extend([False] * len(segment))
    return ids, mask, blocks


def build_inputs(backbone: PortBackbone, rows: list[list[list[int] | torch.Tensor]], placeholder_id: int | None = None,
                 pad_id: int | None = None, device=None, *, heads=None) -> ReadInputs:
    # Placeholder and pad positions are overwritten or masked, so any real token serves; Qwen-family configs (Maple)
    # declare no pad token.
    config = backbone.config
    filler = next((t for t in (getattr(config, 'pad_token_id', None), getattr(config, 'eos_token_id', None), 0)
                   if isinstance(t, int)), 0)
    placeholder_id = filler if placeholder_id is None else placeholder_id
    pad_id = filler if pad_id is None else pad_id
    markers = heads.read_markers if heads is not None else True
    rendered = [render_segments(backbone, row, placeholder_id, markers=markers) for row in rows]
    width = max(len(ids) for ids, _, _ in rendered)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    payload_mask = torch.zeros(len(rows), width, dtype=torch.bool)
    padding = torch.zeros(len(rows), width, dtype=torch.long)
    for b, (row_ids, row_mask, _) in enumerate(rendered):
        ids[b, : len(row_ids)] = torch.tensor(row_ids)
        payload_mask[b, : len(row_mask)] = torch.tensor(row_mask)
        padding[b, : len(row_ids)] = 1
    device = backbone.embedding_weight.device if device is None else device   # the backbone's own device by default
    return ReadInputs(ids.to(device), payload_mask.to(device), padding.to(device), [blocks for _, _, blocks in rendered], markers)


def splice(backbone: PortBackbone, heads: PortHeads, inputs: ReadInputs) -> torch.Tensor:
    """Embeddings with interface-normed payload vectors at placeholder positions."""
    if inputs.markers != heads.read_markers:
        raise ValueError('reader profile mismatch; build inputs with heads=heads')
    embeds = backbone.embed(inputs.ids)
    out = embeds.clone()
    for b, blocks in enumerate(inputs.payloads):
        positions = inputs.payload_mask[b].nonzero().squeeze(-1)
        if not blocks:
            continue
        vectors = torch.cat(blocks, 0).to(embeds.dtype)
        if vectors.shape[0] != positions.shape[0]:
            raise ValueError("payload length does not match placeholder positions")
        out[b, positions] = heads.interface(vectors).to(embeds.dtype)
    return out


def read_forward(backbone: PortBackbone, heads: PortHeads, inputs: ReadInputs) -> dict:
    """Full forward of a consumer sequence with spliced payloads."""
    padding = inputs.padding if bool((inputs.padding == 0).any()) else None
    return backbone.forward_embeds(splice(backbone, heads, inputs), padding=padding)
