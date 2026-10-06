"""One decoding step for many sequences at once (S4 §4.2, continuous batching).

Each sequence keeps its own `PortCache`, so sequences of different lengths, and sequences in the middle of a block
(whose shallow layers have advanced past their upper layers), can share one step. The step stacks the rows: norms,
projections, short convolutions and feed-forward blocks run as one batched tensor; attention runs per row against
that row's own key/value storage (a single query attends to its own prefix, so no padding or masks are needed and
the preallocated per-row storage keeps growing in place). Rotary positions are per row.

`step_rows` agrees with running each row through `PortBackbone.run_layers` on its own; tests check it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model.lfm2_port import AttentionState, ConvState, PortBackbone, PortCache, append_kv


def step_rows(backbone: PortBackbone, h: torch.Tensor, caches: list[PortCache], layers: range
              ) -> tuple[torch.Tensor, list[PortCache]]:
    """Advance N rows by one position through `layers`. `h` is [N, 1, d]; returns [N, 1, d] and the new caches."""
    from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb

    rows = h.shape[0]
    if rows != len(caches):
        raise ValueError("one cache per row")
    states = [list(c.states) for c in caches]
    lengths = [list(c.lengths) for c in caches]
    if not len(layers):
        return h, caches
    if not hasattr(backbone.layers[layers.start], 'operator_norm'):
        # Not the native LFM2 stack (the Qwen/Maple port): each row through the backbone's own runner, which this
        # batched step is checked against. Exact; a batched Qwen step would only be faster.
        outs, new = [], []
        for row in range(rows):
            out, cache = backbone.run_layers(h[row:row + 1], layers, caches[row])
            outs.append(out)
            new.append(cache)
        return torch.cat(outs, 0), new
    starts = [ls[layers.start] for ls in lengths]
    for row, ls in enumerate(lengths):
        if any(ls[i] != starts[row] for i in layers):
            raise ValueError("layers in one range must have processed the same number of positions")
    positions = torch.tensor(starts, device=h.device).unsqueeze(1)
    cos, sin = backbone._rope(h, positions)
    static = not torch.is_grad_enabled()
    for i in layers:
        layer = backbone.layers[i]
        x = layer.operator_norm(h)
        if backbone.is_attention(i):
            attn = layer.self_attn
            hd = attn.head_dim
            q = attn.q_layernorm(attn.q_proj(x).view(rows, 1, -1, hd)).transpose(1, 2)
            k = attn.k_layernorm(attn.k_proj(x).view(rows, 1, -1, hd)).transpose(1, 2)
            v = attn.v_proj(x).view(rows, 1, -1, hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            outs = []
            for row in range(rows):
                state = append_kv(states[row][i], k[row:row + 1], v[row:row + 1], static=static)
                states[row][i] = state
                outs.append(F.scaled_dot_product_attention(q[row:row + 1], state.k, state.v, scale=attn.scaling,
                                                           enable_gqa=True))
            out = torch.cat(outs, 0)
            h = h + attn.out_proj(out.transpose(1, 2).reshape(rows, 1, -1))
        else:
            conv = layer.conv
            bcx = conv.in_proj(x).transpose(-1, -2)
            b, c, xx = bcx.chunk(3, dim=-2)
            gated = b * xx
            windows = []
            for row in range(rows):
                prev = states[row][i]
                windows.append(prev.window if prev is not None
                               else gated.new_zeros(1, gated.shape[1], backbone.conv_window))
            window = torch.cat([torch.cat(windows, 0), gated], dim=-1)
            conv_out = F.conv1d(window, conv.conv.weight, conv.conv.bias, groups=gated.shape[1])
            h = h + conv.out_proj((c * conv_out).transpose(-1, -2).contiguous())
            tail = window[..., -backbone.conv_window:]
            for row in range(rows):
                states[row][i] = ConvState(tail[row:row + 1])
        h = h + layer.feed_forward(layer.ffn_norm(h))
        for row in range(rows):
            lengths[row][i] = starts[row] + 1
    return h, [PortCache(tuple(s), tuple(ls)) for s, ls in zip(states, lengths)]


def split_rows(cache: PortCache, pad: torch.Tensor) -> list[PortCache]:
    """Per-row caches from one left-padded batch prefill (`forward_embeds(..., left_pad=pad)`).

    Row b keeps its keys and values after its `pad[b]` padding positions (rotary positions were already shifted so its
    first real token is at 0) and the conv window at the end, which is all real input or the zeros a fresh row starts
    from. Rows are copied, so the batch's storage is released once the caller drops it; each row then grows on its own.
    """
    rows = []
    for b, p in enumerate(int(x) for x in pad.tolist()):
        states = []
        for state in cache.states:
            if isinstance(state, AttentionState):
                states.append(AttentionState(state.k[b:b + 1, :, p:].clone(), state.v[b:b + 1, :, p:].clone()))
            elif isinstance(state, ConvState):
                states.append(ConvState(state.window[b:b + 1].clone()))
            else:
                states.append(state)
        rows.append(PortCache(tuple(states), tuple(length - p for length in cache.lengths)))
    return rows


__all__ = ["step_rows", "split_rows", "AttentionState"]
