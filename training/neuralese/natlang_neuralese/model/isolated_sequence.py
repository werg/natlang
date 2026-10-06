"""One changed current input per causal position, sharing the unchanged history."""
import torch
import torch.nn.functional as F
from .lfm2_port import AttentionState, ConvState, PortCache, feed_forward_residual


def isolated_sequence(backbone, fixed, replacements, cache, *, cutoff):
    if fixed.shape != replacements.shape or not 0 < cutoff <= backbone.num_layers:
        raise ValueError('isolated sequence requires matching inputs and a valid cutoff')
    batch, steps, _ = fixed.shape
    history, branch = fixed, replacements
    states, lengths = list(cache.states), list(cache.lengths)
    pad = cache.pad
    start = lengths[0]
    if any(length != start for length in lengths):
        raise ValueError('isolated sequence requires aligned layer caches')
    positions = torch.arange(start, start + steps, device=fixed.device)[None].expand(batch, -1)
    real = None if pad is None else positions >= pad[:, None]
    if pad is not None:
        positions = (positions - pad[:, None]).clamp(min=0)
    cos, sin = backbone._rope(fixed, positions)
    from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb, repeat_kv

    for index, layer in enumerate(backbone.layers):
        attention = backbone.is_attention(index)
        previous = states[index]
        fields = (() if previous is None else previous.fields() if attention else (previous.window,))

        def run_one(h, b, *previous_fields, layer=layer, attention=attention):
            xh, xb = layer.operator_norm(h), layer.operator_norm(b)
            if attention:
                attn = layer.self_attn
                def qkv(x):
                    q = attn.q_layernorm(attn.q_proj(x).view(batch, steps, -1, attn.head_dim)).transpose(1, 2)
                    k = attn.k_layernorm(attn.k_proj(x).view(batch, steps, -1, attn.head_dim)).transpose(1, 2)
                    v = attn.v_proj(x).view(batch, steps, -1, attn.head_dim).transpose(1, 2)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    return q, k, v
                qh, kh, vh = qkv(xh)
                qb, kb, vb = qkv(xb)
                if previous_fields:
                    prev = AttentionState.from_fields(previous_fields)
                    kh, vh = torch.cat((prev.k, kh), 2), torch.cat((prev.v, vh), 2)
                total = kh.shape[2]
                keys = torch.arange(total, device=h.device)
                queries = torch.arange(total-steps, total, device=h.device)
                mask = (keys[None] <= queries[:, None])[None, None].expand(batch, 1, -1, -1)
                branch_mask = (keys[None] < queries[:, None])[None, None].expand(batch, 1, -1, -1)
                if pad is not None:
                    valid = keys[None] >= pad[:, None]
                    mask = mask & valid[:, None, None]
                    branch_mask = branch_mask & valid[:, None, None]
                    # Keep padded queries finite, just as the ordinary port does.
                    mask = mask | (keys[None] == queries[:, None])[None, None]
                eye = torch.eye(steps, dtype=torch.bool, device=h.device)[None, None].expand(batch, 1, -1, -1)
                branch_mask = torch.cat((branch_mask, eye), -1)
                def attend(q, k, v, allowed):
                    out = F.scaled_dot_product_attention(q, repeat_kv(k, attn.num_key_value_groups),
                        repeat_kv(v, attn.num_key_value_groups), attn_mask=allowed, scale=attn.scaling)
                    return attn.out_proj(out.transpose(1, 2).reshape(batch, steps, -1))
                oh = attend(qh, kh, vh, mask)
                ob = attend(qb, torch.cat((kh, kb), 2), torch.cat((vh, vb), 2), branch_mask)
                updated = (kh, vh)
            else:
                conv = layer.conv
                if real is not None:
                    xh, xb = xh * real[:, :, None], xb * real[:, :, None]
                bh, ch, xxh = conv.in_proj(xh).transpose(-1, -2).chunk(3, dim=-2)
                bb, cb, xxb = conv.in_proj(xb).transpose(-1, -2).chunk(3, dim=-2)
                gh, gb = bh * xxh, bb * xxb
                prev_window = previous_fields[0] if previous_fields else gh.new_zeros(batch, gh.shape[1], backbone.conv_window)
                window = torch.cat((prev_window, gh), -1)
                conv_h = F.conv1d(window, conv.conv.weight, conv.conv.bias, groups=gh.shape[1])
                conv_b = conv_h + (gb - gh) * conv.conv.weight[:, 0, -1][None, :, None]
                oh = conv.out_proj((ch * conv_h).transpose(-1, -2).contiguous())
                ob = conv.out_proj((cb * conv_b).transpose(-1, -2).contiguous())
                updated = (window[..., -backbone.conv_window:],)
            h = feed_forward_residual(layer, h + oh, getattr(backbone, 'ffn_chunk_tokens', 0))
            b = feed_forward_residual(layer, b + ob, getattr(backbone, 'ffn_chunk_tokens', 0))
            return h, b, *updated

        if getattr(backbone, 'checkpoint_layers', False) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            result = checkpoint(run_one, history, branch, *fields, use_reentrant=False,
                                preserve_rng_state=backbone.checkpoint_preserve_rng)
        else:
            result = run_one(history, branch, *fields)
        history, branch, *updated = result
        states[index] = AttentionState.from_fields(updated) if attention else ConvState(updated[0])
        lengths[index] = start + steps
        if index + 1 == cutoff:
            history_shallow, shallow = history, branch
    return dict(history_shallow=history_shallow, shallow=shallow, final=branch,
                cache=PortCache(tuple(states), tuple(lengths), pad, cache.pad_offsets))
