"""One changed current input per causal position, sharing the unchanged history."""
import torch
import torch.nn.functional as F
from .lfm2_port import (AttentionState, ConvState, PortCache, attention_query, feed_forward_residual,
                        gate_attention_output)


_flex = None


def branch_block_mask(steps, length, start, window_size, pad, device):
    """Build exact sparse block geometry without any token-by-token T² tensor."""
    from torch.nn.attention.flex_attention import BlockMask
    block = 128
    qlo = torch.arange(0, steps, block, device=device)[None, None, :, None]
    qhi = (qlo + block - 1).clamp(max=steps - 1)
    klo = torch.arange(0, length + steps, block, device=device)[None, None, None, :]
    khi = (klo + block - 1).clamp(max=length + steps - 1)
    lower = klo if pad is None else torch.maximum(klo, pad[:, None, None, None])
    upper = khi.clamp(max=length - 1)
    history = (lower <= upper) & (lower < start + qhi)
    if window_size is not None:
        history = history & (upper > start + qlo - window_size) & (window_size > 1)
    own = (klo <= length + qhi) & (khi >= length + qlo)
    full = (khi < length) & (khi < start + qlo)
    if window_size is not None:
        full = full & (klo > start + qhi - window_size)
    if pad is not None:
        full = full & (klo >= pad[:, None, None, None])
    partial = (history | own) & ~full

    def ordered(mask):
        # Stable partition, ascending block indices within each partition.
        indices = torch.argsort(mask.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
        return mask.sum(dim=-1).to(torch.int32), indices

    def mask_mod(b, h, q, kv):
        history = (kv < length) & (kv < start + q)
        if window_size is not None:
            history = history & (kv > start + q - window_size)
        if pad is not None:
            history = history & (kv >= pad[b])
        return history | (kv == length + q)

    counts, indices = ordered(partial)
    full_counts, full_indices = ordered(full)
    return BlockMask.from_kv_blocks(counts, indices, full_counts, full_indices,
        BLOCK_SIZE=block, mask_mod=mask_mod, seq_lengths=(steps, length + steps))


def branch_attention(qb, kh, vh, kb, vb, start, window_size, pad, scale, *, flex=None):
    """Each own query (absolute position start + i) attends to the history keys before it (inside the window, after
    left padding) and to its own key. CUDA uses one block-sparse FlexAttention call (7-12x the tiled masked SDPA,
    which falls back to the math kernel); elsewhere, or with NATLANG_FLEX_BRANCH=0, 512-query tiles."""
    global _flex
    batch, _, steps, _ = qb.shape
    length = kh.shape[2]
    if flex is None:
        import os
        # Spans under 128 queries would dispatch Flex's decode kernel, whose default tile does not divide the
        # 128-token mask blocks (and whose compilation leaks into later fp32 graphs); tiles are cheap there.
        flex = qb.is_cuda and steps >= 128 and os.environ.get('NATLANG_FLEX_BRANCH', '1') != '0'
    if flex:
        from torch.nn.attention.flex_attention import flex_attention
        if _flex is None:
            _flex = torch.compile(flex_attention, dynamic=True)

        from ..common.attention_backend import flex_kernel_options
        block_mask = branch_block_mask(steps, length, start, window_size, pad, qb.device)
        return _flex(qb, torch.cat((kh, kb), 2), torch.cat((vh, vb), 2), block_mask=block_mask, scale=scale,
                     enable_gqa=True,
                     # NGC defaults fp32 matmul to single TF32: on Ada this
                     # loses ~1e-3 against the fp32 SDPA reference. Three TF32
                     # products preserve fp32 accuracy while using tensor cores.
                     kernel_options=flex_kernel_options(float32=qb.dtype == torch.float32, query_tokens=qb.shape[2]))
    outputs = []
    for begin in range(0, steps, 512):
        end = min(begin + 512, steps)
        total = start + end
        keys = torch.arange(total, device=qb.device)
        queries = torch.arange(start + begin, start + end, device=qb.device)
        mask = (keys[None] < queries[:, None])[None, None].expand(batch, 1, -1, -1)
        if window_size is not None:
            mask = mask & (keys[None] > queries[:, None] - window_size)[None, None]
        if pad is not None:
            mask = mask & (keys[None] >= pad[:, None])[:, None, None]
        own = torch.eye(end-begin, dtype=torch.bool, device=qb.device)[None, None].expand(batch, 1, -1, -1)
        mask = torch.cat((mask, own), -1)
        outputs.append(F.scaled_dot_product_attention(qb[:, :, begin:end],
            torch.cat((kh[:, :, :total], kb[:, :, begin:end]), 2),
            torch.cat((vh[:, :, :total], vb[:, :, begin:end]), 2),
            attn_mask=mask, scale=scale, enable_gqa=True))
    return torch.cat(outputs, 2)


def isolated_sequence(backbone, fixed, replacements, cache, *, cutoff):
    if fixed.shape != replacements.shape or not 0 < cutoff <= backbone.num_layers:
        raise ValueError('isolated sequence requires matching inputs and a valid cutoff')
    # Layer forms come from the backbone's declared methods (norm, gate, sandwich residual, rope, window, FFN).
    token_local_ffn = hasattr(backbone, "_feed_forward")
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

    def rotate(index, q, k):
        if hasattr(backbone, "_apply_rope"):
            return backbone._apply_rope(index, q, k, cos, sin)
        from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb
        return apply_rotary_pos_emb(q, k, cos, sin)

    from .layer_staging import tape_for_grad
    tape = tape_for_grad()  # an in-backward update steps each layer once, so this pass joins the layer staging
    for index, layer in enumerate(backbone.layers):
        attention = backbone.is_attention(index)
        if tape is not None:
            history, branch = tape.enter(index, history), tape.enter(index, branch)
        previous = states[index]
        fields = (() if previous is None else previous.fields() if attention else (previous.window,))

        def run_one(h, b, *previous_fields, layer=layer, attention=attention, index=index):
            xh, xb = backbone._operator_norm(layer, h), backbone._operator_norm(layer, b)
            if attention:
                attn = layer.self_attn
                qnorm = getattr(attn, 'q_norm', None) or attn.q_layernorm
                knorm = getattr(attn, 'k_norm', None) or attn.k_layernorm
                projection = getattr(attn, 'o_proj', None) or attn.out_proj
                window_size = backbone.window(index) if hasattr(backbone, 'window') else None
                def qkv(x):
                    q, gate = attention_query(backbone, attn, x)
                    q = qnorm(q).transpose(1, 2)
                    k = knorm(attn.k_proj(x).view(batch, steps, -1, attn.head_dim)).transpose(1, 2)
                    v = attn.v_proj(x).view(batch, steps, -1, attn.head_dim).transpose(1, 2)
                    q, k = rotate(index, q, k)
                    return q, k, v, gate

                def output(out, gate):
                    out = out.transpose(1, 2).reshape(batch, steps, -1)
                    return projection(gate_attention_output(backbone, out, gate, attn.head_dim))
                qh, kh, vh, gh = qkv(xh)
                qb, kb, vb, gb = qkv(xb)
                if previous_fields:
                    prev = AttentionState.from_fields(previous_fields)
                    kh, vh = torch.cat((prev.k, kh), 2), torch.cat((prev.v, vh), 2)
                # Ordinary history uses the port's fused causal attention. Own
                # queries are tiled so longer contexts do not allocate a T² mask.
                if window_size is None:
                    oh = backbone._attend_fast(attn, qh, kh, vh, steps,
                        not previous_fields, None, pad, cache.pad_offsets)
                else:
                    from .hf_port import _attend_window, _window_mask
                    if pad is None:
                        oh = _attend_window(attn, qh, kh, vh, steps, window_size)
                    else:
                        mask = _window_mask(steps, kh.shape[2], window_size, h.device)[None, None]
                        valid = torch.arange(kh.shape[2], device=h.device)[None] >= pad[:, None]
                        mask = mask & valid[:, None, None]
                        oh = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask,
                            scale=attn.scaling, enable_gqa=True)
                oh = output(oh, gh)
                ob = output(branch_attention(qb, kh, vh, kb, vb, start, window_size, pad, attn.scaling), gb)
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
            h, b = backbone._operator_residual(layer, h, oh), backbone._operator_residual(layer, b, ob)
            if token_local_ffn:
                # One token-local feed-forward over both streams: an MoE streams every active expert's weights
                # once per call, so the history and branch share that sweep.
                h, b = backbone._feed_forward(layer, torch.cat((h, b), 1)).split(steps, 1)
            else:
                h = feed_forward_residual(layer, h, getattr(backbone, 'ffn_chunk_tokens', 0))
                b = feed_forward_residual(layer, b, getattr(backbone, 'ffn_chunk_tokens', 0))
            return h, b, *updated

        if getattr(backbone, 'checkpoint_layers', False) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            result = checkpoint(run_one, history, branch, *fields, use_reentrant=False,
                                preserve_rng_state=backbone.checkpoint_preserve_rng)
        else:
            result = run_one(history, branch, *fields)
        history, branch, *updated = result
        if tape is not None:
            history, branch = tape.exit(index, history), tape.exit(index, branch)
        states[index] = AttentionState.from_fields(updated) if attention else ConvState(updated[0])
        lengths[index] = start + steps
        if index + 1 == cutoff:
            history_shallow, shallow = history, branch
    return dict(history_shallow=history_shallow, shallow=shallow, final=branch,
                cache=PortCache(tuple(states), tuple(lengths), pad, cache.pad_offsets))
