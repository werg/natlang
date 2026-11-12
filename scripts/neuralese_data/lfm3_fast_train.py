"""Faster LFM3 training forward/backward on one long sequence (the HF modelling code, same arithmetic, fewer kernels).

Measured on the GB10 (bf16, 11k tokens, `exp0_sft.py bench`): the stock path is memory-bandwidth bound, not matmul bound.
Matmuls run near peak; about half of the GPU time is unfused elementwise work (fp32 RMSNorm, RoPE, gates, SwiGLU, the
width-7 conv as an fp32 depthwise `conv1d`) and about a fifth is attention, which for the sliding-window layers
(window 256) is a dense 11k x 11k masked kernel.

* `banded_sdpa`: sliding-window layers attend with exact 256-token windows (queries in blocks of 256 against the
  previous and current block); full-attention layers use the causal flash kernel. The same key sets and softmax.
* decoder layers are `torch.compile`d (dynamic sequence length), so the norm / RoPE / gate / activation chains fuse; the
  width-7 conv is the same fp32-accumulated 7-tap sum written as shifted adds, which fuses with its gates.

The vendored HF sources are hash-pinned (`third_party/lfm3`), so everything here patches instances, never the files.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

WINDOW = 256


def banded_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, **kwargs):
    """HF attention-interface function: query (1, H, T, D), key/value (1, Hkv, T, D) -> (1, T, H, D). The mask argument is
    ignored: the sequence is one causal document."""
    batch, heads, length, dim = query.shape
    groups = heads // key.shape[1]
    if sliding_window is None or length <= sliding_window:
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scaling,
                                             enable_gqa=groups > 1)
        return out.transpose(1, 2).contiguous(), None
    window = sliding_window
    blocks = -(-length // window)
    pad = blocks * window - length
    if pad:
        query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    kv_heads = key.shape[1]
    q = query.view(heads, blocks, window, dim).transpose(0, 1)                          # (nb, H, W, D)

    def windows(x):
        x = x.view(kv_heads, blocks, window, dim)
        previous = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return torch.cat([previous, x], dim=2).transpose(0, 1)                           # (nb, Hkv, 2W, D)

    k, v = windows(key), windows(value)
    if groups > 1:
        k, v = (x.repeat_interleave(groups, dim=1) for x in (k, v))
    a = torch.arange(window, device=query.device)[:, None]
    b = torch.arange(2 * window, device=query.device)[None, :]
    band = (b > a) & (b <= a + window)                                                    # key within the last `window`
    mask = band.expand(blocks, 1, window, 2 * window).clone()
    mask[0, :, :, :window] = False                                                        # no previous block before block 0
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scaling)         # (nb, H, W, D)
    out = out.transpose(1, 2).reshape(blocks * window, heads, dim)[:length]
    return out[None], None


def _short_conv(self, hidden_states, past_key_values=None, attention_mask=None, seq_idx=None):
    """`Lfm3ShortConv.forward` for one unpadded, unpacked sequence without a cache: the depthwise width-7 causal conv as
    seven shifted multiply-adds accumulated in fp32 and rounded once (as the inference kernel does)."""
    length = hidden_states.shape[1]
    b, c, x = self.in_proj(hidden_states).chunk(3, dim=-1)                                # (1, T, D) each
    h = b * x
    weight = self.conv.weight.squeeze(1).float()                                          # (D, 7)
    padded = F.pad(h.float(), (0, 0, 6, 0))                                               # (1, T + 6, D)
    acc = padded[:, 0:length] * weight[:, 0]
    for k in range(1, 7):
        acc = acc + padded[:, k:k + length] * weight[:, k]
    y = c * acc.to(h.dtype)
    return self.out_proj(y)


def prepare(model, compile_layers=True):
    """Patch a loaded HF `Lfm3ForCausalLM` instance for training one sequence at a time; returns the `forward` helper."""
    from transformers import AttentionInterface
    AttentionInterface.register('banded_sdpa', banded_sdpa)
    model.config._attn_implementation = 'banded_sdpa'
    model.model.config._attn_implementation = 'banded_sdpa'
    for layer in model.model.layers:
        if not layer.is_attention_layer:
            layer.conv.forward = _short_conv.__get__(layer.conv)
    if compile_layers:
        for layer in model.model.layers:
            layer.forward = torch.compile(layer.forward, dynamic=True)
    return model


def hidden(model, ids):
    none = {'full_attention': None, 'sliding_attention': None, 'conv': None}
    return model.model(input_ids=ids[None], attention_mask=none, use_cache=False).last_hidden_state[0]
