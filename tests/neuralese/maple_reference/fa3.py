# SDPA stand-in for deepgrove/maple-preview fa3.py (FlashAttention is not available here); same window rule.
import torch, torch.nn.functional as F

INCLUSIVE = True  # deepgrove: keys at distance <= w; llama.cpp: < w

def flash_attention_forward(module, q, k, v, attention_mask, dropout=0.0, position_ids=None, scaling=None, sliding_window=None, **kw):
    # q,k,v: [B,H,T,hd] -> returns [B,T,H,hd]
    T, S = q.shape[2], k.shape[2]
    qi = torch.arange(S - T, S)[:, None]; ki = torch.arange(S)[None]
    mask = ki <= qi
    if sliding_window is not None and S > sliding_window:
        mask = mask & ((qi - ki <= sliding_window) if INCLUSIVE else (qi - ki < sliding_window))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scaling, enable_gqa=True)
    return out.transpose(1, 2), None
