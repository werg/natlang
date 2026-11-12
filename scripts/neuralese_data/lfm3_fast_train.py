"""Fast LFM3 training forward/backward (the HF modelling arithmetic, fewer and fused kernels), packed documents, fused
linear cross-entropy and a no-grad log-prob pass.

    model = load an HF `Lfm3ForCausalLM` in bf16; prepare(model)
    h = hidden(model, ids)                         # (T, D) for one sequence
    h = hidden(model, ids, lengths=[n1, n2, ...])  # several documents packed into one row, each attending only itself
    nll = linear_nll(h[positions], model.lm_head.weight, targets)        # fp32 per-token NLL, logits never stored
    logp = token_logprobs(model, ids, positions, targets, lengths=...)   # no grad (GRPO old-policy / reference pass)

The HF decoder layer is replaced by two functions over explicit tensors, one per layer kind (conv, attention), each
`torch.compile`d once with a dynamic sequence length and shared by every layer of its kind (compiling the HF
`forward` per instance compiled one code object 24 times: past dynamo's recompile limit the later layers ran eager).
Inside them the norms, RoPE, gates, SwiGLU and the width-7 conv fuse into a few Triton kernels; the matmuls stay cuBLAS.
Attention is the varlen flash kernel (`torch.nn.attention.varlen`): causal for full-attention layers and a 255-left /
0-right window for sliding layers (HF's `sliding_window` 256 keys including the query), so it is exact, needs no mask,
and packed documents are separate sequences (`cu_seqlens`). The conv reads only taps inside the token's document.
Rounding points follow HF: RMSNorm in fp32 rounded before its bf16 weight, RoPE in fp32 with bf16 cos/sin, the conv
accumulated in fp32 from bf16 inputs and rounded once. Layers whose `gradient_checkpointing` flag is set (see
`exp0_sft.set_checkpointing`) are recomputed in backward.

The vendored HF sources are hash-pinned (`third_party/lfm3`), so everything here reads instances, never patches files.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

LOGIT_CHUNK = 4096


def _rms(x, weight, eps):
    """`Lfm3RMSNorm.forward`: normalize in fp32, round to the input dtype, multiply by the weight."""
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return weight * y.to(x.dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope(x, cos, sin):
    """`apply_rotary_pos_emb` for (T, H, D) with (T, D) bf16 cos/sin, in fp32."""
    xf = x.float()
    return (xf * cos[:, None].float() + _rotate_half(xf) * sin[:, None].float()).to(x.dtype)


def _attend(q, k, v, docs, window):
    """Causal attention within each document, keys at most ``window`` positions back (-1: unbounded). (T, H, D) -> same."""
    scale = q.shape[-1] ** -0.5
    cu, max_len, doc, pos = docs
    if q.is_cuda:
        from torch.nn.attention.varlen import varlen_attn
        return varlen_attn(q, k, v, cu, cu, max_len, max_len, scale=scale, window_size=(window, 0))
    # reference path (CPU tests): the same key sets as an explicit mask
    i = torch.arange(q.shape[0], device=q.device)
    mask = (doc[:, None] == doc[None, :]) & (i[None, :] <= i[:, None])
    if window >= 0:
        mask &= i[:, None] - i[None, :] <= window
    out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                         attn_mask=mask, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
    return out[0].transpose(0, 1)


def _ffn(x, ffn_norm, w1, w3, w2, post_ffn_norm, eps):
    f = _rms(x, ffn_norm, eps)
    f = F.linear(F.silu(F.linear(f, w1)) * F.linear(f, w3), w2)
    if post_ffn_norm is not None:
        f = _rms(f, post_ffn_norm, eps)
    return x + f


def conv_layer(x, pos, operator_norm, in_proj, conv_weight, out_proj, post_operator_norm,
               ffn_norm, w1, w3, w2, post_ffn_norm, eps: float):
    """`Lfm3DecoderLayer` with `Lfm3ShortConv` on (T, D): the depthwise width-7 causal conv as seven shifted fp32
    multiply-adds (as the inference kernel accumulates), each tap only inside the token's document (``pos``: position
    within the document)."""
    length = x.shape[0]
    taps = conv_weight.shape[-1]
    h = _rms(x, operator_norm, eps)
    b, c, xx = F.linear(h, in_proj).chunk(3, dim=-1)
    bx = b * xx
    weight = conv_weight.reshape(conv_weight.shape[0], taps).float()
    padded = F.pad(bx.float(), (0, 0, taps - 1, 0))
    acc = padded[taps - 1:taps - 1 + length] * weight[:, taps - 1]
    for back in range(1, taps):                         # the tap ``back`` positions before the token
        k = taps - 1 - back
        acc = acc + (padded[k:k + length] * weight[:, k]) * (pos >= back)[:, None]
    y = F.linear(c * acc.to(bx.dtype), out_proj)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps)


def attention_layer(x, cos, sin, cu, max_len: int, doc, pos, operator_norm, wq, wk, wv, wo, wg, q_norm, k_norm,
                    post_operator_norm, ffn_norm, w1, w3, w2, post_ffn_norm, eps: float, head_dim: int, window: int):
    """`Lfm3DecoderLayer` with `Lfm3Attention` on (T, D): QK-norm, fp32 RoPE, flash attention, sigmoid head gate."""
    length = x.shape[0]
    h = _rms(x, operator_norm, eps)
    q = _rms(F.linear(h, wq).view(length, -1, head_dim), q_norm, eps)
    k = _rms(F.linear(h, wk).view(length, -1, head_dim), k_norm, eps)
    v = F.linear(h, wv).view(length, -1, head_dim)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    o = _attend(q, k, v, (cu, max_len, doc, pos), window)
    o = (o * torch.sigmoid(F.linear(h, wg))[..., None]).reshape(length, -1)
    y = F.linear(o, wo)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps)


_COMPILED = {}


def _kernel(name, compiled):
    function = {'conv': conv_layer, 'attention': attention_layer}[name]
    if not compiled:
        return function
    if name not in _COMPILED:
        _COMPILED[name] = torch.compile(function, dynamic=True, fullgraph=True)
    return _COMPILED[name]


def _weight(module):
    return None if module is None else module.weight


def _layer_call(layer, compiled):
    """(function, static args before x-dependent ones) for one HF decoder layer."""
    post_operator = _weight(getattr(layer, 'post_operator_norm', None))
    post_ffn = _weight(getattr(layer, 'post_ffn_norm', None))
    eps = layer.operator_norm.variance_epsilon
    mlp = layer.feed_forward
    ffn = (layer.ffn_norm.weight, mlp.w1.weight, mlp.w3.weight, mlp.w2.weight, post_ffn, eps)
    if not layer.is_attention_layer:
        conv = layer.conv
        if conv.bias:
            raise ValueError('lfm3_fast_train: conv_bias is not supported')
        return 'conv', (layer.operator_norm.weight, conv.in_proj.weight, conv.conv.weight, conv.out_proj.weight,
                        post_operator) + ffn
    attn = layer.self_attn
    window = -1 if attn.sliding_window is None else attn.sliding_window - 1
    return 'attention', ((layer.operator_norm.weight, attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight,
                          attn.o_proj.weight, attn.gate_proj.weight, attn.q_norm.weight, attn.k_norm.weight,
                          post_operator, layer.ffn_norm.weight, mlp.w1.weight, mlp.w3.weight, mlp.w2.weight, post_ffn,
                          eps, attn.head_dim, window))


def prepare(model, compile_layers=True):
    """Mark a loaded HF `Lfm3ForCausalLM` for the fast path (`hidden`, `token_logprobs`, fused `linear_nll` in
    `exp0_sft.token_nll`); the HF modules and parameters are untouched. Returns the model."""
    for layer in model.model.layers:
        _layer_call(layer, False)                       # refuse unsupported configurations now
    model._lfm3_fast = {'compile': bool(compile_layers)}
    return model


def _documents(ids, lengths):
    """Per-token position within its document, document index, cu_seqlens (int32) and the longest document."""
    device = ids.device
    total = ids.shape[0]
    if lengths is None:
        lengths = [total]
    lengths = [int(n) for n in lengths]
    if sum(lengths) != total or min(lengths) < 1:
        raise ValueError(f'document lengths {lengths} do not tile {total} tokens')
    counts = torch.tensor(lengths, device=device)
    starts = torch.cumsum(counts, 0) - counts
    doc = torch.repeat_interleave(torch.arange(len(lengths), device=device), counts, output_size=total)
    pos = torch.arange(total, device=device) - starts[doc]
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(counts, 0)
    return pos, doc, cu, max(lengths)


def _rope_tables(model, pos, dtype):
    """`Lfm3RotaryEmbedding` per attention layer type: bf16 (cos, sin) of shape (T, head_dim)."""
    rotary = model.model.rotary_emb
    tables = {}
    for layer_type in rotary.layer_types:
        inv_freq = getattr(rotary, f'{layer_type}_inv_freq').float()
        scaling = getattr(rotary, f'{layer_type}_attention_scaling')
        freqs = pos[:, None].float() * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        tables[layer_type] = ((emb.cos() * scaling).to(dtype), (emb.sin() * scaling).to(dtype))
    return tables


def hidden(model, ids, lengths=None):
    """Final hidden states (T, D) of ``ids`` (1-D): one causal document, or ``lengths`` documents packed back to back
    (each attends and convolves only within itself, positions restart at 0)."""
    config = model.model.config
    fast = getattr(model, '_lfm3_fast', None)
    compiled = fast is not None and fast['compile']
    pos, doc, cu, max_len = _documents(ids, lengths)
    x = model.model.embed_tokens(ids)
    tables = _rope_tables(model, pos, x.dtype)
    grad = torch.is_grad_enabled()
    for index, layer in enumerate(model.model.layers):
        kind, static = _layer_call(layer, compiled)
        function = _kernel(kind, compiled)
        if kind == 'conv':
            args = (x, pos) + static
        else:
            cos, sin = tables[config.layer_types[index]]
            args = (x, cos, sin, cu, max_len, doc, pos) + static
        if grad and getattr(layer, 'gradient_checkpointing', False):
            x = torch_checkpoint(function, *args, use_reentrant=False)
        else:
            x = function(*args)
    norm = model.model.embedding_norm
    return _rms(x, norm.weight, norm.variance_epsilon)


# --- loss ------------------------------------------------------------------------------------------------------------

def _ce_forward(logits, target):
    lf = logits.float()
    lse = torch.logsumexp(lf, dim=-1)
    return lse, lse - lf.gather(-1, target[:, None])[:, 0]


def _ce_backward(logits, lse, target, grad):
    """d nll / d logits (bf16, as the stock path's `.float()` backward rounds it): grad * (softmax - onehot)."""
    lf = logits.float()
    p = torch.exp(lf - lse[:, None])
    onehot = torch.arange(lf.shape[-1], device=lf.device)[None, :] == target[:, None]
    return ((p - onehot.float()) * grad[:, None]).to(logits.dtype)


_CE = {}


def _ce(name):
    if name not in _CE:
        _CE[name] = torch.compile({'forward': _ce_forward, 'backward': _ce_backward}[name], dynamic=True)
    return _CE[name]


class _LinearNLL(torch.autograd.Function):
    """Per-row NLL of ``target`` under logits ``h @ weight.T`` (bf16 logits, fp32 softmax), in row chunks: forward keeps
    only the fp32 log-sum-exp per row, backward recomputes each chunk's logits once (the stock checkpointed loss ran the
    head four times and stored fp32 logits); the weight gradient accumulates in fp32."""

    @staticmethod
    def forward(ctx, h, weight, target, chunk):
        rows = h.shape[0]
        nll = torch.empty(rows, dtype=torch.float32, device=h.device)
        lse = torch.empty(rows, dtype=torch.float32, device=h.device)
        forward_ce = _ce('forward') if h.is_cuda else _ce_forward
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            lse[start:stop], nll[start:stop] = forward_ce(F.linear(h[start:stop], weight), target[start:stop])
        ctx.save_for_backward(h, weight, target, lse)
        ctx.chunk = chunk
        return nll

    @staticmethod
    def backward(ctx, grad):
        h, weight, target, lse = ctx.saved_tensors
        grad = grad.float().contiguous()
        backward_ce = _ce('backward') if h.is_cuda else _ce_backward
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        dh = torch.empty_like(h) if need_h else None
        dw = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_w else None
        for start in range(0, h.shape[0], ctx.chunk):
            stop = min(h.shape[0], start + ctx.chunk)
            rows = h[start:stop]
            d = backward_ce(F.linear(rows, weight), lse[start:stop], target[start:stop], grad[start:stop])
            if need_h:
                dh[start:stop] = d @ weight
            if need_w:
                if d.is_cuda:
                    dw += torch.mm(d.t(), rows, out_dtype=torch.float32)
                else:
                    dw += (d.t() @ rows).float()
        return dh, None if dw is None else dw.to(weight.dtype), None, None


def linear_nll(h, weight, target, chunk=LOGIT_CHUNK):
    """fp32 NLL per row of ``target`` (int64) from hidden rows ``h`` (N, D) and the tied head ``weight`` (V, D)."""
    return _LinearNLL.apply(h, weight, target, chunk)


@torch.no_grad()
def token_logprobs(model, ids, positions, targets, lengths=None, chunk=LOGIT_CHUNK):
    """Log-probabilities (fp32) of ``targets`` predicted at rows ``positions`` of the packed ``ids`` (no grad: the
    GRPO behaviour / reference recompute). Pack many sequences per call (``lengths``) for throughput."""
    h = hidden(model, ids, lengths)
    return -linear_nll(h[positions], model.lm_head.weight, targets, chunk)
