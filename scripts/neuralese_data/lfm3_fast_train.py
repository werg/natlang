"""Fast LFM3 training forward/backward (the HF modelling arithmetic, fewer and fused kernels), packed documents, fused
linear cross-entropy and a no-grad log-prob pass.

    model = load an HF `Lfm3ForCausalLM` in bf16; prepare(model)
    h = hidden(model, ids)                         # (T, D) for one sequence
    h = hidden(model, ids, lengths=[n1, n2, ...])  # several documents packed into one row, each attending only itself
    nll = linear_nll(h[positions], model.lm_head.weight, targets)        # fp32 per-token NLL, logits never stored
    logp = token_logprobs(model, ids, positions, targets, lengths=...)   # no grad (GRPO old-policy / reference pass)
    nll, nucleus_nll, outside = linear_nll_nucleus(h[positions], weight, targets, top_p)   # + NLL under the top-p nucleus
    h = hidden(model, ids, imports=import_tensors(model, [(start, block)], device))   # imported native state spliced in
    run_prefix_tree(model, prefix_tree(seqs, caps, 256), ids_of, leaf)   # shared prompts computed once, exact grads

The HF decoder layer is replaced by two functions over explicit tensors, one per layer kind (conv, attention), each
`torch.compile`d (fullgraph, dynamic sequence length) and shared by every layer of its kind, with no HF mask/cache
plumbing in the traced region. Inside them the norms, RoPE, gates, SwiGLU and the width-7 conv fuse into a few Triton
kernels; the matmuls stay cuBLAS.
Attention is the varlen flash kernel (`torch.nn.attention.varlen`): causal for full-attention layers and a 255-left /
0-right window for sliding layers (HF's `sliding_window` 256 keys including the query), so it is exact, needs no mask,
and packed documents are separate sequences (`cu_seqlens`). The conv reads only taps inside the token's document.
Rounding points follow HF: RMSNorm in fp32 rounded before its bf16 weight, RoPE in fp32 with bf16 cos/sin, the conv
accumulated in fp32 from bf16 inputs and rounded once. Layers whose `gradient_checkpointing` flag is set (see
`exp0_sft.set_checkpointing`) are recomputed in backward.

Imported native Neuralese state (a consumer call's prompt as the SGLang actor fed it, `import_tensors`): at the imported
rows every attention layer's keys (after the K norm, before RoPE) and values and every conv layer's input B*x are the
stored block's, exactly the fork's `NatlangImport`; the rows' own outputs reach nothing. The state is a constant (no
gradient into the producer); `exp0_sft.hidden` does the same on the stock HF modules through forward hooks.

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
               ffn_norm, w1, w3, w2, post_ffn_norm, eps: float, import_rows=None, import_bx=None):
    """`Lfm3DecoderLayer` with `Lfm3ShortConv` on (T, D): the depthwise width-7 causal conv as seven shifted fp32
    multiply-adds (as the inference kernel accumulates), each tap only inside the token's document (``pos``: position
    within the document; ``import_rows``: rows whose B*x is the imported ``import_bx``)."""
    length = x.shape[0]
    taps = conv_weight.shape[-1]
    h = _rms(x, operator_norm, eps)
    b, c, xx = F.linear(h, in_proj).chunk(3, dim=-1)
    bx = b * xx
    if import_rows is not None:
        bx = bx.index_put((import_rows,), import_bx)
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
                    post_operator_norm, ffn_norm, w1, w3, w2, post_ffn_norm, eps: float, head_dim: int, window: int,
                    import_rows=None, import_k=None, import_v=None):
    """`Lfm3DecoderLayer` with `Lfm3Attention` on (T, D): QK-norm, fp32 RoPE, flash attention, sigmoid head gate
    (``import_rows``: rows whose normed pre-RoPE keys and values are the imported ``import_k``/``import_v``)."""
    length = x.shape[0]
    h = _rms(x, operator_norm, eps)
    q = _rms(F.linear(h, wq).view(length, -1, head_dim), q_norm, eps)
    k = _rms(F.linear(h, wk).view(length, -1, head_dim), k_norm, eps)
    v = F.linear(h, wv).view(length, -1, head_dim)
    if import_rows is not None:
        k = k.index_put((import_rows,), import_k)
        v = v.index_put((import_rows,), import_v)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    o = _attend(q, k, v, (cu, max_len, doc, pos), window)
    o = (o * torch.sigmoid(F.linear(h, wg))[..., None]).reshape(length, -1)
    y = F.linear(o, wo)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps)


def _attend_past(q, k, v, cu_q, cu_k, window):
    """Attention of ``q`` (the last S of the K key positions) over ``k``/``v`` (K, H, D): causal, keys at most ``window``
    positions back (-1: unbounded), the queries aligned to the end of the keys (flash's causal rule for S < K).
    ``cu_q``/``cu_k``: int32 [0, S] / [0, K], made outside the compiled layer like ``hidden``'s ``cu``."""
    scale = q.shape[-1] ** -0.5
    s, n = q.shape[0], k.shape[0]
    if q.is_cuda:
        from torch.nn.attention.varlen import varlen_attn
        return varlen_attn(q, k, v, cu_q, cu_k, s, n, scale=scale, window_size=(window, 0))
    i = torch.arange(s, device=q.device)[:, None] + (n - s)
    j = torch.arange(n, device=q.device)[None, :]
    mask = j <= i
    if window >= 0:
        mask &= i - j <= window
    out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                         attn_mask=mask, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
    return out[0].transpose(0, 1)


def conv_layer_past(x, past_bx, operator_norm, in_proj, conv_weight, out_proj, post_operator_norm,
                    ffn_norm, w1, w3, w2, post_ffn_norm, eps: float):
    """``conv_layer`` for one document that continues a computed prefix: ``past_bx`` (taps - 1, D) is the B*x of the
    prefix's last taps - 1 tokens (zero rows before the document start). Returns (output, the last taps - 1 B*x)."""
    length = x.shape[0]
    taps = conv_weight.shape[-1]
    h = _rms(x, operator_norm, eps)
    b, c, xx = F.linear(h, in_proj).chunk(3, dim=-1)
    bx = b * xx
    weight = conv_weight.reshape(conv_weight.shape[0], taps).float()
    joined = torch.cat([past_bx, bx])
    padded = joined.float()
    acc = padded[taps - 1:taps - 1 + length] * weight[:, taps - 1]
    for back in range(1, taps):
        k = taps - 1 - back
        acc = acc + padded[k:k + length] * weight[:, k]
    y = F.linear(c * acc.to(bx.dtype), out_proj)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps), joined[length:]


def attention_layer_past(x, cos, sin, cu_q, cu_k, past_k, past_v, operator_norm, wq, wk, wv, wo, wg, q_norm, k_norm,
                         post_operator_norm, ffn_norm, w1, w3, w2, post_ffn_norm, eps: float, head_dim: int,
                         window: int):
    """``attention_layer`` for one document that continues a computed prefix whose keys (after RoPE) and values are
    ``past_k``/``past_v`` (a sliding layer's: at least its window's last keys). Returns (output, keys, values) for a
    continuation of this one (a sliding layer's: the last ``window`` only)."""
    length = x.shape[0]
    h = _rms(x, operator_norm, eps)
    q = _rms(F.linear(h, wq).view(length, -1, head_dim), q_norm, eps)
    k = _rms(F.linear(h, wk).view(length, -1, head_dim), k_norm, eps)
    v = F.linear(h, wv).view(length, -1, head_dim)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    keys, values = torch.cat([past_k, k]), torch.cat([past_v, v])
    o = _attend_past(q, keys, values, cu_q, cu_k, window)
    o = (o * torch.sigmoid(F.linear(h, wg))[..., None]).reshape(length, -1)
    y = F.linear(o, wo)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    if window >= 0:
        keys, values = keys[-window:], values[-window:]
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps), keys, values


_COMPILED = {}
RECOMPILE_LIMIT = 64


def _kernel(name, compiled):
    function = {'conv': conv_layer, 'attention': attention_layer,
                'conv_past': conv_layer_past, 'attention_past': attention_layer_past}[name]
    if not compiled:
        return function
    if name not in _COMPILED:
        # One shared graph per layer kind specializes on window (full/sliding), grad mode, imported rows, packing and the
        # prefix-tree passes: more than dynamo's default 8 variants, and fullgraph turns the 9th into a hard failure
        # (grpo-v2-run1 died in step 1's KL reference pass on attention_layer, 2026-11-13).
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, RECOMPILE_LIMIT)
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


def import_tensors(model, spans, device=None):
    """Imported native state of one (packed) sequence for ``hidden``: ``spans`` [(row start, NativeStateBlock)] ->
    {'rows': int64 (n,), 'layers': [per layer (k, v) for attention, (bx,) for conv, in the model's dtype]}, or None when
    nothing is imported. The block's tensors are what the SGLang actor imports (``sglang_actor._sglang_import_file``)."""
    layers = model.model.layers
    device = device or model.model.embed_tokens.weight.device
    dtype = model.model.embed_tokens.weight.dtype
    rows, parts = [], [[] for _ in layers]
    for start, block in spans:
        count = len(block.token_ids)
        if len(block.layers) != len(layers):
            raise ValueError(f'native state has {len(block.layers)} layers, the model {len(layers)}')
        for index, (layer, state) in enumerate(zip(layers, block.layers)):
            kind = 'attention' if layer.is_attention_layer else 'finite_convolution'
            if state['kind'] != kind:
                raise ValueError(f'native state layer {index} is {state["kind"]}, the model has {kind}')
            parts[index].append((state['raw_k'], state['v']) if kind == 'attention' else (state['gated_inputs'],))
        rows.extend(range(start, start + count))
    if not rows:
        return None
    return {'rows': torch.tensor(rows, dtype=torch.long, device=device),
            'layers': [tuple(torch.cat([p[field] for p in part]).to(device=device, dtype=dtype)
                             for field in range(len(part[0]))) for part in parts]}


class Past:
    """A computed prefix that a continuation attends to: ``length`` tokens; per layer (keys after RoPE, values) for
    attention (a sliding layer keeps its window's last keys) or (B*x of the last taps - 1 tokens,) for conv."""

    def __init__(self, length, layers):
        self.length, self.layers = length, layers

    def tensors(self):
        return [t for layer in self.layers for t in layer]

    def detached(self, requires_grad):
        return Past(self.length, [tuple(t.detach().requires_grad_(requires_grad) for t in layer) for layer in self.layers])


def _empty_past(model, x):
    layers = []
    for layer in model.model.layers:
        if layer.is_attention_layer:
            attn = layer.self_attn
            heads = attn.k_proj.weight.shape[0] // attn.head_dim
            empty = x.new_zeros((0, heads, attn.head_dim))
            layers.append((empty, empty))
        else:
            layers.append((x.new_zeros((layer.conv.conv.weight.shape[-1] - 1, x.shape[-1])),))
    return Past(0, layers)


def _hidden_past(model, ids, past, compiled):
    """``hidden`` of one document continuing ``past`` (None: from position 0) -> (hidden (T, D), Past of past + ids)."""
    config = model.model.config
    x = model.model.embed_tokens(ids)
    if past is None:
        past = _empty_past(model, x)
    pos = torch.arange(past.length, past.length + ids.shape[0], device=ids.device)
    tables = _rope_tables(model, pos, x.dtype)
    cu_q = torch.tensor([0, ids.shape[0]], dtype=torch.int32, device=ids.device)
    grad = torch.is_grad_enabled()
    layers = []
    for index, layer in enumerate(model.model.layers):
        kind, static = _layer_call(layer, compiled)
        function = _kernel(kind + '_past', compiled)
        if kind == 'conv':
            args = (x, past.layers[index][0]) + static
        else:
            cos, sin = tables[config.layer_types[index]]
            keys = past.layers[index][0].shape[0] + ids.shape[0]
            cu_k = torch.tensor([0, keys], dtype=torch.int32, device=ids.device)
            args = (x, cos, sin, cu_q, cu_k) + tuple(past.layers[index]) + static
        if grad and getattr(layer, 'gradient_checkpointing', False):
            x, *state = torch_checkpoint(function, *args, use_reentrant=False)
        else:
            x, *state = function(*args)
        layers.append(tuple(state))
    norm = model.model.embedding_norm
    return _rms(x, norm.weight, norm.variance_epsilon), Past(past.length + ids.shape[0], layers)


def hidden(model, ids, lengths=None, imports=None, past=None, keep_past=False):
    """Final hidden states (T, D) of ``ids`` (1-D): one causal document, or ``lengths`` documents packed back to back
    (each attends and convolves only within itself, positions restart at 0). ``imports``: ``import_tensors`` (rows are
    indices into ``ids``). ``past``: a ``Past`` that ``ids`` continue (one document, positions after it); ``keep_past``:
    return (hidden, Past of the whole document so far) for continuations (``prefix_tree``)."""
    config = model.model.config
    fast = getattr(model, '_lfm3_fast', None)
    compiled = fast is not None and fast['compile']
    if past is not None or keep_past:
        if lengths is not None or imports is not None:
            raise ValueError('hidden: a past continues one document without imported state')
        h, whole = _hidden_past(model, ids, past, compiled)
        return (h, whole) if keep_past else h
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
        if imports is not None:
            args = args + (imports['rows'],) + imports['layers'][index]
        if grad and getattr(layer, 'gradient_checkpointing', False):
            x = torch_checkpoint(function, *args, use_reentrant=False)
        else:
            x = function(*args)
    norm = model.model.embedding_norm
    return _rms(x, norm.weight, norm.variance_epsilon)


# --- shared prefixes -------------------------------------------------------------------------------------------------

def _common(a, b):
    n = min(len(a), len(b))
    diff = (a[:n] != b[:n]).nonzero()[0]
    return int(diff[0]) if len(diff) else n


def prefix_tree(seqs, caps, min_share):
    """A compressed trie over token sequences (numpy 1-D), so that a prefix several of them share is computed once.
    ``caps[i]``: sequence i's prefix may not extend past this position (the first row whose hidden state it needs);
    a shared prefix is at least ``min_share`` tokens. Returns root nodes: a prefix {'start', 'end', 'first' (a member
    sequence), 'children'} or a leaf {'start', 'index'} (sequence ``index`` from ``start`` on)."""
    order = sorted(range(len(seqs)), key=lambda i: seqs[i].tobytes())    # a shared token prefix is a contiguous run
    adjacent = [min(_common(seqs[a], seqs[b]), caps[a], caps[b]) for a, b in zip(order, order[1:])]

    def build(lo, hi, start):                       # order[lo:hi], all sharing [0, start)
        nodes, i = [], lo
        while i < hi:
            j, end = i, None
            while j + 1 < hi and adjacent[j] - start >= min_share:
                end = adjacent[j] if end is None else min(end, adjacent[j])
                j += 1
            if j == i:
                nodes.append({'start': start, 'index': order[i]})
            else:
                nodes.append({'start': start, 'end': end, 'first': order[i], 'children': build(i, j + 1, end)})
            i = j + 1
        return nodes
    return build(0, len(seqs), 0) if min_share > 0 else [{'start': 0, 'index': i} for i in range(len(seqs))]


def tree_tokens(nodes, lengths):
    """Tokens a tree computes (each prefix once, each leaf from its start)."""
    return sum(tree_tokens(n['children'], lengths) + n['end'] - n['start'] if 'children' in n
               else lengths[n['index']] - n['start'] for n in nodes)


def run_prefix_tree(model, nodes, ids_of, leaf, prefix_checkpointing=None, backward=True):
    """Visit a ``prefix_tree`` depth first: a prefix node computes its tokens once (``ids_of(first, start, end)``)
    after its parent's ``Past``; each leaf calls ``leaf(index, start, past)`` (past None at start 0), which runs its own
    forward from ``past`` and, under ``backward``, its own backward. With ``backward`` the children's gradients reach the
    prefix through detached leaf copies of its Past and the prefix's graph backpropagates them once after its last child
    (the exact gradient of the unshared computation); without it each prefix's graph is dropped as soon as its Past
    is detached.
    ``prefix_checkpointing(tokens)`` sets the layers' recompute before each prefix forward."""
    def visit(node, past):
        if 'index' in node:
            leaf(node['index'], node['start'], past)
            return
        if prefix_checkpointing is not None:
            prefix_checkpointing(node['end'] - node['start'])
        _, whole = hidden(model, ids_of(node['first'], node['start'], node['end']), past=past, keep_past=True)
        # under grad mode the leaves require grad even without a backward: the children then compile the policy's
        # own training graph (the KL reference pass must round exactly as the policy does)
        held = whole.detached(requires_grad=torch.is_grad_enabled())
        if not backward:
            whole = None
        for child in node['children']:
            visit(child, held)
        if backward:
            pairs = [(t, h.grad) for t, h in zip(whole.tensors(), held.tensors()) if h.grad is not None]
            if pairs:
                torch.autograd.backward([t for t, _ in pairs], [g for _, g in pairs])
    for node in nodes:
        visit(node, None)


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


NUCLEUS_TOP_K = 1024
NUCLEUS_CHUNK = 2048    # rows per head chunk of the nucleus NLL: it holds about twice the fp32 temporaries of the plain CE


def _nucleus_cut(lf, lse, top_p, top_k):
    """Per row of fp32 logits ``lf`` (log-sum-exp ``lse``), the logit of the last token of the top-p nucleus: tokens by
    decreasing probability, a token kept while the mass before it is <= ``top_p`` (SGLang's rule,
    ``top_k_top_p_min_p_sampling_from_probs_torch``). The nucleus is every token with a logit >= this cut (ties with
    the last kept token are in). Only the ``top_k`` largest logits are sorted; a row whose nucleus may reach past them
    is sorted in full."""
    k = min(top_k, lf.shape[-1])

    def cut_of(values, row_lse):
        p = torch.exp(values - row_lse[:, None])
        count = ((torch.cumsum(p, -1) - p) <= top_p).sum(-1).clamp(min=1)
        return values.gather(1, (count - 1)[:, None])[:, 0]
    values = lf.topk(k, dim=-1).values
    cut = cut_of(values, lse)
    if k < lf.shape[-1]:
        wide = cut <= values[:, -1]
        if wide.any():
            cut[wide] = cut_of(lf[wide].sort(dim=-1, descending=True).values, lse[wide])
    return cut


class _LinearNLLNucleus(torch.autograd.Function):
    """``_LinearNLL`` plus the NLL under the top-p nucleus of each row's own distribution: outputs (nll, nucleus nll,
    target outside the nucleus). The nucleus (``_nucleus_cut``; the target is always in it) is a constant set, so
    nucleus nll = logsumexp over the set - target logit, with gradient softmax-over-the-set - onehot. Per row only
    three scalars are kept (log-sum-exp, the set's log-sum-exp, the cut logit); backward recomputes each chunk's
    logits once and rebuilds the set from the cut."""

    @staticmethod
    def forward(ctx, h, weight, target, chunk, top_p, top_k):
        rows = h.shape[0]
        nll, lse, set_lse, cut = (torch.empty(rows, dtype=torch.float32, device=h.device) for _ in range(4))
        outside = torch.empty(rows, dtype=torch.bool, device=h.device)
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            lf = F.linear(h[start:stop], weight).float()
            own = lf.gather(-1, target[start:stop, None])[:, 0]
            lse[start:stop] = torch.logsumexp(lf, dim=-1)
            cut[start:stop] = _nucleus_cut(lf, lse[start:stop], top_p, top_k)
            inside = torch.logsumexp(lf.masked_fill(lf < cut[start:stop, None], float('-inf')), dim=-1)
            outside[start:stop] = own < cut[start:stop]
            set_lse[start:stop] = torch.where(outside[start:stop], torch.logaddexp(inside, own), inside)
            nll[start:stop] = lse[start:stop] - own
        ctx.save_for_backward(h, weight, target, lse, set_lse, cut)
        ctx.chunk = chunk
        ctx.mark_non_differentiable(outside)
        return nll, set_lse - (lse - nll), outside

    @staticmethod
    def backward(ctx, grad, grad_nucleus, _):
        h, weight, target, lse, set_lse, cut = ctx.saved_tensors
        zeros = torch.zeros(h.shape[0], dtype=torch.float32, device=h.device)
        grad = zeros if grad is None else grad.float()
        grad_nucleus = zeros if grad_nucleus is None else grad_nucleus.float()
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        dh = torch.empty_like(h) if need_h else None
        dw = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_w else None
        for start in range(0, h.shape[0], ctx.chunk):
            stop = min(h.shape[0], start + ctx.chunk)
            rows = h[start:stop]
            lf = F.linear(rows, weight).float()
            onehot = torch.arange(lf.shape[-1], device=lf.device)[None, :] == target[start:stop, None]
            d = (torch.exp(lf - lse[start:stop, None]) - onehot.float()) * grad[start:stop, None]
            kept = (lf >= cut[start:stop, None]) | onehot
            d = d + (torch.exp(lf - set_lse[start:stop, None]) * kept - onehot.float()) * grad_nucleus[start:stop, None]
            d = d.to(rows.dtype)
            if need_h:
                dh[start:stop] = d @ weight
            if need_w:
                if d.is_cuda:
                    dw += torch.mm(d.t(), rows, out_dtype=torch.float32)
                else:
                    dw += (d.t() @ rows).float()
        return dh, None if dw is None else dw.to(weight.dtype), None, None, None, None


def linear_nll_nucleus(h, weight, target, top_p, chunk=NUCLEUS_CHUNK, top_k=NUCLEUS_TOP_K):
    """(nll, nucleus nll, outside): ``linear_nll`` and the NLL of ``target`` under the row's distribution truncated to
    its own top-p nucleus and renormalized (the set is a constant; the gradient flows through the target logit and the
    set's mass), with the target kept in the set even when it lies outside (``outside`` marks those rows).
    log(nucleus mass) = nll - nucleus nll. ``linear_nll`` itself is untouched (top_p = 1)."""
    if not 0. < top_p < 1.:
        raise ValueError(f'top_p must be in (0, 1) for the nucleus NLL, got {top_p} (use linear_nll at 1)')
    return _LinearNLLNucleus.apply(h, weight, target, chunk, float(top_p), int(top_k))


@torch.no_grad()
def token_logprobs(model, ids, positions, targets, lengths=None, chunk=LOGIT_CHUNK, imports=None, top_p=1.0):
    """Log-probabilities (fp32) of ``targets`` predicted at rows ``positions`` of the packed ``ids`` (no grad: the
    GRPO behaviour / reference recompute). Pack many sequences per call (``lengths``) for throughput. ``top_p`` < 1:
    (logp, logp under the row's own top-p nucleus, target outside the nucleus) (``linear_nll_nucleus``)."""
    h = hidden(model, ids, lengths, imports)
    if top_p >= 1.:
        return -linear_nll(h[positions], model.lm_head.weight, targets, chunk)
    nll, nucleus, outside = linear_nll_nucleus(h[positions], model.lm_head.weight, targets, top_p)
    return -nll, -nucleus, outside
