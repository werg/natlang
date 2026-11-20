"""Fast LFM3 training forward/backward (the HF modelling arithmetic, fewer and fused kernels), packed documents, fused
linear cross-entropy and a no-grad log-prob pass.

    model = load an HF `Lfm3ForCausalLM` in bf16; prepare(model)
    h = hidden(model, ids)                         # (T, D) for one sequence
    h = hidden(model, ids, lengths=[n1, n2, ...])  # several documents packed into one row, each attending only itself
    nll = linear_nll(h[positions], model.lm_head.weight, targets)        # fp32 per-token NLL, logits never stored
    logp = token_logprobs(model, ids, positions, targets, lengths=...)   # no grad (GRPO old-policy / reference pass)
    nll, nucleus_nll, outside = linear_nll_nucleus(h[positions], weight, targets, top_p)   # + NLL under the top-p nucleus
    h = hidden(model, ids, imports=import_tensors(model, [(start, block)], device))   # imported native state spliced in
    run_prefix_tree(model, prefix_tree(seqs, caps, 256), ids_of, leaf, tokens, pack_tokens, activation_gb)
                                                   # shared prompts computed once, every batch packed, exact grads

The HF decoder layer is replaced by two functions over explicit tensors, one per layer kind (conv, attention), each
`torch.compile`d (fullgraph, dynamic sequence length) and shared by every layer of its kind, with no HF mask/cache
plumbing in the traced region. Inside them the norms, RoPE, gates, SwiGLU and the width-7 conv fuse into a few Triton
kernels; the matmuls stay cuBLAS.
Attention is the varlen flash kernel (`torch.nn.attention.varlen`): causal for full-attention layers and a 255-left /
0-right window for sliding layers (HF's `sliding_window` 256 keys including the query), so it is exact, needs no mask,
and packed documents are separate sequences (`cu_seqlens`). The conv reads only taps inside the token's document.
Rounding points follow HF: RMSNorm in fp32 rounded before its bf16 weight, RoPE in fp32 with bf16 cos/sin, the conv
accumulated in fp32 from bf16 inputs and rounded once. Layers whose `gradient_checkpointing` flag is set (see
`set_checkpointing`) are recomputed in backward.

Shared prefixes (`run_prefix_tree`, plans/neuralese/GRPO_STEP_THROUGHPUT_20261117.md): each level of a prefix tree's
nodes and then its leaves run as packed varlen batches; a continuation's attention reads its history's cached keys and
values (``cu_seqlens_k`` longer than ``cu_seqlens_q``, flash's bottom-right causal rule) and its conv the history's last
B*x rows. The history state is held as FP32 leaves, so the cotangents of all continuations add up in FP32 and reach the
prefix's graph in one backward.

Imported native Neuralese state (a consumer call's prompt as the SGLang actor fed it, `import_tensors`): at the imported
rows every attention layer's keys (after the K norm, before RoPE) and values and every conv layer's input B*x are the
stored block's, exactly the fork's `NatlangImport`; the rows' own outputs reach nothing. The state is a constant (no
gradient into the producer); `exp0_sft.hidden` does the same on the stock HF modules through forward hooks.

Pass-through (`passthrough_plan`, plans/neuralese/NEURALESE_PASSTHROUGH_20261117.md): the imported state computed in the
same forward from the producer's own tokens. Each import's producer document (the block's recorded ``source_capture``:
for a text write ``[produce_open] + text ids`` from position 0, the span rows 1..n, exactly the actor's capture request)
rides as one more packed document, and in every layer the import rows' normed pre-RoPE keys, values and conv B*x are
that layer's own values at the span rows (``index_put`` of a gather: differentiable), so the consumer's loss reaches the
weights that computed the state. ``frozen`` rows (the opt-out, or a ``stop_gradient`` import) carry the same values
detached.

The vendored HF sources are hash-pinned (`third_party/lfm3`), so everything here reads instances, never patches files.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

LOGIT_CHUNK = 4096

# Inductor options of every compile here (layers, the cross-entropy and nucleus kernels). ``emulate_precision_casts``:
# keep eager's BF16 rounding after each BF16 op; by default Inductor computes a fused chain of BF16 ops in FP32 and rounds
# only what it stores, so the rounding points depended on which values a graph stores: a no-grad (inference) graph stores
# fewer than a training graph, and compiled no-grad log-probabilities drifted from eager and from the grad-mode graph
# (2026-11-17). ``deterministic``: no per-process benchmarking of reduction configs, whose choice changes summation order
# (the same compiled forward differed between processes). Opt-out: NATLANG_FAST_TRAIN_INDUCTOR=fast (Inductor defaults).
import os as _os

COMPILE_OPTIONS = ({} if _os.environ.get('NATLANG_FAST_TRAIN_INDUCTOR') == 'fast'
                   else {'emulate_precision_casts': True, 'deterministic': True})


def _compile(function, **kwargs):
    return torch.compile(function, dynamic=True, options=dict(COMPILE_OPTIONS) or None, **kwargs)


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


def _varlen_gqa():
    """torch >= 2.12 takes grouped-query heads only with ``enable_gqa=True`` (2.11 always did; Pop's SGLang venv runs 2.14)."""
    try:
        import inspect
        from torch.nn.attention.varlen import varlen_attn
        return {'enable_gqa': True} if 'enable_gqa' in inspect.signature(varlen_attn).parameters else {}
    except ImportError:
        return {}


_GQA = _varlen_gqa()


def _attend(q, k, v, docs, window, tree_mask=None):
    """Causal attention within each document, keys at most ``window`` positions back (-1: unbounded). (T, H, D) -> same.
    ``tree_mask`` (a flattened prefix tree, ``tree_masks``): each query sees its own path instead, a flex-attention
    ``BlockMask`` on CUDA, the same mask as a dense bool (T, T) on the CPU."""
    scale = q.shape[-1] ** -0.5
    if tree_mask is not None:
        if q.is_cuda:
            from torch.nn.attention.flex_attention import flex_attention
            out = flex_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                 block_mask=tree_mask, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
            return out[0].transpose(0, 1)
        out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                             attn_mask=tree_mask, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
        return out[0].transpose(0, 1)
    cu, max_len, doc, pos = docs
    if q.is_cuda:
        from torch.nn.attention.varlen import varlen_attn
        return varlen_attn(q, k, v, cu, cu, max_len, max_len, scale=scale, window_size=(window, 0), **_GQA)
    # reference path (CPU tests): the same key sets as an explicit mask
    i = torch.arange(q.shape[0], device=q.device)
    mask = (doc[:, None] == doc[None, :]) & (i[None, :] <= i[:, None])
    if window >= 0:
        mask &= i[:, None] - i[None, :] <= window
    out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                         attn_mask=mask, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
    return out[0].transpose(0, 1)


def _passed(t, source, frozen):
    """The producer span's own rows of ``t`` for a pass-through import; rows marked in ``frozen`` (bool, one per row, or
    None) are detached: their state is fixed context with the same values."""
    value = t[source]
    if frozen is None:
        return value
    return torch.where(frozen.view(-1, *([1] * (value.dim() - 1))), value.detach(), value)


def _ffn(x, ffn_norm, w1, w3, w2, post_ffn_norm, eps):
    f = _rms(x, ffn_norm, eps)
    f = F.linear(F.silu(F.linear(f, w1)) * F.linear(f, w3), w2)
    if post_ffn_norm is not None:
        f = _rms(f, post_ffn_norm, eps)
    return x + f


def conv_layer(x, pos, operator_norm, in_proj, conv_weight, out_proj, post_operator_norm,
               ffn_norm, w1, w3, w2, post_ffn_norm, eps: float, import_rows=None, import_bx=None,
               pass_rows=None, pass_source=None, pass_frozen=None, pred=None):
    """`Lfm3DecoderLayer` with `Lfm3ShortConv` on (T, D): the depthwise width-7 causal conv as seven shifted fp32
    multiply-adds (as the inference kernel accumulates), each tap only inside the token's document (``pos``: position
    within the document; ``import_rows``: rows whose B*x is the imported ``import_bx``; ``pass_rows``: rows whose B*x
    is this layer's own B*x at ``pass_source``, the producer span in the same forward: ``passthrough``). ``pred``
    (taps - 1, T) (a flattened prefix tree): tap ``back``'s row is ``pred[back - 1]``, the token ``back`` steps back
    along the token's own path (T: none, a zero row), in place of the shifted rows."""
    length = x.shape[0]
    taps = conv_weight.shape[-1]
    h = _rms(x, operator_norm, eps)
    b, c, xx = F.linear(h, in_proj).chunk(3, dim=-1)
    bx = b * xx
    if import_rows is not None:
        bx = bx.index_put((import_rows,), import_bx)
    if pass_rows is not None:
        bx = bx.index_put((pass_rows,), _passed(bx, pass_source, pass_frozen))
    weight = conv_weight.reshape(conv_weight.shape[0], taps).float()
    if pred is None:
        padded = F.pad(bx.float(), (0, 0, taps - 1, 0))
        acc = padded[taps - 1:taps - 1 + length] * weight[:, taps - 1]
        for back in range(1, taps):                     # the tap ``back`` positions before the token
            k = taps - 1 - back
            acc = acc + (padded[k:k + length] * weight[:, k]) * (pos >= back)[:, None]
    else:
        padded = F.pad(bx.float(), (0, 0, 0, 1))       # row T: zero (before the path's start)
        acc = padded[:length] * weight[:, taps - 1]
        for back in range(1, taps):                     # the same multiply-adds in the same order
            acc = acc + padded[pred[back - 1]] * weight[:, taps - 1 - back]
    y = F.linear(c * acc.to(bx.dtype), out_proj)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps)


def attention_layer(x, cos, sin, cu, max_len: int, doc, pos, operator_norm, wq, wk, wv, wo, wg, q_norm, k_norm,
                    post_operator_norm, ffn_norm, w1, w3, w2, post_ffn_norm, eps: float, head_dim: int, window: int,
                    import_rows=None, import_k=None, import_v=None, pass_rows=None, pass_source=None,
                    pass_frozen=None, tree_mask=None):
    """`Lfm3DecoderLayer` with `Lfm3Attention` on (T, D): QK-norm, fp32 RoPE, flash attention, sigmoid head gate
    (``import_rows``: rows whose normed pre-RoPE keys and values are the imported ``import_k``/``import_v``;
    ``pass_rows``: rows whose normed pre-RoPE keys and values are this layer's own at ``pass_source``)."""
    length = x.shape[0]
    h = _rms(x, operator_norm, eps)
    q = _rms(F.linear(h, wq).view(length, -1, head_dim), q_norm, eps)
    k = _rms(F.linear(h, wk).view(length, -1, head_dim), k_norm, eps)
    v = F.linear(h, wv).view(length, -1, head_dim)
    if import_rows is not None:
        k = k.index_put((import_rows,), import_k)
        v = v.index_put((import_rows,), import_v)
    if pass_rows is not None:
        k = k.index_put((pass_rows,), _passed(k, pass_source, pass_frozen))
        v = v.index_put((pass_rows,), _passed(v, pass_source, pass_frozen))
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    o = _attend(q, k, v, (cu, max_len, doc, pos), window, tree_mask)
    o = (o * torch.sigmoid(F.linear(h, wg))[..., None]).reshape(length, -1)
    y = F.linear(o, wo)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps)


def _attend_past(q, k, v, docs, window):
    """Causal attention of packed documents' queries over their keys (each document's history rows, then its own): keys
    at most ``window`` positions back (-1: unbounded), each document's queries aligned to the end of its keys (flash's
    causal rule for fewer queries than keys). ``docs``: (cu_q, cu_k, max_q, max_k, then the CPU reference's query
    document, query position, key document, key position), made outside the compiled layer like ``hidden``'s ``cu``."""
    scale = q.shape[-1] ** -0.5
    cu_q, cu_k, max_q, max_k, q_doc, q_pos, k_doc, k_pos = docs
    if q.is_cuda:
        from torch.nn.attention.varlen import varlen_attn
        return varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k, scale=scale, window_size=(window, 0), **_GQA)
    mask = (q_doc[:, None] == k_doc[None, :]) & (k_pos[None, :] <= q_pos[:, None])
    if window >= 0:
        mask &= q_pos[:, None] - k_pos[None, :] <= window
    out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                         attn_mask=mask, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
    return out[0].transpose(0, 1)


def conv_layer_past(x, src_bx, past_index, joined_order, own_rows, operator_norm, in_proj, conv_weight, out_proj,
                    post_operator_norm, ffn_norm, w1, w3, w2, post_ffn_norm, eps: float):
    """``conv_layer`` for packed documents that continue computed histories (``Past``): ``src_bx`` (S, D) FP32 rows of
    the histories' B*x (row 0 zero), ``past_index`` each document's taps - 1 history rows in ``src_bx`` (0 before the
    history's start), ``joined_order`` the rows of [own B*x; history B*x] laid out per document as [its taps - 1 history
    rows, its own rows], ``own_rows`` the documents' own rows in that layout less taps - 1. Returns (output, B*x)."""
    taps = conv_weight.shape[-1]
    h = _rms(x, operator_norm, eps)
    b, c, xx = F.linear(h, in_proj).chunk(3, dim=-1)
    bx = b * xx
    joined = torch.cat([bx, src_bx.index_select(0, past_index).to(bx.dtype)]).index_select(0, joined_order)
    weight = conv_weight.reshape(conv_weight.shape[0], taps).float()
    padded = joined.float()
    rows = padded.shape[0] - (taps - 1)
    acc = padded[taps - 1:taps - 1 + rows] * weight[:, taps - 1]
    for back in range(1, taps):                         # the same fp32 multiply-adds in the same order as conv_layer
        k = taps - 1 - back
        acc = acc + padded[k:k + rows] * weight[:, k]
    acc = acc.index_select(0, own_rows)
    y = F.linear(c * acc.to(bx.dtype), out_proj)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps), bx


def attention_layer_past(x, cos, sin, src_k, src_v, past_index, key_order, cu_q, cu_k, max_q: int, max_k: int, q_doc,
                         q_pos, k_doc, k_pos, operator_norm, wq, wk, wv, wo, wg, q_norm, k_norm, post_operator_norm,
                         ffn_norm, w1, w3, w2, post_ffn_norm, eps: float, head_dim: int, window: int):
    """``attention_layer`` for packed documents that continue computed histories: ``src_k``/``src_v`` FP32 keys (after
    RoPE) and values of the histories' rows, ``past_index`` each document's history rows in them (a sliding layer's: its
    window's last), ``key_order`` the rows of [own keys; history keys] laid out per document as [history, own] (``cu_k``).
    The history's FP32 rows are rounded back to the BF16 they were: the values are the whole document's. Returns
    (output, own keys after RoPE, own values)."""
    length = x.shape[0]
    h = _rms(x, operator_norm, eps)
    q = _rms(F.linear(h, wq).view(length, -1, head_dim), q_norm, eps)
    k = _rms(F.linear(h, wk).view(length, -1, head_dim), k_norm, eps)
    v = F.linear(h, wv).view(length, -1, head_dim)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    keys = torch.cat([k, src_k.index_select(0, past_index).to(k.dtype)]).index_select(0, key_order)
    values = torch.cat([v, src_v.index_select(0, past_index).to(v.dtype)]).index_select(0, key_order)
    o = _attend_past(q, keys, values, (cu_q, cu_k, max_q, max_k, q_doc, q_pos, k_doc, k_pos), window)
    o = (o * torch.sigmoid(F.linear(h, wg))[..., None]).reshape(length, -1)
    y = F.linear(o, wo)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps), k, v


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
        _COMPILED[name] = _compile(function, fullgraph=True)
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
    nothing is imported. The block's file is what the SGLang actor imports (``sglang_actor._sglang_import``)."""
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


def passthrough_plan(parts, total, frozen_all=False):
    """Producer documents and gather indices for the imports of one packed sequence of ``total`` tokens.

    ``parts``: [(offset, imports, sources)] of the sequences packed in it (``imports`` [(start, key)] relative to the
    sequence; ``sources`` {key: {'ids': int[], 'start', 'stop', 'frozen'?}}, the producer documents, ``exp0_sft.chains_of``).
    An import whose key has no source is returned in ``fixed`` [(absolute row, key)] (stored state:
    ``import_tensors``). Each source is one document appended after the sequence (a source used by several imports once).
    Returns {'documents': [int arrays], 'rows': [...], 'source': [...], 'frozen': [bool], 'fixed': [...]}."""
    documents, placed, rows, source, frozen, fixed = [], {}, [], [], [], []
    end = total
    for offset, imports, sources in parts:
        for start, key in imports:
            spec = (sources or {}).get(key)
            if spec is None:
                fixed.append((offset + start, key))
                continue
            count = int(spec['stop']) - int(spec['start'])
            if key not in placed:
                if not 0 < int(spec['start']) <= int(spec['stop']) <= len(spec['ids']):
                    raise ValueError(f'source of {key}: span [{spec["start"]}, {spec["stop"]}) outside its document')
                placed[key] = end + int(spec['start'])
                documents.append(spec['ids'])
                end += len(spec['ids'])
            rows.extend(range(offset + start, offset + start + count))
            source.extend(range(placed[key], placed[key] + count))
            frozen.extend([bool(frozen_all or spec.get('frozen'))] * count)
    return {'documents': documents, 'rows': rows, 'source': source, 'frozen': frozen, 'fixed': fixed}


def passthrough_tensors(plan, device, fixed=None):
    """``hidden``'s ``imports`` from a ``passthrough_plan`` (``fixed``: ``import_tensors`` of its fixed rows), or None."""
    imports = dict(fixed) if fixed is not None else {}
    if plan['rows']:
        imports['pass_rows'] = torch.tensor(plan['rows'], dtype=torch.long, device=device)
        imports['pass_source'] = torch.tensor(plan['source'], dtype=torch.long, device=device)
        if any(plan['frozen']):
            imports['pass_frozen'] = torch.tensor(plan['frozen'], dtype=torch.bool, device=device)
    return imports or None


class Segment:
    """One prefix node's own rows of the state its continuations read, per layer: attention (keys after RoPE, values),
    every row for a full layer and the last ``window`` for a sliding one; conv (the last taps - 1 B*x,). FP32 leaves of
    the prefix's graph (``Segment.held``): the cotangents of all its continuations accumulate in them in FP32."""

    def __init__(self, length, layers):
        self.length, self.layers = length, layers

    def tensors(self):
        return [t for layer in self.layers for t in layer]


class Past(tuple):
    """The history a document continues: its ancestors' ``Segment``s, oldest first (empty: none)."""

    @property
    def length(self):
        return sum(segment.length for segment in self)


def _layer_keys(model):
    """Per layer: ('conv', taps - 1) or ('attention', window)."""
    keys = []
    for layer in model.model.layers:
        if layer.is_attention_layer:
            window = layer.self_attn.sliding_window
            keys.append(('attention', -1 if window is None else window - 1))
        else:
            keys.append(('conv', layer.conv.conv.weight.shape[-1] - 1))
    return keys


def _kept_rows(key, length):
    """How many of a segment's last rows a layer of ``key`` keeps for continuations."""
    kind, size = key
    return length if kind == 'attention' and size < 0 else min(length, size)


def _past_inputs(model, pasts, lengths, device):
    """Index tensors and FP32 history sources of one packed continuation forward: ``pasts`` (a ``Past`` per document)
    and the documents' own ``lengths``."""
    import numpy as np
    keys = _layer_keys(model)
    segments, first = [], {}
    for past in pasts:
        for segment in past:
            if id(segment) not in first:
                first[id(segment)] = len(segments)
                segments.append(segment)
    starts = np.cumsum([0] + lengths[:-1])
    total = int(sum(lengths))
    history = [past.length for past in pasts]
    q_pos = np.concatenate([h + np.arange(n) for h, n in zip(history, lengths)])
    plans = {}
    for key in set(keys):
        kind, size = key
        rows = [_kept_rows(key, segment.length) for segment in segments]
        offset = np.cumsum([1 if kind == 'conv' else 0] + rows[:-1])     # conv sources start with a zero row
        index, order, k_doc, k_pos, past_rows = [], [], [], [], 0
        for doc, (past, n, start) in enumerate(zip(pasts, lengths, starts)):
            mine = [offset[first[id(s)]] + np.arange(_kept_rows(key, s.length)) for s in past]
            mine = np.concatenate(mine) if mine else np.zeros(0, np.int64)
            if kind == 'conv':
                mine = np.concatenate([np.zeros(size, np.int64), mine])[-size:] if size else mine
            elif size >= 0:
                mine = mine[-size:] if size else mine[:0]
            index.append(mine)
            order.append(total + past_rows + np.arange(len(mine)))
            order.append(start + np.arange(n))
            k_doc.append(np.full(len(mine) + n, doc))
            k_pos.append(history[doc] + np.arange(-len(mine), n))
            past_rows += len(mine)
        tensor = lambda a: torch.from_numpy(np.ascontiguousarray(np.concatenate(a), dtype=np.int64)).to(device)
        plan = {'index': tensor(index), 'order': tensor(order)}
        if kind == 'conv':
            joined = np.cumsum([0] + [size + n for n in lengths[:-1]])
            plan['own'] = tensor([j + np.arange(n) for j, n in zip(joined, lengths)])
        else:
            counts = [len(i) + n for i, n in zip(index, lengths)]
            cu_k = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
            cu_k[1:] = torch.from_numpy(np.cumsum(counts)).to(device)
            plan.update(cu_k=cu_k, max_k=int(max(counts)), k_doc=tensor(k_doc), k_pos=tensor(k_pos))
        plan['rows'] = past_rows
        plans[key] = plan
    sources = []
    for index, (layer, key) in enumerate(zip(model.model.layers, keys)):
        if key[0] == 'conv':
            width = layer.conv.conv.weight.shape[0]
            parts = [torch.zeros((1, width), dtype=torch.float32, device=device)]
            sources.append((torch.cat(parts + [s.layers[index][0] for s in segments]),))
        else:
            attn = layer.self_attn
            empty = torch.zeros((0, attn.k_proj.weight.shape[0] // attn.head_dim, attn.head_dim),
                                dtype=torch.float32, device=device)
            sources.append(tuple(torch.cat([empty] + [s.layers[index][field] for s in segments]) for field in (0, 1)))
    cu_q = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu_q[1:] = torch.from_numpy(np.cumsum(lengths)).to(device)
    q_doc = torch.repeat_interleave(torch.arange(len(lengths), device=device),
                                    torch.tensor(lengths, device=device), output_size=total)
    return {'keys': keys, 'plans': plans, 'sources': sources, 'cu_q': cu_q, 'max_q': int(max(lengths)), 'q_doc': q_doc,
            'q_pos': torch.from_numpy(q_pos.astype(np.int64)).to(device)}


def _hidden_past(model, ids, lengths, pasts, compiled, keep):
    """``hidden`` of packed documents continuing their ``pasts`` -> (final hidden (T, D) or None under ``keep``, each
    document's ``Segment`` under ``keep`` (graph tensors, FP32) or None)."""
    config = model.model.config
    device = ids.device
    inputs = _past_inputs(model, pasts, lengths, device)
    x = model.model.embed_tokens(ids)
    tables = _rope_tables(model, inputs['q_pos'], x.dtype)
    grad = torch.is_grad_enabled()
    own = [[] for _ in lengths]
    starts, total = [], 0
    for n in lengths:
        starts.append(total)
        total += n
    for index, layer in enumerate(model.model.layers):
        kind, static = _layer_call(layer, compiled)
        function = _kernel(kind + '_past', compiled)
        key = inputs['keys'][index]
        plan = inputs['plans'][key]
        if kind == 'conv':
            args = (x, inputs['sources'][index][0], plan['index'], plan['order'], plan['own']) + static
        else:
            cos, sin = tables[config.layer_types[index]]
            args = ((x, cos, sin) + inputs['sources'][index] + (plan['index'], plan['order'], inputs['cu_q'],
                    plan['cu_k'], inputs['max_q'], plan['max_k'], inputs['q_doc'], inputs['q_pos'], plan['k_doc'],
                    plan['k_pos']) + static)
        if grad and getattr(layer, 'gradient_checkpointing', False):
            x, *state = torch_checkpoint(function, *args, use_reentrant=False)
        else:
            x, *state = function(*args)
        if keep:
            for doc, (start, n) in enumerate(zip(starts, lengths)):
                rows = _kept_rows(key, n)
                own[doc].append(tuple(t[start + n - rows:start + n].float() for t in state))
    if keep:
        return None, [Segment(n, layers) for n, layers in zip(lengths, own)]
    norm = model.model.embedding_norm
    return _rms(x, norm.weight, norm.variance_epsilon), None


def tree_layout(lengths, parents, taps):
    """A flattened prefix tree: documents back to back (``lengths``), document d continuing the path of
    ``parents[d]`` (an earlier document, or -1: a root) from its end. -> (per token: path position (int64), document,
    the (taps - 1, T) conv predecessors, T for none), the ancestor-or-self matrix (docs x docs, bool)), numpy."""
    import numpy as np
    lengths = [int(n) for n in lengths]
    count = len(lengths)
    rows = np.cumsum([0] + lengths[:-1])
    starts = []
    for d, p in enumerate(parents):
        if not -1 <= p < d:
            raise ValueError(f'document {d}: parent {p} is not an earlier document')
        starts.append(0 if p < 0 else starts[p] + lengths[p])
    total = int(sum(lengths))
    doc = np.repeat(np.arange(count), lengths)
    pos = np.concatenate([s + np.arange(n) for s, n in zip(starts, lengths)]) if count else np.zeros(0, np.int64)
    ancestors = np.zeros((count, count), bool)
    for d in range(count):
        a = d
        while a >= 0:
            ancestors[d, a] = True
            a = parents[a]
    pred = np.empty((taps - 1, total), np.int64)
    flat = np.arange(total)
    for back in range(1, taps):
        pred[back - 1] = flat - back
    for d in range(count):
        for offset in range(min(taps - 1, lengths[d])):
            for back in range(offset + 1, taps):         # taps that reach before the document: walk up its path
                target, a = starts[d] + offset - back, parents[d]
                while a >= 0 and target < starts[a]:
                    a = parents[a]
                pred[back - 1, rows[d] + offset] = total if a < 0 or target < 0 else rows[a] + target - starts[a]
    return pos.astype(np.int64), doc, pred, ancestors


TREE_BLOCK = 128


def _tree_block_mask(pos, doc, ancestors, window, mask_mod, block=TREE_BLOCK):
    """The flex-attention ``BlockMask`` of a path mask, classified from the tree's structure (no T x T evaluation, no
    compile per sequence length: ``create_block_mask`` recompiled for every new length, 2-17 s each): a (query block, key
    block) tile is skipped only when no query can see a key (no ancestor pair, or every key after every query, or out of
    the window), full only when every query sees every key, and otherwise partial (``mask_mod`` per element)."""
    from torch.nn.attention.flex_attention import BlockMask
    total = pos.shape[0]
    blocks = -(-total // block)
    row_block = torch.arange(total, device=pos.device) // block
    members = torch.zeros(blocks, ancestors.shape[0], device=pos.device)
    members[row_block, doc] = 1.
    seen = ancestors.float()
    any_ancestor = (members @ seen @ members.T) > 0
    not_ancestor = (members @ (1. - seen) @ members.T) > 0
    low = torch.full((blocks,), torch.iinfo(torch.int64).max, device=pos.device).scatter_reduce(0, row_block, pos, 'amin')
    high = torch.full((blocks,), -1, device=pos.device).scatter_reduce(0, row_block, pos, 'amax')
    q_low, q_high, k_low, k_high = low[:, None], high[:, None], low[None, :], high[None, :]
    possible = any_ancestor & (k_low <= q_high)
    full = ~not_ancestor & (k_high <= q_low)
    if window >= 0:
        possible &= q_low - k_high <= window
        full &= q_high - k_low <= window
    whole = torch.ones(blocks, dtype=torch.bool, device=pos.device)
    whole[-1] = total % block == 0                     # a last block shorter than ``block`` is never full
    full &= possible & whole[:, None] & whole[None, :]
    partial = possible & ~full

    def listed(tiles):
        order = torch.argsort((~tiles).to(torch.int8), dim=-1, stable=True).to(torch.int32)
        return tiles.sum(-1).to(torch.int32)[None, None], order[None, None]
    return BlockMask.from_kv_blocks(*listed(partial), *listed(full), BLOCK_SIZE=block, mask_mod=mask_mod,
                                    seq_lengths=(total, total))


def tree_masks(pos, doc, ancestors, windows, device):
    """Per attention window (-1: full): the path mask of a flattened tree (query q sees key k when k's document is an
    ancestor-or-self of q's and k is at most ``window`` path positions before q): a flex-attention ``BlockMask`` on CUDA,
    a dense bool (T, T) elsewhere."""
    pos, doc, ancestors = (torch.as_tensor(t, device=device) for t in (pos, doc, ancestors))
    total = pos.shape[0]
    masks = {}

    def path_mask(window):                 # a 4-argument mask_mod (flex tells mask_mod from score_mod by arity)
        def mask_mod(b, h, q, k):
            seen = ancestors[doc[q], doc[k]] & (pos[k] <= pos[q])
            return seen & (pos[q] - pos[k] <= window) if window >= 0 else seen
        return mask_mod
    for window in set(windows):
        mask_mod = path_mask(window)
        if torch.device(device).type == 'cuda':
            masks[window] = _tree_block_mask(pos, doc, ancestors, window, mask_mod)
        else:
            i = torch.arange(total, device=device)
            masks[window] = mask_mod(None, None, i[:, None], i[None, :])
    return masks


def hidden(model, ids, lengths=None, imports=None, pasts=None, parents=None):
    """Final hidden states (T, D) of ``ids`` (1-D): one causal document, or ``lengths`` documents packed back to back
    (each attends and convolves only within itself, positions restart at 0). ``imports``: ``import_tensors`` (rows are
    indices into ``ids``). ``pasts``: a ``Past`` (or None) per leading document, which then continues that history (its
    positions after it; documents beyond the list start fresh): the prefix tree's packed continuations (``levels``).
    ``parents``: a flattened prefix tree (``tree_layout``): document d continues document ``parents[d]``'s path, each
    token attending and convolving along its own path (``tree-mask``)."""
    config = model.model.config
    fast = getattr(model, '_lfm3_fast', None)
    compiled = fast is not None and fast['compile']
    if pasts is not None and any(pasts):
        if imports is not None:
            raise ValueError('hidden: continuations of a past carry no imported state')
        lengths = [int(ids.shape[0])] if lengths is None else [int(n) for n in lengths]
        pasts = [Past(p or ()) for p in pasts] + [Past()] * (len(lengths) - len(pasts))
        return _hidden_past(model, ids, lengths, pasts, compiled, keep=False)[0]
    tree = None
    if parents is not None and any(p >= 0 for p in parents):
        lengths = [int(ids.shape[0])] if lengths is None else [int(n) for n in lengths]
        parents = list(parents) + [-1] * (len(lengths) - len(parents))
        taps = next(layer.conv.conv.weight.shape[-1] for layer in model.model.layers if not layer.is_attention_layer)
        tree_pos, tree_doc, pred, ancestors = tree_layout(lengths, parents, taps)
        windows = [key[1] for key in _layer_keys(model) if key[0] == 'attention']
        tree = {'pred': torch.from_numpy(pred).to(ids.device),
                'masks': tree_masks(tree_pos, tree_doc, ancestors, windows, ids.device)}
    pos, doc, cu, max_len = _documents(ids, lengths)
    if tree is not None:
        pos = torch.from_numpy(tree_pos).to(ids.device)       # path positions: RoPE, the conv taps' reach
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
        kwargs = {}
        if tree is not None:
            kwargs.update(pred=tree['pred']) if kind == 'conv' else kwargs.update(tree_mask=tree['masks'][static[-1]])
        if imports is not None:
            if imports.get('rows') is not None:
                fields = ('import_bx',) if kind == 'conv' else ('import_k', 'import_v')
                kwargs.update(import_rows=imports['rows'], **dict(zip(fields, imports['layers'][index])))
            if imports.get('pass_rows') is not None:
                kwargs.update(pass_rows=imports['pass_rows'], pass_source=imports['pass_source'],
                              pass_frozen=imports.get('pass_frozen'))
        if grad and getattr(layer, 'gradient_checkpointing', False):
            x = torch_checkpoint(function, *args, use_reentrant=False, **kwargs)
        else:
            x = function(*args, **kwargs)
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


def remap_tree(nodes, indices):
    """A ``prefix_tree`` built over a subset, with its sequence numbers mapped back (``indices[i]``)."""
    out = []
    for node in nodes:
        if 'index' in node:
            out.append({'start': node['start'], 'index': indices[node['index']]})
        else:
            out.append({**node, 'first': indices[node['first']], 'children': remap_tree(node['children'], indices)})
    return out


def tree_tokens(nodes, lengths):
    """Tokens a tree computes (each prefix once, each leaf from its start)."""
    return sum(tree_tokens(n['children'], lengths) + n['end'] - n['start'] if 'children' in n
               else lengths[n['index']] - n['start'] for n in nodes)


ACTIVATION_BYTES_PER_TOKEN_LAYER = 88e3   # measured 84.8 KB of saved activations per token per layer (2026-11-12)


def resident_tokens(model, activation_gb):
    """The longest sequence whose saved activations fit ``activation_gb`` with no layer recomputed (the default packed
    row: packing past it trades per-forward overhead for recompute)."""
    return int(activation_gb * 2**30 // (ACTIVATION_BYTES_PER_TOKEN_LAYER * len(model.model.layers)))


def set_checkpointing(model, tokens, activation_gb):
    """Recompute (gradient-checkpoint) only as many layers as keep the saved activations of a ``tokens`` sequence under
    ``activation_gb``: a checkpointed layer costs one extra forward (a third of its compute), an unchecked one its
    activations. Returns the number of recomputed layers."""
    layers = model.model.layers
    free = int(min(len(layers), activation_gb * 2**30 // (ACTIVATION_BYTES_PER_TOKEN_LAYER * max(1, tokens))))
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = index >= free
    return len(layers) - free


def _pack(items, size, limit):
    """Greedy, order-preserving batches of ``items`` of at most ``limit`` total ``size`` (an oversize item alone)."""
    batches, batch, total = [], [], 0
    for item in items:
        n = size(item)
        if batch and total + n > limit:
            batches.append(batch)
            batch, total = [], 0
        batch.append(item)
        total += n
    if batch:
        batches.append(batch)
    return batches


PREFIX_TREE_MODES = ('tree-mask', 'levels', 'off')


def tree_units(nodes, tokens, limit):
    """Batches of a ``prefix_tree`` forest for ``tree-mask``: lists of subtrees of at most ``limit`` tokens each (a
    prefix's own tokens once per batch, a leaf's ``tokens(index, start)``). A subtree over the limit is split among its
    children's batches, each with its own copy of the prefix (exact: each copy gets the gradient of its own
    continuations); a single leaf over the limit is a batch alone."""
    def size(node):
        return _unit_tokens(node, tokens)

    def units(node):            # pruned copies of ``node`` that together hold each of its leaves once
        if 'index' in node or size(node) <= limit:
            return [node]
        own = node['end'] - node['start']
        parts = [unit for child in node['children'] for unit in units(child)]
        return [dict(node, children=group) for group in _pack(parts, size, max(1, limit - own))]
    return _pack([unit for node in nodes for unit in units(node)], size, limit)


def tree_batch_tokens(model, activation_gb):
    """The largest ``tree-mask`` batch: the tokens whose layer inputs (every layer recomputed) and one layer's
    activations fit ``activation_gb``."""
    per_token = len(model.model.layers) * model.config.hidden_size * 2 + ACTIVATION_BYTES_PER_TOKEN_LAYER
    return int(activation_gb * 2**30 // per_token)


def _resident_gb(model, tokens, activation_gb):
    """The budget left for unrecomputed layers once every layer's saved input (recomputed or not) is counted."""
    return max(0., activation_gb - tokens * len(model.model.layers) * model.config.hidden_size * 2 / 2**30)


def _recomputed(model, tokens, activation_gb):
    layers = len(model.model.layers)
    free = int(min(layers, _resident_gb(model, tokens, activation_gb) * 2**30
                   // (ACTIVATION_BYTES_PER_TOKEN_LAYER * max(1, tokens))))
    return layers - free


def plan_tree_batches(model, nodes, tokens, activation_gb):
    """The ``tree-mask`` batch cap with the least estimated compute: small batches copy shared prefixes into each batch
    (``tree_units``), large ones recompute layers (a recomputed layer costs a third more), so each candidate cap
    (resident rows doubling up to ``tree_batch_tokens``) is scored by its batches' tokens times (1 + recomputed
    fraction / 3). Returns (cap, batches)."""
    layers = len(model.model.layers)
    top = max(1, tree_batch_tokens(model, activation_gb))
    caps, cap = [], max(1, min(top, resident_tokens(model, activation_gb)))
    while cap < top:
        caps.append(cap)
        cap *= 2
    caps.append(top)
    best = None
    for cap in caps:
        batches = tree_units(nodes, tokens, cap)
        cost = 0.
        for batch in batches:
            size = sum(_unit_tokens(node, tokens) for node in batch)
            cost += size * (1 + _recomputed(model, size, activation_gb) / layers / 3)
        if best is None or cost < best[0]:
            best = (cost, cap, batches)
    return best[1], best[2]


def _unit_tokens(node, tokens):
    if 'index' in node:
        return tokens(node['index'], node['start'])
    return node['end'] - node['start'] + sum(_unit_tokens(child, tokens) for child in node['children'])


def _run_tree_mask(model, nodes, ids_of, leaf, tokens, batch_tokens, activation_gb):
    """``run_prefix_tree``'s ``tree-mask`` mode: each batch of ``tree_units`` is one flattened forward (its prefixes'
    tokens once, then the leaves' rows as the caller builds them) and the caller's one backward."""
    if batch_tokens is None:
        batch_tokens, batches = plan_tree_batches(model, nodes, tokens, activation_gb)
    else:
        batches = tree_units(nodes, tokens, batch_tokens)
    for batch in batches:
        prefixes, items, leaf_parent = [], [], []

        stack = [(node, -1) for node in reversed(batch)]     # iterative: see prefix_levels
        while stack:
            node, parent = stack.pop()
            if 'index' in node:
                items.append((node['index'], node['start']))
                leaf_parent.append(parent)
                continue
            prefixes.append((node, parent))
            here = len(prefixes) - 1
            stack.extend((child, here) for child in reversed(node['children']))
        prefix_ids = [ids_of(node['first'], node['start'], node['end']) for node, _ in prefixes]

        def forward(ids, lengths=None, imports=None):
            lengths = [int(ids.shape[0])] if lengths is None else [int(n) for n in lengths]
            offset = sum(int(t.shape[0]) for t in prefix_ids)
            if activation_gb is not None:
                rows = offset + int(ids.shape[0])
                set_checkpointing(model, rows, _resident_gb(model, rows, activation_gb))
            if not prefixes:
                return hidden(model, ids, lengths, imports)
            flat = torch.cat(prefix_ids + [ids])
            parents = [p for _, p in prefixes] + leaf_parent + [-1] * (len(lengths) - len(items))
            if imports is not None:
                imports = dict(imports)
                for key in ('rows', 'pass_rows', 'pass_source'):
                    if imports.get(key) is not None:
                        imports[key] = imports[key] + offset
            return hidden(model, flat, [int(t.shape[0]) for t in prefix_ids] + lengths, imports,
                          parents=parents)[offset:]
        leaf(items, forward)
    return {'mode': 'tree-mask', 'batches': len(batches), 'batch_tokens': batch_tokens}


def retained_bytes_per_token(model):
    """What ``levels`` keeps per prefix token until the prefixes' backward: every layer's saved input (BF16, recomputed
    or not) and the FP32 keys and values of the full-attention layers the continuations read, with their FP32 gradients."""
    layers = model.model.layers
    full = sum(layer.self_attn.k_proj.weight.shape[0] for layer, key in zip(layers, _layer_keys(model))
               if key == ('attention', -1))
    return len(layers) * model.config.hidden_size * 2 + full * 2 * 4 * 2


def _prefix_tokens(node):
    return 0 if 'index' in node else node['end'] - node['start'] + sum(_prefix_tokens(c) for c in node['children'])


def prefix_parts(nodes, limit):
    """The forest split into parts of at most ``limit`` prefix tokens each (leaves cost nothing: they retain no graph
    past their own batch). A subtree over the limit is split among its children's parts, each part with its own copy of
    the path (exact, as ``tree_units``: each copy gets the gradient of its own continuations); a single prefix node over
    the limit is a part alone."""
    def units(node):
        if 'index' in node or _prefix_tokens(node) <= limit:
            return [node]
        own = node['end'] - node['start']
        parts = [unit for child in node['children'] for unit in units(child)]
        return [dict(node, children=group) for group in _pack(parts, _prefix_tokens, max(1, limit - own))]
    return _pack([unit for node in nodes for unit in units(node)], _prefix_tokens, max(1, limit))


def prefix_levels(nodes):
    """(levels, continuations, wholes) of a ``prefix_tree`` forest: the prefix nodes by depth (copies that get their parent
    and later their ``segment``), the leaves under a prefix as (index, start, parent) and the whole sequences as
    (index, start, None), each in tree order. Iterative on purpose: the recursive nested function this was had itself in
    its closure, a reference cycle that kept ``levels`` and through it every node's FP32 ``segment`` (7-8 GB per part) alive
    until Python's cycle collector ran, which on the learner's heap is rare: the parts of a step stacked on each other
    (policy pass 60 GB allocated, 13-19 GB of it freed only by the end-of-step ``gc.collect``, grpo-v4-run2 step 24)."""
    levels, continuations, wholes = [], [], []
    stack = [(node, 0, None) for node in reversed(nodes)]
    while stack:
        node, depth, parent = stack.pop()
        if 'index' in node:
            (wholes if parent is None else continuations).append((node['index'], node['start'], parent))
            continue
        node = dict(node, parent=parent)
        while len(levels) <= depth:
            levels.append([])
        levels[depth].append(node)
        stack.extend((child, depth + 1, node) for child in reversed(node['children']))
    return levels, continuations, wholes


def run_prefix_tree(model, nodes, ids_of, leaf, tokens, pack_tokens, activation_gb=None, backward=True,
                    mode='levels', batch_tokens=None, retained_gb=None):
    """Compute a ``prefix_tree``'s sequences packed, each shared prefix once, with exactly the whole sequences' gradients.

    Prefix nodes run level by level, each level's nodes packed into varlen batches of at most ``pack_tokens`` tokens
    that continue their parents' state (``ids_of(first, start, end)``: a node's tokens). Each node's state for its
    continuations becomes a ``Segment`` of FP32 leaves. Then the leaves run packed in tree order: the whole sequences
    (start 0) in batches of their own, the continuations in batches that attend to their prefixes' cached keys, values
    and conv inputs. ``leaf(items, forward)`` gets one batch ``items`` [(index, start)] and builds its rows (each leaf's
    tokens from its start, one document each in order, then any further documents such as pass-through producers),
    calls ``h = forward(ids, lengths, imports)``, takes its loss and, under ``backward``, its backward.
    ``tokens(index, start)``: the rows a leaf adds to a batch. With ``backward`` the continuations' gradients collect in
    the segments in FP32, and each prefix batch backpropagates them once, deepest level first; without it the prefixes'
    graphs are dropped as soon as their segments exist.
    ``activation_gb`` (None: the layers' recompute flags are left as they are) budgets saved activations: the prefixes'
    retained graphs (layer inputs and segments) come first, each leaf batch gets the rest, and what the largest leaf batch
    leaves unused keeps prefix layers unrecomputed. Returns the plan: batch counts, prefix tokens, the retained and
    leaf budgets (GB) and the prefix layer-tokens recomputed.
    ``mode`` 'tree-mask': each batch of whole subtrees (``tree_units``, at most ``batch_tokens``, default
    ``plan_tree_batches``) is one flattened forward whose tokens attend and convolve along their own path
    (``hidden(parents=...)``), with one backward (the caller's); ``leaf`` and ``forward`` are called the same way.
    'levels': the level-by-level continuations above. 'off' is the caller's (no shared prefix: a forest of leaves).
    ``retained_gb`` (levels; None or 0: unbounded) bounds what the prefixes keep until their backward
    (``retained_bytes_per_token`` per prefix token): a step's forest over it runs as ``prefix_parts``, one after the
    other, each to its own prefixes' backward (shared ancestors recomputed once per part). Unbounded, a 24-48-group GRPO
    step kept every prefix of the step at once (grpo-v4-run1 step 1: the unit passed 110 GB, DECISIONS 2026-11-18)."""
    if mode == 'levels' and retained_gb and nodes:
        limit = int(retained_gb * 2**30 // retained_bytes_per_token(model))
        if sum(_prefix_tokens(n) for n in nodes) > limit:
            parts = prefix_parts(nodes, limit)
            if len(parts) > 1:
                plans = [run_prefix_tree(model, part, ids_of, leaf, tokens, pack_tokens, activation_gb, backward, mode,
                                         batch_tokens) for part in parts]
                merged = {key: sum(p[key] for p in plans) for key in ('prefix_batches', 'leaf_batches', 'prefix_tokens',
                                                                       'recomputed_prefix_layer_tokens')}
                return {**merged, 'parts': len(parts), 'retained_gb': round(max(p['retained_gb'] for p in plans), 2),
                        'leaf_gb': min((p['leaf_gb'] for p in plans if p['leaf_gb'] is not None), default=None)}
    if mode == 'tree-mask':
        if batch_tokens is None or batch_tokens <= 0:
            batch_tokens = None if activation_gb is not None else pack_tokens       # None: plan_tree_batches
        return _run_tree_mask(model, nodes, ids_of, leaf, tokens, batch_tokens, activation_gb)
    if mode not in ('levels', 'off'):
        raise ValueError(f'prefix tree mode {mode!r}: one of {PREFIX_TREE_MODES}')
    layers = model.model.layers
    hidden_size = model.config.hidden_size
    # key (= value) width summed over the full-attention layers: the history rows a continuation's attention saves
    full = sum(layer.self_attn.k_proj.weight.shape[0] for layer, key in zip(layers, _layer_keys(model))
               if key == ('attention', -1))
    levels, continuations, wholes = prefix_levels(nodes)

    def past_of(node):
        chain = []
        while node is not None:
            chain.append(node['segment'])
            node = node['parent']
        return Past(reversed(chain))

    def history(node):
        return 0 if node is None else node['end']

    prefix_batches = [_pack(level, lambda n: n['end'] - n['start'], pack_tokens) for level in levels]
    leaf_batches = (_pack(wholes, lambda item: tokens(item[0], item[1]), pack_tokens)
                    + _pack(continuations, lambda item: tokens(item[0], item[1]), pack_tokens))
    bytes_gb = ACTIVATION_BYTES_PER_TOKEN_LAYER * len(layers) / 2**30

    def effective(batch):       # leaf rows plus the history keys and values its attention layers save, in token units
        rows = sum(tokens(index, start) for index, start, _ in batch)
        past = sum(history(parent) for _, _, parent in batch)
        return rows + past * full * 2 * 2 / (ACTIVATION_BYTES_PER_TOKEN_LAYER * len(layers))
    prefix_tokens = sum(n['end'] - n['start'] for level in levels for n in level)
    retained_gb = prefix_tokens * retained_bytes_per_token(model) / 2**30
    leaf_gb = None if activation_gb is None else max(0., activation_gb - retained_gb)
    spare_gb = None if activation_gb is None else max(
        0., leaf_gb - max([effective(b) for b in leaf_batches] + [0]) * bytes_gb)
    grad = torch.is_grad_enabled()
    compiled = getattr(model, '_lfm3_fast', None) is not None and model._lfm3_fast['compile']
    graphs, recomputed = [], 0
    for level in prefix_batches:
        for batch in level:
            lengths = [n['end'] - n['start'] for n in batch]
            if spare_gb is not None:
                count = set_checkpointing(model, sum(lengths), spare_gb)
                recomputed += count * sum(lengths)
                spare_gb = max(0., spare_gb - (len(layers) - count) * sum(lengths) * ACTIVATION_BYTES_PER_TOKEN_LAYER
                               / 2**30)
            ids = torch.cat([ids_of(n['first'], n['start'], n['end']) for n in batch])
            pasts = [past_of(n['parent']) for n in batch]
            _, segments = _hidden_past(model, ids, lengths, pasts, compiled, keep=True)
            outputs = []
            for node, segment in zip(batch, segments):
                outputs += segment.tensors()
                # the continuations read detached FP32 leaves (requiring grad under grad mode even without a backward:
                # they then compile the policy's own training graph, which the KL reference pass must round as)
                node['segment'] = Segment(segment.length, [tuple(t.detach().requires_grad_(grad) for t in layer)
                                                           for layer in segment.layers])
            graphs.append((outputs, [t for n in batch for t in n['segment'].tensors()]) if backward and grad else None)
    for batch in leaf_batches:
        items = [(index, start) for index, start, _ in batch]
        pasts = [Past() if parent is None else past_of(parent) for _, _, parent in batch]

        def forward(ids, lengths=None, imports=None, pasts=pasts, batch=batch):
            if leaf_gb is not None:
                set_checkpointing(model, effective(batch), leaf_gb)
            return hidden(model, ids, lengths, imports, pasts=pasts)
        leaf(items, forward)
    for entry in reversed(graphs):
        if entry is None:
            continue
        outputs, held = entry
        pairs = [(t, h.grad) for t, h in zip(outputs, held) if h.grad is not None]
        if pairs:
            torch.autograd.backward([t for t, _ in pairs], [g for _, g in pairs])
    return {'prefix_batches': sum(len(level) for level in prefix_batches), 'leaf_batches': len(leaf_batches),
            'prefix_tokens': prefix_tokens, 'retained_gb': round(retained_gb, 2),
            'leaf_gb': None if leaf_gb is None else round(leaf_gb, 2),
            'recomputed_prefix_layer_tokens': recomputed}


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
        _CE[name] = _compile({'forward': _ce_forward, 'backward': _ce_backward}[name])
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
NUCLEUS_WIDE_K = 16384   # second top-k for rows whose nucleus reaches past NUCLEUS_TOP_K (before a full sort)
NUCLEUS_CHUNK = 2048    # rows per head chunk of the nucleus NLL: it holds about twice the fp32 temporaries of the plain CE


def _cut_of(values, row_lse, top_p):
    """The cut logit of rows whose ``values`` are sorted descending and hold the whole nucleus."""
    p = torch.exp(values - row_lse[:, None])
    count = ((torch.cumsum(p, -1) - p) <= top_p).sum(-1).clamp(min=1)
    return values.gather(1, (count - 1)[:, None])[:, 0]


def _nucleus_cut_lse(lf, lse, top_p, top_k, wide_k=None):
    """Per row of fp32 logits ``lf`` (log-sum-exp ``lse``): the logit of the last token of the top-p nucleus (tokens by
    decreasing probability, a token kept while the mass before it is <= ``top_p``: SGLang's rule,
    ``top_k_top_p_min_p_sampling_from_probs_torch``; the nucleus is every token with a logit >= this cut, ties with the
    last kept token included) and the log-sum-exp over that nucleus. Only the ``top_k`` largest logits are sorted and
    summed; a row whose nucleus may reach past them takes the ``wide_k`` largest, and past those the full sort. A full
    128k sort costs ~10x a top-k and a masked full-vocab log-sum-exp ~2x (GB10, 2026-11-13), so both run only on rows
    that need them."""
    vocab = lf.shape[-1]
    values = lf.topk(min(top_k, vocab), dim=-1).values.float()   # bf16 logits: the same values, exact in fp32
    cut = _cut_of(values, lse, top_p)
    inside = torch.logsumexp(values.masked_fill(values < cut[:, None], float('-inf')), dim=-1)
    if values.shape[-1] < vocab:
        wide = (cut <= values[:, -1]).nonzero()[:, 0]       # the nucleus (or a tie at its edge) may reach past top_k
        if wide.numel():
            rows, row_lse = lf[wide].float(), lse[wide]
            more = rows.topk(min(wide_k or NUCLEUS_WIDE_K, vocab), dim=-1).values
            wide_cut = _cut_of(more, row_lse, top_p)
            if more.shape[-1] < vocab:
                wider = (wide_cut <= more[:, -1]).nonzero()[:, 0]
                if wider.numel():
                    wide_cut[wider] = _cut_of(rows[wider].sort(dim=-1, descending=True).values, row_lse[wider], top_p)
            cut[wide] = wide_cut
            inside[wide] = torch.logsumexp(rows.masked_fill(rows < wide_cut[:, None], float('-inf')), dim=-1)
    return cut, inside


def _nucleus_cut(lf, lse, top_p, top_k):
    """The cut logit of ``_nucleus_cut_lse``."""
    return _nucleus_cut_lse(lf, lse, top_p, top_k)[0]


def _nucleus_logit_grad(logits, target, lse, set_lse, cut, grad, grad_nucleus):
    """d loss / d logits of one chunk: grad * (softmax - onehot) + grad_nucleus * (softmax over the set - onehot), in fp32
    from the upcast logits and returned in the logits' dtype. Upcast and downcast are inside the (compiled) kernel: done
    outside, a chunk's fp32 logits and fp32 gradient (1 GB each at 2048 rows of a 128k vocabulary) were both live at the
    learner's activation peak (grpo-v2-run2, 2026-11-15). The values are the same: both casts are exact or round once."""
    lf = logits.float()
    onehot = torch.arange(lf.shape[-1], device=lf.device)[None, :] == target[:, None]
    kept = (lf >= cut[:, None]) | onehot
    d = torch.exp(lf - lse[:, None]) * grad[:, None] + torch.where(kept, torch.exp(lf - set_lse[:, None]), 0.) * grad_nucleus[:, None]
    return (d - onehot * (grad + grad_nucleus)[:, None]).to(logits.dtype)


_NUCLEUS_GRAD = {}


def _nucleus_grad_kernel(device_type):
    """``_nucleus_logit_grad`` compiled (one fused pass over the chunk's logits instead of about eight)."""
    if device_type != 'cuda':
        return _nucleus_logit_grad
    if 'cuda' not in _NUCLEUS_GRAD:
        _NUCLEUS_GRAD['cuda'] = _compile(_nucleus_logit_grad)
    return _NUCLEUS_GRAD['cuda']


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
        forward_ce = _ce('forward') if h.is_cuda else _ce_forward
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            # The log-sum-exp and NLL are linear_nll's own kernel on the bf16 logits (the KL reference pass computes its
            # log-probabilities with it, so at equal weights the two agree bit for bit); the nucleus reads the bf16 logits
            # (their fp32 values are exact). No fp32 copy of the chunk's logits is written: that copy and the eager
            # passes over it cost ~3x the head matmul (GB10, 2026-11-17).
            logits = F.linear(h[start:stop], weight)
            own = logits.gather(-1, target[start:stop, None])[:, 0].float()
            lse[start:stop], nll[start:stop] = forward_ce(logits, target[start:stop])
            cut[start:stop], inside = _nucleus_cut_lse(logits, lse[start:stop], top_p, top_k)
            outside[start:stop] = own < cut[start:stop]
            set_lse[start:stop] = torch.where(outside[start:stop], torch.logaddexp(inside, own), inside)
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
            d = _nucleus_grad_kernel(rows.device.type)(F.linear(rows, weight), target[start:stop], lse[start:stop],
                                                       set_lse[start:stop], cut[start:stop], grad[start:stop],
                                                       grad_nucleus[start:stop])
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
