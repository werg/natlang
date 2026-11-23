"""Fast LFM3 training forward/backward (the HF modelling arithmetic, fewer and fused kernels), adapter-aware factored
projections, trainable sparse Neuralese rows, packed documents, fused linear cross-entropy and a no-grad log-prob pass.

    model = load an HF `Lfm3ForCausalLM` in bf16; prepare(model)
    h = hidden(model, ids)                         # (T, D) for one sequence
    h = hidden(model, ids, lengths=[n1, n2, ...])  # several documents packed into one row, each attending only itself
    nll = linear_nll(h[positions], model.lm_head.weight, targets)        # fp32 per-token NLL, logits never stored
    logp = token_logprobs(model, ids, positions, targets, lengths=...)   # no grad (GRPO old-policy / reference pass)
    nll, nucleus_nll, outside = linear_nll_nucleus(h[positions], weight, targets, top_p)   # + NLL under the top-p nucleus
    h = hidden(model, ids, imports=import_tensors(model, [(start, block)], device))   # imported native state spliced in
    run_prefix_tree(model, prefix_tree(seqs, caps, 256), ids_of, leaf, tokens, pack_tokens, activation_gb)
                                                   # shared prompts computed once, every batch packed, exact grads

Adapter bundles (`LayerAdapterBundle`, flat leaf tuples `(a, b, scale)`) are passed explicitly through all four fast
layer paths (`conv_layer`, `attention_layer`, `conv_layer_past`, `attention_layer_past`) and MLPs (`_ffn`), flat through
`torch.utils.checkpoint(..., use_reentrant=False)` so multi-child prefix-tree backwards never hit freed-graph errors and
`torch.compile(fullgraph=True)` traces without graph breaks.

Blackwell optimizations (`lfm3opti.md` §6):
- Attention uses FlashAttention/varlen_attn (cuDNN SDPA fallback on sm_120+).
- Causal-conv launch configuration adapts to GPU SM count (`causal_conv_launch_config`).
- Optional token-length bucketing (`TOKEN_BUCKETS`, `bucket_tokens`, opt-out `NATLANG_COMPILE_BUCKETS=0`) and compile
  health tracking (`CompileHealthTracker`: compile seconds, recompile counts, graph breaks).
- Optional FP8 frozen-base GEMMs (`fp8_base=True` or `NATLANG_FP8_BASE=1`; default off: DECISIONS "LoRA core fixes").

The vendored HF sources are hash-pinned (`third_party/lfm3`), so everything here reads instances, never patches files.
"""
from __future__ import annotations

import os as _os
import time as _time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from ..model.native_adapters import factored_linear, fp8_base_default

LOGIT_CHUNK = 4096

# Inductor options of every compile here (layers, the cross-entropy and nucleus kernels). ``emulate_precision_casts``:
# keep eager's BF16 rounding after each BF16 op; by default Inductor computes a fused chain of BF16 ops in FP32 and rounds
# only what it stores, so the rounding points depended on which values a graph stores: a no-grad (inference) graph stores
# fewer than a training graph, and compiled no-grad log-probabilities drifted from eager and from the grad-mode graph
# (2026-11-17). ``deterministic``: no per-process benchmarking of reduction configs, whose choice changes summation order
# (the same compiled forward differed between processes). Opt-out: NATLANG_FAST_TRAIN_INDUCTOR=fast (Inductor defaults).
COMPILE_OPTIONS = ({} if _os.environ.get('NATLANG_FAST_TRAIN_INDUCTOR') == 'fast'
                   else {'emulate_precision_casts': True, 'deterministic': True})

# Token-length buckets for torch.compile shape stability (lfm3opti.md §6 item 2; NATLANG_COMPILE_BUCKETS=0 opts out).
TOKEN_BUCKETS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768)


def bucket_tokens(length: int, buckets: tuple[int, ...] = TOKEN_BUCKETS) -> int:
    """Smallest bucket >= ``length`` (or ``length`` when bucketing is opted out or above the largest bucket)."""
    if _os.environ.get('NATLANG_COMPILE_BUCKETS', '1') == '0':
        return int(length)
    for b in buckets:
        if length <= b:
            return b
    return int(length)


def blackwell_attention_config(device=None) -> dict:
    """Attention kernel configuration for the active CUDA device (sm_120 G4 / sm_121 GB10 / fallback)."""
    if not torch.cuda.is_available():
        return {'sm': None, 'backend': 'cpu_sdpa', 'varlen_gqa': bool(_GQA), 'cudnn_sdpa': False}
    dev = torch.device(device) if device is not None else torch.device('cuda', torch.cuda.current_device())
    if dev.type != 'cuda':
        return {'sm': None, 'backend': 'cpu_sdpa', 'varlen_gqa': bool(_GQA), 'cudnn_sdpa': False}
    major, minor = torch.cuda.get_device_capability(dev)
    sm = major * 10 + minor
    cudnn_ok = hasattr(torch.backends.cuda, 'cudnn_sdp_enabled') and torch.backends.cuda.cudnn_sdp_enabled()
    return {'sm': sm, 'backend': 'varlen_flash_sm120' if major >= 12 else 'varlen_flash',
            'varlen_gqa': bool(_GQA), 'cudnn_sdpa': bool(cudnn_ok), 'is_blackwell': major >= 12}


def causal_conv_launch_config(device=None) -> dict:
    """SM-count-aware launch parameters for LFM3's depthwise causal convolution on G4 (~188 SMs) vs GB10 (~48 SMs)."""
    if not torch.cuda.is_available():
        return {'device': 'cpu', 'sm_count': 0, 'tile_tokens': 128, 'warps': 4, 'unroll_taps': True}
    dev = torch.device(device) if device is not None else torch.device('cuda', torch.cuda.current_device())
    if dev.type != 'cuda':
        return {'device': str(dev), 'sm_count': 0, 'tile_tokens': 128, 'warps': 4, 'unroll_taps': True}
    props = torch.cuda.get_device_properties(dev)
    sm_count = int(props.multi_processor_count)
    # High-SM discrete Blackwell (G4: RTX PRO 6000 Blackwell) benefits from smaller token tiles per CTA to saturate SMs;
    # GB10 (fewer SMs, unified memory) prefers larger tiles per CTA to amortize memory fetches.
    tile_tokens = 64 if sm_count >= 128 else 128 if sm_count >= 64 else 256
    warps = 8 if sm_count >= 128 else 4
    return {'device': props.name, 'sm_count': sm_count, 'sm': props.major * 10 + props.minor,
            'tile_tokens': tile_tokens, 'warps': warps, 'unroll_taps': True}


class CompileHealthTracker:
    """Tracks torch.compile compile time, recompile count and graph breaks via ``torch._dynamo.utils.counters``."""

    def __init__(self):
        self._start_frames = self._read_counters()
        self.compile_seconds = 0.0
        self.steps = 0

    @staticmethod
    def _read_counters() -> dict:
        try:
            counters = torch._dynamo.utils.counters
            return {
                'frames_total': int(counters.get('frames', {}).get('total', 0)),
                'frames_ok': int(counters.get('frames', {}).get('ok', 0)),
                'graph_breaks': int(sum(counters.get('graph_break', {}).values())),
                'recompiles': int(sum(counters.get('recompiles', {}).values())),
            }
        except Exception:
            return {'frames_total': 0, 'frames_ok': 0, 'graph_breaks': 0, 'recompiles': 0}

    def snapshot(self) -> dict:
        now = self._read_counters()
        return {k: max(0, now[k] - self._start_frames[k]) for k in now}

    def record_step(self, step_seconds: float, *, is_warmup: bool = False) -> dict:
        self.steps += 1
        snap = self.snapshot()
        if is_warmup or self.steps == 1:
            self.compile_seconds += step_seconds
        return {**snap, 'compile_seconds': round(self.compile_seconds, 2)}


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


def _dense_attend(q, k, v, mask, scale):
    """The CPU reference attention under an explicit bool mask; grouped-query heads repeated explicitly (the same
    arithmetic as ``enable_gqa``, which torch.compile cannot pass a symbolic head count to)."""
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
    out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                         attn_mask=mask, scale=scale)
    return out[0].transpose(0, 1)


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
        return _dense_attend(q, k, v, tree_mask, scale)
    cu, max_len, doc, pos = docs
    if q.is_cuda:
        from torch.nn.attention.varlen import varlen_attn
        return varlen_attn(q, k, v, cu, cu, max_len, max_len, scale=scale, window_size=(window, 0), **_GQA)
    # reference path (CPU tests): the same key sets as an explicit mask
    i = torch.arange(q.shape[0], device=q.device)
    mask = (doc[:, None] == doc[None, :]) & (i[None, :] <= i[:, None])
    if window >= 0:
        mask &= i[:, None] - i[None, :] <= window
    return _dense_attend(q, k, v, mask, scale)


def _passed(t, source, frozen):
    """The producer span's own rows of ``t`` for a pass-through import; rows marked in ``frozen`` (bool, one per row, or
    None) are detached: their state is fixed context with the same values."""
    value = t[source]
    if frozen is None:
        return value
    return torch.where(frozen.view(-1, *([1] * (value.dim() - 1))), value.detach(), value)


def _ffn(x, ffn_norm, w1, w3, w2, post_ffn_norm, eps,
         w1_a=None, w1_b=None, w1_s: float = 1.0,
         w3_a=None, w3_b=None, w3_s: float = 1.0,
         w2_a=None, w2_b=None, w2_s: float = 1.0,
         fp8_base='bf16'):
    f = _rms(x, ffn_norm, eps)
    gate = factored_linear(f, w1, w1_a, w1_b, w1_s, fp8_base=fp8_base)
    up = factored_linear(f, w3, w3_a, w3_b, w3_s, fp8_base=fp8_base)
    f = factored_linear(F.silu(gate) * up, w2, w2_a, w2_b, w2_s, fp8_base=fp8_base)
    if post_ffn_norm is not None:
        f = _rms(f, post_ffn_norm, eps)
    return x + f


def conv_layer(x, pos, operator_norm, in_proj, conv_weight, out_proj, post_operator_norm,
               ffn_norm, w1, w3, w2, post_ffn_norm, eps: float,
               in_a=None, in_b=None, in_s: float = 1.0,
               out_a=None, out_b=None, out_s: float = 1.0,
               w1_a=None, w1_b=None, w1_s: float = 1.0,
               w3_a=None, w3_b=None, w3_s: float = 1.0,
               w2_a=None, w2_b=None, w2_s: float = 1.0,
               fp8_base='bf16',
               import_rows=None, import_bx=None,
               pass_rows=None, pass_source=None, pass_frozen=None, pred=None):
    """`Lfm3DecoderLayer` with `Lfm3ShortConv` on (T, D): depthwise width-7 causal conv with adapter-aware projections."""
    length = x.shape[0]
    taps = conv_weight.shape[-1]
    h = _rms(x, operator_norm, eps)
    b, c, xx = factored_linear(h, in_proj, in_a, in_b, in_s, fp8_base=fp8_base).chunk(3, dim=-1)
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
    y = factored_linear(c * acc.to(bx.dtype), out_proj, out_a, out_b, out_s, fp8_base=fp8_base)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps,
                w1_a, w1_b, w1_s, w3_a, w3_b, w3_s, w2_a, w2_b, w2_s, fp8_base)


def attention_layer(x, cos, sin, cu, max_len: int, doc, pos, operator_norm, wq, wk, wv, wo, wg, q_norm, k_norm,
                    post_operator_norm, ffn_norm, w1, w3, w2, post_ffn_norm, eps: float, head_dim: int, window: int,
                    q_a=None, q_b=None, q_s: float = 1.0,
                    k_a=None, k_b=None, k_s: float = 1.0,
                    v_a=None, v_b=None, v_s: float = 1.0,
                    o_a=None, o_b=None, o_s: float = 1.0,
                    g_a=None, g_b=None, g_s: float = 1.0,
                    w1_a=None, w1_b=None, w1_s: float = 1.0,
                    w3_a=None, w3_b=None, w3_s: float = 1.0,
                    w2_a=None, w2_b=None, w2_s: float = 1.0,
                    fp8_base='bf16',
                    import_rows=None, import_k=None, import_v=None, pass_rows=None, pass_source=None,
                    pass_frozen=None, tree_mask=None):
    """`Lfm3DecoderLayer` with `Lfm3Attention` on (T, D): QK-norm, fp32 RoPE, flash attention, sigmoid head gate,
    with explicit flat LoRA factor leaves."""
    length = x.shape[0]
    h = _rms(x, operator_norm, eps)
    q = _rms(factored_linear(h, wq, q_a, q_b, q_s, fp8_base=fp8_base).view(length, -1, head_dim), q_norm, eps)
    k = _rms(factored_linear(h, wk, k_a, k_b, k_s, fp8_base=fp8_base).view(length, -1, head_dim), k_norm, eps)
    v = factored_linear(h, wv, v_a, v_b, v_s, fp8_base=fp8_base).view(length, -1, head_dim)
    if import_rows is not None:
        k = k.index_put((import_rows,), import_k)
        v = v.index_put((import_rows,), import_v)
    if pass_rows is not None:
        k = k.index_put((pass_rows,), _passed(k, pass_source, pass_frozen))
        v = v.index_put((pass_rows,), _passed(v, pass_source, pass_frozen))
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    o = _attend(q, k, v, (cu, max_len, doc, pos), window, tree_mask)
    o = (o * torch.sigmoid(factored_linear(h, wg, g_a, g_b, g_s, fp8_base=fp8_base))[..., None]).reshape(length, -1)
    y = factored_linear(o, wo, o_a, o_b, o_s, fp8_base=fp8_base)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps,
                w1_a, w1_b, w1_s, w3_a, w3_b, w3_s, w2_a, w2_b, w2_s, fp8_base)


def _attend_past(q, k, v, docs, window):
    """Causal attention of packed documents' queries over their keys (each document's history rows, then its own)."""
    scale = q.shape[-1] ** -0.5
    cu_q, cu_k, max_q, max_k, q_doc, q_pos, k_doc, k_pos = docs
    if q.is_cuda:
        from torch.nn.attention.varlen import varlen_attn
        return varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k, scale=scale, window_size=(window, 0), **_GQA)
    mask = (q_doc[:, None] == k_doc[None, :]) & (k_pos[None, :] <= q_pos[:, None])
    if window >= 0:
        mask &= q_pos[:, None] - k_pos[None, :] <= window
    return _dense_attend(q, k, v, mask, scale)


def conv_layer_past(x, src_bx, past_index, joined_order, own_rows, operator_norm, in_proj, conv_weight, out_proj,
                    post_operator_norm, ffn_norm, w1, w3, w2, post_ffn_norm, eps: float,
                    in_a=None, in_b=None, in_s: float = 1.0,
                    out_a=None, out_b=None, out_s: float = 1.0,
                    w1_a=None, w1_b=None, w1_s: float = 1.0,
                    w3_a=None, w3_b=None, w3_s: float = 1.0,
                    w2_a=None, w2_b=None, w2_s: float = 1.0,
                    fp8_base='bf16'):
    """``conv_layer`` for packed documents that continue computed histories (``Past``) with explicit flat LoRA leaves."""
    taps = conv_weight.shape[-1]
    h = _rms(x, operator_norm, eps)
    b, c, xx = factored_linear(h, in_proj, in_a, in_b, in_s, fp8_base=fp8_base).chunk(3, dim=-1)
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
    y = factored_linear(c * acc.to(bx.dtype), out_proj, out_a, out_b, out_s, fp8_base=fp8_base)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps,
                w1_a, w1_b, w1_s, w3_a, w3_b, w3_s, w2_a, w2_b, w2_s, fp8_base), bx


def attention_layer_past(x, cos, sin, src_k, src_v, past_index, key_order, cu_q, cu_k, max_q: int, max_k: int, q_doc,
                         q_pos, k_doc, k_pos, operator_norm, wq, wk, wv, wo, wg, q_norm, k_norm, post_operator_norm,
                         ffn_norm, w1, w3, w2, post_ffn_norm, eps: float, head_dim: int, window: int,
                         q_a=None, q_b=None, q_s: float = 1.0,
                         k_a=None, k_b=None, k_s: float = 1.0,
                         v_a=None, v_b=None, v_s: float = 1.0,
                         o_a=None, o_b=None, o_s: float = 1.0,
                         g_a=None, g_b=None, g_s: float = 1.0,
                         w1_a=None, w1_b=None, w1_s: float = 1.0,
                         w3_a=None, w3_b=None, w3_s: float = 1.0,
                         w2_a=None, w2_b=None, w2_s: float = 1.0,
                         fp8_base='bf16'):
    """``attention_layer`` for packed documents that continue computed histories with explicit flat LoRA leaves."""
    length = x.shape[0]
    h = _rms(x, operator_norm, eps)
    q = _rms(factored_linear(h, wq, q_a, q_b, q_s, fp8_base=fp8_base).view(length, -1, head_dim), q_norm, eps)
    k = _rms(factored_linear(h, wk, k_a, k_b, k_s, fp8_base=fp8_base).view(length, -1, head_dim), k_norm, eps)
    v = factored_linear(h, wv, v_a, v_b, v_s, fp8_base=fp8_base).view(length, -1, head_dim)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    keys = torch.cat([k, src_k.index_select(0, past_index).to(k.dtype)]).index_select(0, key_order)
    values = torch.cat([v, src_v.index_select(0, past_index).to(v.dtype)]).index_select(0, key_order)
    o = _attend_past(q, keys, values, (cu_q, cu_k, max_q, max_k, q_doc, q_pos, k_doc, k_pos), window)
    o = (o * torch.sigmoid(factored_linear(h, wg, g_a, g_b, g_s, fp8_base=fp8_base))[..., None]).reshape(length, -1)
    y = factored_linear(o, wo, o_a, o_b, o_s, fp8_base=fp8_base)
    if post_operator_norm is not None:
        y = _rms(y, post_operator_norm, eps)
    return _ffn(x + y, ffn_norm, w1, w3, w2, post_ffn_norm, eps,
                w1_a, w1_b, w1_s, w3_a, w3_b, w3_s, w2_a, w2_b, w2_s, fp8_base), k, v


_COMPILED = {}
RECOMPILE_LIMIT = 64


def _kernel(name, compiled):
    function = {'conv': conv_layer, 'attention': attention_layer,
                'conv_past': conv_layer_past, 'attention_past': attention_layer_past}[name]
    if not compiled:
        return function
    if name not in _COMPILED:
        # One shared graph per layer kind specializes on window (full/sliding), grad mode, imported rows, packing,
        # adapter presence and prefix-tree passes: more than dynamo's default 8 variants.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, RECOMPILE_LIMIT)
        _COMPILED[name] = _compile(function, fullgraph=True)
    return _COMPILED[name]


def _weight(module):
    return None if module is None else module.weight


def _leaf_triple(leaf):
    return (None, None, 1.0) if leaf is None else (leaf.a, leaf.b, float(leaf.scale))


def _bundle_args(kind: str, bundle, fp8_base) -> tuple:
    """Flatten a ``LayerAdapterBundle`` (or None) into explicit positional leaf arguments for compiled/checkpointed layers."""
    if bundle is None:
        if kind == 'conv':
            return (None, None, 1.0) * 5 + (fp8_base,)
        return (None, None, 1.0) * 8 + (fp8_base,)
    if kind == 'conv':
        return (*_leaf_triple(bundle.in_proj), *_leaf_triple(bundle.out_proj),
                *_leaf_triple(bundle.w1), *_leaf_triple(bundle.w3), *_leaf_triple(bundle.w2),
                fp8_base)
    return (*_leaf_triple(bundle.q_proj), *_leaf_triple(bundle.k_proj), *_leaf_triple(bundle.v_proj),
            *_leaf_triple(bundle.o_proj), *_leaf_triple(getattr(bundle, 'gate_proj', None)),
            *_leaf_triple(bundle.w1), *_leaf_triple(bundle.w3), *_leaf_triple(bundle.w2),
            fp8_base)


def _resolve_adapters(model, adapters):
    """Explicit ``adapters`` win; ``()`` means the bare base; ``None`` means the model's current adapters (which
    ``Lfm3Learner.reference_view`` swaps for the reference while it is active)."""
    if adapters is None:
        adapters = getattr(model, '_lfm3_adapters', None)
    if adapters is None or len(adapters) == 0:
        return None
    return adapters


def _resolve_trainable_rows(model, trainable_rows):
    if trainable_rows is False:
        return None
    if trainable_rows is not None:
        return trainable_rows
    return getattr(model, '_lfm3_trainable_rows', None)


def _resolve_fp8(model, fp8_base):
    """The frozen-base GEMM mode passed to every ``factored_linear`` (``native_adapters.BASE_MODES``): ``'fp8'`` when FP8
    is on (explicit ``fp8_base`` or the prepared default); ``'f32acc'`` for a ``lora+rows`` model, so the bare base (a
    reference view) and B = 0 factors round the same FP32 accumulator once and agree exactly; else ``'bf16'``."""
    fast = getattr(model, '_lfm3_fast', None) or {}
    use = bool(fp8_base) if fp8_base is not None else bool(fast.get('fp8_base', False))
    if use:
        return 'fp8'
    return 'f32acc' if fast.get('parameterization') == 'lora+rows' else 'bf16'


def _apply_trainable_embed_rows(x: torch.Tensor, ids: torch.Tensor, trainable_rows) -> torch.Tensor:
    """Differentiably splice FP32 trainable Neuralese marker/override rows into input embeddings."""
    if trainable_rows is None:
        return x
    token_ids = trainable_rows.token_ids
    rows = trainable_rows.rows
    if token_ids.numel() == 0:
        return x
    if token_ids.device != ids.device:
        token_ids = token_ids.to(ids.device)
    matches = ids[:, None] == token_ids[None, :]
    hit = matches.any(dim=-1)
    if not bool(hit.any()):
        return x
    pos = hit.nonzero(as_tuple=False)[:, 0]
    which = matches[pos].to(torch.int64).argmax(dim=-1)
    return x.index_put((pos,), rows.index_select(0, which).to(x.dtype))


def _layer_call(layer, compiled):
    """(kind, static args before adapter/x-dependent ones) for one HF decoder layer."""
    post_operator = _weight(getattr(layer, 'post_operator_norm', None))
    post_ffn = _weight(getattr(layer, 'post_ffn_norm', None))
    eps = layer.operator_norm.variance_epsilon
    mlp = layer.feed_forward
    ffn = (layer.ffn_norm.weight, mlp.w1.weight, mlp.w3.weight, mlp.w2.weight, post_ffn, eps)
    if not layer.is_attention_layer:
        conv = layer.conv
        if conv.bias:
            raise ValueError('lfm3_fast: conv_bias is not supported')
        return 'conv', (layer.operator_norm.weight, conv.in_proj.weight, conv.conv.weight, conv.out_proj.weight,
                        post_operator) + ffn
    attn = layer.self_attn
    window = -1 if attn.sliding_window is None else attn.sliding_window - 1
    return 'attention', ((layer.operator_norm.weight, attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight,
                          attn.o_proj.weight, attn.gate_proj.weight, attn.q_norm.weight, attn.k_norm.weight,
                          post_operator, layer.ffn_norm.weight, mlp.w1.weight, mlp.w3.weight, mlp.w2.weight, post_ffn,
                          eps, attn.head_dim, window))


def prepare(model, compile_layers=True, *, fp8_base: bool | None = None, parameterization: str | None = None):
    """Mark a loaded HF `Lfm3ForCausalLM` for the fast path (`hidden`, `token_logprobs`, fused `linear_nll`).
    The HF modules and parameters are untouched. Returns the model."""
    for layer in model.model.layers:
        _layer_call(layer, False)                       # refuse unsupported configurations now
    device = model.model.embed_tokens.weight.device
    use_fp8 = fp8_base_default(device) if fp8_base is None else bool(fp8_base)
    # Only enable FP8 frozen-base GEMMs when the base backbone is actually frozen (LoRA) or explicitly requested.
    if fp8_base is None and any(p.requires_grad for p in model.model.layers.parameters()):
        use_fp8 = False
    param_mode = parameterization_key(parameterization) or (
        'lora+rows' if getattr(model, '_lfm3_adapters', None) is not None else 'full')
    if _os.environ.get('NATLANG_BF16_REDUCED_REDUCTION') != '1':
        # cuBLAS may otherwise reduce split-K partial sums of BF16-output GEMMs (head logits, low-rank branch) in BF16
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    import weakref
    model.lm_head.weight._lfm3_owner = weakref.ref(model)     # linear_nll's default trainable rows (_head_rows)
    model._lfm3_fast = {
        'compile': bool(compile_layers),
        'fp8_base': bool(use_fp8),
        'parameterization': param_mode,
        'attention_config': blackwell_attention_config(device),
        'conv_config': causal_conv_launch_config(device),
    }
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
    {'rows': int64 (n,), 'layers': [per layer (k, v) for attention, (bx,) for conv, in the model's dtype]}, or None."""
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
    """Producer documents and gather indices for the imports of one packed sequence of ``total`` tokens."""
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
                fixed.extend((end + int(at), state) for at, state in spec.get('imports') or ())
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
    """One prefix node's own rows of the state its continuations read, per layer (FP32 leaves)."""

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
    """Index tensors and FP32 history sources of one packed continuation forward."""
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


def _hidden_past(model, ids, lengths, pasts, compiled, keep, *, adapters=None, trainable_rows=None, fp8_base=None):
    """``hidden`` of packed documents continuing their ``pasts`` -> (final hidden or None, segments or None)."""
    config = model.model.config
    device = ids.device
    inputs = _past_inputs(model, pasts, lengths, device)
    x = _apply_trainable_embed_rows(model.model.embed_tokens(ids), ids, _resolve_trainable_rows(model, trainable_rows))
    tables = _rope_tables(model, inputs['q_pos'], x.dtype)
    grad = torch.is_grad_enabled()
    bundles = _resolve_adapters(model, adapters)
    use_fp8 = _resolve_fp8(model, fp8_base)
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
        bundle = None if bundles is None else bundles[index]
        adapter_flat = _bundle_args(kind, bundle, use_fp8)
        if kind == 'conv':
            args = (x, inputs['sources'][index][0], plan['index'], plan['order'], plan['own']) + static + adapter_flat
        else:
            cos, sin = tables[config.layer_types[index]]
            args = ((x, cos, sin) + inputs['sources'][index] + (plan['index'], plan['order'], inputs['cu_q'],
                    plan['cu_k'], inputs['max_q'], plan['max_k'], inputs['q_doc'], inputs['q_pos'], plan['k_doc'],
                    plan['k_pos']) + static + adapter_flat)
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
    """A flattened prefix tree: documents back to back (``lengths``), document d continuing the path of ``parents[d]``."""
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
            for back in range(offset + 1, taps):
                target, a = starts[d] + offset - back, parents[d]
                while a >= 0 and target < starts[a]:
                    a = parents[a]
                pred[back - 1, rows[d] + offset] = total if a < 0 or target < 0 else rows[a] + target - starts[a]
    return pos.astype(np.int64), doc, pred, ancestors


TREE_BLOCK = 128


def _tree_block_mask(pos, doc, ancestors, window, mask_mod, block=TREE_BLOCK):
    """The flex-attention ``BlockMask`` of a path mask, classified from the tree's structure."""
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
    whole[-1] = total % block == 0
    full &= possible & whole[:, None] & whole[None, :]
    partial = possible & ~full

    def listed(tiles):
        order = torch.argsort((~tiles).to(torch.int8), dim=-1, stable=True).to(torch.int32)
        return tiles.sum(-1).to(torch.int32)[None, None], order[None, None]
    return BlockMask.from_kv_blocks(*listed(partial), *listed(full), BLOCK_SIZE=block, mask_mod=mask_mod,
                                    seq_lengths=(total, total))


def tree_masks(pos, doc, ancestors, windows, device):
    """Per attention window (-1: full): the path mask of a flattened tree."""
    pos, doc, ancestors = (torch.as_tensor(t, device=device) for t in (pos, doc, ancestors))
    total = pos.shape[0]
    masks = {}

    def path_mask(window):
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


def hidden(model, ids, lengths=None, imports=None, pasts=None, parents=None, *,
           adapters=None, trainable_rows=None, fp8_base=None):
    """Final hidden states (T, D) of ``ids`` (1-D), with optional packing, imports, pasts, flattened tree parents,
    adapter bundles (``adapters``), trainable Neuralese rows (``trainable_rows``), and Blackwell FP8 frozen base."""
    config = model.model.config
    fast = getattr(model, '_lfm3_fast', None)
    compiled = fast is not None and fast['compile']
    if pasts is not None and any(pasts):
        if imports is not None:
            raise ValueError('hidden: continuations of a past carry no imported state')
        lengths = [int(ids.shape[0])] if lengths is None else [int(n) for n in lengths]
        pasts = [Past(p or ()) for p in pasts] + [Past()] * (len(lengths) - len(pasts))
        extra = {k: v for k, v in (('adapters', adapters), ('trainable_rows', trainable_rows),
                                   ('fp8_base', fp8_base)) if v is not None}
        return _hidden_past(model, ids, lengths, pasts, compiled, keep=False, **extra)[0]
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
        pos = torch.from_numpy(tree_pos).to(ids.device)
    x = _apply_trainable_embed_rows(model.model.embed_tokens(ids), ids, _resolve_trainable_rows(model, trainable_rows))
    tables = _rope_tables(model, pos, x.dtype)
    grad = torch.is_grad_enabled()
    bundles = _resolve_adapters(model, adapters)
    use_fp8 = _resolve_fp8(model, fp8_base)
    for index, layer in enumerate(model.model.layers):
        kind, static = _layer_call(layer, compiled)
        function = _kernel(kind, compiled)
        bundle = None if bundles is None else bundles[index]
        adapter_flat = _bundle_args(kind, bundle, use_fp8)
        if kind == 'conv':
            args = (x, pos) + static + adapter_flat
        else:
            cos, sin = tables[config.layer_types[index]]
            args = (x, cos, sin, cu, max_len, doc, pos) + static + adapter_flat
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
    """A compressed trie over token sequences (numpy 1-D), so that a prefix several of them share is computed once."""
    order = sorted(range(len(seqs)), key=lambda i: seqs[i].tobytes())
    adjacent = [min(_common(seqs[a], seqs[b]), caps[a], caps[b]) for a, b in zip(order, order[1:])]

    def build(lo, hi, start):
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


# --- parameterized activation memory table (lfm3opti.md §3 & §6 item 1) -----------------------------------------------
# Keyed by (parameterization, layer_kind, device_arch): CUDA bytes per token per layer that one unrecomputed layer adds
# to the step's PEAK (saved activations plus the backward's transient gradients), so `--activation-gb` bounds the
# activation part of the real peak. Measured 2026-11-22 on the GB10, the 1.7B model, r = 128 on all 9 families, trainable
# rows, compiled layers, no recompute, slope between 1024 and 3072 tokens: saved 105 KB, peak 127 KB per token per layer
# (eager: saved 127 KB, peak 150 KB; `EAGER_ACTIVATION_FACTOR`). The LoRA branch saves its BF16 input for dA and the
# rank-wide intermediate beside everything the frozen layer itself saves, so LoRA is not cheaper than full FT here.
# The bytes are a property of the saved tensors (shapes, dtypes), not of the SM, so the G4 (sm_120) entry is the same
# per-layer byte model. Full FT: measured 84.8 KB on the GB10 (88 KB kept). Update at runtime: `calibrate_activation_bytes`.
LORA_PEAK_BYTES_PER_TOKEN_LAYER = 128_000.0
EAGER_ACTIVATION_FACTOR = 150.0 / 127.0
ACTIVATION_BYTES_TABLE: dict[tuple[str, str, str], float] = {
    **{('full', kind, arch): 88_000.0 for kind in ('conv', 'attention') for arch in ('default', 'sm_121', 'sm_120')},
    **{('lora+rows', kind, arch): LORA_PEAK_BYTES_PER_TOKEN_LAYER
       for kind in ('conv', 'attention') for arch in ('default', 'sm_121', 'sm_120')},
}

# Retained as default reference constant for full-FT callers/tests.
ACTIVATION_BYTES_PER_TOKEN_LAYER = 88e3


def _device_arch(device=None) -> str:
    if not torch.cuda.is_available():
        return 'default'
    try:
        dev = torch.device(device) if device is not None else torch.device('cuda', torch.cuda.current_device())
        if dev.type != 'cuda':
            return 'default'
        major, minor = torch.cuda.get_device_capability(dev)
        return f'sm_{major}{minor}'
    except Exception:
        return 'default'


def calibrate_activation_bytes(layer_kind: str, bytes_per_token_layer: float, *,
                               parameterization: str = 'lora+rows', device_arch: str | None = None) -> None:
    """Update ``ACTIVATION_BYTES_TABLE`` from a measured ``cuda_peak_by_phase_gib`` observation."""
    arch = device_arch or _device_arch()
    ACTIVATION_BYTES_TABLE[(parameterization, layer_kind, arch)] = float(bytes_per_token_layer)


def activation_bytes_per_token_layer(model=None, *, parameterization: str | None = None,
                                     layer_kind: str | None = None, device_arch: str | None = None) -> float:
    """Mean saved activation bytes per token per layer from ``ACTIVATION_BYTES_TABLE``."""
    if parameterization is None and model is not None:
        fast = getattr(model, '_lfm3_fast', None) or {}
        parameterization = fast.get('parameterization') or (
            'lora+rows' if getattr(model, '_lfm3_adapters', None) is not None else 'full'
        )
    parameterization = parameterization_key(parameterization) or 'lora+rows'
    if device_arch is None and model is not None:
        try:
            device_arch = _device_arch(model.model.embed_tokens.weight.device)
        except Exception:
            device_arch = 'default'
    device_arch = device_arch or 'default'

    def lookup(kind: str) -> float:
        return (ACTIVATION_BYTES_TABLE.get((parameterization, kind, device_arch))
                or ACTIVATION_BYTES_TABLE.get((parameterization, kind, 'default'))
                or ACTIVATION_BYTES_PER_TOKEN_LAYER)

    eager = 1.0
    if parameterization == 'lora+rows' and model is not None and not (getattr(model, '_lfm3_fast', None) or {}).get('compile', True):
        eager = EAGER_ACTIVATION_FACTOR                 # eager layers keep more intermediates than the compiled graphs
    if layer_kind is not None:
        return lookup(layer_kind) * eager
    if model is None:
        return 0.5 * (lookup('conv') + lookup('attention'))
    layers = model.model.layers
    if not layers:
        return lookup('conv') * eager
    total = sum(lookup('attention' if layer.is_attention_layer else 'conv') for layer in layers)
    return total / len(layers) * eager


PARAMETERIZATION_ALIASES = {'lora': 'lora+rows', 'lora+rows': 'lora+rows', 'full': 'full'}


def parameterization_key(parameterization: str | None) -> str | None:
    """``'lora'`` (the name callers use) and ``'lora+rows'`` (the table key) are one parameterization."""
    if parameterization is None:
        return None
    if parameterization not in PARAMETERIZATION_ALIASES:
        raise ValueError(f'unknown parameterization {parameterization!r}: one of {sorted(PARAMETERIZATION_ALIASES)}')
    return PARAMETERIZATION_ALIASES[parameterization]


def resident_tokens(model, activation_gib, *, parameterization: str | None = None):
    """The longest sequence whose saved activations fit ``activation_gib`` (GiB) with no layer recomputed."""
    per_layer = activation_bytes_per_token_layer(model, parameterization=parameterization)
    return int(max(0., activation_gib) * 2**30 // (per_layer * len(model.model.layers)))


def set_checkpointing(model, tokens, activation_gib, *, parameterization: str | None = None):
    """Recompute (gradient-checkpoint) only as many layers as keep saved activations of ``tokens`` under ``activation_gib``."""
    layers = model.model.layers
    per_layer = activation_bytes_per_token_layer(model, parameterization=parameterization)
    count = recomputed_layers(len(layers), tokens, activation_gib, bytes_per_token_layer=per_layer)
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = index >= len(layers) - count
    return count


def recomputed_layers(layers, tokens, activation_gib, *, bytes_per_token_layer: float | None = None):
    """How many of ``layers`` a ``tokens`` sequence recomputes under ``activation_gib`` (GiB)."""
    per_layer = ACTIVATION_BYTES_PER_TOKEN_LAYER if bytes_per_token_layer is None else bytes_per_token_layer
    return layers - int(min(layers, activation_gib * 2**30 // (per_layer * max(1, tokens))))


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
    """Batches of a ``prefix_tree`` forest for ``tree-mask``."""
    def size(node):
        return _unit_tokens(node, tokens)

    def units(node):
        if 'index' in node or size(node) <= limit:
            return [node]
        own = node['end'] - node['start']
        parts = [unit for child in node['children'] for unit in units(child)]
        return [dict(node, children=group) for group in _pack(parts, size, max(1, limit - own))]
    return _pack([unit for node in nodes for unit in units(node)], size, limit)


def tree_batch_tokens(model, activation_gib):
    """The largest ``tree-mask`` batch: the tokens whose layer inputs and one layer's activations fit ``activation_gib``."""
    per_layer = activation_bytes_per_token_layer(model)
    per_token = len(model.model.layers) * model.config.hidden_size * 2 + per_layer
    return int(activation_gib * 2**30 // per_token)


def _resident_gib(model, tokens, activation_gib):
    """The budget left for unrecomputed layers once every layer's saved input (recomputed or not) is counted."""
    return max(0., activation_gib - tokens * len(model.model.layers) * model.config.hidden_size * 2 / 2**30)


def _recomputed(model, tokens, activation_gib):
    layers = len(model.model.layers)
    per_layer = activation_bytes_per_token_layer(model)
    free = int(min(layers, _resident_gib(model, tokens, activation_gib) * 2**30
                   // (per_layer * max(1, tokens))))
    return layers - free


def plan_tree_batches(model, nodes, tokens, activation_gib):
    """The ``tree-mask`` batch cap with the least estimated compute. Returns (cap, batches)."""
    layers = len(model.model.layers)
    top = max(1, tree_batch_tokens(model, activation_gib))
    caps, cap = [], max(1, min(top, resident_tokens(model, activation_gib)))
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
            cost += size * (1 + _recomputed(model, size, activation_gib) / layers / 3)
        if best is None or cost < best[0]:
            best = (cost, cap, batches)
    return best[1], best[2]


def _unit_tokens(node, tokens):
    if 'index' in node:
        return tokens(node['index'], node['start'])
    return node['end'] - node['start'] + sum(_unit_tokens(child, tokens) for child in node['children'])


def _run_tree_mask(model, nodes, ids_of, leaf, tokens, batch_tokens, activation_gib, *,
                   adapters=None, trainable_rows=None, fp8_base=None):
    """``run_prefix_tree``'s ``tree-mask`` mode."""
    if batch_tokens is None:
        batch_tokens, batches = plan_tree_batches(model, nodes, tokens, activation_gib)
    else:
        batches = tree_units(nodes, tokens, batch_tokens)
    for batch in batches:
        prefixes, items, leaf_parent = [], [], []
        stack = [(node, -1) for node in reversed(batch)]
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
            if activation_gib is not None:
                rows = offset + int(ids.shape[0])
                set_checkpointing(model, rows, _resident_gib(model, rows, activation_gib))
            if not prefixes:
                return hidden(model, ids, lengths, imports, adapters=adapters,
                              trainable_rows=trainable_rows, fp8_base=fp8_base)
            flat = torch.cat(prefix_ids + [ids])
            parents = [p for _, p in prefixes] + leaf_parent + [-1] * (len(lengths) - len(items))
            if imports is not None:
                imports = dict(imports)
                for key in ('rows', 'pass_rows', 'pass_source'):
                    if imports.get(key) is not None:
                        imports[key] = imports[key] + offset
            return hidden(model, flat, [int(t.shape[0]) for t in prefix_ids] + lengths, imports,
                          parents=parents, adapters=adapters, trainable_rows=trainable_rows, fp8_base=fp8_base)[offset:]
        leaf(items, forward)
    return {'mode': 'tree-mask', 'batches': len(batches), 'batch_tokens': batch_tokens}


def retained_bytes_per_token(model):
    """What ``levels`` keeps per prefix token until the prefixes' backward."""
    layers = model.model.layers
    full = sum(layer.self_attn.k_proj.weight.shape[0] for layer, key in zip(layers, _layer_keys(model))
               if key == ('attention', -1))
    return len(layers) * model.config.hidden_size * 2 + full * 2 * 4 * 2


def _prefix_tokens(node):
    return 0 if 'index' in node else node['end'] - node['start'] + sum(_prefix_tokens(c) for c in node['children'])


def prefix_parts(nodes, limit):
    """The forest split into parts of at most ``limit`` prefix tokens each."""
    def units(node):
        if 'index' in node or _prefix_tokens(node) <= limit:
            return [node]
        own = node['end'] - node['start']
        parts = [unit for child in node['children'] for unit in units(child)]
        return [dict(node, children=group) for group in _pack(parts, _prefix_tokens, max(1, limit - own))]
    return _pack([unit for node in nodes for unit in units(node)], _prefix_tokens, max(1, limit))


def prefix_levels(nodes):
    """(levels, continuations, wholes) of a ``prefix_tree`` forest (iterative to avoid closure reference cycles)."""
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
                    mode='levels', batch_tokens=None, retained_gb=None, *,
                    adapters=None, trainable_rows=None, fp8_base=None, parameterization=None):
    """Compute a ``prefix_tree``'s sequences packed, each shared prefix once, with exact gradients.
    Reports retained and leaf memory budgets in GiB (``retained_gib``, ``leaf_gib``, ``retained_limit_gib``).
    ``parameterization`` ('lora' / 'lora+rows' / 'full'), when given, must be the prepared model's: the memory plan
    uses the model's own activation table entry."""
    if parameterization is not None:
        prepared = (getattr(model, '_lfm3_fast', None) or {}).get('parameterization')
        if prepared is not None and parameterization_key(parameterization) != prepared:
            raise ValueError(f'run_prefix_tree: parameterization {parameterization!r} but the model is prepared '
                             f'as {prepared!r}')
    if mode == 'levels' and retained_gb is not None and retained_gb < 0:
        retained_gb = choose_retained_gib(model, nodes, tokens, pack_tokens, activation_gb)
    if mode == 'levels' and retained_gb and nodes:
        parts = _parts(model, nodes, retained_gb)
        if len(parts) > 1:
            plans = [run_prefix_tree(model, part, ids_of, leaf, tokens, pack_tokens, activation_gb, backward, mode,
                                     batch_tokens, adapters=adapters, trainable_rows=trainable_rows, fp8_base=fp8_base)
                     for part in parts]
            merged = {key: sum(p[key] for p in plans) for key in ('prefix_batches', 'leaf_batches', 'prefix_tokens',
                                                                   'recomputed_prefix_layer_tokens',
                                                                   'recomputed_leaf_layer_tokens')}
            retained_max = round(max(p['retained_gib'] for p in plans), 2)
            leaf_min = min((p['leaf_gib'] for p in plans if p['leaf_gib'] is not None), default=None)
            return {**merged, 'parts': len(parts), 'retained_gib': retained_max,
                    'retained_limit_gib': round(retained_gb, 2), 'leaf_gib': leaf_min}
    if mode == 'tree-mask':
        if batch_tokens is None or batch_tokens <= 0:
            batch_tokens = None if activation_gb is not None else pack_tokens
        return _run_tree_mask(model, nodes, ids_of, leaf, tokens, batch_tokens, activation_gb,
                              adapters=adapters, trainable_rows=trainable_rows, fp8_base=fp8_base)
    if mode not in ('levels', 'off'):
        raise ValueError(f'prefix tree mode {mode!r}: one of {PREFIX_TREE_MODES}')
    plan = levels_plan(model, nodes, tokens, pack_tokens, activation_gb)
    layers = model.model.layers
    leaf_gib, effective = plan['leaf_gib'], plan['effective']

    def past_of(node):
        chain = []
        while node is not None:
            chain.append(node['segment'])
            node = node['parent']
        return Past(reversed(chain))
    grad = torch.is_grad_enabled()
    compiled = getattr(model, '_lfm3_fast', None) is not None and model._lfm3_fast['compile']
    graphs = []
    for level, counts in zip(plan['prefix_batches'], plan['prefix_recompute']):
        for batch, count in zip(level, counts):
            lengths = [n['end'] - n['start'] for n in batch]
            if count is not None:
                for index, layer in enumerate(layers):
                    layer.gradient_checkpointing = index >= len(layers) - count
            ids = torch.cat([ids_of(n['first'], n['start'], n['end']) for n in batch])
            pasts = [past_of(n['parent']) for n in batch]
            extra = {k: v for k, v in (('adapters', adapters), ('trainable_rows', trainable_rows),
                                       ('fp8_base', fp8_base)) if v is not None}
            _, segments = _hidden_past(model, ids, lengths, pasts, compiled, keep=True, **extra)
            outputs = []
            for node, segment in zip(batch, segments):
                outputs += segment.tensors()
                node['segment'] = Segment(segment.length, [tuple(t.detach().requires_grad_(grad) for t in layer)
                                                           for layer in segment.layers])
            graphs.append((outputs, [t for n in batch for t in n['segment'].tensors()]) if backward and grad else None)
    for batch in plan['leaf_batches']:
        items = [(index, start) for index, start, _ in batch]
        pasts = [Past() if parent is None else past_of(parent) for _, _, parent in batch]

        def forward(ids, lengths=None, imports=None, pasts=pasts, batch=batch):
            if leaf_gib is not None:
                set_checkpointing(model, effective(batch), leaf_gib)
            return hidden(model, ids, lengths, imports, pasts=pasts,
                          adapters=adapters, trainable_rows=trainable_rows, fp8_base=fp8_base)
        leaf(items, forward)
    for entry in reversed(graphs):
        if entry is None:
            continue
        outputs, held = entry
        pairs = [(t, h.grad) for t, h in zip(outputs, held) if h.grad is not None]
        if pairs:
            torch.autograd.backward([t for t, _ in pairs], [g for _, g in pairs])
    return {'prefix_batches': sum(len(level) for level in plan['prefix_batches']),
            'leaf_batches': len(plan['leaf_batches']), 'prefix_tokens': plan['prefix_tokens'],
            'retained_gib': round(plan['retained_gib'], 2), 'leaf_gib': None if leaf_gib is None else round(leaf_gib, 2),
            'recomputed_prefix_layer_tokens': plan['recomputed_prefix_layer_tokens'],
            'recomputed_leaf_layer_tokens': plan['recomputed_leaf_layer_tokens']}


def _full_width(model):
    """Key (= value) width summed over the full-attention layers."""
    return sum(layer.self_attn.k_proj.weight.shape[0] for layer, key in zip(model.model.layers, _layer_keys(model))
               if key == ('attention', -1))


def _parts(model, nodes, retained_gib):
    """The forest as ``prefix_parts`` of at most ``retained_gib`` retained prefix state."""
    limit = int(retained_gib * 2**30 // retained_bytes_per_token(model))
    return prefix_parts(nodes, limit) if sum(_prefix_tokens(n) for n in nodes) > limit else [nodes]


def levels_plan(model, nodes, tokens, pack_tokens, activation_gib):
    """The ``levels`` schedule of one part (no model call), with GiB budgets."""
    layers = len(model.model.layers)
    full = _full_width(model)
    per_layer = activation_bytes_per_token_layer(model)
    levels, continuations, wholes = prefix_levels(nodes)

    def history(node):
        return 0 if node is None else node['end']
    prefix_batches = [_pack(level, lambda n: n['end'] - n['start'], pack_tokens) for level in levels]
    leaf_batches = (_pack(wholes, lambda item: tokens(item[0], item[1]), pack_tokens)
                    + _pack(continuations, lambda item: tokens(item[0], item[1]), pack_tokens))
    bytes_gib = per_layer * layers / 2**30

    def effective(batch):
        rows = sum(tokens(index, start) for index, start, _ in batch)
        past = sum(history(parent) for _, _, parent in batch)
        return rows + past * full * 2 * 2 / (per_layer * layers)
    prefix_tokens = sum(n['end'] - n['start'] for level in levels for n in level)
    retained_gib = prefix_tokens * retained_bytes_per_token(model) / 2**30
    leaf_gib = None if activation_gib is None else max(0., activation_gib - retained_gib)
    spare_gib = None if activation_gib is None else max(
        0., leaf_gib - max([effective(b) for b in leaf_batches] + [0]) * bytes_gib)
    recompute, recomputed = [], 0
    for level in prefix_batches:
        counts = []
        for batch in level:
            size = sum(n['end'] - n['start'] for n in batch)
            count = None
            if spare_gib is not None:
                count = recomputed_layers(layers, size, spare_gib, bytes_per_token_layer=per_layer)
                recomputed += count * size
                spare_gib = max(0., spare_gib - (layers - count) * size * per_layer / 2**30)
            counts.append(count)
        recompute.append(counts)
    leaf_tokens, leaf_recomputed = 0, 0
    for batch in leaf_batches:
        size = sum(tokens(index, start) for index, start, _ in batch)
        leaf_tokens += size
        if leaf_gib is not None:
            leaf_recomputed += recomputed_layers(layers, effective(batch), leaf_gib, bytes_per_token_layer=per_layer) * size
    return {'levels': levels, 'prefix_batches': prefix_batches, 'prefix_recompute': recompute,
            'leaf_batches': leaf_batches, 'effective': effective, 'leaf_gib': leaf_gib, 'prefix_tokens': prefix_tokens,
            'retained_gib': retained_gib, 'recomputed_prefix_layer_tokens': recomputed,
            'recomputed_leaf_layer_tokens': leaf_recomputed,
            'cost': 3 * layers * (prefix_tokens + leaf_tokens) + recomputed + leaf_recomputed}


PREFIX_RETAINED_GIB = 8.
PREFIX_RETAINED_GB = PREFIX_RETAINED_GIB
# The retained-prefix bound under a zero activation budget (full recompute, e.g. the OOM retry's first fallback): a
# zero budget must shrink what the prefixes keep, never fall back to the 8 GiB no-budget default.
PREFIX_RETAINED_MIN_GIB = 1.
RETAINED_SHARES = (1 / 32, 1 / 16, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3)


def choose_retained_gib(model, nodes, tokens, pack_tokens, activation_gib):
    """The ``retained_gib`` (``run_prefix_tree``, levels) with the least estimated compute under ``activation_gib``
    (None: no budget, the 8 GiB default; <= 0: full recompute, ``PREFIX_RETAINED_MIN_GIB``)."""
    if activation_gib is None:
        return PREFIX_RETAINED_GIB
    if activation_gib <= 0:
        return PREFIX_RETAINED_MIN_GIB
    if not nodes:
        return PREFIX_RETAINED_GIB
    best = None
    for share in sorted(RETAINED_SHARES, reverse=True):
        gib = activation_gib * share
        cost = sum(levels_plan(model, part, tokens, pack_tokens, activation_gib)['cost']
                   for part in _parts(model, nodes, gib))
        if best is None or cost < best[0]:
            best = (cost, gib)
    return best[1]


choose_retained_gb = choose_retained_gib


# --- loss ------------------------------------------------------------------------------------------------------------

def _ce_forward(logits, target):
    lf = logits.float()
    lse = torch.logsumexp(lf, dim=-1)
    return lse, lse - lf.gather(-1, target[:, None])[:, 0]


def _ce_backward(logits, lse, target, grad, out_dtype=None):
    """d nll / d logits in ``out_dtype`` (default the logits' dtype; the head's rows' dtype, as the stock path's
    `.float()` backward rounds it): grad * (softmax - onehot)."""
    lf = logits.float()
    p = torch.exp(lf - lse[:, None])
    onehot = torch.arange(lf.shape[-1], device=lf.device)[None, :] == target[:, None]
    return ((p - onehot.float()) * grad[:, None]).to(out_dtype or logits.dtype)


_CE = {}


def _ce(name):
    if name not in _CE:
        _CE[name] = _compile({'forward': _ce_forward, 'backward': _ce_backward}[name])
    return _CE[name]


# Head logits leave the GEMM in FP32 (BF16 inputs, FP32 accumulate and output; torch.mm out_dtype): a BF16 logit near
# |20| is rounded to a multiple of 0.125, which moved per-token log-probs by up to ~0.06 nats and the native engine's own
# full-forward and token-stepped scores of one sequence apart by 0.4 nats per decision (decision-parity hunt 2026-11-23).
# The serving engines read FP32 head logits too (shared_state.logits, SGLang --enable-fp32-lm-head). The gradient into
# the head GEMMs is still rounded to the rows' dtype, as before. Opt-out: NATLANG_HEAD_LOGITS=bf16.
HEAD_LOGITS_FP32 = _os.environ.get('NATLANG_HEAD_LOGITS', 'fp32') != 'bf16'


def _head_linear(rows, weight):
    if HEAD_LOGITS_FP32 and rows.is_cuda and rows.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
        return torch.mm(rows, weight.t(), out_dtype=torch.float32)
    return F.linear(rows, weight)


def _head_chunk(chunk, h):
    """Rows per head chunk: half of ``chunk`` when the logits are FP32, so a chunk's logits keep their bytes."""
    return max(1, chunk // 2) if HEAD_LOGITS_FP32 and h.is_cuda and h.dtype == torch.bfloat16 else chunk


def _head_logits_chunk(rows, weight, sparse_ids, sparse_rows):
    logits = _head_linear(rows, weight)
    if sparse_ids is not None and sparse_rows is not None and sparse_ids.numel():
        logits = logits.clone()
        logits[:, sparse_ids] = _head_linear(rows, sparse_rows.to(rows.dtype)).to(logits.dtype)
    return logits


class HeadMask:
    """A sampling-law mask on some rows of a head computation (common.tool_grammar): at rows ``rows`` (int64, sorted, into
    ``h``/``target``) only the tokens of ``table[which]`` (packed int32 bitmask rows, xgrammar's layout) are allowed.
    The NLL, the nucleus and their gradients are then those of the renormalized law over the allowed set, as the engine
    drew the token; other rows are unchanged. ``None`` everywhere means no mask."""

    UNPACK_BLOCK = 64

    def __init__(self, rows, table, which):
        self.rows = torch.as_tensor(rows, dtype=torch.long)
        self.rows_host = self.rows.cpu().numpy()
        if len(self.rows_host) > 1 and (self.rows_host[1:] <= self.rows_host[:-1]).any():
            raise ValueError('HeadMask rows must be strictly increasing')
        self.table = torch.as_tensor(table, dtype=torch.int32)
        self.which = torch.as_tensor(which, dtype=torch.long)
        if self.which.shape != self.rows.shape:
            raise ValueError('HeadMask rows and table indexes differ in length')
        self._allowed = {}

    def __len__(self):
        return int(self.rows.numel())

    def allowed_table(self, width, device):
        """bool [masks, width] on ``device`` (unpacked once per device and width)."""
        key = (str(device), int(width))
        if key not in self._allowed:
            table = self.table.to(device)
            shifts = torch.arange(32, device=device, dtype=torch.int32)
            parts = [((table[i:i + self.UNPACK_BLOCK].unsqueeze(-1) >> shifts) & 1).reshape(
                min(self.UNPACK_BLOCK, table.shape[0] - i), -1)[:, :width].bool()
                for i in range(0, table.shape[0], self.UNPACK_BLOCK)]
            self._allowed[key] = (torch.cat(parts) if parts else
                                  torch.zeros((0, width), dtype=torch.bool, device=device))
        return self._allowed[key]

    def apply(self, logits, start, stop):
        """``logits`` of rows [start, stop) with the disallowed tokens of their masked rows at -inf (in place)."""
        lo, hi = (int(x) for x in np.searchsorted(self.rows_host, [start, stop]))
        if lo == hi:
            return logits
        local = self.rows[lo:hi].to(logits.device) - start
        allowed = self.allowed_table(logits.shape[-1], logits.device)[self.which[lo:hi].to(logits.device)]
        logits[local] = logits[local].masked_fill(~allowed, float('-inf'))
        return logits

    def allowed_rows(self, start, stop, width, device):
        """bool [stop - start, width]: the allowed tokens of rows [start, stop) (every token at an unmasked row)."""
        out = torch.ones(stop - start, width, dtype=torch.bool, device=device)
        lo, hi = (int(x) for x in np.searchsorted(self.rows_host, [start, stop]))
        if lo < hi:
            out[self.rows[lo:hi].to(device) - start] = self.allowed_table(width, device)[self.which[lo:hi].to(device)]
        return out

    def subset(self, start, stop):
        """The mask of rows [start, stop), renumbered from 0 (a slice of the rows it covers)."""
        lo, hi = (int(x) for x in np.searchsorted(self.rows_host, [start, stop]))
        out = HeadMask(self.rows[lo:hi] - start, self.table, self.which[lo:hi])
        out._allowed = self._allowed
        return out


def _masked(logits, mask, start, stop):
    return logits if mask is None else mask.apply(logits, start, stop)


class _LinearNLL(torch.autograd.Function):
    """Per-row NLL of ``target`` under logits ``h @ weight.T`` with optional tied trainable sparse rows, in chunks."""

    @staticmethod
    def forward(ctx, h, weight, target, chunk, sparse_ids, sparse_rows, mask=None):
        rows, chunk = h.shape[0], _head_chunk(chunk, h)
        nll = torch.empty(rows, dtype=torch.float32, device=h.device)
        lse = torch.empty(rows, dtype=torch.float32, device=h.device)
        forward_ce = _ce('forward') if h.is_cuda else _ce_forward
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            logits = _masked(_head_logits_chunk(h[start:stop], weight, sparse_ids, sparse_rows), mask, start, stop)
            lse[start:stop], nll[start:stop] = forward_ce(logits, target[start:stop])
        ctx.save_for_backward(h, weight, target, lse,
                              sparse_ids if sparse_ids is not None else torch.empty(0, dtype=torch.long, device=h.device),
                              sparse_rows if sparse_rows is not None else torch.empty(0, dtype=torch.float32, device=h.device))
        ctx.has_sparse = sparse_ids is not None and sparse_rows is not None and sparse_ids.numel() > 0
        ctx.chunk, ctx.mask = chunk, mask
        return nll

    @staticmethod
    def backward(ctx, grad):
        h, weight, target, lse, sparse_ids, sparse_rows = ctx.saved_tensors
        grad = grad.float().contiguous()
        backward_ce = _ce('backward') if h.is_cuda else _ce_backward
        need_h, need_w, need_s = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[5]
        dh = torch.empty_like(h) if need_h else None
        dw = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_w else None
        ds = torch.zeros(sparse_rows.shape, dtype=torch.float32, device=sparse_rows.device) if (need_s and ctx.has_sparse) else None
        s_ids = sparse_ids if ctx.has_sparse else None
        s_rows = sparse_rows if ctx.has_sparse else None
        w_delta = (s_rows.to(weight.dtype) - weight.index_select(0, s_ids)) if (ctx.has_sparse and need_h) else None
        for start in range(0, h.shape[0], ctx.chunk):
            stop = min(h.shape[0], start + ctx.chunk)
            rows = h[start:stop]
            logits = _masked(_head_logits_chunk(rows, weight, s_ids, s_rows), ctx.mask, start, stop)
            d = backward_ce(logits, lse[start:stop], target[start:stop], grad[start:stop], rows.dtype)
            if need_h:
                chunk_dh = d @ weight
                if w_delta is not None:
                    chunk_dh = chunk_dh + d.index_select(1, s_ids) @ w_delta
                dh[start:stop] = chunk_dh
            if need_w:
                if d.is_cuda:
                    dw += torch.mm(d.t(), rows, out_dtype=torch.float32)
                else:
                    dw += (d.t() @ rows).float()
            if ds is not None:
                d_s = d.index_select(1, s_ids)
                ds += d_s.t().float() @ rows.float()
        return dh, None if dw is None else dw.to(weight.dtype), None, None, None, ds, None


def _head_rows(weight, trainable_rows):
    """The trainable rows a head ``weight`` scores with: explicit rows win, ``False`` means none, ``None`` means the
    rows of the prepared model that owns ``weight`` (``prepare`` tags the head), like ``hidden`` resolves its rows, so
    a caller that omits ``trainable_rows=`` never scores the live policy's head with the start rows."""
    if trainable_rows is False:
        return None
    if trainable_rows is not None:
        return trainable_rows
    owner = getattr(weight, '_lfm3_owner', None)
    model = owner() if owner is not None else None
    return None if model is None else getattr(model, '_lfm3_trainable_rows', None)


def linear_nll(h, weight, target, chunk=LOGIT_CHUNK, *, trainable_rows=None, mask=None):
    """fp32 NLL per row of ``target`` (int64) from hidden rows ``h`` (N, D) and the tied head ``weight`` (V, D).
    ``trainable_rows`` (``_head_rows``: default the owning model's rows, ``False`` none) splices FP32 rows into the head
    and accumulates gradients directly into them without touching the frozen head weights. ``mask`` (``HeadMask``):
    rows drawn under a sampling-law mask take the NLL of the renormalized law over their allowed tokens."""
    trainable_rows = _head_rows(weight, trainable_rows)
    s_ids = trainable_rows.token_ids.to(h.device) if trainable_rows is not None else None
    s_rows = trainable_rows.rows if trainable_rows is not None else None
    return _LinearNLL.apply(h, weight, target, chunk, s_ids, s_rows, mask if mask is not None and len(mask) else None)


NUCLEUS_TOP_K = 1024
NUCLEUS_WIDE_K = 16384
NUCLEUS_CHUNK = 2048


def _cut_of(values, row_lse, top_p):
    """The cut logit of rows whose ``values`` are sorted descending and hold the whole nucleus."""
    p = torch.exp(values - row_lse[:, None])
    count = ((torch.cumsum(p, -1) - p) <= top_p).sum(-1).clamp(min=1)
    return values.gather(1, (count - 1)[:, None])[:, 0]


def _nucleus_cut_lse(lf, lse, top_p, top_k, wide_k=None):
    """Per row of fp32 logits ``lf`` (log-sum-exp ``lse``): the cut logit of the top-p nucleus and its log-sum-exp."""
    vocab = lf.shape[-1]
    values = lf.topk(min(top_k, vocab), dim=-1).values.float()
    cut = _cut_of(values, lse, top_p)
    inside = torch.logsumexp(values.masked_fill(values < cut[:, None], float('-inf')), dim=-1)
    if values.shape[-1] < vocab:
        wide = (cut <= values[:, -1]).nonzero()[:, 0]
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


def _nucleus_logit_grad(logits, target, lse, set_lse, cut, grad, grad_nucleus, out_dtype=None):
    """d loss / d logits of one chunk in ``out_dtype`` (default the logits' dtype): grad * (softmax - onehot) +
    grad_nucleus * (softmax over the set - onehot)."""
    lf = logits.float()
    onehot = torch.arange(lf.shape[-1], device=lf.device)[None, :] == target[:, None]
    kept = (lf >= cut[:, None]) | onehot
    d = torch.exp(lf - lse[:, None]) * grad[:, None] + torch.where(kept, torch.exp(lf - set_lse[:, None]), 0.) * grad_nucleus[:, None]
    return (d - onehot * (grad + grad_nucleus)[:, None]).to(out_dtype or logits.dtype)


_NUCLEUS_GRAD = {}


def _nucleus_grad_kernel(device_type):
    """``_nucleus_logit_grad`` compiled (one fused pass over the chunk's logits)."""
    if device_type != 'cuda':
        return _nucleus_logit_grad
    if 'cuda' not in _NUCLEUS_GRAD:
        _NUCLEUS_GRAD['cuda'] = _compile(_nucleus_logit_grad)
    return _NUCLEUS_GRAD['cuda']


class _LinearNLLNucleus(torch.autograd.Function):
    """``_LinearNLL`` plus the NLL under the top-p nucleus of each row's own distribution, with optional sparse rows."""

    @staticmethod
    def forward(ctx, h, weight, target, chunk, top_p, top_k, sparse_ids, sparse_rows, mask=None):
        rows, chunk = h.shape[0], _head_chunk(chunk, h)
        nll, lse, set_lse, cut = (torch.empty(rows, dtype=torch.float32, device=h.device) for _ in range(4))
        outside = torch.empty(rows, dtype=torch.bool, device=h.device)
        forward_ce = _ce('forward') if h.is_cuda else _ce_forward
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            logits = _masked(_head_logits_chunk(h[start:stop], weight, sparse_ids, sparse_rows), mask, start, stop)
            own = logits.gather(-1, target[start:stop, None])[:, 0].float()
            lse[start:stop], nll[start:stop] = forward_ce(logits, target[start:stop])
            cut[start:stop], inside = _nucleus_cut_lse(logits, lse[start:stop], top_p, top_k)
            outside[start:stop] = own < cut[start:stop]
            set_lse[start:stop] = torch.where(outside[start:stop], torch.logaddexp(inside, own), inside)
        ctx.save_for_backward(h, weight, target, lse, set_lse, cut,
                              sparse_ids if sparse_ids is not None else torch.empty(0, dtype=torch.long, device=h.device),
                              sparse_rows if sparse_rows is not None else torch.empty(0, dtype=torch.float32, device=h.device))
        ctx.has_sparse = sparse_ids is not None and sparse_rows is not None and sparse_ids.numel() > 0
        ctx.chunk, ctx.mask = chunk, mask
        ctx.mark_non_differentiable(outside)
        return nll, set_lse - (lse - nll), outside

    @staticmethod
    def backward(ctx, grad, grad_nucleus, _):
        h, weight, target, lse, set_lse, cut, sparse_ids, sparse_rows = ctx.saved_tensors
        zeros = torch.zeros(h.shape[0], dtype=torch.float32, device=h.device)
        grad = zeros if grad is None else grad.float()
        grad_nucleus = zeros if grad_nucleus is None else grad_nucleus.float()
        need_h, need_w, need_s = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[7]
        dh = torch.empty_like(h) if need_h else None
        dw = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_w else None
        ds = torch.zeros(sparse_rows.shape, dtype=torch.float32, device=sparse_rows.device) if (need_s and ctx.has_sparse) else None
        s_ids = sparse_ids if ctx.has_sparse else None
        s_rows = sparse_rows if ctx.has_sparse else None
        w_delta = (s_rows.to(weight.dtype) - weight.index_select(0, s_ids)) if (ctx.has_sparse and need_h) else None
        for start in range(0, h.shape[0], ctx.chunk):
            stop = min(h.shape[0], start + ctx.chunk)
            rows = h[start:stop]
            logits = _masked(_head_logits_chunk(rows, weight, s_ids, s_rows), ctx.mask, start, stop)
            d = _nucleus_grad_kernel(rows.device.type)(logits, target[start:stop], lse[start:stop],
                                                       set_lse[start:stop], cut[start:stop], grad[start:stop],
                                                       grad_nucleus[start:stop], rows.dtype)
            if need_h:
                chunk_dh = d @ weight
                if w_delta is not None:
                    chunk_dh = chunk_dh + d.index_select(1, s_ids) @ w_delta
                dh[start:stop] = chunk_dh
            if need_w:
                if d.is_cuda:
                    dw += torch.mm(d.t(), rows, out_dtype=torch.float32)
                else:
                    dw += (d.t() @ rows).float()
            if ds is not None:
                d_s = d.index_select(1, s_ids)
                ds += d_s.t().float() @ rows.float()
        return dh, None if dw is None else dw.to(weight.dtype), None, None, None, None, None, ds, None


def linear_nll_nucleus(h, weight, target, top_p, chunk=NUCLEUS_CHUNK, top_k=NUCLEUS_TOP_K, *, trainable_rows=None,
                       mask=None):
    """(nll, nucleus nll, outside): ``linear_nll`` and the NLL of ``target`` under the row's top-p nucleus (of the
    masked law at ``mask``'s rows, as the engines draw: mask, then temperature, then nucleus)."""
    if not 0. < top_p < 1.:
        raise ValueError(f'top_p must be in (0, 1) for the nucleus NLL, got {top_p} (use linear_nll at 1)')
    trainable_rows = _head_rows(weight, trainable_rows)
    s_ids = trainable_rows.token_ids.to(h.device) if trainable_rows is not None else None
    s_rows = trainable_rows.rows if trainable_rows is not None else None
    return _LinearNLLNucleus.apply(h, weight, target, chunk, float(top_p), int(top_k), s_ids, s_rows,
                                   mask if mask is not None and len(mask) else None)


@torch.no_grad()
def token_logprobs(model, ids, positions, targets, lengths=None, chunk=LOGIT_CHUNK, imports=None, top_p=1.0, *,
                   adapters=None, trainable_rows=None, fp8_base=None, mask=None):
    """Log-probabilities (fp32) of ``targets`` predicted at rows ``positions`` of the packed ``ids`` (``mask``: a
    ``HeadMask`` over the rows of ``positions``)."""
    resolved_rows = _resolve_trainable_rows(model, trainable_rows) or False
    h = hidden(model, ids, lengths, imports, adapters=adapters, trainable_rows=trainable_rows, fp8_base=fp8_base)
    if top_p >= 1.:
        return -linear_nll(h[positions], model.lm_head.weight, targets, chunk, trainable_rows=resolved_rows, mask=mask)
    nll, nucleus, outside = linear_nll_nucleus(h[positions], model.lm_head.weight, targets, top_p,
                                               trainable_rows=resolved_rows, mask=mask)
    return -nll, -nucleus, outside
