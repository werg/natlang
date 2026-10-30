"""Shared backbone primitives from the Maple port: the model config, RMS norm (with its fused/compiled kernels), partial
rotary, and the stacked MoE experts (ternary and dense) with their routing.

The Maple/Mellum causal LMs built from these were retired with the Mellum line (plans/neuralese/
LFM3_COMMITTED_DECISIONS_20261028.md item 16; git history is the archive). What stays is imported by the shared
native trainer, serving and kernels: ``RMSNorm``/``eager_rms_norm`` by the LFM ports, ``DenseExperts``/``DENSE_MOE``
by the QAT and FP8 backends, ``SparseMoE`` by the MoE kernel tests. Experts route with an FP32 router, softmax,
top-k, renormalised; ``down(silu(min(gate, 7)) * clamp(up, -7, 7))``.
"""

from __future__ import annotations

import json
import contextlib
from contextvars import ContextVar
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..common.profiling import scope as _profile_scope
from .ternary import ternary_codes

SWIGLU_CLAMP = 7.0


@dataclass
class MapleConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    moe_intermediate_size: int = 512
    num_experts: int = 256
    num_experts_per_tok: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.5
    sliding_window: int = 512
    layer_types: list = field(default_factory=lambda: (["sliding_attention"] * 3 + ["full_attention"]) * 6)
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False
    bos_token_id: int | None = None  # the tokenizer declares none; raw ports start a context-free write here
    # Same family, other checkpoints (Mellum 2.x): full rotary with per-layer-type RoPE (YaRN on global layers),
    # no SwiGLU clamp, and BF16 experts (``dense_experts``) until QAT ternarizes them.
    model_type: str = "maple"
    swiglu_clamp: float | None = SWIGLU_CLAMP
    rope_parameters: dict | None = None
    dense_experts: bool = False

    @staticmethod
    def from_dir(path) -> "MapleConfig":
        data = json.loads((Path(path) / "config.json").read_text())
        names = MapleConfig.__dataclass_fields__
        if data.get("model_type") == "mellum":
            data = {"partial_rotary_factor": 1.0, "swiglu_clamp": None, **data}
        if "num_experts" not in data and "num_local_experts" in data:  # transformers' serialized alias
            data["num_experts"] = data["num_local_experts"]
        config = MapleConfig(**{k: v for k, v in data.items() if k in names})
        # Keys the model does not use (token ids such as pad_token_id, dropout, ...) stay readable as attributes, as on
        # an HF config: the port reads them through `backbone.config`.
        for key, value in data.items():
            if key not in names:
                setattr(config, key, value)
        return config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.private = nn.ParameterDict()  # per family member: additive gain correction, zero-initialised

    def forward(self, x):
        with _profile_scope("norm"):
            return self._forward(x)

    def gain(self):
        """The active gain: the shared weight plus the active family member's private correction."""
        from .ternary import STATE

        weight = self.weight
        member = STATE["size"]
        if member is not None and STATE["enabled"] and str(member) in self.private:
            weight = weight + self.private[str(member)].to(weight.dtype)
        return weight

    def _forward(self, x):
        weight = self.gain()
        from ..common.native_higher_order import active_reference_arithmetic
        if active_reference_arithmetic() is not None:
            return _rms_norm(x,weight,self.eps)
        if (x.is_cuda and not _eager_depth and x.dtype in (torch.bfloat16, torch.float16)
                and os.environ.get("NATLANG_MAPLE_FUSED_NORM", "1") != "0"):
            # One Triton kernel each way (maple/norm_kernels.py), same rule; NATLANG_MAPLE_FUSED_NORM=0: eager/compiled.
            from .norm_kernels import fused_rms_norm
            return fused_rms_norm(x, weight, self.eps)
        if (x.is_cuda and not getattr(self, 'native_stable_norm', False)
                and not _eager_depth and not _compiled_eager_depth.get()
                and os.environ.get("NATLANG_MAPLE_COMPILE", "1") != "0"):
            global _rms_norm_compiled
            if _rms_norm_compiled is None:
                _rms_norm_compiled = torch.compile(_rms_norm, dynamic=True)
            return _rms_norm_compiled(x, weight, self.eps)
        return _rms_norm(x, weight, self.eps)


_rms_norm_compiled = None
_eager_depth = 0
_compiled_eager_depth = ContextVar("maple_compiled_norm_eager_depth",default=0)


@contextlib.contextmanager
def eager_rms_norm(*, compiled_only=False):
    """Run RMSNorm eagerly inside this block. A checkpointed region must recompute exactly the graph it saved:
    the dynamic-shape compiled norm can select a different graph on recompute for an odd chunk length (seen: a
    39-position readout chunk saved bf16 inputs but recomputed float32 ones), which breaks the saved-tensor match."""
    if compiled_only:
        # Keep the already stable custom BF16/FP16 fused kernel; bypass only
        # dynamic compile dispatch, whose fallback changes saved-tensor graphs.
        token=_compiled_eager_depth.set(_compiled_eager_depth.get()+1)
        try:
            yield
        finally:
            _compiled_eager_depth.reset(token)
        return
    global _eager_depth
    _eager_depth += 1
    try:
        yield
    finally:
        _eager_depth -= 1


def _rms_norm(x, weight, eps):
    """One fused kernel under torch.compile (CUDA) instead of five passes over the activations."""
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    # A gain trained in FP32 (QAT) keeps the activation dtype; with a BF16 gain this is the same value.
    return (weight * y.to(dtype)).to(dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_partial_rope(q, k, cos, sin):
    """Rotate the first ``cos.shape[-1]`` dimensions of each head; q, k: [B, H, T, hd]; cos, sin: [B, T, r]."""
    r = cos.shape[-1]
    cos, sin = cos[:, None], sin[:, None]

    def rope(x):
        rot, rest = x[..., :r], x[..., r:]
        return torch.cat([rot * cos + _rotate_half(rot) * sin, rest], dim=-1)

    return rope(q), rope(k)


def _dense(codes: torch.Tensor, scale: torch.Tensor, block: int, dtype) -> torch.Tensor:
    return codes.to(dtype) * scale.to(dtype).repeat_interleave(block, dim=-1)[..., :codes.shape[-1]]


class _TernaryMatmul(torch.autograd.Function):
    """``x @ (codes * scale)ᵀ`` that keeps only the int8 codes and the scales for backward: the dense expert weight
    is rebuilt there instead of being saved (saved dense weights of every routed expert, in every layer and member
    pass, cost tens of GB). Gradients: exact to ``x``; to a scale of shape [rows, blocks], the block sums of
    ``(gᵀ x) ⊙ codes``."""

    @staticmethod
    def forward(ctx, x, codes, scale, block):
        ctx.save_for_backward(x, codes, scale)
        ctx.block = block
        return x @ _dense(codes, scale, block, x.dtype).T

    @staticmethod
    def backward(ctx, grad):
        x, codes, scale = ctx.saved_tensors
        grad_x = grad_scale = None
        if ctx.needs_input_grad[0]:
            grad_x = grad @ _dense(codes, scale, ctx.block, grad.dtype)
        if ctx.needs_input_grad[2]:
            product = (grad.reshape(-1, grad.shape[-1]).float().T @ x.reshape(-1, x.shape[-1]).float()) * codes.float()
            pad = (-product.shape[-1]) % ctx.block
            if pad:
                product = F.pad(product, (0, pad))
            grad_scale = product.reshape(product.shape[0], -1, ctx.block).sum(-1).to(scale.dtype)
        return grad_x, None, grad_scale, None


class TernaryExperts(nn.Module):
    """All experts of one layer: int8 codes [E, 2*ff, d] (gate rows then up rows) and [E, d, ff], BF16 row scales."""

    def __init__(self, experts: int, hidden: int, ff: int):
        super().__init__()
        self.ff = ff
        self.register_buffer("gate_up_codes", torch.zeros(experts, 2 * ff, hidden, dtype=torch.int8))
        self.register_buffer("gate_up_scale", torch.zeros(experts, 2 * ff, 1, dtype=torch.bfloat16))
        self.register_buffer("down_codes", torch.zeros(experts, hidden, ff, dtype=torch.int8))
        self.register_buffer("down_scale", torch.zeros(experts, hidden, 1, dtype=torch.bfloat16))

    @torch.no_grad()
    def set_expert(self, index: int, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
        codes, scale = ternary_codes(torch.cat([gate, up], 0))
        self.gate_up_codes[index], self.gate_up_scale[index] = codes, scale
        codes, scale = ternary_codes(down)
        self.down_codes[index], self.down_scale[index] = codes, scale

    @torch.no_grad()
    def permute(self, order: torch.Tensor):
        for name in ("gate_up_codes", "gate_up_scale", "down_codes", "down_scale"):
            setattr(self, name, getattr(self, name)[order.to(getattr(self, name).device)].contiguous())

    def learn_scales(self, block: int = 256) -> list[nn.Parameter]:
        """Scale-only QAT for the experts: a learned FP32 scale per ``block`` columns of every row (TQ2_0 block),
        initialised at the row scale. Codes stay fixed; the forward uses the scales rounded to FP16."""
        self.block = block
        self.gate_up_blocks = nn.Parameter(self.gate_up_scale.float().expand(
            -1, -1, -(-self.gate_up_codes.shape[-1] // block)).clone())
        self.down_blocks = nn.Parameter(self.down_scale.float().expand(
            -1, -1, -(-self.down_codes.shape[-1] // block)).clone())
        return [self.gate_up_blocks, self.down_blocks]

    def _scaled(self, codes, row_scale, blocks, index, dtype):
        from .ternary import STATE

        if blocks is None or not STATE["enabled"]:
            return codes[index].to(dtype) * row_scale[index].to(dtype)
        scale = blocks[index].half().to(dtype)
        scale = scale.repeat_interleave(self.block, dim=-1)[..., :codes.shape[-1]]
        return codes[index].to(dtype) * scale

    def weights(self, index, dtype):
        gate_up = self._scaled(self.gate_up_codes, self.gate_up_scale, getattr(self, "gate_up_blocks", None), index,
                               dtype)
        down = self._scaled(self.down_codes, self.down_scale, getattr(self, "down_blocks", None), index, dtype)
        return gate_up, down

    def _matmul(self, x, codes, row_scale, blocks, index):
        from .ternary import STATE

        if blocks is None or not STATE["enabled"]:
            return _TernaryMatmul.apply(x, codes[index], row_scale[index], codes.shape[-1])
        # The deployed block scale is FP16 (TQ2_0): train against it; .half() passes the gradient to the FP32 master.
        return _TernaryMatmul.apply(x, codes[index], blocks[index].half(), self.block)

    def run(self, x: torch.Tensor, index: int, dtype) -> torch.Tensor:
        y = self._matmul(x, self.gate_up_codes, self.gate_up_scale, getattr(self, "gate_up_blocks", None), index)
        gate, up = y[..., :self.ff], y[..., self.ff:]
        h = swiglu(gate, up, getattr(self, "clamp", SWIGLU_CLAMP))
        return self._matmul(h, self.down_codes, self.down_scale, getattr(self, "down_blocks", None), index)


def swiglu(gate, up, clamp):
    if clamp is None:
        return F.silu(gate) * up
    return F.silu(gate.clamp(max=clamp)) * up.clamp(-clamp, clamp)


class DenseExperts(nn.Module):
    """All experts of one layer as BF16 weights [E, 2*ff, d] (gate rows then up rows) and [E, d, ff]: the exact
    published checkpoint of a non-ternary family member (Mellum) before QAT, with the TernaryExperts interface."""

    def __init__(self, experts: int, hidden: int, ff: int, clamp: float | None = None):
        super().__init__()
        self.ff, self.clamp = ff, clamp
        self.register_buffer("gate_up", torch.zeros(experts, 2 * ff, hidden, dtype=torch.bfloat16))
        self.register_buffer("down", torch.zeros(experts, hidden, ff, dtype=torch.bfloat16))

    @torch.no_grad()
    def set_expert(self, index: int, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
        self.gate_up[index] = torch.cat([gate, up], 0).to(self.gate_up.dtype)
        self.down[index] = down.to(self.down.dtype)

    @torch.no_grad()
    def permute(self, order: torch.Tensor):
        for name in ("gate_up", "down"):
            value = getattr(self, name)
            permuted = value[order.to(value.device)].contiguous()
            if isinstance(value, nn.Parameter):
                value.data = permuted  # a latent keeps its identity (optimizer state is per parameter)
            else:
                setattr(self, name, permuted)

    quantize = False

    def make_latent(self, quantize: bool = True) -> list[nn.Parameter]:
        """Full-latent QAT (plans/mellum-port.md): the BF16 weights become trainable latents and every forward uses
        their ternary codes (Maple's per-row rule, straight-through), so training moves codes, not only scales.
        ``quantize=False``: trainable BF16 experts (the ``latent`` backbone policy), quantized only by the recipe's
        precision points (``precision_group``)."""
        for name in ("gate_up", "down"):
            value = getattr(self, name)
            if not isinstance(value, nn.Parameter):
                del self._buffers[name]
                setattr(self, name, nn.Parameter(value.detach().clone(), requires_grad=True))
        self.quantize = quantize
        return [self.gate_up, self.down]

    # Set by train/quantization.py: the experts follow the recipe's precision points (maple/ternary.precision_value)
    # instead of the conversion's single ternary ramp.
    precision_group = None

    def _value(self, weight, dtype):
        from ..model.native_adapters import native_forward_storage_dtype
        if native_forward_storage_dtype(weight) is not None:
            if self.precision_group is not None or self.quantize:
                raise ValueError('native coefficient carriers require unquantized dense expert source weights')
            return weight
        if self.precision_group is not None:
            from .ternary import precision_value
            return precision_value(weight, self.precision_group).to(dtype)
        if not self.quantize:
            return weight.to(dtype)
        from .ternary import ramped_ternarize_ste
        return ramped_ternarize_ste(weight).to(dtype)

    def weights(self, index, dtype):
        return self._value(self.gate_up[index], dtype), self._value(self.down[index], dtype)

    def run(self, x: torch.Tensor, index: int, dtype) -> torch.Tensor:
        gate_up, down = self._value(self.gate_up[index], x.dtype), self._value(self.down[index], x.dtype)
        from ..model.native_adapters import native_forward_storage_dtype
        from ..model.native_weight_carrier import stored_linear
        gate_storage = native_forward_storage_dtype(gate_up)
        down_storage = native_forward_storage_dtype(down)
        y = (stored_linear(x, gate_up, storage_dtype=gate_storage) if gate_storage is not None
             else x @ gate_up.T)
        hidden = swiglu(y[..., :self.ff], y[..., self.ff:], self.clamp)
        return (stored_linear(hidden, down, storage_dtype=down_storage) if down_storage is not None
                else hidden @ down.T)

    def unbound(self, dtype) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Every expert's (ramped ternary) weights from one op per tensor. Indexing the latent per expert instead
        makes each expert's backward materialise a zero gradient the size of all experts (64x per tensor per
        layer: most of a Mellum QAT step); unbind's backward stacks all expert gradients once."""

        gate_up, down = self.stacked(dtype)
        return gate_up.unbind(0), down.unbind(0)

    def stacked(self, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """All experts' (precision-ramped) weights as [E, 2*ff, d] and [E, d, ff], one op per tensor."""
        from .ternary import QUANT_MIX

        # A functional native adapter replaces these exact bank tensors for
        # one call. Its effective weights must neither read nor populate the
        # immutable base generation cache.
        from ..common.native_higher_order import active_reference_arithmetic
        if QUANT_MIX.get("cache") is not None and active_reference_arithmetic() is None:
            from ..model.native_adapters import substituted_native_weight
            adapted = substituted_native_weight(self.gate_up) or substituted_native_weight(self.down)
            if not adapted:
                key = (id(self), dtype)
                if key not in QUANT_MIX["cache"]:
                    with torch.no_grad():
                        QUANT_MIX["cache"][key] = (self._value(self.gate_up, dtype), self._value(self.down, dtype))
                return QUANT_MIX["cache"][key]
        return self._value(self.gate_up, dtype), self._value(self.down, dtype)

    def run_weights(self, x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        from ..model.native_adapters import native_forward_storage_dtype
        from ..model.native_weight_carrier import stored_linear
        def project(value, weight):
            from ..common.native_higher_order import active_reference_arithmetic,reference_linear
            if active_reference_arithmetic() is not None:
                return reference_linear(value,weight)
            dtype = native_forward_storage_dtype(weight)
            return stored_linear(value, weight, storage_dtype=dtype) if dtype is not None else value @ weight.T
        y = project(x, gate_up)
        return project(swiglu(y[..., :self.ff], y[..., self.ff:], self.clamp), down)


# The dense-expert MoE kernel (a trainer's --moe-kernel; recipe parameter ``moe_kernel``): "invariant" (the default:
# one fixed Triton tile per routed row, so a token's output and routing never depend on the other tokens of the call;
# maple/fused_moe.invariant_dense_experts), "loop" (the per-expert reference) or "grouped" (torch's grouped GEMM).
# The last two choose cuBLAS kernels by shape (NUMERICAL_PARITY.md, "Mellum batch-row dragnet").
DENSE_MOE_KERNELS = ("invariant", "loop", "grouped")
DENSE_MOE = {"kernel": os.environ.get("NATLANG_MAPLE_DENSE_MOE", "invariant"),
             # "grouped": FP8 grouped expert GEMMs (maple/expert_gemm.py), set by train/fp8_backend.FP8ExpertAdapter
             "fp8": None, "fp8_stats": None, "fp8_input_gradient_precision": "fp8"}


def set_dense_moe_kernel(kernel: str) -> None:
    if kernel not in DENSE_MOE_KERNELS:
        raise ValueError(f"unknown dense MoE kernel {kernel!r}; expected one of {', '.join(DENSE_MOE_KERNELS)}")
    DENSE_MOE["kernel"] = kernel


class SparseMoE(nn.Module):
    def __init__(self, config: MapleConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.clamp = config.swiglu_clamp
        if config.dense_experts:
            self.experts = DenseExperts(config.num_experts, config.hidden_size, config.moe_intermediate_size,
                                        config.swiglu_clamp)
        else:
            self.experts = TernaryExperts(config.num_experts, config.hidden_size, config.moe_intermediate_size)
            self.experts.clamp = config.swiglu_clamp
        self.active_experts: int | None = None
        # The active member's expert ids in this layer (``MapleForCausalLM.set_member_order``): a member of an
        # unpermuted model routes over these experts (the first ``active_experts`` of the N0 order) instead of the
        # first ``active_experts`` indices; an expert keeps its own id, so weights, optimizer state and kernels are
        # untouched. ``None``: the prefix (a model permuted with ``order_experts``).
        self.member_experts: torch.Tensor | None = None
        self.statistics: dict | None = None
        # Private router rows per family member (MAPLE_NESTED §4): a trainable copy of the rows of the member's
        # experts. The member's GGUF carries them as its own ffn_gate_inp, so they export exactly.
        self.private_gate = nn.ParameterDict()

    def route(self, x):
        from .ternary import STATE

        key = str(STATE["size"])
        ids = self.member_experts if self.active_experts is not None else None
        if self.active_experts is not None and STATE["enabled"] and key in self.private_gate:
            logits = F.linear(x.float(), self.private_gate[key])  # the member's own rows, in its expert order
        else:
            from ..model.native_weight_carrier import _scope_anchors
            anchors=_scope_anchors(self.gate.weight)
            if anchors and x.is_cuda and isinstance(self.experts, DenseExperts):
                # Keep the established FP32 router operand while routing its
                # BF16 weight derivative through the shared scope.
                from .fused_moe import invariant_linear
                logits=invariant_linear(x.float(),self.gate.weight,forward_storage_dtype=torch.float32,
                                        cotangent_anchors=anchors)
            elif anchors:
                from ..model.native_weight_carrier import stored_linear
                logits=stored_linear(x.float(),self.gate.weight,storage_dtype=torch.float32)
            elif x.is_cuda and isinstance(self.experts, DenseExperts):
                # Exact FP32 with one fixed tile per row: top-k routing flips near ties on any shape-dependent
                # rounding (and the NGC containers' TF32 override would otherwise run this FP32 product in TF32).
                # Expert precision/dispatch must not change the router's
                # arithmetic contract when grouped FP8 execution is selected.
                from .fused_moe import invariant_linear
                logits = invariant_linear(x.float(), self.gate.weight.float())
            else:
                logits = F.linear(x.float(), self.gate.weight.float())
            if self.active_experts is not None:
                logits = (logits.index_select(1, ids.to(logits.device)) if ids is not None
                          else logits[:, :self.active_experts])
        top_logits, top_index = logits.topk(self.top_k, dim=-1)
        if ids is not None:  # member positions -> the experts' own ids
            top_index = ids.to(top_index.device)[top_index]
        weights = torch.softmax(top_logits, dim=-1)  # = softmax over all, renormalised over the top k
        return top_index, weights

    def _fused(self, x):
        """The fused grouped kernel (maple/fused_moe.py) for frozen experts on CUDA: their weights, or None (the
        reference loop below, which also trains learned block scales)."""
        from ..common.native_higher_order import active_reference_arithmetic
        if (active_reference_arithmetic() is not None or not x.is_cuda or isinstance(self.experts, DenseExperts)
                or os.environ.get("NATLANG_MAPLE_FUSED_MOE", "1") == "0"):
            return None
        from .fused_moe import expert_weights
        return expert_weights(self.experts)

    def _invariant(self, x) -> bool:
        """Dense experts on CUDA under the "invariant" kernel (the default): maple/fused_moe.invariant_dense_experts."""
        from ..common.native_higher_order import active_reference_arithmetic
        return active_reference_arithmetic() is None and DENSE_MOE["kernel"] == "invariant" and x.is_cuda and isinstance(self.experts, DenseExperts)

    def _grouped(self, x) -> bool:
        """Dense (BF16, trainable) experts on CUDA under ``set_dense_moe_kernel("grouped")`` run as two grouped GEMMs
        (maple/fused_moe.dense_grouped_experts) instead of the per-expert loop below (the reference, the default)."""
        from ..common.native_higher_order import active_reference_arithmetic
        return (active_reference_arithmetic() is None and DENSE_MOE["kernel"] == "grouped" and x.is_cuda and x.dtype == torch.bfloat16
                and isinstance(self.experts, DenseExperts) and hasattr(torch, "_grouped_mm"))

    def forward(self, h):
        shape = h.shape
        x = h.reshape(-1, shape[-1])
        with _profile_scope("moe.route"):
            index, weights = self.route(x)
        if self.statistics is not None:
            self._record(index, weights)
        with _profile_scope("moe.experts"):
            return self._experts(h, x, index, weights, shape)

    def _experts(self, h, x, index, weights, shape):
        projections = self._fused(x)
        if projections is not None:
            from .fused_moe import fused_experts
            return fused_experts(self.experts, x, index, weights, self.clamp, projections).to(h.dtype).view(shape)
        if self._invariant(x) and x.dtype == torch.bfloat16:
            from .fused_moe import invariant_dense_experts
            gate_up, down = self.experts.stacked(x.dtype)
            return invariant_dense_experts(x, index, weights, gate_up, down, self.clamp).to(h.dtype).view(shape)
        if self._grouped(x):
            from .fused_moe import dense_grouped_experts
            gate_up, down = self.experts.stacked(x.dtype)
            return dense_grouped_experts(x, index, weights, gate_up, down, self.clamp).to(h.dtype).view(shape)
        flat = index.reshape(-1)
        order = flat.argsort()
        token = order // self.top_k
        counts = torch.bincount(flat, minlength=self.gate.weight.shape[0]).tolist()
        out = torch.zeros(flat.numel(), shape[-1], device=x.device, dtype=x.dtype)
        if isinstance(self.experts, DenseExperts):
            gate_ups, downs = self.experts.unbound(x.dtype)
            run = lambda xs, e: self.experts.run_weights(xs, gate_ups[e], downs[e])
        else:
            run = lambda xs, e: self.experts.run(xs, e, x.dtype)
        start = 0
        for expert, count in enumerate(counts):
            if count:
                rows = order[start:start + count]
                out[rows] = run(x[token[start:start + count]], expert)
                start += count
        out = (out.view(-1, self.top_k, shape[-1]).float() * weights[..., None]).sum(1)
        return out.to(h.dtype).view(shape)

    @torch.no_grad()
    def _record(self, index, weights):
        n = self.gate.weight.shape[0]
        stats = self.statistics
        mass = torch.zeros(n, device=weights.device).index_add_(0, index.reshape(-1), weights.reshape(-1).float())
        count = torch.bincount(index.reshape(-1), minlength=n).float()
        stats["mass"] = stats.get("mass", 0) + mass.cpu()
        stats["count"] = stats.get("count", 0) + count.cpu()
        stats["tokens"] = stats.get("tokens", 0) + index.shape[0]
