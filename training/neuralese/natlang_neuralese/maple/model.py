"""Maple in plain PyTorch: the deployed (ternary) model, nested expert subsets, routing statistics.

Not deepgrove's remote code (written for Transformers 4.57, imports FlashAttention). The published BF16 checkpoint
is already ternary (maple-qat §2.0), so loading ternarizes nothing in practice; the rule is applied anyway so a
merged or modified checkpoint loads in deployed form. This module follows the llama.cpp graph, which is what we
deploy:

- attention: Q/K/V, per-head RMS norm on Q and K, partial rotary (first half of each head, NeoX layout) on sliding
  layers only, no position encoding on global layers; sliding layers see keys at distance < 512 (llama.cpp
  ``LLAMA_SWA_TYPE_STANDARD``; deepgrove's FlashAttention call uses ``window_size=(512, 0)``, i.e. distance <= 512:
  they differ only on keys exactly 512 back, and llama.cpp is what ships);
- MoE: FP32 router, softmax, top 8, renormalised; experts ``down(silu(min(gate, 7)) * clamp(up, -7, 7))``;
- every linear weight ternarized with Maple's rule (``ternary.py``); embedding, head, norms and router stay floating.

Experts are stored stacked per layer as int8 codes plus a BF16 scale per row (half the memory of BF16; exact) and
expanded per used expert. ``active_experts`` restricts routing to the first n experts of every layer, which is
exactly the nested model (MAPLE_NESTED §2: renormalisation over the top 8 cancels the softmax denominator).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from .ternary import ternary_codes, ternarize

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

    @staticmethod
    def from_dir(path) -> "MapleConfig":
        data = json.loads((Path(path) / "config.json").read_text())
        names = MapleConfig.__dataclass_fields__
        return MapleConfig(**{k: v for k, v in data.items() if k in names})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.private = nn.ParameterDict()  # per family member: additive gain correction, zero-initialised

    def forward(self, x):
        from .ternary import STATE

        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        weight = self.weight
        member = STATE["size"]
        if member is not None and STATE["enabled"] and str(member) in self.private:
            weight = weight + self.private[str(member)].to(weight.dtype)
        return weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """cos/sin over the rotary part of a head (``partial_rotary_factor * head_dim`` dimensions)."""

    def __init__(self, config: MapleConfig):
        super().__init__()
        self.dim = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        freqs = position_ids[..., None].float() * self.inv_freq.to(x.device).float()  # [B, T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


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


class Attention(nn.Module):
    def __init__(self, config: MapleConfig):
        super().__init__()
        d, hd = config.hidden_size, config.head_dim
        self.head_dim = hd
        self.scaling = hd ** -0.5
        self.q_proj = nn.Linear(d, config.num_attention_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, config.num_key_value_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, config.num_key_value_heads * hd, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * hd, d, bias=False)
        self.q_norm = RMSNorm(hd, config.rms_norm_eps)
        self.k_norm = RMSNorm(hd, config.rms_norm_eps)


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
        h = F.silu(gate.clamp(max=SWIGLU_CLAMP)) * up.clamp(-SWIGLU_CLAMP, SWIGLU_CLAMP)
        return self._matmul(h, self.down_codes, self.down_scale, getattr(self, "down_blocks", None), index)


class SparseMoE(nn.Module):
    def __init__(self, config: MapleConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = TernaryExperts(config.num_experts, config.hidden_size, config.moe_intermediate_size)
        self.active_experts: int | None = None
        self.statistics: dict | None = None
        # Private router rows per family member (MAPLE_NESTED §4): a trainable copy of the rows of the member's
        # experts. The member's GGUF carries them as its own ffn_gate_inp, so they export exactly.
        self.private_gate = nn.ParameterDict()

    def route(self, x):
        from .ternary import STATE

        key = str(STATE["size"])
        if self.active_experts is not None and STATE["enabled"] and key in self.private_gate:
            logits = F.linear(x.float(), self.private_gate[key])
        else:
            logits = F.linear(x.float(), self.gate.weight.float())
            if self.active_experts is not None:
                logits = logits[:, :self.active_experts]
        top_logits, top_index = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(top_logits, dim=-1)  # = softmax over all, renormalised over the top k
        return top_index, weights

    def forward(self, h):
        shape = h.shape
        x = h.reshape(-1, shape[-1])
        index, weights = self.route(x)
        if self.statistics is not None:
            self._record(index, weights)
        flat = index.reshape(-1)
        order = flat.argsort()
        token = order // self.top_k
        counts = torch.bincount(flat, minlength=self.gate.weight.shape[0]).tolist()
        out = torch.zeros(flat.numel(), shape[-1], device=x.device, dtype=x.dtype)
        start = 0
        for expert, count in enumerate(counts):
            if count:
                rows = order[start:start + count]
                out[rows] = self.experts.run(x[token[start:start + count]], expert, x.dtype)
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


class DecoderLayer(nn.Module):
    def __init__(self, config: MapleConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SparseMoE(config)


class MapleModel(nn.Module):
    def __init__(self, config: MapleConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.active_layers: int | None = None  # depth-nested member: early exit after this many layers

    def forward(self, input_ids=None, inputs_embeds=None, capture=(), layers: int | None = None):
        """Final-norm hidden states after ``layers`` layers (default: all; fewer = a depth-nested member's early
        exit), plus the residual stream after each layer count in ``capture``."""
        from .maple_port import MaplePortBackbone

        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        port = MaplePortBackbone.runner(self)
        depth = layers or self.active_layers or self.config.num_hidden_layers
        cache = _empty_cache(self.config.num_hidden_layers)
        captured, start = {}, 0
        for stop in sorted({c for c in capture if 0 < c < depth} | {depth}):
            h, cache = port.run_layers(h, range(start, stop), cache)
            captured[stop] = h
            start = stop
        return SimpleNamespace(last_hidden_state=self.norm(h), captured=captured)


def _empty_cache(n):
    from ..model.lfm2_port import PortCache

    return PortCache.empty(n)


class MapleForCausalLM(nn.Module):
    def __init__(self, config: MapleConfig):
        super().__init__()
        self.config = SimpleNamespace(**config.__dict__)
        self.model = MapleModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids=input_ids).last_hidden_state))

    def set_active_experts(self, n: int | None):
        """Width-only member (all layers, first ``n`` experts), keyed by ``n``."""
        self.set_member(None if n is None else str(n), experts=n)

    def set_member(self, key: str | None, experts: int | None = None, layers: int | None = None):
        """Run as family member ``key``: first ``experts`` experts per layer, first ``layers`` layers (early exit),
        and the member's private parts (router bias, norm gains, attention deltas). ``key=None``: the full model."""
        from .ternary import STATE

        STATE["size"] = key
        self.model.active_layers = layers
        for layer in self.model.layers:
            layer.mlp.active_experts = experts

    def add_private_router(self, n: int, key: str | None = None, layers: int | None = None) -> list[nn.Parameter]:
        """Trainable private router rows for member ``key`` (default ``str(n)``): a copy of the first ``n`` rows."""
        added = []
        for layer in self.model.layers[:layers]:
            parameter = nn.Parameter(layer.mlp.gate.weight.detach()[:n].float().clone())
            layer.mlp.private_gate[key or str(n)] = parameter
            added.append(parameter)
        return added

    def add_private_norms(self, key: str, layers: int | None = None) -> list[nn.Parameter]:
        """Zero-initialised gain corrections for member ``key``: every layer norm it uses and the final norm."""
        added = []
        norms = [self.model.norm]
        for layer in self.model.layers[:layers]:
            norms += [layer.input_layernorm, layer.post_attention_layernorm]
        for norm in norms:
            parameter = nn.Parameter(torch.zeros_like(norm.weight, dtype=torch.float32))
            norm.private[key] = parameter
            added.append(parameter)
        return added

    def collect_routing(self, on: bool = True) -> list[dict] | None:
        for layer in self.model.layers:
            layer.mlp.statistics = {} if on else None
        return [layer.mlp.statistics for layer in self.model.layers] if on else None

    @torch.no_grad()
    def order_experts(self, orders: list[torch.Tensor]):
        """Permute every layer's experts (and router rows) so that ``orders[layer][j]`` becomes expert j."""
        for layer, order in zip(self.model.layers, orders):
            layer.mlp.experts.permute(order)
            layer.mlp.gate.weight.copy_(layer.mlp.gate.weight[order.to(layer.mlp.gate.weight.device)])

    @torch.no_grad()
    def ternarize_attention(self):
        """Replace the attention weights by their ternary values (for frozen use; QAT instead parametrizes them)."""
        for layer in self.model.layers:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = getattr(layer.self_attn, proj)
                module.weight.copy_(ternarize(module.weight))


CHECKPOINT_RENAMES = {"model.word_embeddings.weight": "model.embed_tokens.weight"}


def load_maple(path, device="cpu", dtype=torch.bfloat16, ternary_attention: bool = True, layers: int | None = None,
               experts: int | None = None, cache: str | Path | None = None) -> MapleForCausalLM:
    """Load the BF16 checkpoint into the deployed form: experts as ternary codes, attention ternarized unless
    ``ternary_attention=False`` (QAT keeps the latent). ``layers``/``experts`` load a truncated model (tests).

    ``cache``: a directory of converted-state shards (codes and scales, ~21 GB in ~1 GB files). Read when it exists
    (sequential reads instead of 18,651 memory-mapped tensors), written after a fresh conversion otherwise. The published checkpoint
    is already ternary, so the cached attention equals the latent and serves QAT too."""
    if cache is not None and Path(cache).is_dir() and layers is None and experts is None:
        config = MapleConfig.from_dir(path)
        with torch.device("meta"):
            model = MapleForCausalLM(config)
        moved = {}
        # Shards of ~1 GB, each mapped, moved to the device and released before the next: the memory guard counts
        # page cache charged to the job's cgroup, so a single 21 GB mapping would count twice while loading.
        for shard in sorted(Path(cache).glob("shard-*.pt")):
            state = torch.load(shard, map_location="cpu", mmap=True, weights_only=True)
            for name in list(state):
                moved[name] = state.pop(name).to(device)
            del state
            with open(shard, "rb") as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        model.load_state_dict(moved, strict=True, assign=True)  # keeps the stored dtypes (BF16, FP32 router, int8)
        model.model.rotary_emb = RotaryEmbedding(config).to(device)
        model.requires_grad_(False)
        return model.eval()
    model = _convert(path, device, dtype, ternary_attention, layers, experts)
    if cache is not None and layers is None and experts is None:
        save_cache(model, cache)
    return model


def save_cache(model, cache, shard_bytes: int = 1 << 30):
    """Write the converted state as ~1 GB shards in directory ``cache``."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    shard, size, index = {}, 0, 0
    for name, tensor in model.state_dict().items():
        shard[name] = tensor.detach().cpu()
        size += tensor.numel() * tensor.element_size()
        if size >= shard_bytes:
            torch.save(shard, cache / f"shard-{index:04d}.pt")
            shard, size, index = {}, 0, index + 1
    if shard:
        torch.save(shard, cache / f"shard-{index:04d}.pt")


def _convert(path, device, dtype, ternary_attention, layers, experts) -> MapleForCausalLM:
    from safetensors import safe_open

    path = Path(path)
    config = MapleConfig.from_dir(path)
    if layers is not None:
        config.num_hidden_layers = layers
        config.layer_types = config.layer_types[:layers]
    if experts is not None:
        config.num_experts = experts
    with torch.device("meta"):
        model = MapleForCausalLM(config)
    model = model.to_empty(device=device)
    model.model.rotary_emb = RotaryEmbedding(config).to(device)
    for buffer_owner in model.modules():
        if isinstance(buffer_owner, TernaryExperts):
            for name, buf in list(buffer_owner._buffers.items()):
                buffer_owner._buffers[name] = torch.zeros_like(buf)
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, file in index.items():
        by_file.setdefault(file, []).append(name)
    params = dict(model.named_parameters())
    pending: dict[tuple[int, int], dict] = {}
    for file, names in sorted(by_file.items()):
        with safe_open(path / file, framework="pt", device="cpu") as f:
            for name in names:
                parts = name.split(".")
                if name.startswith("model.layers."):
                    layer = int(parts[2])
                    if layer >= config.num_hidden_layers:
                        continue
                    if parts[4] == "experts":
                        expert = int(parts[5])
                        if expert >= config.num_experts:
                            continue
                        slot = pending.setdefault((layer, expert), {})
                        slot[parts[6]] = f.get_tensor(name).to(device)
                        if len(slot) == 3:
                            model.model.layers[layer].mlp.experts.set_expert(
                                expert, slot["gate_proj"], slot["up_proj"], slot["down_proj"])
                            del pending[(layer, expert)]
                        continue
                    if parts[4] == "gate" and experts is not None:
                        params[name].data.copy_(f.get_tensor(name)[:experts].to(params[name].dtype))
                        continue
                target = CHECKPOINT_RENAMES.get(name, name)
                params[target].data.copy_(f.get_tensor(name).to(params[target].dtype))
    if pending:
        raise ValueError(f"incomplete experts in the checkpoint: {sorted(pending)[:4]}")
    model.to(dtype=dtype)
    model.model.rotary_emb = RotaryEmbedding(config).to(device)  # FP32 frequencies (BF16 would corrupt positions)
    for layer in model.model.layers:
        layer.mlp.gate.weight.data = layer.mlp.gate.weight.data.float()
    if ternary_attention:
        model.ternarize_attention()
    model.requires_grad_(False)
    return model.eval()
