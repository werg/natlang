"""Layer-range execution of an HF LFM2 model with an explicit, snapshot-able cache.

The Neuralese write procedure runs the shallow layers 0..k-1 position by position,
then the upper layers k..D-1 blockwise over the collected residuals, then restores
every layer to the block start and prefills the completed payload. HF's own cache
cannot do this: its short-convolution path treats any chunk that follows cached state
as a single token. This module therefore runs the HF modules' weights through its own
forward, with a cache that never changes in place, so a snapshot is just a reference.

Equivalence with HF's forward is a tested invariant (tests/neuralese).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_BASE = "LiquidAI/LFM2.5-350M"
DEFAULT_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"

# Proposed in S0 §10: two of LFM2.5's unused reserved tokens carry the block markers.
OPEN_TOKEN = "<|reserved_7|>"
CLOSE_TOKEN = "<|reserved_8|>"
OPEN_NAME = "<|neuralese|>"
CLOSE_NAME = "<|/neuralese|>"


class KVBuffer:
    """Preallocated key/value storage shared by the attention states of one decoding chain.

    Used only without autograd. `frontier` is how far the storage has been written. A state
    may append in place only when it ends exactly at the frontier; any other state (an
    earlier snapshot) copies its prefix into fresh storage first. So a state's view never
    changes after it is created, and a snapshot is still just a reference.
    """

    __slots__ = ("k", "v", "frontier")

    def __init__(self, k: torch.Tensor, v: torch.Tensor, frontier: int):
        self.k, self.v, self.frontier = k, v, frontier

    @property
    def capacity(self) -> int:
        return self.k.shape[2]


@dataclass(frozen=True)
class AttentionState:
    k: torch.Tensor  # [B, Hkv, T, hd], rotary already applied (a view into `buffer` when there is one)
    v: torch.Tensor
    buffer: KVBuffer | None = None

    @property
    def length(self) -> int:
        return self.k.shape[2]

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor, static: bool) -> "AttentionState":
        return append_kv(self, k_new, v_new, static)


def append_kv(prev: AttentionState | None, k_new: torch.Tensor, v_new: torch.Tensor, static: bool,
              headroom: int = 256) -> AttentionState:
    """Grow a layer's keys and values. `static` uses preallocated storage (no autograd)."""
    if not static:
        if prev is None:
            return AttentionState(k_new, v_new)
        return AttentionState(torch.cat([prev.k, k_new], 2), torch.cat([prev.v, v_new], 2))
    length = 0 if prev is None else prev.length
    end = length + k_new.shape[2]
    buffer = None if prev is None else prev.buffer
    if buffer is None or buffer.frontier != length or buffer.capacity < end or buffer.k.shape[0] != k_new.shape[0]:
        capacity = max(2 * end, end + headroom)
        shape = (k_new.shape[0], k_new.shape[1], capacity, k_new.shape[3])
        fresh = KVBuffer(k_new.new_empty(shape), v_new.new_empty(shape), length)
        if length:
            fresh.k[:, :, :length] = prev.k
            fresh.v[:, :, :length] = prev.v
        buffer = fresh
    buffer.k[:, :, length:end] = k_new
    buffer.v[:, :, length:end] = v_new
    buffer.frontier = end
    return AttentionState(buffer.k[:, :, :end], buffer.v[:, :, :end], buffer)


@dataclass(frozen=True)
class ConvState:
    window: torch.Tensor  # [B, C, L_cache - 1]: the last gated inputs (B * x) before the conv


@dataclass(frozen=True)
class PortCache:
    """Per-layer state plus the number of positions each layer has processed.

    Shallow layers advance during sketching while upper layers stay at the block
    start, so lengths are per layer. Tensors are never modified in place: every
    update builds new tensors, so keeping a reference to a cache is a snapshot.
    """

    states: tuple[AttentionState | ConvState | None, ...]
    lengths: tuple[int, ...]

    @staticmethod
    def empty(num_layers: int) -> "PortCache":
        return PortCache(states=(None,) * num_layers, lengths=(0,) * num_layers)

    def length(self, layer: int) -> int:
        return self.lengths[layer]

    def merge_layers(self, other: "PortCache", layers: range) -> "PortCache":
        """This cache with `layers` taken from `other`."""
        states, lengths = list(self.states), list(self.lengths)
        for i in layers:
            states[i], lengths[i] = other.states[i], other.lengths[i]
        return PortCache(tuple(states), tuple(lengths))

    def select(self, rows: slice | list[int]) -> "PortCache":
        """The cache restricted to some batch rows."""
        def pick(state):
            if state is None:
                return None
            if isinstance(state, AttentionState):
                return AttentionState(state.k[rows], state.v[rows])
            return ConvState(state.window[rows])
        return replace(self, states=tuple(pick(s) for s in self.states))


@dataclass(frozen=True)
class ControlTokens:
    open_id: int
    close_id: int

    @staticmethod
    def from_tokenizer(tokenizer) -> "ControlTokens":
        ids = tokenizer.convert_tokens_to_ids([OPEN_TOKEN, CLOSE_TOKEN])
        if any(i is None or i == tokenizer.unk_token_id for i in ids):
            raise ValueError("tokenizer lacks the reserved tokens used for Neuralese markers")
        return ControlTokens(open_id=ids[0], close_id=ids[1])

    def render(self, text: str) -> str:
        """Map the marker names in runtime-inserted structure to their reserved tokens.

        Only apply this to text the runtime itself produced: marker text inside
        ordinary content must stay text (S0 §3.3).
        """
        return text.replace(OPEN_NAME, OPEN_TOKEN).replace(CLOSE_NAME, CLOSE_TOKEN)


def resolve_base(path: str | None) -> str:
    """A local snapshot of the pinned base revision when it is cached, else the hub ID."""
    if path:
        return path
    cache = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots" / DEFAULT_REVISION
    return str(cache) if cache.exists() else DEFAULT_BASE


def load_backbone(base: str | None = None, lora: str | None = None, dtype=torch.bfloat16, device="cpu"):
    """Load the base model, optionally merging a LoRA adapter into it, frozen.

    S3 §1: the selected crisp-student LoRA is merged into the base to form the frozen
    crisp base; S3's own changes are separate deltas.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = resolve_base(base)
    # Load on the CPU first and move explicitly: no implicit device placement on a shared machine.
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, device_map="cpu")
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora, torch_device="cpu").merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(source)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


# Pinned revision of the hub kernel (works offline once cached; has an aarch64 sm_121 build).
CONV_KERNEL_REPO = "kernels-community/causal-conv1d"
CONV_KERNEL_REVISION = "2a73868ad241e86b8ebd9917d7be07986bfa645d"


def load_conv_kernel():
    """The hub `causal-conv1d` kernel (it has an sm_121/aarch64 build), or None.

    Opt out with NATLANG_NEURALESE_KERNELS=0. Only used for fresh prefills on CUDA.
    """
    import os

    if os.environ.get("NATLANG_NEURALESE_KERNELS", "1") == "0" or not torch.cuda.is_available():
        return None
    try:
        from kernels import get_kernel

        return get_kernel(CONV_KERNEL_REPO, revision=CONV_KERNEL_REVISION)
    except Exception:
        return None


class PortBackbone(nn.Module):
    """A frozen LFM2 causal LM run layer range by layer range, plus the trainable control-token rows.

    `fast=True` (the default) uses SDPA's native GQA and causal handling, so flash/cuDNN
    backends apply: `is_causal` for fresh prefills, no mask for single-token steps, a
    lower-right causal bias for chunks after cached state, and an explicit mask only for
    right-padded batches. Without autograd it also grows KV in preallocated storage and uses
    the `causal-conv1d` kernel for fresh prefills when available. `fast=False` is the
    original reference path (explicit boolean masks, `repeat_kv`, `torch.cat` growth),
    kept for equivalence tests and benchmarks.
    """

    def __init__(self, hf_model, controls: ControlTokens, noise: float = 0.02, seed: int = 0,
                 fast: bool = True, conv_kernel=None):
        super().__init__()
        self.fast = fast
        self.conv_kernel = conv_kernel
        self.hf = hf_model
        self.config = hf_model.config
        self.controls = controls
        self.num_layers = self.config.num_hidden_layers
        self.layer_types = list(self.config.layer_types)
        self.conv_window = self.config.conv_L_cache - 1
        embed = self.embedding_weight
        generator = torch.Generator().manual_seed(seed)
        mean = embed.detach().float().mean(0).cpu()
        rows = mean + noise * torch.randn(2, embed.shape[1], generator=generator)
        target_rms = embed.detach().float().pow(2).mean(-1).sqrt().mean().cpu()
        rows = rows * (target_rms / rows.pow(2).mean(-1, keepdim=True).sqrt())
        # Only these two rows of the (tied) embedding are trainable (S3 §2).
        self.control_rows = nn.Parameter(rows.to(device=embed.device, dtype=embed.dtype))

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self.hf.model.embed_tokens.weight

    @property
    def layers(self):
        return self.hf.model.layers

    def is_attention(self, layer: int) -> bool:
        return self.layer_types[layer] == "full_attention"

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        out = F.embedding(ids, self.embedding_weight)
        rows = self.control_rows.to(out.dtype)
        out = torch.where((ids == self.controls.open_id)[..., None], rows[0], out)
        return torch.where((ids == self.controls.close_id)[..., None], rows[1], out)

    def final_norm(self, h: torch.Tensor) -> torch.Tensor:
        return self.hf.model.embedding_norm(h)

    def logits(self, h_final: torch.Tensor) -> torch.Tensor:
        """LM-head logits from the last residual (before the final norm)."""
        normed = self.final_norm(h_final)
        logits = normed @ self.embedding_weight.t().to(normed.dtype)
        rows = self.control_rows.to(normed.dtype)
        logits = logits.clone()
        logits[..., self.controls.open_id] = normed @ rows[0]
        logits[..., self.controls.close_id] = normed @ rows[1]
        return logits

    def _rope(self, h: torch.Tensor, positions: torch.Tensor):
        return self.hf.model.rotary_emb(h, position_ids=positions)

    def run_layers(
        self,
        h: torch.Tensor,
        layers: range,
        cache: PortCache,
        positions: torch.Tensor | None = None,
        padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PortCache]:
        """Run `h` ([B, T, d]) through `layers`, continuing each layer from `cache`.

        All rows share the same per-layer cache length. `padding` ([B, T], 1 = real
        token) masks padded positions, as in HF; it is only meaningful for a fresh
        prefill whose padding is on the right.
        """
        from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb, repeat_kv

        states, lengths = list(cache.states), list(cache.lengths)
        batch, steps, _ = h.shape
        start = lengths[layers.start] if len(layers) else 0
        if any(lengths[i] != start for i in layers):
            raise ValueError("layers in one range must have processed the same number of positions")
        if positions is None:
            positions = torch.arange(start, start + steps, device=h.device).unsqueeze(0).expand(batch, -1)
        cos, sin = self._rope(h, positions)
        for i in layers:
            layer = self.layers[i]
            x = layer.operator_norm(h)
            if self.is_attention(i):
                attn = layer.self_attn
                hd = attn.head_dim
                q = attn.q_layernorm(attn.q_proj(x).view(batch, steps, -1, hd)).transpose(1, 2)
                k = attn.k_layernorm(attn.k_proj(x).view(batch, steps, -1, hd)).transpose(1, 2)
                v = attn.v_proj(x).view(batch, steps, -1, hd).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                prev = states[i]
                if self.fast:
                    state = append_kv(prev, k, v, static=not torch.is_grad_enabled())
                    out = self._attend_fast(attn, q, state.k, state.v, steps, prev is None, padding)
                    states[i] = state
                    h = h + attn.out_proj(out.transpose(1, 2).reshape(batch, steps, -1))
                    h = h + layer.feed_forward(layer.ffn_norm(h))
                    lengths[i] = start + steps
                    continue
                if prev is not None:
                    k = torch.cat([prev.k, k], dim=2)
                    v = torch.cat([prev.v, v], dim=2)
                total = k.shape[2]
                key_index = torch.arange(total, device=h.device)
                query_index = torch.arange(total - steps, total, device=h.device)
                allowed = key_index[None, :] <= query_index[:, None]  # [T, total]
                mask = allowed[None, None].expand(batch, 1, steps, total)
                if padding is not None:
                    key_valid = torch.ones(batch, total, dtype=torch.bool, device=h.device)
                    key_valid[:, total - steps:] = padding.bool()
                    mask = mask & key_valid[:, None, None, :]
                    # A padded query row would see nothing; let it see itself to stay finite.
                    mask = mask | torch.eye(total, device=h.device, dtype=torch.bool)[None, None, total - steps:]
                out = F.scaled_dot_product_attention(
                    q, repeat_kv(k, attn.num_key_value_groups), repeat_kv(v, attn.num_key_value_groups),
                    attn_mask=mask, scale=attn.scaling,
                )
                out = attn.out_proj(out.transpose(1, 2).reshape(batch, steps, -1))
                states[i] = AttentionState(k, v)
            else:
                conv = layer.conv
                if padding is not None:
                    x = x * padding[:, :, None].to(x.dtype)
                bcx = conv.in_proj(x).transpose(-1, -2)
                b, c, xx = bcx.chunk(3, dim=-2)
                gated = b * xx
                prev = states[i]
                if (self.fast and self.conv_kernel is not None and prev is None and h.is_cuda
                        and steps > self.conv_window):
                    conv_out = self.conv_kernel.causal_conv1d_fn(
                        gated.contiguous(), conv.conv.weight.squeeze(1), conv.conv.bias)
                    out = conv.out_proj((c * conv_out).transpose(-1, -2).contiguous())
                    states[i] = ConvState(gated[..., -self.conv_window:])
                    h = h + out
                    h = h + layer.feed_forward(layer.ffn_norm(h))
                    lengths[i] = start + steps
                    continue
                if prev is None:
                    prev_window = gated.new_zeros(batch, gated.shape[1], self.conv_window)
                else:
                    prev_window = prev.window
                window = torch.cat([prev_window, gated], dim=-1)
                conv_out = F.conv1d(window, conv.conv.weight, conv.conv.bias, groups=gated.shape[1])
                out = conv.out_proj((c * conv_out).transpose(-1, -2).contiguous())
                states[i] = ConvState(window[..., -self.conv_window:])
            h = h + out
            h = h + layer.feed_forward(layer.ffn_norm(h))
            lengths[i] = start + steps
        return h, PortCache(tuple(states), tuple(lengths))

    @staticmethod
    def _attend_fast(attn, q, k, v, steps: int, fresh: bool, padding: torch.Tensor | None) -> torch.Tensor:
        from torch.nn.attention.bias import causal_lower_right

        total = k.shape[2]
        if padding is None:
            if steps == 1:
                return F.scaled_dot_product_attention(q, k, v, scale=attn.scaling, enable_gqa=True)
            if fresh:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=attn.scaling, enable_gqa=True)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=causal_lower_right(steps, total),
                                                  scale=attn.scaling, enable_gqa=True)
        batch = q.shape[0]
        key_index = torch.arange(total, device=q.device)
        query_index = torch.arange(total - steps, total, device=q.device)
        mask = (key_index[None, :] <= query_index[:, None])[None, None].expand(batch, 1, steps, total)
        key_valid = torch.ones(batch, total, dtype=torch.bool, device=q.device)
        key_valid[:, total - steps:] = padding.bool()
        mask = (mask & key_valid[:, None, None, :]) | torch.eye(total, device=q.device, dtype=torch.bool)[None, None, total - steps:]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=attn.scaling, enable_gqa=True)

    def forward_embeds(
        self,
        embeds: torch.Tensor,
        cache: PortCache | None = None,
        cutoff: int | None = None,
        padding: torch.Tensor | None = None,
    ) -> dict:
        """Run all layers. Returns the final residual, logits, cache, and the residual after `cutoff` layers."""
        cache = cache or PortCache.empty(self.num_layers)
        result = {}
        h = embeds
        if cutoff is not None:
            h, cache = self.run_layers(h, range(0, cutoff), cache, padding=padding)
            result["h_cut"] = h
            h, cache = self.run_layers(h, range(cutoff, self.num_layers), cache, padding=padding)
        else:
            h, cache = self.run_layers(h, range(0, self.num_layers), cache, padding=padding)
        result.update(h_final=h, logits=self.logits(h), cache=cache)
        return result

    def forward_ids(self, ids: torch.Tensor, cache: PortCache | None = None, cutoff: int | None = None,
                    padding: torch.Tensor | None = None) -> dict:
        return self.forward_embeds(self.embed(ids), cache=cache, cutoff=cutoff, padding=padding)
