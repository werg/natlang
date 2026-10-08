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
    tail_k: torch.Tensor  # [B, Hkv, T, hd], rotary already applied
    tail_v: torch.Tensor
    buffer: KVBuffer | None = None
    prefix_k: torch.Tensor | None = None
    prefix_v: torch.Tensor | None = None

    @property
    def k(self):
        return self.tail_k if self.prefix_k is None else torch.cat([self.prefix_k, self.tail_k], 2)

    @property
    def v(self):
        return self.tail_v if self.prefix_v is None else torch.cat([self.prefix_v, self.tail_v], 2)

    @property
    def length(self) -> int:
        return self.tail_k.shape[2] + (self.prefix_k.shape[2] if self.prefix_k is not None else 0)

    def fields(self):
        """Checkpoint inputs share a long prefix rather than copying it per token."""
        if self.prefix_k is None:
            return self.tail_k, self.tail_v
        return self.prefix_k, self.prefix_v, self.tail_k, self.tail_v

    @staticmethod
    def from_fields(fields):
        if len(fields) == 2:
            return AttentionState(*fields)
        if len(fields) == 4:
            return AttentionState(fields[2], fields[3], prefix_k=fields[0], prefix_v=fields[1])
        raise ValueError('invalid attention cache tensor fields')

    def select(self, rows):
        return AttentionState(self.tail_k[rows], self.tail_v[rows],
                              prefix_k=None if self.prefix_k is None else self.prefix_k[rows],
                              prefix_v=None if self.prefix_v is None else self.prefix_v[rows])

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor, static: bool) -> "AttentionState":
        return append_kv(self, k_new, v_new, static)


def append_kv(prev: AttentionState | None, k_new: torch.Tensor, v_new: torch.Tensor, static: bool,
              headroom: int = 256) -> AttentionState:
    """Grow a layer's keys and values. `static` uses preallocated storage (no autograd)."""
    if not static:
        if prev is None:
            return AttentionState(k_new, v_new)
        # Keep the differentiable prompt prefix once. Saving a full concatenated
        # prefix as each token's checkpoint input costs O(context * recurrence).
        # Attention still sees identical materialized K/V; gradients accumulate
        # into the shared prefix and every generated suffix without detaching.
        if prev.prefix_k is None:
            return AttentionState(k_new, v_new, prefix_k=prev.tail_k, prefix_v=prev.tail_v)
        return AttentionState(torch.cat([prev.tail_k, k_new], 2), torch.cat([prev.tail_v, v_new], 2),
                              prefix_k=prev.prefix_k, prefix_v=prev.prefix_v)
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
    # [B] number of left-padding positions per row (batched prefixes of different lengths),
    # or None. Padded keys are masked for every later position, and rotary positions are
    # shifted so each row's first real token is at position 0.
    pad: torch.Tensor | None = None
    # Immutable host copy of fixed left-padding layout. Ragged CUDA attention
    # must not synchronize on int(pad[row]) at every layer and recurrent token.
    pad_offsets: tuple[int, ...] | None = None

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
        return PortCache(tuple(states), tuple(lengths), self.pad, self.pad_offsets)

    def select(self, rows: slice | list[int]) -> "PortCache":
        """The cache restricted to some batch rows."""
        def pick(state):
            if state is None:
                return None
            if isinstance(state, AttentionState):
                return state.select(rows)
            return ConvState(state.window[rows])
        pad = None if self.pad is None else self.pad[rows]
        offsets = None if self.pad_offsets is None else (self.pad_offsets[rows] if isinstance(rows, slice)
                   else tuple(self.pad_offsets[i] for i in rows))
        return replace(self, states=tuple(pick(s) for s in self.states), pad=pad, pad_offsets=offsets)

    def repeat_interleave(self, repeats: int) -> "PortCache":
        """Branch each row without changing causal positions or scope adjoints.

        Singleton prefixes use expanded views, avoiding N copies of a long
        context in saved checkpoint inputs. Never reuse a mutable KV buffer
        between branches. Differentiable layer updates remain immutable.
        """
        if repeats < 1:
            raise ValueError('cache branch count must be positive')
        def repeat(tensor):
            if tensor is None:
                return None
            return (tensor.expand(repeats, *tensor.shape[1:]) if tensor.shape[0] == 1
                    else tensor.repeat_interleave(repeats, dim=0))
        def branch(state):
            if state is None:
                return None
            if isinstance(state, AttentionState):
                return AttentionState(repeat(state.tail_k), repeat(state.tail_v),
                                      prefix_k=repeat(state.prefix_k), prefix_v=repeat(state.prefix_v))
            return ConvState(repeat(state.window))
        offsets = (None if self.pad_offsets is None
                   else tuple(offset for offset in self.pad_offsets for _ in range(repeats)))
        return replace(self, states=tuple(branch(s) for s in self.states),
                       pad=repeat(self.pad), pad_offsets=offsets)


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
    import os
    hub = Path(os.environ.get('HF_HUB_CACHE') or
               str(Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'hub'))
    cache = hub / "models--LiquidAI--LFM2.5-350M/snapshots" / DEFAULT_REVISION
    return str(cache) if cache.exists() else DEFAULT_BASE


def load_backbone(base: str | None = None, lora: str | None = None, dtype=torch.bfloat16, device="cpu"):
    """Load the base model, optionally merging a LoRA adapter into it, frozen.

    S3 §1: the selected crisp-student LoRA is merged into the base to form the frozen
    crisp base; S3's own changes are separate deltas.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = resolve_base(base)
    pinned = {"revision": DEFAULT_REVISION} if source == DEFAULT_BASE else {}
    # Load on the CPU first and move explicitly: no implicit device placement on a shared machine.
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, device_map="cpu", **pinned)
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora, torch_device="cpu").merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(source, **pinned)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def backbone_identity(lora: str | None, base: str | None = None) -> dict:
    """What a port checkpoint was trained on: the base (and its pinned revision) and the merged student LoRA with
    the SHA-256 of its weights. Empty for the plain default base, as checkpoints from before this was recorded."""
    import hashlib

    if not lora and not base:
        return {}
    identity = {"base": base or DEFAULT_BASE, **({"revision": DEFAULT_REVISION} if not base else {})}
    if lora:
        weights = Path(lora) / "adapter_model.safetensors"
        digest = hashlib.sha256()
        with open(weights, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        identity.update({"student_lora": str(Path(lora).resolve()), "student_lora_sha256": digest.hexdigest()})
    return identity


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


def feed_forward_residual(layer, h, chunk_tokens=0):
    """Token-local FFN chunks preserve gradients without recomputing activations.

    Attention and convolution retain their full context. Chunking the norm and
    FFN bounds temporary expanded-width projections; saved activation storage
    still scales with total tokens unless the caller enables exact CPU offload.
    """
    if chunk_tokens <= 0 or h.shape[1] <= chunk_tokens:
        return h + layer.feed_forward(layer.ffn_norm(h))
    return torch.cat([part + layer.feed_forward(layer.ffn_norm(part))
                      for part in h.split(chunk_tokens, dim=1)], dim=1)


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

    attention_checkpoint_prefixes = True

    def __init__(self, hf_model, controls: ControlTokens, noise: float = 0.02, seed: int = 0,
                 fast: bool = True, conv_kernel=None):
        super().__init__()
        self.fast = fast
        self.conv_kernel = conv_kernel
        self.checkpoint_preserve_rng = True
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
    def output_weight(self) -> torch.Tensor:
        """The LM head's matrix (tied to the embedding in LFM2)."""
        return self.embedding_weight

    @property
    def final_norm_weight(self) -> torch.Tensor:
        return self.hf.model.embedding_norm.weight

    @property
    def norm_eps(self) -> float:
        return float(self.config.norm_eps)

    @property
    def layers(self):
        return self.hf.model.layers

    def is_attention(self, layer: int) -> bool:
        return self.layer_types[layer] == "full_attention"

    def elide_checkpoint_rng(self):
        """Opt out of RNG snapshots only for the deterministic native layer path.

        This forward uses explicit zero-dropout SDPA and deterministic conv/FFN
        operations. PEFT wrappers may add dropout; refuse those configurations.
        Call after adapter installation, before training. Sampling outside these
        checkpoint closures still owns and preserves its ordinary RNG state.
        """
        for module in self.layers.modules():
            if isinstance(module, nn.modules.dropout._DropoutNd) and module.p:
                raise ValueError('checkpoint RNG elision requires zero adapter/layer dropout')
            if isinstance(module, nn.RReLU):
                raise ValueError('checkpoint RNG elision does not support stochastic activations')
        self.checkpoint_preserve_rng = False

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

    def isolated_sequence(self, fixed, replacements, cache, *, cutoff):
        from .isolated_sequence import isolated_sequence
        return isolated_sequence(self, fixed, replacements, cache, cutoff=cutoff)

    def run_layers(
        self,
        h: torch.Tensor,
        layers: range,
        cache: PortCache,
        positions: torch.Tensor | None = None,
        padding: torch.Tensor | None = None,
        left_pad: torch.Tensor | None = None,
        _checkpoint_layer: bool = False,
    ) -> tuple[torch.Tensor, PortCache]:
        """Run `h` ([B, T, d]) through `layers`, continuing each layer from `cache`.

        All rows share the same per-layer cache length. `padding` ([B, T], 1 = real
        token) masks padded positions, as in HF; it is only meaningful for a fresh
        prefill whose padding is on the right. `left_pad` ([B], fresh prefill only) or a
        cache that carries `pad` handles left-padded rows: their padded keys stay masked for
        every later position, so rows of different prefix lengths can write in lockstep.
        """
        pad = cache.pad if cache.pad is not None else left_pad
        pad_offsets = cache.pad_offsets
        if pad is not None and pad_offsets is None:
            pad_offsets = tuple(pad.detach().cpu().tolist())
        if getattr(self, 'checkpoint_layers', False) and torch.is_grad_enabled() and not _checkpoint_layer:
            from torch.utils.checkpoint import checkpoint
            for i in layers:
                if getattr(self, 'checkpoint_attention_only', False) and not self.is_attention(i):
                    h, cache = self.run_layers(h, range(i, i + 1), cache, positions=positions,
                                               padding=padding, left_pad=left_pad, _checkpoint_layer=True)
                    continue
                # Only this layer's state participates in its computation.
                # Passing every layer's cache multiplies checkpoint bookkeeping
                # and exposes unrelated tensors to saved hooks. Keep the used
                # fields explicit and return tensors, never a captured cache.
                previous = cache.states[i]
                attention = self.is_attention(i)
                fields = (() if previous is None else previous.fields()
                          if attention else (previous.window,))
                start, size = cache.lengths[i], self.num_layers
                def run_one(value, pos, padding_arg, left_pad_arg, pad, *tensors,
                            index=i, attention=attention, start=start, size=size, pad_offsets=pad_offsets):
                    states = [None] * size
                    lengths = [0] * size
                    if tensors:
                        states[index] = AttentionState.from_fields(tensors) if attention else ConvState(tensors[0])
                    lengths[index] = start
                    current_cache = PortCache(tuple(states), tuple(lengths), pad, pad_offsets)
                    # Forward and recompute must build the same graph: Maple's dynamic-shape compiled RMSNorm can
                    # pick a different graph on recompute (CheckpointError at Maple recurrence step 89), so the
                    # checkpointed layer runs it eagerly both times. LFM2 norms are unaffected.
                    from ..maple.model import eager_rms_norm
                    with eager_rms_norm():
                        output, updated = self.run_layers(value, range(index, index + 1), current_cache,
                                                         positions=pos, padding=padding_arg, left_pad=left_pad_arg,
                                                         _checkpoint_layer=True)
                    state = updated.states[index]
                    return (output, *state.fields()) if attention else (output, state.window)
                result = checkpoint(run_one, h, positions, padding, left_pad, cache.pad,
                                    *fields, use_reentrant=False,
                                    preserve_rng_state=self.checkpoint_preserve_rng)
                h, *updated = result
                states, lengths = list(cache.states), list(cache.lengths)
                states[i] = AttentionState.from_fields(updated) if attention else ConvState(updated[0])
                lengths[i] = start + h.shape[1]
                cache = PortCache(tuple(states), tuple(lengths), cache.pad if cache.pad is not None else left_pad,
                                  pad_offsets)
            return h, cache

        from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb, repeat_kv

        states, lengths = list(cache.states), list(cache.lengths)
        batch, steps, _ = h.shape
        start = lengths[layers.start] if len(layers) else 0
        if any(lengths[i] != start for i in layers):
            raise ValueError("layers in one range must have processed the same number of positions")
        pad = cache.pad if cache.pad is not None else left_pad
        if left_pad is not None and cache.pad is not None and left_pad is not cache.pad and not torch.equal(left_pad, cache.pad):
            raise ValueError("left_pad disagrees with the cache")
        if pad is not None and padding is not None:
            raise ValueError("left and right padding cannot be combined")
        if positions is None:
            positions = torch.arange(start, start + steps, device=h.device).unsqueeze(0).expand(batch, -1)
            if pad is not None:
                positions = (positions - pad[:, None]).clamp(min=0)
        chunk_real = None
        if pad is not None:
            # [B, T]: 1 where this chunk's position is a real (non-pad) token.
            chunk_real = (torch.arange(start, start + steps, device=h.device)[None] >= pad[:, None])
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
                    out = self._attend_fast(attn, q, state.k, state.v, steps, prev is None, padding, pad, pad_offsets)
                    states[i] = state
                    h = h + attn.out_proj(out.transpose(1, 2).reshape(batch, steps, -1))
                    h = feed_forward_residual(layer, h, getattr(self, 'ffn_chunk_tokens', 0))
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
                if pad is not None:
                    mask = _left_pad_mask(pad, steps, total, h.device)
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
                if chunk_real is not None:
                    x = x * chunk_real[:, :, None].to(x.dtype)
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
                    h = feed_forward_residual(layer, h, getattr(self, 'ffn_chunk_tokens', 0))
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
            h = feed_forward_residual(layer, h, getattr(self, 'ffn_chunk_tokens', 0))
            lengths[i] = start + steps
        return h, PortCache(tuple(states), tuple(lengths), pad, pad_offsets)

    @staticmethod
    def _attend_ragged(attn, q, k, v, *, left_pad=None, right_padding=None, left_offsets=None):
        """Causal attention on each row's real tokens, without a quadratic pad mask.

        Real positions equal masked attention. Padded query outputs are zero and
        are never used as real tokens; padding is also excluded from cached keys.
        Unpadding allows CUDA's fused causal/GQA kernels at long context lengths.
        """
        from torch.nn.attention.bias import causal_lower_right

        steps, total = q.shape[2], k.shape[2]
        rows = []
        for b in range(q.shape[0]):
            if left_pad is not None:
                key_start = min(total, left_offsets[b] if left_offsets is not None else int(left_pad[b]))
                query_start = max(0, min(steps, key_start - (total - steps)))
                real_queries = steps - query_start
                key_end = total
            else:
                query_start = 0
                real_queries = int(right_padding[b].sum())
                key_start = 0
                key_end = total - steps + real_queries
            if real_queries == 0:
                rows.append(q[b:b+1].new_zeros(1, q.shape[1], steps, q.shape[3]))
                continue
            qb = q[b:b+1, :, query_start:query_start+real_queries]
            kb, vb = k[b:b+1, :, key_start:key_end], v[b:b+1, :, key_start:key_end]
            options = dict(scale=attn.scaling, enable_gqa=True)
            if real_queries == kb.shape[2]:
                options['is_causal'] = True
            elif real_queries > 1:
                options['attn_mask'] = causal_lower_right(real_queries, kb.shape[2])
            out = F.scaled_dot_product_attention(qb, kb, vb, **options)
            rows.append(F.pad(out, (0, 0, query_start, steps-query_start-real_queries)))
        return torch.cat(rows)

    @staticmethod
    def _attend_fast(attn, q, k, v, steps: int, fresh: bool, padding: torch.Tensor | None,
                     pad: torch.Tensor | None = None, pad_offsets=None) -> torch.Tensor:
        from torch.nn.attention.bias import causal_lower_right

        total = k.shape[2]
        if q.is_cuda and (pad is not None or padding is not None):
            return PortBackbone._attend_ragged(attn, q, k, v, left_pad=pad, right_padding=padding,
                                             left_offsets=pad_offsets)
        if pad is not None:
            mask = _left_pad_mask(pad, steps, total, q.device)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=attn.scaling, enable_gqa=True)
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
        left_pad: torch.Tensor | None = None,
        logits: bool = True,
    ) -> dict:
        """Run all layers. Returns the final residual, logits, cache, and the residual after `cutoff` layers.

        `logits=False` skips the vocabulary projection (callers that need only a few positions
        project those themselves with `logits()`)."""
        cache = cache or PortCache.empty(self.num_layers)
        result = {}
        h = embeds
        if cutoff is not None:
            h, cache = self.run_layers(h, range(0, cutoff), cache, padding=padding, left_pad=left_pad)
            result["h_cut"] = h
            h, cache = self.run_layers(h, range(cutoff, self.num_layers), cache, padding=padding, left_pad=left_pad)
        else:
            h, cache = self.run_layers(h, range(0, self.num_layers), cache, padding=padding, left_pad=left_pad)
        result.update(h_final=h, logits=self.logits(h) if logits else None, cache=cache)
        return result

    def forward_ids(self, ids: torch.Tensor, cache: PortCache | None = None, cutoff: int | None = None,
                    padding: torch.Tensor | None = None, left_pad: torch.Tensor | None = None,
                    logits: bool = True) -> dict:
        return self.forward_embeds(self.embed(ids), cache=cache, cutoff=cutoff, padding=padding, left_pad=left_pad,
                                   logits=logits)


def _left_pad_mask(pad: torch.Tensor, steps: int, total: int, device) -> torch.Tensor:
    """[B, 1, T, total] boolean mask: causal, padded keys hidden; a padded query sees itself."""
    key_index = torch.arange(total, device=device)
    query_index = torch.arange(total - steps, total, device=device)
    causal = key_index[None, :] <= query_index[:, None]
    key_valid = key_index[None, :] >= pad[:, None]                      # [B, total]
    mask = causal[None] & key_valid[:, None, :]                          # [B, T, total]
    own = key_index[None, :] == query_index[:, None]
    mask = mask | (own[None] & (query_index[None, :] < pad[:, None])[:, :, None])
    return mask[:, None]
