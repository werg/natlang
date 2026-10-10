"""Port backbones for Qwen3-family decoders (Qwen3-0.6B; Maple reuses the attention path, maple-qat §5).

Same interface as the LFM2 ``PortBackbone`` (embed, logits, run_layers over a layer range with a ``PortCache``,
forward_embeds with a cutoff), so the S3 trainer, heads and server run unchanged. Differences from LFM2: every layer
is attention (optionally sliding-window), the output head may be untied (then the two Neuralese markers get their
own trainable output rows), and the markers are unused vocabulary rows shared by Maple and Qwen3 (same IDs).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .lfm2_port import AttentionState, ControlTokens, PortBackbone, PortCache, _left_pad_mask, append_kv
from .layer_staging import staged_layers

# Unused rows of the shared Qwen3/Maple vocabulary (151,669..151,935 have no token): the marker IDs are the same
# in every model of the shared Neuralese space (MAPLE_QWEN_JOINT §1).
QWEN_OPEN_ID = 151669
QWEN_CLOSE_ID = 151670


def qwen_controls(tokenizer=None) -> ControlTokens:
    if tokenizer is not None:
        taken = set(tokenizer.get_vocab().values())
        if QWEN_OPEN_ID in taken or QWEN_CLOSE_ID in taken:
            raise ValueError("the Neuralese marker rows are assigned tokens in this tokenizer")
    return ControlTokens(open_id=QWEN_OPEN_ID, close_id=QWEN_CLOSE_ID)


# Mellum 2.x has no unused vocabulary rows: its markers are two spare added tokens (never produced by its tokenizer
# from ordinary text, so they cannot collide with content).
MELLUM_MARKERS = ("<|extra_token_7|>", "<|extra_token_8|>")


def family_controls(model, tokenizer=None) -> ControlTokens:
    """The Neuralese markers of a Maple-family backbone: Qwen/Maple unused rows, or Mellum's spare added tokens."""
    if getattr(model.config, "model_type", "maple") == "mellum":
        if tokenizer is None:
            raise ValueError("Mellum markers are resolved through its tokenizer")
        vocab = tokenizer.get_vocab()
        if any(m not in vocab for m in MELLUM_MARKERS):
            raise ValueError("tokenizer lacks the Mellum Neuralese marker tokens")
        return ControlTokens(open_id=vocab[MELLUM_MARKERS[0]], close_id=vocab[MELLUM_MARKERS[1]])
    return qwen_controls(tokenizer)


def load_qwen_backbone(path: str, dtype=torch.bfloat16, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


class QwenPortBackbone(PortBackbone):
    """A frozen Qwen3 causal LM run layer range by layer range, plus the trainable marker rows."""

    attention_checkpoint_prefixes = False

    def __init__(self, hf_model, controls: ControlTokens | None = None, noise: float = 0.02, seed: int = 0,
                 fast: bool = True, markers: bool = True):
        nn.Module.__init__(self)
        self.fast = fast
        # Training memory controls shared with the LFM2 port (set by the trainers): per-layer activation checkpointing
        # and token-chunked feed-forward. Both are exact; checkpointing trades recomputation for activation memory.
        self.checkpoint_layers = False
        self.checkpoint_preserve_rng = True
        self.ffn_chunk_tokens = 0
        self.conv_kernel = None
        self.hf = hf_model
        self.config = hf_model.config
        self.controls = controls or qwen_controls()
        self.num_layers = self.config.num_hidden_layers
        self.layer_types = list(getattr(self.config, "layer_types", None) or ["full_attention"] * self.num_layers)
        self.conv_window = 0
        self.tied = self.output_weight is self.embedding_weight
        if not markers:  # a plain stack runner (no Neuralese markers): embed/logits are not used
            return
        self.control_rows = nn.Parameter(self._marker_rows(self.embedding_weight, noise, seed))
        if not self.tied:
            self.control_head_rows = nn.Parameter(self._marker_rows(self.output_weight, noise, seed + 1))

    @staticmethod
    def _marker_rows(table: torch.Tensor, noise: float, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        mean = table.detach().float().mean(0).cpu()
        rows = mean + noise * torch.randn(2, table.shape[1], generator=generator)
        target_rms = table.detach().float().pow(2).mean(-1).sqrt().mean().cpu()
        rows = rows * (target_rms / rows.pow(2).mean(-1, keepdim=True).sqrt())
        return rows.to(device=table.device, dtype=table.dtype)

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self.hf.model.embed_tokens.weight

    @property
    def output_weight(self) -> torch.Tensor:
        return self.hf.get_output_embeddings().weight

    @property
    def final_norm_weight(self) -> torch.Tensor:
        return self.hf.model.norm.weight

    @property
    def norm_eps(self) -> float:
        return float(self.config.rms_norm_eps)

    def is_attention(self, layer: int) -> bool:
        return True

    def window(self, layer: int) -> int | None:
        if self.layer_types[layer] == "sliding_attention":
            return int(self.config.sliding_window)
        return None

    def final_norm(self, h: torch.Tensor) -> torch.Tensor:
        return self.hf.model.norm(h)

    def _rope(self, h, positions):
        return self.hf.model.rotary_emb(h, position_ids=positions)

    def _apply_rope(self, layer: int, q, k, cos, sin):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        return apply_rotary_pos_emb(q, k, cos, sin)

    def logits(self, h_final: torch.Tensor) -> torch.Tensor:
        normed = self.final_norm(h_final)
        logits = (normed @ self.output_weight.t().to(normed.dtype)).clone()
        rows = (self.control_rows if self.tied else self.control_head_rows).to(normed.dtype)
        logits[..., self.controls.open_id] = normed @ rows[0]
        logits[..., self.controls.close_id] = normed @ rows[1]
        return logits

    @staged_layers
    def run_layers(self, h, layers, cache, positions=None, padding=None, left_pad=None, _checkpoint_layer=False):
        if self.checkpoint_layers and torch.is_grad_enabled() and not _checkpoint_layer:
            return self._run_checkpointed(h, layers, cache, positions, padding, left_pad)
        states, lengths = list(cache.states), list(cache.lengths)
        batch, steps, _ = h.shape
        start = lengths[layers.start] if len(layers) else 0
        if any(lengths[i] != start for i in layers):
            raise ValueError("layers in one range must have processed the same number of positions")
        pad = cache.pad if cache.pad is not None else left_pad
        pad_offsets = cache.pad_offsets
        if pad is not None and pad_offsets is None:
            pad_offsets = tuple(pad.detach().cpu().tolist())
        if left_pad is not None and cache.pad is not None and left_pad is not cache.pad and not torch.equal(left_pad, cache.pad):
            raise ValueError("left_pad disagrees with the cache")
        if pad is not None and padding is not None:
            raise ValueError("left and right padding cannot be combined")
        if positions is None:
            positions = torch.arange(start, start + steps, device=h.device).unsqueeze(0).expand(batch, -1)
            if pad is not None:
                positions = (positions - pad[:, None]).clamp(min=0)
        cos, sin = self._rope(h, positions)
        for i in layers:
            layer = self.layers[i]
            attn = layer.self_attn
            hd = attn.head_dim
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).view(batch, steps, -1, hd)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).view(batch, steps, -1, hd)).transpose(1, 2)
            v = attn.v_proj(x).view(batch, steps, -1, hd).transpose(1, 2)
            q, k = self._apply_rope(i, q, k, cos, sin)
            prev = states[i]
            if self.fast:
                state = append_kv(prev, k, v, static=not torch.is_grad_enabled())
            else:
                state = AttentionState(k, v) if prev is None else AttentionState(
                    torch.cat([prev.k, k], 2), torch.cat([prev.v, v], 2))
            window = self.window(i)
            total = state.k.shape[2]
            if window is not None and total > window and pad is None and padding is None:
                out = _attend_window(attn, q, state.k, state.v, steps, window)
            elif window is not None and total > window:
                mask = _window_mask(steps, total, window, h.device)[None, None].expand(batch, 1, steps, total)
                if pad is not None:
                    mask = mask & _left_pad_mask(pad, steps, total, h.device)
                if padding is not None:
                    mask = mask & _right_pad_keys(padding, steps, total)
                    mask = mask | torch.eye(total, device=h.device, dtype=torch.bool)[None, None, total - steps:]
                out = F.scaled_dot_product_attention(q, state.k, state.v, attn_mask=mask, scale=attn.scaling,
                                                     enable_gqa=True)
            else:
                out = self._attend_fast(attn, q, state.k, state.v, steps, prev is None, padding, pad, pad_offsets)
            states[i] = state
            h = h + attn.o_proj(out.transpose(1, 2).reshape(batch, steps, -1))
            h = self._feed_forward(layer, h)
            lengths[i] = start + steps
        return h, PortCache(tuple(states), tuple(lengths), pad, pad_offsets)


    def _feed_forward(self, layer, h):
        """Residual feed-forward, in token chunks when `ffn_chunk_tokens` is set (token-local, so exact): bounds the
        expanded-width and expert-dispatch temporaries of long sequences."""
        chunk = self.ffn_chunk_tokens
        if chunk <= 0 or h.shape[1] <= chunk:
            return h + layer.mlp(layer.post_attention_layernorm(h))
        return torch.cat([part + layer.mlp(layer.post_attention_layernorm(part)) for part in h.split(chunk, dim=1)], 1)

    def _run_checkpointed(self, h, layers, cache, positions, padding, left_pad):
        """`run_layers` with each layer under activation checkpointing (as the LFM2 port): only the layer's own
        cache tensors enter the checkpoint, and the updated state is returned as tensors, never a captured cache."""
        from torch.utils.checkpoint import checkpoint

        pad = cache.pad if cache.pad is not None else left_pad
        pad_offsets = cache.pad_offsets
        if pad is not None and pad_offsets is None:
            pad_offsets = tuple(pad.detach().cpu().tolist())
        for i in layers:
            previous = cache.states[i]
            fields = () if previous is None else previous.fields()
            start, size = cache.lengths[i], self.num_layers

            def run_one(value, pos, padding_arg, left_pad_arg, pad_arg, *tensors, index=i, start=start, size=size):
                states, lengths = [None] * size, [0] * size
                if tensors:
                    states[index] = AttentionState.from_fields(tensors)
                lengths[index] = start
                single = PortCache(tuple(states), tuple(lengths), pad_arg, pad_offsets)
                output, updated = self.run_layers(value, range(index, index + 1), single, positions=pos,
                                                  padding=padding_arg, left_pad=left_pad_arg, _checkpoint_layer=True)
                return (output, *updated.states[index].fields())

            h, *updated = checkpoint(run_one, h, positions, padding, left_pad, cache.pad, *fields,
                                     use_reentrant=False, preserve_rng_state=self.checkpoint_preserve_rng)
            states, lengths = list(cache.states), list(cache.lengths)
            states[i] = AttentionState.from_fields(updated)
            lengths[i] = start + h.shape[1]
            cache = PortCache(tuple(states), tuple(lengths), pad, pad_offsets)
        return h, cache



def _attend_window(attn, q, k, v, steps: int, window: int) -> torch.Tensor:
    """Sliding-window causal attention without padding, banded: each block of `window` queries attends to its own
    keys only (at most 2 * window - 1), so the cost is linear in length rather than a masked full T x total. One query
    needs no mask at all (its keys are exactly the last `window`), which keeps decoding on the fast SDPA kernels; a
    block passes a small band mask with the key heads expanded, since a mask with GQA would fall back to SDPA's math
    backend. Same result as the explicit `_window_mask` path."""
    total = k.shape[2]
    start = total - steps
    repeats = q.shape[1] // k.shape[1]
    outputs = []
    for first in range(0, steps, window):
        last = min(steps, first + window)
        query_start = start + first
        key_start = max(0, query_start - window + 1)
        qq = q[:, :, first:last]
        kk, vv = k[:, :, key_start:start + last], v[:, :, key_start:start + last]
        if last - first == 1:
            outputs.append(F.scaled_dot_product_attention(qq, kk, vv, scale=attn.scaling, enable_gqa=True))
            continue
        query_index = torch.arange(query_start, start + last, device=q.device)
        key_index = torch.arange(key_start, start + last, device=q.device)
        distance = query_index[:, None] - key_index[None, :]
        mask = (distance >= 0) & (distance < window)
        outputs.append(F.scaled_dot_product_attention(qq, kk.repeat_interleave(repeats, 1),
                                                      vv.repeat_interleave(repeats, 1), attn_mask=mask,
                                                      scale=attn.scaling))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, 2)


def _window_mask(steps: int, total: int, window: int, device) -> torch.Tensor:
    """[T, total]: causal and at most ``window`` positions back (the query itself included)."""
    key_index = torch.arange(total, device=device)
    query_index = torch.arange(total - steps, total, device=device)
    distance = query_index[:, None] - key_index[None, :]
    return (distance >= 0) & (distance < window)


def _right_pad_keys(padding: torch.Tensor, steps: int, total: int) -> torch.Tensor:
    key_valid = torch.ones(padding.shape[0], total, dtype=torch.bool, device=padding.device)
    key_valid[:, total - steps:] = padding.bool()
    return key_valid[:, None, None, :]
