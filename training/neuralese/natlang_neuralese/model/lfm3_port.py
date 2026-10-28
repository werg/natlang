"""LiquidAI LFM3 as a native port backbone: the shared Qwen-family layer-range runner with LFM3's layer forms.

LFM3 (owner switch 2026-10-28, ``plans/neuralese/LFM3_PORT_20261028.md``) interleaves gated short convolutions
(width ``conv_L_cache`` = 7), full attention (partial "zero-tail" RoPE) and sliding-window attention (its own RoPE
theta), each attention layer with q/k RMSNorm, ``o_proj`` and a per-head sigmoid output gate projected from the
normed attention input, and a sandwich norm around both the operator and the feed-forward block. Everything that is
not LFM3-specific (attention kernels, windows, caches, checkpointing, short convolution, gates) is the shared
mechanism in ``hf_port``/``lfm2_port``; this class only declares the forms and the LFM2-style layer names.

The exact parity reference is the vendored pinned remote code (``natlang_neuralese/third_party/lfm3``), never network
``trust_remote_code``. Neuralese role markers are the four provisioned role tokens, which land in LFM3's unused
vocabulary rows (125,033 onwards) without a resize; LFM2's ``<|reserved_7|>``/``<|reserved_8|>`` markers are not used.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .hf_port import QwenPortBackbone
from .lfm2_port import ControlTokens, feed_forward_residual

DEFAULT_BASE = "LiquidAI/LFM3-1.7B"
DEFAULT_REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
# The tokenizer's own vocabulary ends at 125,032; rows 125,033..127,999 of the tied embedding are unused.
FIRST_UNUSED_ROW = 125033


def resolve_base(path: str | None = None) -> str:
    """A local snapshot of the pinned revision. LFM3 is never fetched implicitly."""
    if path:
        return path
    hub = Path(os.environ.get('HF_HUB_CACHE') or
               str(Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'hub'))
    cache = hub / "models--LiquidAI--LFM3-1.7B/snapshots" / DEFAULT_REVISION
    if not cache.exists():
        raise FileNotFoundError(f"pinned LFM3 snapshot {DEFAULT_REVISION} is not in the local Hub cache")
    return str(cache)


def load_lfm3_backbone(path: str | None = None, dtype=torch.bfloat16, device="cpu"):
    """Load a local LFM3 checkpoint through the vendored modeling code, frozen (as ``load_qwen_backbone``)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from ..third_party import lfm3
    lfm3.register()
    source = resolve_base(path)
    config = AutoConfig.from_pretrained(source, trust_remote_code=False)
    if config.model_type != 'lfm3':
        raise ValueError(f"{source} is not an LFM3 checkpoint (model_type {config.model_type!r})")
    model = AutoModelForCausalLM.from_pretrained(source, config=config, dtype=dtype, device_map="cpu",
                                                 trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


class Lfm3PortBackbone(QwenPortBackbone):
    """LFM3 decoder run layer range by layer range (conv / full / sliding layers, gates, sandwich norms)."""

    attention_gate = 'per_head_input'

    def __init__(self, hf_model, controls: ControlTokens | None = None, noise: float = 0.02, seed: int = 0,
                 fast: bool = True, markers: bool = True, conv_kernel=None):
        if controls is None:
            raise ValueError('LFM3 markers are its provisioned role tokens: pass their ControlTokens explicitly')
        super().__init__(hf_model, controls, noise=noise, seed=seed, fast=fast, markers=markers)
        config = self.config
        if config.model_type != 'lfm3':
            raise ValueError(f'Lfm3PortBackbone needs an LFM3 model, not {config.model_type!r}')
        vocab = self.embedding_weight.shape[0]
        if any(not 0 <= marker < vocab for marker in (controls.open_id, controls.close_id)):
            raise ValueError('LFM3 Neuralese markers must be vocabulary rows of this checkpoint (no resize)')
        self.conv_window = int(config.conv_L_cache) - 1
        # causal-conv1d supports widths 2-4 only; the shared short convolution checks the width before using it.
        self.conv_kernel = conv_kernel
        self.sandwich_norm = bool(config.use_sandwich_norm)

    @property
    def final_norm_weight(self) -> torch.Tensor:
        return self.text.embedding_norm.weight

    @property
    def norm_eps(self) -> float:
        return float(self.config.norm_eps)

    def final_norm(self, h: torch.Tensor) -> torch.Tensor:
        return self.text.embedding_norm(h)

    def _operator_norm(self, layer, h: torch.Tensor) -> torch.Tensor:
        return layer.operator_norm(h)

    def _feed_forward(self, layer, h):
        return feed_forward_residual(layer, h, self.ffn_chunk_tokens, sandwich=self.sandwich_norm)

    def _rope(self, h, positions):
        """Per-layer-type (cos, sin) tables from the reference rotary module: full attention rotates the first
        half of each head (zero-tail frequencies, theta 1e6), sliding attention the whole head (theta 1e4)."""
        rotary = self.text.rotary_emb
        pairs = {kind: rotary(h, positions, kind) for kind in rotary.layer_types}
        return {k: c for k, (c, _) in pairs.items()}, {k: s for k, (_, s) in pairs.items()}

    def _apply_rope(self, layer: int, q, k, cos, sin):
        # The reference rotates in FP32 and casts back to the input dtype; zero-tail frequencies leave the
        # unrotated half unchanged (cos 1, sin 0).
        from ..third_party.lfm3.modeling_lfm3 import apply_rotary_pos_emb
        kind = self.layer_types[layer]
        return apply_rotary_pos_emb(q, k, cos[kind], sin[kind])
