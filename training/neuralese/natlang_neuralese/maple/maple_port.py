"""Maple as a Neuralese port backbone: the Qwen3 port's layer-range runner with Maple's position encoding
(partial rotary on sliding layers, none on global layers) and MoE feed-forward. The same runner serves Mellum 2.x
(full rotary per layer type, YaRN on global layers) through its per-layer-type cos/sin tables."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from ..model.hf_port import QwenPortBackbone
from .model import apply_partial_rope


class MaplePortBackbone(QwenPortBackbone):
    ternary = True  # phase F adapters are ternary QAT adapters (train/adapters.py)

    def _apply_rope(self, layer: int, q, k, cos, sin):
        if isinstance(cos, dict):
            # Per-layer-type tables (Mellum): full rotary, YaRN-scaled on global layers.
            kind = self.layer_types[layer]
            return apply_partial_rope(q, k, cos[kind], sin[kind])
        if self.layer_types[layer] == "sliding_attention":
            return apply_partial_rope(q, k, cos, sin)
        return q, k

    def _fused_rope_tables(self, layer: int, attn, x, cos, sin):
        """Mellum's full rotary (per-layer-type tables covering the whole head) with Maple's RMSNorm q/k norms on
        BF16 CUDA activations: maple/rope_kernels.qk_norm_rope (NATLANG_FUSED_ROPE=0: the eager path)."""
        import os

        from .model import RMSNorm
        if (not isinstance(cos, dict) or not x.is_cuda or x.dtype != torch.bfloat16
                or os.environ.get("NATLANG_FUSED_ROPE", "1") == "0"
                or not isinstance(attn.q_norm, RMSNorm) or not isinstance(attn.k_norm, RMSNorm)):
            return None
        kind = self.layer_types[layer]
        if cos[kind].shape[-1] != attn.head_dim or cos[kind].dtype != torch.bfloat16:
            return None
        return cos[kind], sin[kind]

    @staticmethod
    def runner(maple_model) -> "MaplePortBackbone":
        """A marker-free runner over a bare ``MapleModel`` (used by its own forward)."""
        hf = SimpleNamespace(model=maple_model, config=SimpleNamespace(**maple_model.config.__dict__),
                             get_output_embeddings=lambda: maple_model.embed_tokens)
        runner = MaplePortBackbone(hf, markers=False)
        runner.checkpoint_layers = getattr(maple_model, "checkpoint_layers", False)
        return runner
