"""Maple as a Neuralese port backbone: the Qwen3 port's layer-range runner with Maple's position encoding
(partial rotary on sliding layers, none on global layers) and MoE feed-forward. The same runner serves Mellum 2.x
(full rotary per layer type, YaRN on global layers) through its per-layer-type cos/sin tables."""

from __future__ import annotations

from types import SimpleNamespace

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

    @staticmethod
    def runner(maple_model) -> "MaplePortBackbone":
        """A marker-free runner over a bare ``MapleModel`` (used by its own forward)."""
        hf = SimpleNamespace(model=maple_model, config=SimpleNamespace(**maple_model.config.__dict__),
                             get_output_embeddings=lambda: maple_model.embed_tokens)
        return MaplePortBackbone(hf, markers=False)
