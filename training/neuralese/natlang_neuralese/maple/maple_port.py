"""Maple as a Neuralese port backbone: the Qwen3 port's layer-range runner with Maple's position encoding
(partial rotary on sliding layers, none on global layers) and MoE feed-forward."""

from __future__ import annotations

from types import SimpleNamespace

from ..model.hf_port import QwenPortBackbone
from .model import apply_partial_rope


class MaplePortBackbone(QwenPortBackbone):
    ternary = True  # phase F adapters are ternary QAT adapters (train/adapters.py)

    def _apply_rope(self, layer: int, q, k, cos, sin):
        if self.layer_types[layer] == "sliding_attention":
            return apply_partial_rope(q, k, cos, sin)
        return q, k

    @staticmethod
    def runner(maple_model) -> "MaplePortBackbone":
        """A marker-free runner over a bare ``MapleModel`` (used by its own forward)."""
        hf = SimpleNamespace(model=maple_model, config=SimpleNamespace(**maple_model.config.__dict__),
                             get_output_embeddings=lambda: maple_model.embed_tokens)
        return MaplePortBackbone(hf, markers=False)
