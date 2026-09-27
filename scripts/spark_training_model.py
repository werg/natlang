"""Make a trainable copy of Spark-X2.5-4B (XHToken/Spark-X2.5-4B) for LoRA on an 8 GB GPU.

    python scripts/spark_training_model.py models/candidates/spark-x25-4b-hf models/candidates/spark-x25-4b-train

The released modeling code (written for Transformers 4.57) is patched; the weights are linked, not copied:
- attention dispatches on the configured implementation (SDPA): it always ran eagerly, materializing
  heads x L x L scores per layer, which cannot fit at training lengths; sliding-window layers attend block by block
  (no SDPA kernel takes a long sliding mask efficiently, and the fallback materializes it);
- the MLP runs over long sequences in checkpointed chunks of tokens;
- in training, the loss is taken block by block over the tied embedding head: full logits over the 131k vocabulary in
  fp32 cost about half a GB per thousand positions, twice over with their gradient;
- tied weights are declared as Transformers 5 expects them, and activations take the norms' dtype, not a weight's
  that 4-bit loading packs as uint8;
- decoder layers checkpoint themselves (GradientCheckpointingLayer), as Transformers 5 models do, so the trainer can
  count and choose them.
"""
import argparse
import os
import shutil
from pathlib import Path

SUPPORTS_OLD = "    supports_gradient_checkpointing = True\n"
SUPPORTS_NEW = "    supports_gradient_checkpointing = True\n    _supports_sdpa = True\n"

ATTENTION_OLD = """        attn_output, attn_weights = eager_attention_forward(
            self, q, k, v,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
        )
"""
ATTENTION_NEW = """        if self.config._attn_implementation == "eager":
            attn_output, attn_weights = eager_attention_forward(
                self, q, k, v,
                attention_mask=attention_mask,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
            )
        elif self.sliding_window and past_key_values is None and seq_len > self.sliding_window:
            attn_output, attn_weights = _banded_attention(q, k, v, self.sliding_window, self.scaling,
                                                          self.attention_dropout if self.training else 0.0), None
        else:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            attn_output, attn_weights = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation](
                self, q, k, v, attention_mask,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
            )
            # Transformers' implementations return (batch, positions, heads, dim); the gate below expects heads first.
            attn_output = attn_output.transpose(1, 2)
"""
BANDED_FN = '''

def _banded_attention(q, k, v, window, scaling, dropout=0.0):
    """Causal attention within a sliding window, one block of `window` queries at a time: each block attends to its
    own positions and the `window - 1` before them, so memory grows with the window, not the square of the sequence.
    q (B, H, L, D), k and v (B, KV, L, D); returns (B, H, L, D)."""
    length = q.shape[2]
    outputs = []
    for start in range(0, length, window):
        stop = min(start + window, length)
        first = max(0, start - window + 1)
        query_positions = torch.arange(start, stop, device=q.device)[:, None]
        key_positions = torch.arange(first, stop, device=q.device)[None, :]
        allowed = (key_positions <= query_positions) & (query_positions - key_positions < window)
        outputs.append(F.scaled_dot_product_attention(
            q[:, :, start:stop], k[:, :, first:stop], v[:, :, first:stop], attn_mask=allowed, dropout_p=dropout,
            scale=scaling, enable_gqa=q.shape[1] != k.shape[1]))
    return torch.cat(outputs, dim=2)
'''
BANDED_ANCHOR = "\n\nclass Spark2_5Attention(nn.Module):"

LOGITS_OLD = """        if self.config.tie_word_embeddings:
            embed_weight = self.model.embedding.weight
            logits = F.linear(hidden_states, embed_weight)
        else:
            logits = self.lm_head(hidden_states)
"""
LOGITS_NEW = """        if self.config.tie_word_embeddings:
            head = lambda h: F.linear(h, self.model.embedding.weight)
        else:
            head = self.lm_head
        if labels is not None and self.training:
            loss = _chunked_causal_lm_loss(head, hidden_states, labels)
            return CausalLMOutputWithPast(loss=loss, past_key_values=outputs.past_key_values)
        logits = head(hidden_states)
"""
LOSS_FN = '''

def _chunked_causal_lm_loss(head, hidden_states, labels, block=512):
    """Transformers' causal LM loss (labels shifted left, -100 ignored, mean over counted tokens), one block of
    positions at a time, each block's logits recomputed in backward."""
    from torch.utils.checkpoint import checkpoint
    labels = F.pad(labels, (0, 1), value=-100)[..., 1:].reshape(-1).to(hidden_states.device)
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

    def block_loss(h, y):
        return F.cross_entropy(head(h).float(), y, ignore_index=-100, reduction="sum")

    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, labels.shape[0], block):
        y = labels[start:start + block]
        if (y != -100).any():
            total = total + checkpoint(block_loss, hidden_states[start:start + block], y, use_reentrant=False)
    return total / (labels != -100).sum().clamp(min=1)
'''
LOSS_ANCHOR = "\n\nclass Spark2_5ForCausalLM("

LAYER_OLD = "class Spark2_5DecoderLayer(nn.Module):"
LAYER_NEW = """from transformers.modeling_layers import GradientCheckpointingLayer


class Spark2_5DecoderLayer(GradientCheckpointingLayer):"""
LOOP_OLD = """            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    layer_attention_mask,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_ids=position_ids,
                )
"""
LOOP_NEW = """            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_ids=position_ids,
            )
"""

# Transformers 5 takes tied weights as a mapping from each tied parameter to its source.
TIED_OLD = '    _tied_weights_keys = ["lm_head.weight"]  # noqa: RUF012\n'
TIED_NEW = '    _tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}  # noqa: RUF012\n'
COMPAT = ((TIED_OLD, TIED_NEW),)

# The MLP is per token; over a long sequence its 10,240-wide activations (and their LoRA branches) run in chunks,
# each checkpointed in training, so a layer's recompute holds one chunk's.
MLP_OLD = """    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
"""
MLP_NEW = """    token_chunk = 4096

    def forward(self, x):
        if x.shape[-2] <= self.token_chunk:
            return self._forward(x)
        from torch.utils.checkpoint import checkpoint
        parts = [x[..., start:start + self.token_chunk, :] for start in range(0, x.shape[-2], self.token_chunk)]
        return torch.cat([checkpoint(self._forward, part, use_reentrant=False) if torch.is_grad_enabled()
                          else self._forward(part) for part in parts], dim=-2)

    def _forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
"""

# The layer casts its activations to the MLP weight's dtype, which a 4-bit load stores packed as uint8; the norms'
# weights are never quantized, and have the dtype the layer computes in.
CAST_OLD = "        hidden_states = hidden_states.to(self.mlp.gate_proj.weight.dtype)\n"
CAST_NEW = "        hidden_states = hidden_states.to(self.input_layernorm.weight.dtype)\n"

PATCHES = ((SUPPORTS_OLD, SUPPORTS_NEW), (MLP_OLD, MLP_NEW), (BANDED_ANCHOR, BANDED_FN + BANDED_ANCHOR), (ATTENTION_OLD, ATTENTION_NEW),
           (LOGITS_OLD, LOGITS_NEW), (LOSS_ANCHOR, LOSS_FN + LOSS_ANCHOR), (LAYER_OLD, LAYER_NEW), (LOOP_OLD, LOOP_NEW))


def _apply(source: str, patches) -> str:
    for old, _ in patches:
        if source.count(old) != 1:
            raise SystemExit(f"modeling code changed; expected exactly one of:\n{old}")
    for old, new in patches:
        source = source.replace(old, new)
    return source


def compat_modeling(source: str) -> str:
    """The released code, loadable by Transformers 5 and otherwise unchanged."""
    return _apply(source, COMPAT)


def patch_modeling(source: str) -> str:
    source = _apply(compat_modeling(source), PATCHES)
    if source.count(CAST_OLD) != 2:
        raise SystemExit(f"modeling code changed; expected two of:\n{CAST_OLD}")
    return source.replace(CAST_OLD, CAST_NEW)


def main(source: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        target = out / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if item.name == "modeling_spark.py":
            target.write_text(patch_modeling(item.read_text()))
        elif item.suffix == ".safetensors":
            os.symlink(os.path.relpath(item.resolve(), out.resolve()), target)
        else:
            shutil.copy2(item, target)
    print(f"patched {source} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    main(a.source, a.out)
