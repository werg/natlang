"""Convert a Ling (bailing_hybrid / BailingMoeV3) checkpoint so each MoE layer keeps its routed experts stacked.

The released model holds every routed expert as its own three Linear layers and, in training, runs them one expert
at a time in a Python loop: 128 experts x 23 layers of small matmuls per forward pass, and a LoRA adapter per expert.
Stacked, a layer's experts are two parameters in the layout Transformers 5 MoE models use:

    gate_up_proj (experts, 2 * intermediate, hidden)   gate rows first, then up
    down_proj    (experts, hidden, intermediate)

which Unsloth quantizes to 4 bits at load, runs as one grouped matmul, and adapts with LoRA on the stacked parameters.
The modeling code is patched to route through that module (with a plain loop when Unsloth is absent); everything
else is copied unchanged. Usage: python scripts/ling_grouped_experts.py SOURCE_DIR OUT_DIR
"""
import json
import re
import shutil
import sys
from pathlib import Path

EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")
MODELING = "modeling_bailing_moe_v3.py"

EXPERTS_CLASS = '''

def _unsloth_moe_backend():
    """Unsloth's MoE dispatcher (4-bit grouped matmul, LoRA on stacked experts), when Unsloth is loaded."""
    import importlib
    import sys
    if "unsloth" not in sys.modules:
        return None
    # importlib, not an import statement: Transformers requires every module a remote-code file imports.
    moe_utils = importlib.import_module("unsloth_zoo.temporary_patches.moe_utils")
    moe_utils.patch_param_wrapper_for_moe()
    return moe_utils.forward_moe_backend


def _unsloth_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
    import importlib
    extract = importlib.import_module("unsloth_zoo.temporary_patches.moe_utils").extract_moe_lora_weights_for_grouped_mm
    base = wrapper.get_base_layer()
    dims = {"gate_up_proj": (base.hidden_dim, 2 * base.intermediate_dim),
            "down_proj": (base.intermediate_dim, base.hidden_dim)}.get(getattr(wrapper, "parameter_name", None))
    input_dim, output_dim = dims or (None, None)
    return extract(wrapper, weight_A, weight_B, scaling, num_experts,
                   input_dim=input_dim, output_dim=output_dim, model_name="Ling MoE")


class BailingMoeV3Experts(nn.Module):
    """A layer's routed experts, stacked: gate_up_proj (E, 2I, H) with gate rows first, down_proj (E, H, I)."""
    _unsloth_lora_extractor_fn = staticmethod(_unsloth_lora_extractor)

    def __init__(self, config: BailingMoeV3Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        for weight in (self.gate_up_proj, self.down_proj):  # as Ling initializes its Linear layers
            nn.init.normal_(weight, mean=0.0, std=config.initializer_range)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Tokens (T, H), each routed to top_k experts with weights (T, top_k); returns the weighted sum (T, H)."""
        backend = _unsloth_moe_backend()
        if backend is not None:
            return backend(self, hidden_states, top_k_index, top_k_weights)
        out = torch.zeros_like(hidden_states)
        for expert in top_k_index.unique().tolist():
            token, slot = torch.where(top_k_index == expert)
            gate, up = F.linear(hidden_states[token], self.gate_up_proj[expert]).chunk(2, dim=-1)
            y = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
            out.index_add_(0, token, (y * top_k_weights[token, slot, None]).to(out.dtype))
        return out
'''

SETUP_OLD = '''    def _setup_experts(self):
        self.experts = nn.ModuleList(
            [
                BailingMoeV3MLP(config=self.config, intermediate_size=self.config.moe_intermediate_size)
                for _ in range(self.config.num_experts)
            ]
        )
'''
SETUP_NEW = '''    def _setup_experts(self):
        self.experts = BailingMoeV3Experts(self.config)
'''
FORWARD_OLD = '''        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)
'''
FORWARD_NEW = '''        y = self.experts(hidden_states, topk_idx, topk_weight).to(hidden_states.dtype).view(bsz, seq_len, h)
'''
ANCHOR = "\n\nclass BailingMoeV3SparseMoeBlock(nn.Module):"
# Transformers 5 dropped is_torch_fx_available; the released code uses it only to mark a mask helper for FX tracing.
FX_OLD = "from transformers.utils.import_utils import is_torch_fx_available\n"
FX_NEW = """try:
    from transformers.utils.import_utils import is_torch_fx_available
except ImportError:  # Transformers 5
    def is_torch_fx_available():
        return False
"""


# Transformers 5 dropped "default" from ROPE_INIT_FUNCTIONS: plain RoPE over the rotary dims (Transformers 4's
# default, with the partial factor the rotary embedding sets to 1 over qk_rope_head_dim).
ROPE_OLD = "        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]\n"
ROPE_NEW = """        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type) or _default_rope_init
"""
ROPE_FN = '''

def _default_rope_init(config, device=None):
    """Unscaled RoPE: inverse frequencies base ** (-2i / dim) over config.head_dim, and no attention scaling."""
    base = getattr(config, "rope_theta", None) or config.rope_parameters["rope_theta"]
    dim = int(config.head_dim * getattr(config, "partial_rotary_factor", 1.0))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0
'''
ROPE_ANCHOR = "\n\nclass BailingMoeV3RotaryEmbedding(nn.Module):"
# Transformers 5 fills rope_scaling in even when the config has none ({"rope_type": "default", ...}); YaRN's
# attention scale applies only when a scaling factor is set.
MSCALE_OLD = "        if self.config.rope_scaling is not None:\n"
MSCALE_NEW = "        if self.config.rope_scaling is not None and \"factor\" in self.config.rope_scaling:\n"

# Logits only for the positions asked for (logits_to_keep, as Transformers' own models do): a whole prompt's logits
# over Ling's 157k-token vocabulary, in fp32, would not fit beside the model.
LOGITS_OLD = """        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
"""
LOGITS_NEW = """        hidden_states = outputs[0]
        logits_to_keep = kwargs.get("logits_to_keep", 0)
        if isinstance(logits_to_keep, int) and logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:]
        if labels is not None and self.training and self.num_nextn_predict_layers == 0:
            loss = _chunked_causal_lm_loss(self.lm_head, hidden_states, labels)
            return MoEV3CausalLMOutputWithPast(loss=loss, past_key_values=outputs.past_key_values)
        logits = self.lm_head(hidden_states)
"""
# In training, the loss block by block, each block's logits recomputed in backward: full-width fp32 logits over the
# vocabulary, and their gradient, cost about 1.2 GB per thousand positions.
LOSS_FN = '''

def _chunked_causal_lm_loss(lm_head, hidden_states, labels, block=512):
    """Transformers' causal LM loss (labels shifted left, -100 ignored, mean over counted tokens), one block of
    positions at a time."""
    from torch.utils.checkpoint import checkpoint
    labels = F.pad(labels, (0, 1), value=-100)[..., 1:].reshape(-1).to(hidden_states.device)
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

    def block_loss(h, y):
        return F.cross_entropy(lm_head(h).float(), y, ignore_index=-100, reduction="sum")

    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, labels.shape[0], block):
        y = labels[start:start + block]
        if (y != -100).any():
            total = total + checkpoint(block_loss, hidden_states[start:start + block], y, use_reentrant=False)
    return total / (labels != -100).sum().clamp(min=1)
'''
LOSS_ANCHOR = "\n\nclass BailingMoeV3ForCausalLM("

# Checkpointing per decoder layer, as Transformers 5 models do it (GradientCheckpointingLayer), in place of the
# model-level wrapper: layers can then be checkpointed individually, and the trainer can see and count them.
LAYER_OLD = "class BailingMoeV3DecoderLayer(nn.Module):"
LAYER_NEW = """from transformers.modeling_layers import GradientCheckpointingLayer


class BailingMoeV3DecoderLayer(GradientCheckpointingLayer):"""
LOOP_OLD = """            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    cache_position,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
"""
LOOP_NEW = """            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
"""

# The released MLA attention always runs eagerly, materializing heads x L x L scores (half a GB at 4k tokens);
# dispatch on the configured implementation, as Transformers models do.
ATTENTION_OLD = "        attention_interface: Callable = eager_attention_forward\n"
ATTENTION_NEW = """        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
"""


def patch_modeling(source: str) -> str:
    for old in (SETUP_OLD, FORWARD_OLD, ANCHOR, FX_OLD, ROPE_OLD, ROPE_ANCHOR, MSCALE_OLD, LOGITS_OLD,
                LOSS_ANCHOR, LAYER_OLD, LOOP_OLD, ATTENTION_OLD):
        if source.count(old) != 1:
            raise SystemExit(f"modeling code changed; expected exactly one of:\n{old}")
    return (source.replace(SETUP_OLD, SETUP_NEW).replace(FORWARD_OLD, FORWARD_NEW)
            .replace(ANCHOR, EXPERTS_CLASS + ANCHOR).replace(FX_OLD, FX_NEW).replace(ROPE_OLD, ROPE_NEW)
            .replace(ROPE_ANCHOR, ROPE_FN + ROPE_ANCHOR).replace(MSCALE_OLD, MSCALE_NEW)
            .replace(LOGITS_OLD, LOGITS_NEW).replace(LOSS_ANCHOR, LOSS_FN + LOSS_ANCHOR)
            .replace(LAYER_OLD, LAYER_NEW).replace(LOOP_OLD, LOOP_NEW).replace(ATTENTION_OLD, ATTENTION_NEW))


def stack_experts(parts, experts):
    """{(expert, "gate_proj" | "up_proj" | "down_proj"): weight} -> (gate_up_proj, down_proj), stacked."""
    import torch
    gate_up = torch.stack([torch.cat([parts[(e, "gate_proj")], parts[(e, "up_proj")]]) for e in range(experts)])
    return gate_up, torch.stack([parts[(e, "down_proj")] for e in range(experts)])


def main(source: Path, out: Path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((source / "config.json").read_text())
    experts, dense = config["num_experts"], config["first_k_dense_replace"]
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    out.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json":
            shutil.copy2(path, out / path.name)
    (out / MODELING).write_text(patch_modeling((source / MODELING).read_text()))

    # One output shard per layer (plus one for everything outside the layers), read one tensor at a time.
    groups: dict[str, list[str]] = {}
    for name in weight_map:
        layer = re.match(r"model\.layers\.(\d+)\.", name)
        groups.setdefault(f"layer{int(layer.group(1)):02d}" if layer else "rest", []).append(name)
    handles = {}
    def read(name):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(str(source / shard), framework="pt")
        return handles[shard].get_tensor(name)

    index, total = {}, 0
    for group in sorted(groups):
        tensors, stacked = {}, {}
        for name in groups[group]:
            match = EXPERT.fullmatch(name)
            if match:
                stacked.setdefault(int(match.group(1)), {})[(int(match.group(2)), match.group(3))] = read(name)
            else:
                tensors[name] = read(name)
        for layer, parts in stacked.items():
            if layer < dense or len(parts) != 3 * experts:
                raise SystemExit(f"layer {layer}: {len(parts)} expert tensors, expected {3 * experts}")
            prefix = f"model.layers.{layer}.mlp.experts"
            tensors[f"{prefix}.gate_up_proj"], tensors[f"{prefix}.down_proj"] = stack_experts(parts, experts)
        shard = f"model-{group}.safetensors"
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / shard), metadata={"format": "pt"})
        for name, tensor in tensors.items():
            index[name] = shard
            total += tensor.numel() * tensor.element_size()
        print(f"{shard}: {len(tensors)} tensors, {len(stacked)} stacked expert layers", flush=True)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(index.items()))}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
