"""Backbone deltas for phase F (S3 §5.1): LoRA on selected layers, and the deltas-off teacher.

The port's own forward calls the HF modules (`q_proj`, `in_proj`, `feed_forward`, ...), so
LoRA injected in place into those modules applies on every path. The self-distillation
teacher and the ordinary-text replay reference are the same model with every adapter
disabled ("deltas off"), so no separate teacher copy is kept.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from ..model.lfm2_port import PortBackbone

# Linear modules adapted per layer kind. The short convolution's depthwise kernel is used by
# weight directly and is never adapted.
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")
CONV_TARGETS = ("in_proj", "out_proj")
MLP_TARGETS = ("w1", "w2", "w3")
# LFM3 keeps LFM2's conv/feed-forward names; its attention has `o_proj` and the per-head output gate `gate_proj`.
ATTENTION_TARGETS_BY_TYPE = {"lfm3": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj")}


def lora_target_names(backbone: PortBackbone, layers: list[int]) -> list[str]:
    names = []
    if getattr(backbone, 'architecture', None) == 'qwen3_5':
        prefix = 'model.language_model.layers'
        for i in layers:
            root = f'{prefix}.{i}'
            if backbone.is_attention(i):
                names += [f'{root}.self_attn.{target}' for target in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
            else:
                names += [f'{root}.linear_attn.{target}' for target in
                          ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj')]
            names += [f'{root}.mlp.{target}' for target in ('gate_proj', 'up_proj', 'down_proj')]
        return names
    attention_targets = ATTENTION_TARGETS_BY_TYPE.get(getattr(backbone.config, 'model_type', None), ATTENTION_TARGETS)
    for i in layers:
        if backbone.is_attention(i):
            names += [f"model.layers.{i}.self_attn.{t}" for t in attention_targets]
        else:
            names += [f"model.layers.{i}.conv.{t}" for t in CONV_TARGETS]
        names += [f"model.layers.{i}.feed_forward.{t}" for t in MLP_TARGETS]
    return names


def inject_lora(backbone: PortBackbone, layers: list[int], rank: int = 16, alpha: int = 32,
                adapter_name: str = "neuralese") -> dict[int, list[torch.nn.Parameter]]:
    """Inject LoRA into `layers` (in place) and return its parameters grouped by layer."""
    from peft import LoraConfig, inject_adapter_in_model

    existing = {name for name, _ in backbone.hf.named_modules() if name.endswith(f".lora_A.{adapter_name}")}
    targets = [n for n in lora_target_names(backbone, layers) if f"{n}.lora_A.{adapter_name}" not in existing]
    if targets:
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, bias="none")
        inject_adapter_in_model(config, backbone.hf, adapter_name=adapter_name)
    grouped: dict[int, list[torch.nn.Parameter]] = {}
    for name, param in backbone.hf.named_parameters():
        if "lora_" not in name:
            continue
        param.requires_grad_(True)
        layer = _layer_index(name)
        if layer in layers:
            grouped.setdefault(layer, []).append(param)
    # Everything that is not a port module or an adapter stays frozen.
    for name, param in backbone.hf.named_parameters():
        if "lora_" not in name:
            param.requires_grad_(False)
    device = backbone.embedding_weight.device
    dtype = backbone.embedding_weight.dtype
    for module in backbone.hf.modules():
        if hasattr(module, "lora_A"):
            module.to(device=device)
            for sub in (module.lora_A, module.lora_B):
                sub.to(dtype=dtype)
    return grouped


def lora_state(backbone: PortBackbone) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in backbone.hf.named_parameters() if "lora_" in n}


def restore_lora_checkpoint(backbone: PortBackbone, checkpoint: dict, *, alpha: int | None = None) -> dict:
    """Install and restore an embedded LFM LoRA checkpoint with exact structure checks.

    Current serving exports declare ``lora_layers`` and ``lora_rank``. Older
    recurrence checkpoints declare rank in ``identity.options.rank`` and only
    carry the named tensors; for those, layers are recovered from the complete
    tensor-name set and then checked against the injected structure.
    """
    saved = checkpoint.get("lora") or {}
    if not isinstance(saved, dict) or not saved:
        raise ValueError("checkpoint must contain a nonempty LoRA tensor mapping")
    layers = checkpoint.get("lora_layers")
    if layers is None:
        import re
        matches = [re.search(r"\.layers\.(\d+)\..+\.lora_[AB]\.neuralese\.weight$", name)
                   for name in saved]
        if any(match is None for match in matches):
            raise ValueError("legacy LoRA tensor names do not identify decoder layers")
        layers = sorted({int(match.group(1)) for match in matches})
    elif (not isinstance(layers, (list, tuple)) or not layers or
          any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 0 for layer in layers) or
          len(set(layers)) != len(layers)):
        raise ValueError("checkpoint LoRA layers must be a nonempty unique list of nonnegative integers")
    else:
        layers = sorted(layers)
    identity = checkpoint.get("identity") or {}
    rank = checkpoint.get("lora_rank")
    if rank is None:
        rank = (identity.get("options") or {}).get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("checkpoint must declare a positive LoRA rank")
    if alpha is not None and (not isinstance(alpha, int) or isinstance(alpha, bool) or alpha < 1):
        raise ValueError("LoRA alpha must be a positive integer when declared")
    inject_lora(backbone, layers, rank=rank, alpha=alpha if alpha is not None else 2 * rank)
    current = lora_state(backbone)
    if set(current) != set(saved):
        missing = sorted(set(current) - set(saved))[:8]
        extra = sorted(set(saved) - set(current))[:8]
        raise ValueError(f"checkpoint LoRA names do not exactly match injected structure; missing={missing}, extra={extra}")
    parameters = dict(backbone.hf.named_parameters())
    with torch.no_grad():
        for name, value in saved.items():
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(current[name].shape):
                raise ValueError(f"checkpoint LoRA tensor shape differs from adapter structure: {name}")
            parameters[name].copy_(value.to(parameters[name]))
    return {"layers": layers, "rank": rank, "tensor_count": len(saved)}


def adapter_layers(backbone: PortBackbone) -> list[int]:
    return sorted({_layer_index(name) for name, _ in backbone.hf.named_parameters() if 'lora_' in name})


def adapter_layer_indices(names) -> list[int]:
    """Layer IDs represented by exact named adapter tensors, including nested hybrid-model paths."""
    return sorted({_layer_index(name) for name in names})


def _layer_index(name):
    marker = '.layers.'
    if marker not in name:
        raise ValueError('LoRA parameter is outside the decoder layer namespace: ' + name)
    suffix = name.split(marker, 1)[1]
    try:
        return int(suffix.split('.', 1)[0])
    except ValueError as exc:
        raise ValueError('LoRA parameter has a malformed decoder layer name: ' + name) from exc


@contextmanager
def deltas_off(backbone: PortBackbone):
    """Disable every adapter (the self-distillation teacher and replay reference)."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [m for m in backbone.hf.modules() if isinstance(m, BaseTunerLayer)]
    for m in layers:
        m.enable_adapters(False)
    try:
        yield
    finally:
        for m in layers:
            m.enable_adapters(True)
