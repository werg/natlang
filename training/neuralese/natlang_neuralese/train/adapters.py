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


def lora_target_names(backbone: PortBackbone, layers: list[int]) -> list[str]:
    names = []
    for i in layers:
        if backbone.is_attention(i):
            names += [f"model.layers.{i}.self_attn.{t}" for t in ATTENTION_TARGETS]
        else:
            names += [f"model.layers.{i}.conv.{t}" for t in CONV_TARGETS]
        names += [f"model.layers.{i}.feed_forward.{t}" for t in MLP_TARGETS]
    return names


# Maple (ternary backbone): phase F trains the student's own ternary QAT adapters on attention (codes(W + dW) times
# learned FP16 block scales, exactly exportable to TQ2_0) instead of PEFT LoRA. Released adapters snapshot themselves
# first, so "deltas off" (the self-distillation teacher) is the student as it was, not the original Maple.
MAPLE_ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


def _ternary(backbone) -> bool:
    return bool(getattr(backbone, "ternary", False))


def _maple_adapters(backbone, layers=None):
    """{layer: [(module name, QATTernaryLoRA)]} over attention projections (released ones only if layers is None)."""
    from ..maple.ternary import QATTernaryLoRA

    found: dict[int, list] = {}
    for name, module in backbone.hf.named_modules():
        if not isinstance(module, QATTernaryLoRA):
            continue
        layer = int(name.split("model.layers.")[1].split(".")[0])
        if (layers is None and getattr(module, "released", False)) or (layers is not None and layer in layers):
            found.setdefault(layer, []).append((name, module))
    return found


def _inject_maple(backbone, layers: list[int], rank: int, alpha: int) -> dict[int, list[torch.nn.Parameter]]:
    from torch.nn.utils import parametrize

    from ..maple.ternary import add_qat_lora

    for i in layers:
        attention = backbone.layers[i].self_attn
        for proj in MAPLE_ATTENTION:
            module = getattr(attention, proj)
            if not parametrize.is_parametrized(module, "weight"):
                adapter = add_qat_lora(module, rank=rank, alpha=float(alpha))
                adapter.learn_scales(module.parametrizations.weight.original)
    grouped: dict[int, list[torch.nn.Parameter]] = {}
    for layer, adapters in _maple_adapters(backbone, layers).items():
        for _, adapter in adapters:
            if not getattr(adapter, "released", False):
                adapter.snapshot_teacher()
                adapter.released = True
            params = [adapter.lora_A, adapter.lora_B] + ([adapter.learned_scale] if adapter.learned_scale is not None
                                                          else [])
            for p in params:
                p.requires_grad_(True)
            grouped.setdefault(layer, []).extend(params)
    return grouped


def inject_lora(backbone: PortBackbone, layers: list[int], rank: int = 16, alpha: int = 32,
                adapter_name: str = "neuralese") -> dict[int, list[torch.nn.Parameter]]:
    """Inject LoRA into `layers` (in place) and return its parameters grouped by layer."""
    if _ternary(backbone):
        return _inject_maple(backbone, layers, rank, alpha)
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
        layer = int(name.split("model.layers.")[1].split(".")[0])
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
    if _ternary(backbone):  # the released adapters; the frozen student is identified by the backbone identity
        return {f"{name}.{attr}": getattr(adapter, attr).detach().cpu()
                for adapters in _maple_adapters(backbone).values() for name, adapter in adapters
                for attr in ("lora_A", "lora_B", "learned_scale") if getattr(adapter, attr) is not None}
    return {n: p.detach().cpu() for n, p in backbone.hf.named_parameters() if "lora_" in n}


def adapter_layers(backbone: PortBackbone) -> list[int]:
    if _ternary(backbone):
        return sorted(_maple_adapters(backbone))
    return sorted({int(n.split("model.layers.")[1].split(".")[0]) for n, _ in backbone.hf.named_parameters()
                   if "lora_" in n})


@contextmanager
def deltas_off(backbone: PortBackbone):
    """Disable every adapter (the self-distillation teacher and replay reference)."""
    if _ternary(backbone):
        from ..maple.ternary import teacher_mode

        with teacher_mode():
            yield
        return
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [m for m in backbone.hf.modules() if isinstance(m, BaseTunerLayer)]
    for m in layers:
        m.enable_adapters(False)
    try:
        yield
    finally:
        for m in layers:
            m.enable_adapters(True)
