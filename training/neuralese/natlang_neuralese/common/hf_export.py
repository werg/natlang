"""Model-independent export helpers for ordinary Hugging Face models."""
from __future__ import annotations

import torch


def merged_peft_state_dict(model) -> dict | None:
    """Return plain state names with active PEFT LoRA deltas merged, or None.

    The model is left unchanged. Import PEFT only when the model actually has
    tuner layers so ordinary HF export has no optional PEFT runtime dependency.
    """
    if not any(hasattr(module, "base_layer") and hasattr(module, "lora_A")
               for _, module in model.named_modules()):
        return None
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        return None
    tuners = {name: module for name, module in model.named_modules() if isinstance(module, BaseTunerLayer)}
    if not tuners:
        return None
    state = {}
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if "lora_" in name:
                continue
            owner = next((t for t in tuners if name.startswith(t + ".base_layer.")), None)
            if owner is None:
                state[name] = tensor
                continue
            key = name.replace(".base_layer.", ".", 1)
            if name.endswith(".base_layer.weight"):
                layer = tuners[owner]
                delta = sum(layer.get_delta_weight(adapter) for adapter in layer.active_adapters
                            if adapter in layer.lora_A)
                state[key] = (tensor + delta.to(tensor.dtype)) if not isinstance(delta, int) else tensor
            else:
                state[key] = tensor
    return state
