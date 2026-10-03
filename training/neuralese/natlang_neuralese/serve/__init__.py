"""Reference Neuralese model server (S4 §4): chat completions with block parts, the write procedure during
decoding, and a content-addressed tensor store behind `/v1/neuralese/blocks`."""

from __future__ import annotations


def load_engine(base: str | None = None, lora: str | None = None, heads_checkpoint: str | None = None,
                cutoff: int = 6, max_block: int = 64, device: str = "cpu", dialect: str | None = None,
                dtype=None):
    """Load the backbone (optionally merging a LoRA), the port heads (untrained unless a trainer checkpoint is
    given) and an empty store."""
    import torch

    from ..model.dialect import DIALECT
    from ..model.heads import PortHeads
    from ..model.lfm2_port import ControlTokens, PortBackbone, load_backbone, load_conv_kernel
    from .engine import Engine
    from .store import TensorStore

    dtype = dtype or (torch.float32 if device == "cpu" else torch.bfloat16)
    model, tokenizer = load_backbone(base, lora, dtype=dtype, device="cpu")
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer), conv_kernel=load_conv_kernel())
    heads = PortHeads(backbone, cutoff=cutoff, max_length=max_block)
    if heads_checkpoint:
        # Trainer checkpoints include optimizer tensors that inference never uses.
        # Map their storage instead of allocating a second resident CPU copy.
        state = torch.load(heads_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        heads.load_state_dict(state["heads"])
        with torch.no_grad():
            backbone.control_rows.copy_(state["control_rows"].to(backbone.control_rows))
        del state
    heads.to(device=device, dtype=dtype).eval()
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    backbone.control_rows.requires_grad_(False)
    return Engine(backbone, heads, tokenizer, TensorStore(), dialect or DIALECT, max_block=max_block, device=device)
