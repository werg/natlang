"""Reference Neuralese model server (S4 §4): chat completions with block parts, the write procedure during
decoding, and a content-addressed tensor store behind `/v1/neuralese/blocks`."""

from __future__ import annotations


def load_engine(base: str | None = None, lora: str | None = None, heads_checkpoint: str | None = None,
                cutoff: int | None = None, max_block: int | None = None, device: str = "cpu", dialect: str | None = None,
                dtype=None):
    """Load the backbone (optionally merging a LoRA), the port heads (untrained unless a trainer checkpoint is
    given) and an empty store."""
    import torch

    from ..model.dialect import DIALECT
    from ..model.heads import PortHeads
    from pathlib import Path

    from ..model.lfm2_port import DEFAULT_BASE, ControlTokens, PortBackbone, backbone_identity, load_backbone, load_conv_kernel
    from .engine import Engine
    from .store import TensorStore

    dtype = dtype or (torch.float32 if device == "cpu" else torch.bfloat16)
    state = torch.load(heads_checkpoint, map_location="cpu", weights_only=False, mmap=True) if heads_checkpoint else None
    metadata = (state or {}).get("port_config", {})
    cutoff = cutoff if cutoff is not None else metadata.get("cutoff")
    if state is not None and cutoff is None:
        raise ValueError("Legacy checkpoint has no cutoff metadata; supply its actual cutoff")
    cutoff = cutoff if cutoff is not None else 6
    saved_length = int(state["heads"]["stop.position.weight"].shape[0]) - 1 if state else None
    max_block = max_block if max_block is not None else saved_length or 64
    if saved_length is not None and max_block != saved_length:
        raise ValueError(f"max_block must match checkpoint length {saved_length}")
    if metadata.get("cutoff") is not None and cutoff != metadata["cutoff"]:
        raise ValueError("cutoff differs from the trained checkpoint")
    saved = (state or {}).get("backbone") or {}
    if saved.get("student_lora"):
        # The port was trained on a merged student: rebuild exactly that backbone.
        if lora and Path(lora).resolve() != Path(saved["student_lora"]):
            raise ValueError(f"the checkpoint was trained on student {saved['student_lora']}, not {lora}")
        lora = saved["student_lora"]
        if backbone_identity(lora, saved.get("base") if saved.get("base") != DEFAULT_BASE else None).get(
                "student_lora_sha256") != saved["student_lora_sha256"]:
            raise ValueError(f"student LoRA {lora} changed since the port was trained on it")
    if saved.get("backbone") == "maple":
        # A Maple port (maple/student.py): published Maple plus the nested-family state it was trained on.
        from ..maple.maple_port import MaplePortBackbone
        from ..maple.student import load_student, student_identity
        from ..model.hf_port import qwen_controls

        if saved.get("student_state") and student_identity(saved["base"], saved["student_state"]) != saved:
            raise ValueError(f"Maple student state {saved['student_state']} changed since the port was trained on it")
        model, tokenizer = load_student(saved["base"], saved.get("student_state"), device=device)
        backbone = MaplePortBackbone(model, qwen_controls(tokenizer))
    else:
        model, tokenizer = load_backbone(base, lora, dtype=dtype, device="cpu")
        model.to(device)
        backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer), conv_kernel=load_conv_kernel())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Checkpoints from before stop sources were recorded read sketch states and the count.
    heads = PortHeads(backbone, cutoff=cutoff, max_length=max_block, stop_source=metadata.get("stop_source", "shallow"),
                      stop_position=metadata.get("stop_position", True))
    if state is not None:
        if state.get("lora"):
            from ..train.adapters import inject_lora, lora_state
            inject_lora(backbone, state["lora_layers"], rank=state["lora_rank"],
                        alpha=metadata.get("lora_alpha", 2 * state["lora_rank"]))
            if set(lora_state(backbone)) != set(state["lora"]):
                raise ValueError("Checkpoint backbone adapter names do not match the model")
            parameters = dict(backbone.hf.named_parameters())
            with torch.no_grad():
                for name, value in state["lora"].items():
                    parameters[name].copy_(value.to(parameters[name]))
        heads.load_state_dict(state["heads"])
        with torch.no_grad():
            backbone.control_rows.copy_(state["control_rows"].to(backbone.control_rows))
            if state.get("control_head_rows") is not None:
                backbone.control_head_rows.copy_(state["control_head_rows"].to(backbone.control_head_rows))
        del state
    # Match the trainer: head parameters stay float32, table buffer follows the base.
    heads.to(device=device).eval()
    for parameter in backbone.hf.parameters():
        parameter.requires_grad_(False)
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    backbone.control_rows.requires_grad_(False)
    engine = Engine(backbone, heads, tokenizer, TensorStore(), dialect or DIALECT, max_block=max_block, device=device)
    engine.base_dir = base  # the HF base, for exports (adapter LoRAs)
    return engine
