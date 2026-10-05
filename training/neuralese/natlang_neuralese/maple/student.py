"""The Maple student the Neuralese port trains on (MAPLE_NESTED §9): published Maple plus a nested-family state
(``nested-state.pt`` from ``nested_train``: shared attention QAT LoRA, learned block scales, expert block scales and
every member's private parts), all frozen. Phase F later releases the shared attention adapters of chosen layers
(``train/adapters.py``); the members stay exact because their private parts ride on the same modules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .model import load_maple
from .nested_train import Member, setup

DEFAULT_MODEL = "/home/werg/data/models/maple-preview-bf16"
DEFAULT_CACHE = "/home/werg/data/models/maple-preview-converted"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def student_identity(model: str, state: str | None) -> dict:
    """What the port's checkpoints record as their frozen base (a resume refuses a different one)."""
    identity = {"backbone": "maple", "base": str(model)}
    if state:
        identity.update(student_state=str(Path(state).resolve()), student_sha256=_sha256(Path(state)))
    return identity


def load_student(model: str = DEFAULT_MODEL, state: str | None = None, device: str = "cuda",
                 cache: str | None = DEFAULT_CACHE, order: str | None = None):
    """Returns (MapleForCausalLM, tokenizer). Without ``state``: published Maple with attention ternarized once.
    With it: the adapters, scales and member parts of the state, in its expert order, every parameter frozen."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    if state is None:
        maple = load_maple(model, device=device, ternary_attention=True, cache=cache)
    else:
        saved = torch.load(state, map_location="cpu")
        args = saved.get("args", {})
        maple = load_maple(model, device=device, ternary_attention=False, cache=cache)
        order = order or saved.get("order")
        if order:
            maple.order_experts(torch.load(order)["orders"])
        total = maple.config.num_hidden_layers
        members = [Member.parse(m, total) for m in saved.get("members", [])]
        setup(maple, members, args.get("rank", 8), args.get("private_rank", 4),
              not args.get("no_learned_scales", False), not args.get("no_expert_scales", False))
        result = maple.load_state_dict({k: v.to(device) for k, v in saved["trainable"].items()}, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"student state has keys this model lacks: {result.unexpected_keys[:5]}")
        missing = [k for k in result.missing_keys if "lora_" in k or "blocks" in k or "learned_scale" in k
                   or "private" in k]
        if missing:
            raise ValueError(f"student state lacks trained parameters: {missing[:5]}")
        maple.members = members
    for parameter in maple.parameters():
        parameter.requires_grad_(False)
    maple.eval()
    return maple, tokenizer
