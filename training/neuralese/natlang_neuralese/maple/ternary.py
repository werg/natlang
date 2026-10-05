"""Maple's ternary weight rule, quantization-aware LoRA, and the export path to llama.cpp TQ2_0.

Maple's rule (deepgrove ``mlx_lm/ternary.py``), per output row of a BF16 weight, computed in FP32:
``threshold = 0.7 * mean|w|``, ``mask = |w| > threshold``, ``alpha = BF16(mean |w| over mask)`` (0 for an empty
mask), ``Q(w) = sign(w) * mask * alpha``.

llama.cpp's TQ2_0 quantizer uses a different rule (per 256-weight block, scale = block absmax, codes = round(x /
absmax)). Applied to weights already in Maple's ternary form it is exact: every nonzero in a row equals +-alpha, so
each block's absmax is alpha (or the block is all zero). ``export_ternary`` produces that form; ``tq2_0_roundtrip``
reproduces the reference quantizer to check it.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils import parametrize

THRESHOLD_FACTOR = 0.7
QK_K = 256


def ternary_codes(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Maple's rule on ``weight`` (rows = output features): int8 codes in {-1, 0, 1} and a BF16 scale per row."""
    w = weight.to(torch.bfloat16).float()
    magnitude = w.abs()
    threshold = THRESHOLD_FACTOR * magnitude.mean(dim=-1, keepdim=True)
    mask = magnitude > threshold
    count = mask.sum(dim=-1, keepdim=True)
    alpha = torch.where(mask, magnitude, torch.zeros_like(magnitude)).sum(dim=-1, keepdim=True)
    alpha = torch.where(count > 0, alpha / count.clamp_min(1), torch.zeros_like(alpha))
    codes = (torch.sign(w) * mask).to(torch.int8)
    return codes, alpha.to(torch.bfloat16)


def ternarize(weight: torch.Tensor) -> torch.Tensor:
    """``Q(BF16(weight))`` as a dense tensor of ``weight``'s dtype: the values the deployed model multiplies by."""
    codes, alpha = ternary_codes(weight)
    return (codes.float() * alpha.float()).to(weight.dtype)


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return ternarize(weight)

    @staticmethod
    def backward(ctx, grad):
        return grad


def ternarize_ste(weight: torch.Tensor) -> torch.Tensor:
    """Ternarize in the forward pass; identity gradient (straight-through) in the backward pass."""
    return _StraightThrough.apply(weight)


# Global switches read by every adapter: which nested member is running (its private deltas apply) and whether
# adapters apply at all (off = the frozen original model, used as the anchor teacher).
STATE = {"size": None, "enabled": True, "teacher": False}


class adapters_disabled:
    """Context manager: run the frozen original model (no shared or private deltas, no learned scales)."""

    def __enter__(self):
        self.previous = STATE["enabled"]
        STATE["enabled"] = False

    def __exit__(self, *exc):
        STATE["enabled"] = self.previous


class teacher_mode:
    """Context manager: adapters with a teacher snapshot (``snapshot_teacher``) run their snapshot instead of the live
    parameters; the others run as they are. The Neuralese port's self-distillation teacher ("deltas off") on Maple:
    the student as it was before phase F, not the original Maple."""

    def __enter__(self):
        self.previous = STATE["teacher"]
        STATE["teacher"] = True

    def __exit__(self, *exc):
        STATE["teacher"] = self.previous


class _CodesTimesScale(torch.autograd.Function):
    """``codes(w) * s``: straight-through to ``w`` (identity), exact gradient to the scale ``s``."""

    @staticmethod
    def forward(ctx, merged, scale, block):
        codes = ternary_codes(merged)[0].to(merged.dtype)
        # The deployed scale is FP16 (TQ2_0 block scale): train against it, update the FP32 master.
        expanded = _expand_blocks(scale.half().to(merged.dtype), block, merged.shape[-1])
        ctx.save_for_backward(codes)
        ctx.block = block
        return codes * expanded

    @staticmethod
    def backward(ctx, grad):
        (codes,) = ctx.saved_tensors
        product = grad * codes
        pad = (-product.shape[-1]) % ctx.block
        if pad:
            product = torch.nn.functional.pad(product, (0, pad))
        grad_scale = product.reshape(*grad.shape[:-1], -1, ctx.block).sum(-1)
        return grad, grad_scale, None


def _expand_blocks(scale: torch.Tensor, block: int, columns: int) -> torch.Tensor:
    return scale.repeat_interleave(block, dim=-1)[..., :columns]


def block_scales(weight: torch.Tensor, block: int = QK_K) -> torch.Tensor:
    """Initial learned scales: Maple's row alpha broadcast to every ``block``-wide column block (FP32)."""
    _, alpha = ternary_codes(weight)
    blocks = -(-weight.shape[-1] // block)
    return alpha.float().expand(*weight.shape[:-1], blocks).clone()


class PrivateLoRA(nn.Module):
    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float):
        super().__init__()
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def delta(self) -> torch.Tensor:
        return self.scale * (self.lora_B @ self.lora_A)


class QATTernaryLoRA(nn.Module):
    """Parametrization of a linear layer's ``weight``: ``Q(BF16(W_base + (alpha / r) * B @ A))``.

    The adapter is FP32 and B starts at zero, so the initial forward is exactly the deployed ternary model. The base
    weight stays frozen; the quantizer sees the merged weight (``Q(W + dW) != Q(W) + dW``). ``quantize=False`` gives an
    ordinary LoRA through the same code path (for stand-in models that were not trained ternary)."""

    def __init__(self, out_features: int, in_features: int, rank: int = 8, alpha: float = 16.0,
                 quantize: bool = True):
        super().__init__()
        self.rank = rank
        self.quantize = quantize
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.private = nn.ModuleDict()          # per nested size: zero-initialised private deltas
        self.learned_scale: nn.Parameter | None = None
        self.block = QK_K

    def add_private(self, size: int, rank: int, alpha: float) -> PrivateLoRA:
        module = PrivateLoRA(self.lora_B.shape[0], self.lora_A.shape[1], rank, alpha).to(self.lora_A.device)
        self.private[str(size)] = module
        return module

    def learn_scales(self, base: torch.Tensor, block: int = QK_K) -> nn.Parameter:
        """Replace Maple's mean rule for the scale by a learned scale per ``block`` columns (TQ2_0's block size:
        exported exactly). Initialised at the base's row alpha, so the forward is unchanged at the start."""
        self.block = block
        self.learned_scale = nn.Parameter(block_scales(base, block).to(self.lora_A.device))
        return self.learned_scale

    def snapshot_teacher(self) -> None:
        """Freeze a copy of the current shared delta and scales as this adapter's teacher (``teacher_mode``)."""
        self.register_buffer("teacher_A", self.lora_A.detach().clone(), persistent=False)
        self.register_buffer("teacher_B", self.lora_B.detach().clone(), persistent=False)
        scale = self.learned_scale.detach().clone() if self.learned_scale is not None else None
        self.register_buffer("teacher_scale", scale, persistent=False)

    def _as_teacher(self) -> bool:
        return STATE["teacher"] and getattr(self, "teacher_A", None) is not None

    def delta(self) -> torch.Tensor:
        if self._as_teacher():
            return self.scale * (self.teacher_B @ self.teacher_A)
        total = self.scale * (self.lora_B @ self.lora_A)
        size = STATE["size"]
        if size is not None and str(size) in self.private:
            total = total + self.private[str(size)].delta()
        return total

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        if not STATE["enabled"]:
            return ternarize(base) if self.quantize else base
        merged = base.float() + self.delta()
        if not self.quantize:
            return merged.to(base.dtype)
        scale = self.teacher_scale if self._as_teacher() else self.learned_scale
        if scale is not None:
            return _CodesTimesScale.apply(merged, scale, self.block).to(base.dtype)
        return ternarize_ste(merged).to(base.dtype)


@torch.no_grad()
def flip_fraction(module: nn.Linear) -> float:
    """Fraction of a QAT-parametrized weight's entries whose ternary state differs from the frozen base's."""
    base = module.parametrizations.weight.original
    adapter = next(p for p in module.parametrizations.weight if isinstance(p, QATTernaryLoRA))
    before = ternary_codes(base)[0]
    after = ternary_codes(base.float() + adapter.delta())[0]
    return (before != after).float().mean().item()


class FrozenTernary(nn.Module):
    """Parametrization for a frozen matrix outside the trainable scope: ternarized once, returned as is."""

    def __init__(self, base: torch.Tensor):
        super().__init__()
        self.register_buffer("value", ternarize(base.detach()), persistent=False)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        return self.value


def add_qat_lora(module: nn.Linear, rank: int = 8, alpha: float = 16.0, quantize: bool = True) -> QATTernaryLoRA:
    """Freeze ``module.weight`` and attach a ternary QAT LoRA to it. Returns the adapter."""
    module.weight.requires_grad_(False)
    out_features, in_features = module.weight.shape
    adapter = QATTernaryLoRA(out_features, in_features, rank, alpha, quantize).to(module.weight.device)
    parametrize.register_parametrization(module, "weight", adapter)
    return adapter


def freeze_ternary(module: nn.Linear) -> None:
    """Replace ``module.weight``'s forward value by its ternarization, once."""
    module.weight.requires_grad_(False)
    parametrize.register_parametrization(module, "weight", FrozenTernary(module.weight), unsafe=True)


def qat_adapters(model: nn.Module) -> dict[str, QATTernaryLoRA]:
    """Every QAT adapter in ``model``, keyed by the owning module's name."""
    found = {}
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module, "weight"):
            for p in module.parametrizations.weight:
                if isinstance(p, QATTernaryLoRA):
                    found[name] = p
    return found


def export_ternary(base: torch.Tensor, adapter: QATTernaryLoRA | None = None, size: int | None = None) -> torch.Tensor:
    """The exported weight (for nested member ``size``, with its private delta): merge in FP32, round to BF16, take
    ternary codes; scale by Maple's row rule, or by the learned FP16 block scales. FP32 tensor whose every 256-block
    holds {-s, 0, +s}: TQ2_0 stores it exactly."""
    merged = base.float()
    if adapter is not None:
        previous = STATE["size"]
        STATE["size"] = size
        try:
            merged = merged + adapter.delta().detach().to(merged.device)
        finally:
            STATE["size"] = previous
        if adapter.learned_scale is not None:
            codes = ternary_codes(merged)[0].float()
            scale = _expand_blocks(adapter.learned_scale.detach().half().float(), adapter.block, merged.shape[-1])
            return codes * scale.to(codes.device)
    return ternarize(merged).float()


def tq2_0_roundtrip(weight: torch.Tensor) -> torch.Tensor:
    """llama.cpp's ``quantize_row_tq2_0_ref`` followed by dequantization, over rows of ``weight`` (FP32)."""
    rows, columns = weight.shape
    if columns % QK_K:
        raise ValueError(f"row length {columns} is not a multiple of {QK_K}")
    blocks = weight.float().reshape(rows, columns // QK_K, QK_K)
    d = blocks.abs().amax(dim=-1, keepdim=True)
    inverse = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    # lroundf rounds halves away from zero; torch.round rounds them to even.
    scaled = blocks * inverse
    q = torch.sign(scaled) * torch.floor(scaled.abs() + 0.5)
    stored = d.to(torch.float16).float()
    return (q.clamp(-1, 1) * stored).reshape(rows, columns)


def tq2_0_exact(weight: torch.Tensor) -> bool:
    """True when TQ2_0 stores ``weight`` without change (expects Maple's ternary form)."""
    return bool(torch.equal(tq2_0_roundtrip(weight), weight.float()))


def ternary_tensor_bytes(shape: tuple[int, ...], storage: str) -> int:
    """Bytes to hold a frozen ternary matrix in training: 'bf16' dense, 'fp8' codes, 'nvfp4' packed codes; plus a
    4-byte FP32 scale per row for the code formats (NVFP4 block scales are all 1: the row scale is applied after)."""
    rows = shape[0]
    count = math.prod(shape)
    if storage == "bf16":
        return 2 * count
    if storage == "fp8":
        return count + 4 * rows
    if storage == "nvfp4":
        return count // 2 + count // 16 + 4 * rows
    raise ValueError(storage)
