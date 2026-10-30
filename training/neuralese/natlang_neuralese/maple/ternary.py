"""Maple's ternary weight rule and the shared precision points (QAT fake quantization) of train/quantization.py.

Maple's rule (deepgrove ``mlx_lm/ternary.py``), per output row of a BF16 weight, computed in FP32:
``threshold = 0.7 * mean|w|``, ``mask = |w| > threshold``, ``alpha = BF16(mean |w| over mask)`` (0 for an empty
mask), ``Q(w) = sign(w) * mask * alpha``.
"""

from __future__ import annotations


import torch
from torch import nn

THRESHOLD_FACTOR = 0.7
QK_K = 256


def ternary_row_stats(weight: torch.Tensor):
    """The reductions of Maple's rule, per row: (BF16-rounded FP32 weight, |w|, FP32 threshold, mask, BF16 alpha). The
    one implementation behind ``ternary_codes`` (export) and the fused training ramp (maple/precision_kernels.py), so
    training forwards and the exported codes agree bit for bit."""
    w = weight.to(torch.bfloat16).float()
    magnitude = w.abs()
    threshold = THRESHOLD_FACTOR * magnitude.mean(dim=-1, keepdim=True)
    mask = magnitude > threshold
    count = mask.sum(dim=-1, keepdim=True)
    alpha = torch.where(mask, magnitude, torch.zeros_like(magnitude)).sum(dim=-1, keepdim=True)
    alpha = torch.where(count > 0, alpha / count.clamp_min(1), torch.zeros_like(alpha))
    return w, magnitude, threshold, mask, alpha.to(torch.bfloat16)


def ternary_codes(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Maple's rule on ``weight`` (rows = output features): int8 codes in {-1, 0, 1} and a BF16 scale per row."""
    w, _, _, mask, alpha = ternary_row_stats(weight)
    codes = (torch.sign(w) * mask).to(torch.int8)
    return codes, alpha


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


# Gradual ternarization (full-latent QAT of a BF16 model): forwards use w + mix·(Q(w) − w), straight-through, with mix
# ramped 0 → 1 over the conversion (HF 1.58-bit fine-tuning: an abrupt switch loses most of the model). 1 = deployed.
# ``fused``: one elementwise kernel per BF16 CUDA latent after the rule's eager row reductions (the precision points'
# kernel, ``_FusedPrecisionRamp``): w + mix·(Q(w) − w) in FP32, cast once, bit-identical codes. Training forwards only
# (the conversion sets it).
QUANT_MIX = {"value": 1.0, "fused": False}


def ramped_ternarize_ste(weight: torch.Tensor) -> torch.Tensor:
    mix = QUANT_MIX["value"]
    if mix <= 0.0:
        return weight
    if QUANT_MIX["fused"] and weight.is_cuda and weight.dtype == torch.bfloat16:
        return _FusedPrecisionRamp.apply(weight, "ternary", {}, float(mix))
    if mix >= 1.0:
        return ternarize_ste(weight)
    return weight + mix * (ternarize(weight.detach()) - weight.detach())


# Precision points of the recipe ``quantization`` component (train/quantization.py; owner 2026-10-10: QAT is woven
# into every training stage). While a point is active, ``PRECISION["groups"]`` maps a module group (experts, attention,
# ...) to (format, mix, options) and every selected weight's forward value is w + mix·(Q(w) − w), straight-through
# to the BF16 latent. ``None`` (the default, and always for the BF16 teacher) leaves weights untouched.
PRECISION: dict = {"groups": None, "point": None,
                   # One elementwise kernel per weight on CUDA (bit-identical to the eager rule; ~20x less memory
                   # traffic). False: the eager rule everywhere (a reference for the exactness tests).
                   "fused": True}
QK4_0 = 32


def q4_0(weight: torch.Tensor, group: int = QK4_0) -> torch.Tensor:
    """llama.cpp's ``quantize_row_q4_0_ref`` then dequantization, along the last dim in blocks of ``group``: per block
    ``d = max / -8`` (``max`` the signed value of largest magnitude), ``q = min(15, floor(x / d + 8.5)) - 8``, ``d``
    stored as FP16. The values a Q4_0 GGUF of this weight multiplies by (bit-exact rule, so int4 is deployable)."""
    d = q4_scales(weight, group)
    columns = weight.shape[-1]
    blocks = weight.float().reshape(*weight.shape[:-1], columns // group, group)
    inverse = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.floor(blocks * inverse + 8.5), max=15.0) - 8.0
    return (q * d.to(torch.float16).float()).reshape(weight.shape)


def q4_scales(weight: torch.Tensor, group: int = QK4_0) -> torch.Tensor:
    """The Q4_0 reduction, per block of ``group`` along the last dim: FP32 ``d = max / -8`` with ``max`` the signed value
    of largest magnitude (the first one on ties), shape (..., blocks, 1). Taken in ``weight``'s own dtype (abs, argmax and
    gather are exact in any float dtype), so a BF16 latent is never widened for it. Shared by ``q4_0`` and the fused
    training ramp (maple/precision_kernels.py)."""
    columns = weight.shape[-1]
    if columns % group:
        raise ValueError(f"row length {columns} is not a multiple of the Q4_0 block {group}")
    blocks = weight.reshape(*weight.shape[:-1], columns // group, group)
    index = blocks.abs().argmax(dim=-1, keepdim=True)
    return blocks.gather(-1, index).float() / -8.0


def quantized_value(weight: torch.Tensor, fmt: str, options: dict | None = None) -> torch.Tensor:
    """The deployed value of ``weight`` at one precision format (no gradient; FP32):

    - ``int4``: Q4_0 (``options["group"]``, default 32);
    - ``ternary``: Maple's per-row rule; with ``options["nested_group"]`` it is applied to the Q4_0 values of that
      block size, so the ternary weights are a function of the int4 codes and scales (the coarsest level of the int4
      grid: an int4 artifact determines the ternary one, Matryoshka style);
    - ``nvfp4``: ModelOpt NVFP4 (group 16, E4M3 group scales, FP32 global scale; ``options["global_amax"]`` when the
      global scale is shared with fused siblings); ``fp8_block``: compressed-tensors FP8 128x128 blocks
      (train/qformats.py);
    - ``bf16``: the weight itself."""
    options = options or {}
    w = weight.detach()
    if fmt == "bf16":
        return w.float()
    if fmt == "int4":
        return q4_0(w, int(options.get("group", QK4_0)))
    if fmt == "ternary":
        if options.get("nested_group"):
            w = q4_0(w, int(options["nested_group"]))
        return ternarize(w.float()).float()
    if fmt in ("nvfp4", "fp8_block"):
        from ..train.qformats import weight_value
        return weight_value(w, fmt, options)
    raise ValueError(f"unknown precision format: {fmt}")


def parse_group_key(key: str):
    """``group[@layer][:leaf]`` -> (group, layer or None, leaf or None)."""
    head, _, leaf = key.partition(":")
    name, _, layer = head.partition("@")
    return name, (int(layer) if layer else None), (leaf or None)


def _selected(key: str):
    """The active (fmt, mix, options) for a module key, or None when the active point leaves it unquantized (no point,
    group not selected, λ = 0, or excluded by the point's ``layers`` / ``names`` selection)."""
    groups = PRECISION["groups"]
    if not groups:
        return None
    name, layer, leaf = parse_group_key(key)
    if name not in groups:
        return None
    fmt, mix, options = groups[name]
    if mix <= 0.0:
        return None
    if layer is not None and options.get("layers") is not None and layer not in options["layers"]:
        return None
    if leaf is not None and options.get("names") is not None and leaf not in options["names"]:
        return None
    return fmt, mix, options


def precision_value(weight: torch.Tensor, group: str, peers=()) -> torch.Tensor:
    """``weight`` under the active precision point for its module ``group`` (``group@layer`` for a layer's module,
    ``:leaf`` its module name): w + mix·(Q(w) − w), straight-through (identity gradient to the latent). Unchanged when
    no point is active, the point does not select the group, or the point's layer selection (``options["layers"]``)
    or module names (``options["names"]``) exclude it. ``peers``: (leaf, latent) of the sibling projections a format
    may share a scale with (NVFP4 ``shared_global``: one fused deploy GEMM, one global scale)."""
    selected = _selected(group)
    if selected is None:
        return weight
    fmt, mix, options = selected
    if fmt == "bf16":
        return weight
    if fmt == "nvfp4" and peers and options.get("shared_global"):
        leaf = parse_group_key(group)[2]
        shared = next((set(s) for s in options["shared_global"] if leaf in s), set())
        amaxes = [weight.detach().abs().amax()] + [p.detach().abs().amax() for n, p in peers if n in shared]
        if len(amaxes) > 1:
            options = {**options, "global_amax": torch.stack(amaxes).amax()}
    from ..common.profiling import scope
    with scope("qat"):
        if PRECISION["fused"] and weight.is_cuda and weight.dtype == torch.bfloat16 and (
                fmt == "int4" or (fmt == "ternary" and not options.get("nested_group"))):
            return _FusedPrecisionRamp.apply(weight, fmt, options, float(mix))
        return _PrecisionRamp.apply(weight, quantized_value(weight, fmt, options), float(mix))


class _FusedPrecisionRamp(torch.autograd.Function):
    """``_PrecisionRamp(weight, quantized_value(weight))`` for a BF16 CUDA latent: the eager scales of the rule, then one
    elementwise kernel (maple/precision_kernels.py), bit-identical; identity gradient to ``weight``."""

    @staticmethod
    def forward(ctx, weight, fmt, options, mix):
        from .precision_kernels import q4_ramp, ternary_ramp

        w = weight.detach()
        if fmt == "int4":
            group = int(options.get("group", QK4_0))
            return q4_ramp(w, q4_scales(w, group), mix, group)
        _, _, threshold, _, alpha = ternary_row_stats(w)
        return ternary_ramp(w, threshold, alpha, mix)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None


class _PrecisionRamp(torch.autograd.Function):
    """``w + mix·(q − w)`` computed in FP32 and cast once (λ = 1 gives exactly the deployed values), identity
    gradient to ``w``."""

    @staticmethod
    def forward(ctx, weight, quantized, mix):
        x = weight.detach().float()
        return (x + mix * (quantized.float() - x)).to(weight.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class active_precision:
    """Context manager: run the forward passes inside at one resolved precision point (``{group: (format, mix,
    options)}``, from ``train.quantization``); ``None`` is BF16 (the teacher and the BF16 gate column)."""

    def __init__(self, groups: dict | None, point: str | None = None):
        self.groups, self.point = groups, point

    def __enter__(self):
        self.previous = (PRECISION["groups"], PRECISION["point"])
        PRECISION["groups"], PRECISION["point"] = self.groups, self.point
        return self

    def __exit__(self, *exc):
        PRECISION["groups"], PRECISION["point"] = self.previous


class PrecisionSTE(nn.Module):
    """Parametrization of a selected weight: its value under the active precision point (``precision_value``).
    ``peers`` (set by the quantization component): [(leaf, module)] siblings whose latents a format's shared scale reads
    (a plain list, so the modules are not registered as children)."""

    def __init__(self, group: str):
        super().__init__()
        self.group = group
        self.peers = []

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        peers = [(leaf, module.parametrizations.weight.original) for leaf, module in self.peers]
        return precision_value(latent, self.group, peers)


class InputPrecision:
    """The input-activation quantizer of one selected linear module (a forward pre-hook installed by the quantization
    component): under an active point whose module config has ``activations``, the input becomes
    x + mix·(Q(x) − x), straight-through. ``amax`` is the static calibrated range for formats with a static scale
    (NVFP4: ``input_scale = amax / (6·448)``); ``observe`` records max |x| for calibration (BF16 forwards)."""

    def __init__(self, key: str):
        self.key, self.amax, self.observed = key, None, None
        self.observe = False

    def __call__(self, module, args):
        x = args[0]
        value = self.apply(x)
        return None if value is x else (value,) + tuple(args[1:])

    def apply(self, x):
        """``x`` under the active point (observing it for calibration when ``observe`` is set); ``x`` itself when the
        active point leaves this module's input unquantized."""
        if self.observe:
            current = x.detach().abs().amax().float()
            self.observed = current if self.observed is None else torch.maximum(self.observed, current)
        selected = _selected(self.key)
        if selected is None or not selected[2].get("activations"):
            return x
        _, mix, options = selected
        from ..train.qformats import activation_value, nvfp4_global_scale
        config = options["activations"]
        scale = None
        if config["format"] == "nvfp4":
            if self.amax is None:
                raise RuntimeError(f"{self.key}: static NVFP4 activations are not calibrated (Quantization.calibrate)")
            if self.amax.device != x.device:
                self.amax = self.amax.to(x.device)
            scale = nvfp4_global_scale(self.amax)
        value = activation_value(x.detach(), config, scale)
        return _PrecisionRamp.apply(x, value, float(mix))


# Inline linear precision (native trainer): the module keeps its plain ``weight`` Parameter (no parametrization, so
# parameter names, optimizer routing and checkpoints are unchanged); every linear execution path reads its effective
# operands through ``linear_precision``. Attributes set by ``Quantization.install(..., inline=True)``:
# ``native_precision_key`` (group key), ``native_precision_peers`` ([(leaf, module)] sharing a scale) and
# ``native_precision_input`` (an ``InputPrecision`` or None).


class _OperandStraightThrough(torch.autograd.Function):
    """Forward ``value`` (a detached, already ramped operand), identity gradient to ``latent``."""

    @staticmethod
    def forward(ctx, latent, value):
        return value

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class inline_precision:
    """``Quantization.context`` for inline modules: enters the point, then resolves each module's pass operands once
    (``native_precision_active``): the ramped weight ``W + λ·(Q(W) − W)`` of the current latents (detached; computed
    once per pass, so activation-checkpoint recomputation reuses it), the activation config, λ as a device scalar and
    the static input scale. Inside the pass, ``linear_precision`` only applies them, so compiled layer pieces see the
    same graph at every λ (no recompilation as the ramp moves) and fuse the activation fake quant."""

    def __init__(self, context, modules):
        self.context, self.modules = context, modules

    @torch.no_grad()
    def __enter__(self):
        self.context.__enter__()
        mixes = {}
        try:
            for module in self.modules:
                module.__dict__["native_precision_active"] = _resolve_inline(module, mixes)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for module in self.modules:
            module.__dict__.pop("native_precision_active", None)
        return self.context.__exit__(*exc)


def _resolve_inline(module, mixes):
    selected = _selected(module.native_precision_key)
    holder = module.__dict__.get("native_precision_input")
    if selected is None or selected[0] == "bf16":
        return None
    fmt, mix, options = selected
    weight = module.weight
    peers = [(leaf, peer.weight) for leaf, peer in module.__dict__.get("native_precision_peers", ())]
    if fmt == "nvfp4" and peers and options.get("shared_global"):
        leaf = parse_group_key(module.native_precision_key)[2]
        shared = next((set(s) for s in options["shared_global"] if leaf in s), set())
        amaxes = [weight.detach().abs().amax()] + [p.detach().abs().amax() for n, p in peers if n in shared]
        if len(amaxes) > 1:
            options = {**options, "global_amax": torch.stack(amaxes).amax()}
    w = weight.detach()
    key = (float(mix), w.device)
    if key not in mixes:
        mixes[key] = torch.tensor(float(mix), dtype=torch.float32, device=w.device)
    ramped = _ramped_weight(w, fmt, options, mixes[key])
    activation, scale = options.get("activations"), None
    if activation is not None and holder is not None:
        if activation["format"] == "nvfp4":
            if holder.amax is None:
                raise RuntimeError(f"{holder.key}: static NVFP4 activations are not calibrated (Quantization.calibrate)")
            from ..train.qformats import nvfp4_global_scale
            scale = nvfp4_global_scale(holder.amax.to(w.device))
    else:
        activation = None
    return ramped, activation, mixes[key], scale


_RAMP = {}
EAGER_NUMERICS = {"eager_numerics.division_rounding": True, "eager_numerics.disable_ftz": True,
                  "emulate_precision_casts": True}


def _ramp_rule(w, fmt, block_or_group, global_amax, mix):
    from ..train.qformats import weight_value
    options = ({"group": block_or_group, "global_amax": global_amax} if fmt == "nvfp4" else {"block": block_or_group})
    wf = w.float()
    return (wf + mix * (weight_value(w, fmt, options) - wf)).to(w.dtype)


def _ramped_weight(w, fmt, options, mix):
    """``W + λ·(Q(W) − W)`` once per pass: one compiled kernel set per weight shape on CUDA (the eager codec is ~60
    launches per weight; bit-identical arithmetic), the eager rule elsewhere. ``mix`` is a device scalar."""
    if fmt not in ("nvfp4", "fp8_block"):
        wf = w.float()
        return (wf + mix * (quantized_value(w, fmt, options) - wf)).to(w.dtype)
    size = int(options.get("group", 16) if fmt == "nvfp4" else options.get("block", 128))
    global_amax = options.get("global_amax")
    if fmt == "nvfp4" and global_amax is None:
        global_amax = w.abs().amax()
    if global_amax is not None:
        global_amax = torch.as_tensor(global_amax, device=w.device)
    if not w.is_cuda or not PRECISION["fused"]:
        return _ramp_rule(w, fmt, size, global_amax, mix)
    if "kernel" not in _RAMP:
        # Eager division rounding and no flush-to-zero: the codec's ties must round as the eager (deploy) rule does
        # (Triton's default approximate division moved 3,672 of 15.7M NVFP4 codes of one LFM3 weight).
        _RAMP["kernel"] = torch.compile(_ramp_rule, dynamic=False, options=EAGER_NUMERICS)
    return _RAMP["kernel"](w, fmt, size, global_amax, mix)


def _inline_operands(module, x, active):
    ramped, activation, mix, scale = active
    holder = module.__dict__.get("native_precision_input")
    if holder is not None and holder.observe:
        current = x.detach().abs().amax().float()
        holder.observed = current if holder.observed is None else torch.maximum(holder.observed, current)
    if activation is not None:
        from ..train.qformats import activation_value
        xf = x.detach().float()
        value = activation_value(x.detach(), activation, scale)
        x = _OperandStraightThrough.apply(x, (xf + mix * (value - xf)).to(x.dtype))
    return x, _OperandStraightThrough.apply(module.weight, ramped)


def linear_precision(module, x):
    """(input, weight) of a linear ``module`` under the active precision point. Without an installed key, or with
    no active point selecting it, returns ``x`` and ``module.weight`` themselves (the unquantized path, unchanged)."""
    state = module.__dict__
    if "native_precision_active" in state:
        active = state["native_precision_active"]
        if active is not None:
            return _inline_operands(module, x, active)
    holder = state.get("native_precision_input")
    if holder is not None:
        x = holder.apply(x)
    key = state.get("native_precision_key")
    weight = module.weight
    if key is None or PRECISION["groups"] is None:
        return x, weight
    peers = [(leaf, peer.weight) for leaf, peer in state.get("native_precision_peers", ())]
    return x, precision_value(weight, key, peers)


def precision_linear_forward(self, x):
    """Instance ``forward`` of an inline-quantized ``nn.Linear`` (marked so native carriers may replace it; they apply
    ``linear_precision`` themselves)."""
    x, weight = linear_precision(self, x)
    return torch.nn.functional.linear(x, weight, self.bias)


precision_linear_forward.natlang_precision_forward = True


# Global switches read by the MoE experts: which nested member is running (its private deltas apply) and whether
# deltas apply at all.
STATE = {"size": None, "enabled": True, "teacher": False}


def _expand_blocks(scale: torch.Tensor, block: int, columns: int) -> torch.Tensor:
    return scale.repeat_interleave(block, dim=-1)[..., :columns]


def block_scales(weight: torch.Tensor, block: int = QK_K) -> torch.Tensor:
    """Initial learned scales: Maple's row alpha broadcast to every ``block``-wide column block (FP32)."""
    _, alpha = ternary_codes(weight)
    blocks = -(-weight.shape[-1] // block)
    return alpha.float().expand(*weight.shape[:-1], blocks).clone()
