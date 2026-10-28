"""Deployable low-precision formats of the quantization component: NVFP4 (ModelOpt) and FP8 block (compressed-tensors).

Each format has an exact codec (codes and scales as the deployed checkpoint stores them, and their dequantization) and a
fake-quant value (quantize -> dequantize, FP32, no gradient) that ``maple.ternary.quantized_value`` returns for the
straight-through precision ramp. The conventions are those of the engines and producers of Liquid's own LFM3 releases
(owner 2026-10-28), so a trained latent exports to the same layout and the QAT forward sees the deployed values:

- ``nvfp4`` weights (ModelOpt ``NVFP4QTensor``, ``LiquidAI/LFM3-1.7B-NVFP4``): along the input dim in groups of 16,
  a per-tensor FP32 global scale ``s2 = amax(W) / (6 * 448)``, per-group scales ``s = amax(group) / (6 * s2)`` cast to
  FP8 E4M3 (round to nearest even; a zero scale is stored as 1.0), codes ``E2M1(W / (s * s2))``: round to nearest on the
  grid {0, .5, 1, 1.5, 2, 3, 4, 6} with ties to the even code, saturating at 6, sign bit 8 (negative values that round
  to zero keep code 8, as ModelOpt's ``_cast_fp4``). Two codes per byte, the even element in the low nibble.
- ``nvfp4`` input activations (the NVFP4 GEMM's activation quantizer, vLLM/FlashInfer ``scaled_fp4_quant``): the
  global scale is static, ``input_scale = amax_calibrated / (6 * 448)`` (calibrated, see ``Calibration``); per
  group of 16 along the hidden dim the scale ``amax(group) / 6 / input_scale`` is cast to E4M3 with saturation at 448
  (runtime groups above the calibrated range saturate, as deployed), the values ``E2M1(x / (s * input_scale))``.
  The deploy kernels use approximate reciprocals (``rcp.approx.ftz``); this rule is the correctly rounded form, so
  rare rounding ties may differ from a given kernel by one code.
- ``fp8_block`` weights (compressed-tensors ``FP8_BLOCK``, ``LiquidAI/LFM3-1.7B-FP8``): 128x128 blocks, symmetric
  scale ``amax(block) / 448`` (the block range is first widened to include 0) stored in the weight's dtype (BF16),
  codes ``E4M3(clamp(W / s, -448, 448))`` with the division in the weight's dtype (compressed-tensors' ``_quantize``).
- ``fp8_block`` input activations (``fp8_group``: dynamic, vLLM ``per_token_group_quant_fp8``): per token and group of
  128 along the hidden dim, FP32 scale ``max(amax, 1e-10) / 448``, values ``E4M3(clamp(x / s, -448, 448))``.

The exactness of the weight rules against Liquid's checkpoints is measured by ``lfm3_qat_check`` (tests and report).
"""
from __future__ import annotations

import torch

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
FP4_MAX, FP8_MAX = 6.0, 448.0
NVFP4_GROUP, FP8_BLOCK, FP8_ACT_GROUP = 16, 128, 128
WEIGHT_FORMATS = ('nvfp4', 'fp8_block')
ACTIVATION_FORMATS = ('nvfp4', 'fp8_group')


def _table(values, device):
    return torch.tensor(values, dtype=torch.float32, device=device)


# E2M1 --------------------------------------------------------------------------------------------------------------
def e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 codes (uint8 0..15) of FP32 values already divided by their scales: nearest grid value, ties to the
    even code, saturating at 6; bit 3 is the sign of the input (``x < 0``)."""
    x = x.float()
    magnitude = x.abs()
    bounds = _table(_E2M1_BOUNDS, x.device)
    code = torch.bucketize(magnitude, bounds, right=False)  # bounds[i-1] < |x| <= bounds[i] -> i
    tie_up = (magnitude == bounds[code.clamp(max=len(_E2M1_BOUNDS) - 1)]) & (code % 2 == 1)
    code = code + tie_up.to(code.dtype)
    return (code + 8 * (x < 0).to(code.dtype)).to(torch.uint8)


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    grid = _table(E2M1_GRID, codes.device)
    value = grid[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -value, value)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Two codes per byte along the last dim: element 2i in the low nibble, 2i+1 in the high one (ModelOpt layout)."""
    codes = codes.to(torch.uint8)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    low, high = packed & 15, packed >> 4
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    """FP32 values rounded to E4M3 (nearest even), saturating at ±448."""
    return x.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def _groups(x: torch.Tensor, group: int) -> torch.Tensor:
    columns = x.shape[-1]
    if columns % group:
        raise ValueError(f'row length {columns} is not a multiple of the group {group}')
    return x.reshape(*x.shape[:-1], columns // group, group)


# NVFP4 weights (ModelOpt) -----------------------------------------------------------------------------------------
def nvfp4_global_scale(amax: torch.Tensor) -> torch.Tensor:
    """``weight_scale_2`` / ``input_scale``: FP32 ``amax / (6 * 448)``."""
    return amax.float() / (FP4_MAX * FP8_MAX)


def nvfp4_weight_encode(weight: torch.Tensor, group: int = NVFP4_GROUP, global_amax=None):
    """ModelOpt NVFP4 export of one weight: (packed uint8 codes [out, in/2], E4M3 group scales [out, in/group],
    FP32 global scale []). ``global_amax``: the amax the global scale is taken from when it is shared by projections
    that deploy as one fused GEMM (ModelOpt unifies w1/w3, the gate/up pair reading the same input); default the
    weight's own."""
    w = weight.detach()
    blocks = _groups(w, group)
    scale_2 = nvfp4_global_scale(w.abs().amax() if global_amax is None else torch.as_tensor(global_amax))
    amax = blocks.abs().amax(dim=-1).float()
    if float(scale_2) == 0.0:
        scales = torch.ones_like(amax).to(torch.float8_e4m3fn)
        codes = torch.zeros(blocks.shape, dtype=torch.uint8, device=w.device)
    else:
        per_group = amax / (FP4_MAX * scale_2)
        per_group[per_group == 0] = 1.0
        scales = per_group.to(torch.float8_e4m3fn)
        codes = e2m1_codes(blocks.float() / (scales.float() * scale_2).unsqueeze(-1))
    return pack_fp4(codes.reshape(w.shape)), scales, scale_2


def nvfp4_weight_decode(packed, scales, scale_2, group: int = NVFP4_GROUP):
    """FP32 dequantized weight of a ModelOpt NVFP4 tensor."""
    values = _groups(e2m1_values(unpack_fp4(packed)), group)
    return (values * (scales.float() * scale_2.float()).unsqueeze(-1)).reshape(*packed.shape[:-1], -1)


def nvfp4_weight_value(weight: torch.Tensor, group: int = NVFP4_GROUP, global_amax=None) -> torch.Tensor:
    return nvfp4_weight_decode(*nvfp4_weight_encode(weight, group, global_amax), group=group)


# NVFP4 activations (static global scale) ---------------------------------------------------------------------------
def nvfp4_activation_value(x: torch.Tensor, input_scale: torch.Tensor, group: int = NVFP4_GROUP) -> torch.Tensor:
    """FP32 fake-quantized activations at a static ``input_scale`` (FP32 scalar): dynamic E4M3 group scales
    ``amax(group) / 6 / input_scale`` (saturating at 448), E2M1 values of ``x / (s * input_scale)``."""
    input_scale = input_scale.float().to(x.device)
    blocks = _groups(x.float(), group)
    amax = blocks.abs().amax(dim=-1)
    scales = _e4m3(amax / FP4_MAX / input_scale).float()
    divisor = (scales * input_scale).unsqueeze(-1)
    codes = e2m1_codes(torch.where(divisor > 0, blocks / divisor, torch.zeros_like(blocks)))
    return (e2m1_values(codes) * divisor).reshape(x.shape)


# FP8 block weights (compressed-tensors) ---------------------------------------------------------------------------
def _block_view(weight: torch.Tensor, block: int):
    rows, columns = weight.shape
    if rows % block or columns % block:
        raise ValueError(f'weight {tuple(weight.shape)} is not tiled by {block}x{block} blocks')
    return weight.reshape(rows // block, block, columns // block, block)


def fp8_block_encode(weight: torch.Tensor, block: int = FP8_BLOCK):
    """compressed-tensors FP8_BLOCK export of one weight: (E4M3 codes [out, in], scales [out/block, in/block] in the
    weight's dtype)."""
    w = weight.detach()
    tiles = _block_view(w, block)
    low = tiles.amin(dim=(1, 3)).float().clamp(max=0)
    high = tiles.amax(dim=(1, 3)).float().clamp(min=0)
    scales = torch.clamp(torch.maximum(low.abs(), high) / FP8_MAX, min=torch.finfo(torch.float32).eps).to(w.dtype)
    expanded = scales[:, None, :, None]
    codes = torch.clamp(tiles / expanded, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes.reshape(w.shape), scales


def fp8_block_decode(codes, scales, block: int = FP8_BLOCK):
    tiles = _block_view(codes.float(), block)
    return (tiles * scales.float()[:, None, :, None]).reshape(codes.shape)


def fp8_block_value(weight: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    return fp8_block_decode(*fp8_block_encode(weight, block), block=block)


# FP8 dynamic group activations (vLLM) ------------------------------------------------------------------------------
def fp8_group_activation_value(x: torch.Tensor, group: int = FP8_ACT_GROUP) -> torch.Tensor:
    blocks = _groups(x.float(), group)
    scales = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / FP8_MAX
    return (_e4m3(blocks / scales).float() * scales).reshape(x.shape)


# Dispatch ---------------------------------------------------------------------------------------------------------
def weight_value(weight: torch.Tensor, fmt: str, options: dict) -> torch.Tensor:
    """FP32 deployed value of ``weight`` at a low-precision weight format."""
    if fmt == 'nvfp4':
        return nvfp4_weight_value(weight, int(options.get('group', NVFP4_GROUP)), options.get('global_amax'))
    if fmt == 'fp8_block':
        return fp8_block_value(weight, int(options.get('block', FP8_BLOCK)))
    raise ValueError(f'unknown low-precision weight format: {fmt}')


def activation_value(x: torch.Tensor, config: dict, input_scale=None) -> torch.Tensor:
    """FP32 deployed value of a module input at an activation format (``config`` is the point's ``activations``)."""
    fmt = config['format']
    if fmt == 'nvfp4':
        if input_scale is None:
            raise ValueError('nvfp4 activations need a calibrated static input scale (Quantization.calibrate)')
        return nvfp4_activation_value(x, input_scale, int(config.get('group', NVFP4_GROUP)))
    if fmt == 'fp8_group':
        return fp8_group_activation_value(x, int(config.get('group', FP8_ACT_GROUP)))
    raise ValueError(f'unknown activation format: {fmt}')
