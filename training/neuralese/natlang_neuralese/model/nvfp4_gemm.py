"""Real NVFP4 forward GEMMs for an inline QAT pass at λ = 1 (Blackwell: ``F.scaled_mm`` BlockWise1x16 + TensorWise).

The NVFP4 deploy format (ModelOpt; ``train/qformats.py``) is exactly what the Blackwell block-scaled FP4 GEMM
multiplies: E2M1 codes with E4M3 scales per 16 values along K and one FP32 global scale per operand (the weight's
``weight_scale_2``, the activation's static ``input_scale``). At λ = 1 the fake-quant forward multiplies the same
values after rounding each operand to BF16; the FP4 GEMM multiplies them unrounded with FP32 accumulation, which is
the deployed arithmetic. The backward is the fake-quant pass's: straight-through, dX = dY·W_q and dW = dYᵀ·x_q with
the BF16 dequantized operands.

The activation encode (codes, packing, swizzled scales and the BF16 value the backward needs) is ordinary torch code,
so inside the compiled layer pieces (``shared_state._plain_compiled``) Inductor fuses it with the producing norm or
SiLU product, as it fuses the fake-quant codec; only the GEMM itself is an opaque custom op (no graph break). The weight
operand is encoded once per pass on entry. ``NATLANG_NVFP4_GEMM=0`` opts out (fake quant everywhere).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def supported(device) -> bool:
    if os.environ.get('NATLANG_NVFP4_GEMM', '1') == '0' or not hasattr(F, 'scaled_mm'):
        return False
    if not hasattr(torch, 'float4_e2m1fn_x2') or torch.device(device).type != 'cuda':
        return False
    return torch.cuda.get_device_capability(device) >= (10, 0)


def to_blocked(scales):
    """cuBLAS's swizzled block-scale layout (128x4 tiles, as torchao's ``to_blocked``) of [rows, K/16] scales."""
    rows, cols = scales.shape
    row_blocks, col_blocks = -(-rows // 128), -(-cols // 4)
    padded = F.pad(scales.view(torch.uint8), (0, col_blocks * 4 - cols, 0, row_blocks * 128 - rows)).view(scales.dtype)
    blocks = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _weight_operand(w, global_amax):
    from ..train import qformats as Q
    packed, scales, scale_2 = Q.nvfp4_weight_encode(w, global_amax=global_amax)
    return packed, to_blocked(scales), scale_2.reshape(1)


def activation_operand(x, input_scale):
    """(packed FP4 codes [M, K/2], swizzled E4M3 block scales, BF16 deployed value [M, K]) of 2-D activations."""
    from ..train import qformats as Q
    codes, scales = Q.nvfp4_activation_encode(x, input_scale)
    divisor = (scales.float() * input_scale.float()).unsqueeze(-1)
    value = (Q._groups(Q.e2m1_values(codes), Q.NVFP4_GROUP) * divisor).reshape(x.shape).to(x.dtype)
    return Q.pack_fp4(codes), to_blocked(scales), value


_COMPILED = {}


def _compiled(name, function):
    if name not in _COMPILED:
        from ..maple.ternary import EAGER_NUMERICS
        _COMPILED[name] = (torch.compile(function, dynamic=True) if name == 'activation'
                           else torch.compile(function, dynamic=False, options=EAGER_NUMERICS))
    return _COMPILED[name]


@torch.no_grad()
def weight_operand(w, global_amax):
    """(packed FP4 [N, K/2], swizzled block scales, global scale [1]) of a BF16 latent, once per pass."""
    return _compiled('weight', _weight_operand)(w.detach(), global_amax)


@torch.library.custom_op('natlang::nvfp4_gemm', mutates_args=())
def nvfp4_gemm(value: torch.Tensor, ramped: torch.Tensor, packed_x: torch.Tensor, blocked_x: torch.Tensor,
               input_scale: torch.Tensor, packed_w: torch.Tensor, blocked_w: torch.Tensor,
               global_w: torch.Tensor) -> torch.Tensor:
    """BF16 ``x_q @ W_qᵀ`` of the NVFP4 operands with FP32 accumulation (cuBLASLt block-scaled FP4). ``value`` and
    ``ramped`` are the same operands dequantized to BF16: the differentiable inputs of the straight-through backward
    (dX = dY·W_q, dW = dYᵀ·x_q, as the fake-quant pass); the GEMM itself reads only the codes and scales."""
    from torch.nn.functional import ScalingType, SwizzleType
    f4 = torch.float4_e2m1fn_x2
    return F.scaled_mm(packed_x.view(f4), packed_w.view(f4).t(),
                       [blocked_x.view(torch.float8_e4m3fn), input_scale.reshape(1).float()],
                       [ScalingType.BlockWise1x16, ScalingType.TensorWise],
                       [blocked_w.view(torch.float8_e4m3fn), global_w.reshape(1).float()],
                       [ScalingType.BlockWise1x16, ScalingType.TensorWise],
                       swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
                       swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
                       output_dtype=torch.bfloat16)


@nvfp4_gemm.register_fake
def _(value, ramped, packed_x, blocked_x, input_scale, packed_w, blocked_w, global_w):
    return value.new_empty((value.shape[0], ramped.shape[0]), dtype=torch.bfloat16)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])


def _backward(ctx, grad):
    value, ramped = ctx.saved_tensors
    dx = grad @ ramped if ctx.needs_input_grad[0] else None
    dw = grad.t() @ value if ctx.needs_input_grad[1] else None
    return dx, dw, None, None, None, None, None, None


nvfp4_gemm.register_autograd(_backward, setup_context=_setup)


def nvfp4_linear(x, weight, ramped, operand, input_scale, bias=None):
    """``x @ Q(W)ᵀ`` through the FP4 GEMM, straight-through to ``x`` and the latent ``weight`` (dX with the BF16
    ``ramped`` = Q(W), dW with the BF16 deployed activations, as the fake-quant pass)."""
    from ..maple.ternary import _OperandStraightThrough
    flat = x.reshape(-1, x.shape[-1])
    # Traced inside a compiled layer piece, the encode fuses there; eager callers use one compiled encode.
    encode = activation_operand if torch.compiler.is_compiling() else _compiled('activation', activation_operand)
    packed_x, blocked_x, value = encode(flat.detach(), input_scale)
    out = nvfp4_gemm(_OperandStraightThrough.apply(flat, value), _OperandStraightThrough.apply(weight, ramped),
                     packed_x, blocked_x, input_scale, *operand)
    out = out.reshape(*x.shape[:-1], out.shape[-1])
    return out if bias is None else out + bias
