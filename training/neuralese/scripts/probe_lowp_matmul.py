"""Which real low-precision GEMMs this torch build runs on this GPU (QAT performance note, owner 2026-10-28).

Tries FP8 ``torch._scaled_mm`` with tensorwise, rowwise and 128-block scales (the FP8_BLOCK deploy format) and
NVFP4 (E2M1 x2 with swizzled E4M3 1x16 block scales), checks each result against the dequantized BF16/FP32 product
and times it against a BF16 GEMM of the same shape. Prints one JSON line per case. No kernels are implemented here.
"""
import json
import time

import torch

from natlang_neuralese.train import qformats as F


def timed(fn, repeat=20):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def to_blocked(scales):
    """cuBLAS's swizzled block-scale layout (128x4 tiles), as torchao's ``to_blocked``."""
    rows, cols = scales.shape
    row_blocks, col_blocks = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(row_blocks * 128, col_blocks * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    blocks = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def report(case, fn, reference, bf16_seconds):
    try:
        out = fn()
        torch.cuda.synchronize()
        error = float((out.float() - reference).norm() / reference.norm())
        seconds = timed(fn)
        print(json.dumps({'case': case, 'ok': True, 'rel_err_vs_dequantized': error,
                          'ms': seconds * 1e3, 'speedup_vs_bf16': bf16_seconds / seconds}), flush=True)
    except Exception as error:  # noqa: BLE001 - the probe records every failure mode
        print(json.dumps({'case': case, 'ok': False, 'error': f'{type(error).__name__}: {str(error)[:300]}'}),
              flush=True)


def main():
    torch.manual_seed(0)
    device = 'cuda'
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': props.name,
                      'capability': f'{props.major}.{props.minor}',
                      'has_float4_e2m1fn_x2': hasattr(torch, 'float4_e2m1fn_x2'),
                      'has_functional_scaled_mm': hasattr(torch.nn.functional, 'scaled_mm')}), flush=True)
    m, n, k = 4096, 4096, 4096
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
    bf16_seconds = timed(lambda: a @ b.t())
    print(json.dumps({'case': 'bf16', 'ms': bf16_seconds * 1e3,
                      'tflops': 2 * m * n * k / bf16_seconds / 1e12}), flush=True)

    # FP8 tensorwise.
    sa, sb = a.abs().amax().float() / 448, b.abs().amax().float() / 448
    a8, b8 = (a.float() / sa).to(torch.float8_e4m3fn), (b.float() / sb).to(torch.float8_e4m3fn)
    reference = (a8.float() * sa) @ (b8.float() * sb).t()
    report('fp8_tensorwise', lambda: torch._scaled_mm(a8, b8.t(), sa, sb, out_dtype=torch.bfloat16), reference,
           bf16_seconds)
    # FP8 rowwise.
    ra, rb = a.abs().amax(1, keepdim=True).float() / 448, b.abs().amax(1, keepdim=True).float() / 448
    a8r, b8r = (a.float() / ra).to(torch.float8_e4m3fn), (b.float() / rb).to(torch.float8_e4m3fn)
    reference = (a8r.float() * ra) @ (b8r.float() * rb).t()
    report('fp8_rowwise', lambda: torch._scaled_mm(a8r, b8r.t(), ra, rb.t(), out_dtype=torch.bfloat16), reference,
           bf16_seconds)
    # FP8 blockwise: activations 1x128 (dynamic per group), weights 128x128 (FP8_BLOCK), FP32 scales.
    ga = a.float().reshape(m, k // 128, 128).abs().amax(-1).clamp(min=1e-10) / 448  # [m, k/128]
    a8g = (a.float().reshape(m, k // 128, 128) / ga[..., None]).to(torch.float8_e4m3fn).reshape(m, k)
    codes, block_scales = F.fp8_block_encode(b)
    block_scales = block_scales.float()
    reference = (a8g.float().reshape(m, k // 128, 128) * ga[..., None]).reshape(m, k) @ F.fp8_block_decode(codes, block_scales).t()
    functional = getattr(torch.nn.functional, 'scaled_mm', None)
    if functional is not None:
        from torch.nn.functional import ScalingType, SwizzleType  # noqa: F401
        report('fp8_block_1x128_128x128 (F.scaled_mm)',
               lambda: functional(a8g, codes.t(), ga.t().contiguous().t(), ScalingType.BlockWise1x128,
                                  block_scales, ScalingType.BlockWise128x128, output_dtype=torch.bfloat16),
               reference, bf16_seconds)
    report('fp8_block_1x128_128x128 (_scaled_mm)',
           lambda: torch._scaled_mm(a8g, codes.t(), ga.t().contiguous().t(), block_scales.t().contiguous().t(),
                                    out_dtype=torch.bfloat16), reference, bf16_seconds)
    # NVFP4: E2M1 x2 with E4M3 1x16 block scales (swizzled) and FP32 global scales.
    if hasattr(torch, 'float4_e2m1fn_x2'):
        packed_b, scales_b, global_b = F.nvfp4_weight_encode(b)
        packed_a, scales_a, global_a = F.nvfp4_weight_encode(a)
        reference = F.nvfp4_weight_decode(packed_a, scales_a, global_a) @ F.nvfp4_weight_decode(
            packed_b, scales_b, global_b).t()
        fa, fb = packed_a.view(torch.float4_e2m1fn_x2), packed_b.view(torch.float4_e2m1fn_x2)
        blocked_a, blocked_b = to_blocked(scales_a), to_blocked(scales_b)
        report('nvfp4 (_scaled_mm, block scales only, global applied after)',
               lambda: torch._scaled_mm(fa, fb.t(), blocked_a, blocked_b, out_dtype=torch.bfloat16).float()
               * (global_a * global_b), reference, bf16_seconds)
        if functional is not None:
            from torch.nn.functional import ScalingType, SwizzleType
            report('nvfp4 (F.scaled_mm, BlockWise1x16 + TensorWise)',
                   lambda: functional(fa, fb.t(), [blocked_a, global_a.reshape(1)],
                                      [ScalingType.BlockWise1x16, ScalingType.TensorWise],
                                      [blocked_b, global_b.reshape(1)],
                                      [ScalingType.BlockWise1x16, ScalingType.TensorWise],
                                      swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
                                      swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
                                      output_dtype=torch.bfloat16), reference, bf16_seconds)


if __name__ == '__main__':
    main()
