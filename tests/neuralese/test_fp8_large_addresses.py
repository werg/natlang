"""Exact address-width and unchanged-gradient checks for the real MoE dW kernel."""
import pytest
import torch
import triton
import triton.language as tl
from natlang_neuralese.maple.fp8_experts import grouped_weight_gradient


@triton.jit
def _historical_grouped_weight_gradient(g_ptr, a_ptr, ends_ptr, out_ptr, N: tl.constexpr, K: tl.constexpr,
                                       SG0: tl.constexpr, SG1: tl.constexpr, SA0: tl.constexpr, SA1: tl.constexpr,
                                       FP32_INPUTS: tl.constexpr, B: tl.constexpr):
    # Frozen df15aa423 reference: its arithmetic is valid below the signed32 bound.
    expert = tl.program_id(0)
    n = tl.program_id(1) * B + tl.arange(0, B)
    k = tl.program_id(2) * B + tl.arange(0, B)
    start = tl.load(ends_ptr + expert - 1, mask=expert > 0, other=0)
    stop = tl.load(ends_ptr + expert)
    acc = tl.zeros((B, B), tl.float32)
    for p0 in range(start, stop, B):
        p = p0 + tl.arange(0, B)
        grad = tl.load(g_ptr + p[None, :] * SG0 + n[:, None] * SG1,
                       mask=(p[None, :] < stop) & (n[:, None] < N), other=0)
        value = tl.load(a_ptr + p[:, None] * SA0 + k[None, :] * SA1,
                        mask=(p[:, None] < stop) & (k[None, :] < K), other=0)
        if FP32_INPUTS:
            grad, value = grad.to(tl.float32), value.to(tl.float32)
        acc += tl.dot(grad, value, input_precision='ieee')
    tl.store(out_ptr + expert * N * K + n[:, None] * K + k[None, :], acc,
             mask=(n[:, None] < N) & (k[None, :] < K))


@triton.jit
def _address_law(out, ROW: tl.constexpr, STRIDE: tl.constexpr):
    row = tl.full((1,), ROW, dtype=tl.int32)
    narrow = row * STRIDE
    wide = row.to(tl.int64) * STRIDE
    lane = tl.arange(0, 1)
    tl.store(out + lane, narrow.to(tl.int64))
    tl.store(out + lane + 1, wide)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel evidence')
def test_actual_pending_routed_row_overflows_old_address_but_not_widened_address():
    row, stride = 161972 * 8 - 32, 2304
    offsets = torch.empty(2, device='cuda', dtype=torch.int64)
    _address_law[(1,)](offsets, row, stride)
    old, actual = offsets.cpu().tolist()
    assert row * stride > 2**31 - 1
    assert actual == row * stride
    assert old == row * stride - 2**32 and old < 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel evidence')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_wide_addresses_preserve_historical_complete_gradient_bits(dtype):
    torch.manual_seed(261019)
    g = torch.randn((74, 35), device='cuda', dtype=dtype)
    a = torch.randn((74, 29), device='cuda', dtype=dtype)
    ends = torch.tensor([0, 3, 37, 74], device='cuda', dtype=torch.int32)
    old = torch.empty((4, 35, 29), device='cuda', dtype=torch.float32)
    _historical_grouped_weight_gradient[(4, 2, 1)](g, a, ends, old, 35, 29,
        *g.stride(), *a.stride(), FP32_INPUTS=dtype==torch.float32, B=32,
        num_warps=4, num_stages=2)
    actual = grouped_weight_gradient(g, a, ends, old.shape)
    assert torch.equal(actual, old)


@pytest.mark.gpu_heavy
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel evidence')
def test_real_grouped_gradient_reads_actual_large_routed_address_without_aliasing():
    # One genuine dense-width2304 storage, but read only its first32 columns.
    # This exercises the exact production row*stride address with <6GB storage,
    # rather than loading a full model or intentionally launching an invalid kernel.
    rows, stride, tile = 161972 * 8, 2304, 32
    storage = torch.empty(rows * stride, device='cuda', dtype=torch.bfloat16)
    g = storage.as_strided((rows, tile), (stride, 1))
    g[-tile:].fill_(1)
    a = torch.ones((1, tile), device='cuda', dtype=torch.bfloat16).expand(rows, tile)
    ends = torch.tensor([rows - tile, rows], device='cuda', dtype=torch.int32)
    result = grouped_weight_gradient(g, a, ends, (2, tile, tile))
    assert torch.equal(result[1], torch.full((tile, tile), float(tile), device='cuda'))
