"""Physical CUDA addressing across the 2**31-element boundary."""
import gc

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual CUDA large-address allocation')
def test_grouped_dense_reads_and_writes_64bit_strides():
    from natlang_neuralese.maple.fused_moe import _grouped_dense_mm

    # The 407k-token convolution projection crosses this boundary even though
    # every individual matmul tile is small. Touch only two values in each
    # >4 GiB virtual tensor; the CUDA allocator still reserves the full span.
    stride = 2**31 + 17
    x = torch.empty_strided((2, 1), (stride, 1), device='cuda', dtype=torch.bfloat16)
    out = torch.empty_strided((2, 1), (stride, 1), device='cuda', dtype=torch.bfloat16)
    x[0, 0] = 3; x[1, 0] = 5
    weight = torch.tensor([2.], device='cuda', dtype=torch.bfloat16)
    _grouped_dense_mm[(1, 1)](x, x, weight, out, x, 2, 1, 1,
        stride, 1, 1, 1, 1, stride, 1, HAS_ROWS=False, SINGLE_EXPERT=True,
        INNER_CONTIGUOUS=True, DOT_PRECISION='native', BM=16, BN=16, BK=32,
        num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    assert out[0, 0].item() == 6 and out[1, 0].item() == 10
    del x, out
    gc.collect(); torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual CUDA large-address allocation')
def test_ordered_dense_gradient_reads_beyond_32bit_element_offset():
    from natlang_neuralese.maple.fused_moe import _ordered_dense_weight_gradient

    n, k = 6144, 1
    start = 2**31 // n + 1
    gradient = torch.empty((start + 1, n), device='cuda', dtype=torch.bfloat16)
    x = torch.empty((start + 1, k), device='cuda', dtype=torch.bfloat16)
    gradient[start].zero_(); gradient[start, 0] = 2; x[start, 0] = 3
    offsets = torch.tensor([start], device='cuda', dtype=torch.int32)
    counts = torch.tensor([1], device='cuda', dtype=torch.int32)
    acc = torch.zeros((n, k), device='cuda', dtype=torch.float32)
    _ordered_dense_weight_gradient[(n // 64, 1)](gradient, x, offsets, counts, acc,
        n, k, 1, acc.stride(0), acc.stride(1), BM=64, BN=64, BK=32,
        DOT_PRECISION='native', num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    assert acc[0, 0].item() == 6 and torch.count_nonzero(acc).item() == 1
    del gradient, x
    gc.collect(); torch.cuda.empty_cache()
