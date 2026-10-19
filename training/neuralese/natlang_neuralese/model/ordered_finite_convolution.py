"""Exact reverse-token finite-convolution cotangent accumulation."""
from __future__ import annotations

from functools import lru_cache


_CPU_PRODUCT_BATCH_BYTES = 16 * 1024 * 1024


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def weight_kernel(target_ptr, history_ptr, gradient_ptr,
                      q_count, channels: tl.constexpr, width: tl.constexpr,
                      hs_q: tl.constexpr, hs_w: tl.constexpr, hs_c: tl.constexpr,
                      gs_q: tl.constexpr, gs_c: tl.constexpr,
                      ts_c: tl.constexpr, ts_w: tl.constexpr,
                      BLOCK: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < channels * width
        channel = offset // width
        window = offset % width
        target_offset = channel * ts_c + window * ts_w
        accumulator = tl.load(target_ptr + target_offset, valid, other=0.0)
        for step in range(q_count):
            row = q_count - 1 - step
            upstream = tl.load(gradient_ptr + row * gs_q + channel * gs_c, valid, other=0.0)
            history = tl.load(history_ptr + row * hs_q + window * hs_w + channel * hs_c,
                              valid, other=0.0)
            # The forward/carrier derivative casts both operands to FP32.
            contribution = upstream.to(tl.float32) * history.to(tl.float32)
            accumulator = accumulator + contribution
        tl.store(target_ptr + target_offset, accumulator, valid)

    @triton.jit
    def bias_kernel(target_ptr, gradient_ptr,
                    q_count, channels: tl.constexpr,
                    gs_q: tl.constexpr, gs_c: tl.constexpr, ts_c: tl.constexpr,
                    BLOCK: tl.constexpr):
        channel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = channel < channels
        accumulator = tl.load(target_ptr + channel * ts_c, valid, other=0.0)
        for step in range(q_count):
            row = q_count - 1 - step
            contribution = tl.load(gradient_ptr + row * gs_q + channel * gs_c, valid, other=0.0)
            accumulator = accumulator + contribution.to(tl.float32)
        tl.store(target_ptr + channel * ts_c, accumulator, valid)

    return weight_kernel, bias_kernel


def _ordered_gpu_update(target, histories, gradient, *, bias=False):
    import triton

    weight_kernel, bias_kernel = _kernels()
    if bias:
        bias_kernel[(triton.cdiv(target.numel(), 128),)](
            target, gradient, gradient.shape[1], target.shape[0],
            gradient.stride(1), gradient.stride(2), target.stride(0), 128,
            num_warps=4, enable_fp_fusion=False)
    else:
        channels, _, width = target.shape
        weight_kernel[(triton.cdiv(channels * width, 128),)](
            target, histories, gradient, gradient.shape[1], channels, width,
            histories.stride(1), histories.stride(2), histories.stride(3),
            gradient.stride(1), gradient.stride(2), target.stride(0), target.stride(2), 128,
            num_warps=4, enable_fp_fusion=False)


def _cpu_update(target, histories, gradient, *, bias=False):
    import numpy as np
    import torch

    q_count = gradient.shape[1]
    if q_count == 0:
        return
    if bias:
        row_bytes = max(4, target.numel() * 4)
    else:
        row_bytes = max(4, target.numel() * 4)
    rows_per_batch = max(1, _CPU_PRODUCT_BATCH_BYTES // row_bytes)
    for stop in range(q_count, 0, -rows_per_batch):
        start = max(0, stop - rows_per_batch)
        grad = gradient[:, start:stop].detach().float()
        if bias:
            values = grad.squeeze(0)
        else:
            if histories is None or histories.shape[0] != 1:
                raise RuntimeError('finite-convolution ordered accumulation requires one graph row')
            history = histories[:, start:stop].detach().float()
            values = (grad.unsqueeze(2) * history).squeeze(0).permute(0, 2, 1).unsqueeze(2)
        values = values.to(device='cpu', dtype=torch.float32).contiguous()
        if bias:
            # The bias rows are the incoming gradient itself; CPU `.to()` and
            # `.contiguous()` may both return aliases. NumPy accumulation below
            # is in-place, so take one bounded owned batch first.
            values = values.clone()
        reverse_rows = values.numpy()[::-1]
        # Ufunc accumulate is a coordinate-wise left fold. Seed its first row
        # with the preexisting accumulator so multiple batches/microbatches keep
        # the same non-associative order as separate cached autograd calls.
        np.add(reverse_rows[0], target.numpy(), out=reverse_rows[0])
        np.add.accumulate(reverse_rows, axis=0, dtype=np.float32, out=reverse_rows)
        target.copy_(values[0])


def ordered_finite_convolution_accumulate(target, histories, gradient, *, backend, bias=False):
    """Add reverse-token FP32 contributions to an existing accumulator in place.

    ``gradient`` is the exact FP32 effective upstream gradient for each token;
    ``histories`` has shape [1,Q,W,C] for convolution weights and is omitted for
    bias. Existing accumulator contents are the starting value, so the operation
    also preserves ordering across microbatches.
    """
    import torch

    if not isinstance(target, torch.Tensor) or target.dtype != torch.float32:
        raise TypeError('ordered finite-convolution target must be an FP32 accumulator')
    if gradient.ndim != 3 or gradient.shape[0] != 1:
        raise RuntimeError('finite-convolution ordered accumulation requires one batch row')
    if bias:
        if target.ndim != 1 or target.shape[0] != gradient.shape[2]:
            raise RuntimeError('finite-convolution bias accumulator shape changed')
    elif (target.ndim != 3 or target.shape[1] != 1 or histories is None
          or histories.ndim != 4 or histories.shape[0] != 1
          or histories.shape[1] != gradient.shape[1]
          or histories.shape[2] != target.shape[2]
          or histories.shape[3] != target.shape[0]
          or gradient.shape[2] != target.shape[0]):
        raise RuntimeError('finite-convolution ordered accumulator shape changed')

    if backend == 'device_fp32':
        if target.device.type != 'cuda' or gradient.device != target.device:
            raise RuntimeError('device_fp32 finite-convolution accumulator is not on the parameter device')
        if histories is not None and histories.device != target.device:
            raise RuntimeError('finite-convolution history and accumulator devices differ')
        _ordered_gpu_update(target, histories, gradient, bias=bias)
    elif backend == 'cpu_file':
        if target.device.type != 'cpu':
            raise RuntimeError('cpu_file finite-convolution accumulator is not CPU-backed')
        _cpu_update(target, histories, gradient, bias=bias)
    else:
        raise ValueError('unknown cotangent accumulator backend')
