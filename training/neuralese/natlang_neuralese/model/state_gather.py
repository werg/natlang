"""FP32 history values with wide accumulation at shared convolution sources."""
from __future__ import annotations
import torch


class _WideStateGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, index):
        if (source.ndim != 3 or source.dtype != torch.float64 or index.dtype != torch.int64
                or index.ndim < 1 or index.device != source.device or source.shape[1] < 1):
            raise ValueError('wide state gather needs FP64 [batch,source,channels] and colocated int64 indices')
        if index.numel():
            valid = ((index >= -source.shape[1]) & (index < source.shape[1])).all()
            if source.is_cuda:
                torch._assert_async(valid, 'wide state gather index is outside source')
            elif not bool(valid):
                raise IndexError('wide state gather index is outside source')
        ctx.save_for_backward(index)
        ctx.source_shape = source.shape
        if source.is_cuda:
            from ..maple.state_gather_kernels import gather_values
            return gather_values(source, index)
        return source[:, index].float()

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        if grad.is_cuda:
            from ..maple.state_gather_kernels import accumulate_sources
            return accumulate_sources(grad, index, ctx.source_shape), None
        batch, sources, channels = ctx.source_shape
        result = torch.zeros(ctx.source_shape, device=grad.device, dtype=torch.float64)
        result.index_add_(1, index.reshape(-1).remainder(sources), grad.reshape(batch,-1,channels).double())
        return result, None


def gather_convolution_state(source, index):
    """Gather histories without materializing FP64 history values on CUDA.

    Differentiable source state stays wide until all consumers have contributed;
    each finite convolution receives the original FP32 values. The saved index
    and one FP64 source-gradient tensor bound backward workspace by source size.
    """
    from ..common.native_higher_order import active_reference_arithmetic
    if active_reference_arithmetic() is not None:
        # The existing higher-order reference deliberately uses traceable torch
        # operations instead of first-order Triton adjoints.
        return source[:,index].float()
    return _WideStateGather.apply(source, index)
