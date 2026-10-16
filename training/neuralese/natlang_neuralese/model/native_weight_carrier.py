"""Separate exact forward weight storage from a coefficient's FP32 cotangent.

Only explicit native adapter scopes install these carriers. Ordinary model
execution keeps its original module objects, dtypes, kernels and autograd.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F


class _RoundedWeightCarrier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, previous, delta, storage_dtype):
        # Preserve the existing ordered BF16 cast/add semantics exactly.
        return (previous.to(storage_dtype) + delta.to(storage_dtype)).float()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient, None


def rounded_weight_carrier(previous, delta, storage_dtype):
    return _RoundedWeightCarrier.apply(previous, delta, storage_dtype)


class _TorchStoredLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, storage_dtype):
        stored = weight.to(storage_dtype)
        ctx.save_for_backward(x, stored)
        ctx.bias = bias is not None
        return F.linear(x, stored, bias)

    @staticmethod
    def backward(ctx, grad):
        x, stored = ctx.saved_tensors
        flat_grad, flat_x = grad.reshape(-1, grad.shape[-1]), x.reshape(-1, x.shape[-1])
        dx = (grad @ stored) if ctx.needs_input_grad[0] else None
        dw = None
        if ctx.needs_input_grad[1]:
            if grad.is_cuda:
                from ..maple.fused_moe import invariant_weight_gradient
                dw = invariant_weight_gradient(flat_x, flat_grad)
            else:
                dw = flat_grad.float().T @ flat_x.float()
        db = flat_grad.sum(0) if ctx.bias and ctx.needs_input_grad[2] else None
        return dx, dw, db, None


def stored_linear(x, weight, bias=None, *, storage_dtype, fixed=False, precision='ieee'):
    if fixed:
        if x.is_cuda and x.dtype in (torch.float32, torch.bfloat16):
            from ..maple.fused_moe import invariant_linear
            shape = x.shape
            out = invariant_linear(x.reshape(-1, shape[-1]), weight, precision=precision,
                                   forward_storage_dtype=storage_dtype).reshape(*shape[:-1], weight.shape[0])
        else:
            out = _TorchStoredLinear.apply(x, weight, None, storage_dtype)
        return out if bias is None else out + bias
    return _TorchStoredLinear.apply(x, weight, bias, storage_dtype)



class NativeCarrierLinear(nn.Linear):
    """Preserve native fixed projections and ordinary HF Linear arithmetic."""
    def __init__(self, original):
        # Preserve hooks/config/ties without resetting or cloning parameters.
        self.__dict__ = original.__dict__.copy()
        self._parameters = original._parameters.copy()
        self._buffers = original._buffers.copy()
        self._modules = original._modules.copy()
        self.forward_storage_dtype = original.weight.dtype

    def forward(self, x):
        return stored_linear(x, self.weight, self.bias, storage_dtype=self.forward_storage_dtype)

    def native_fixed_forward(self, x):
        return stored_linear(x, self.weight, self.bias, storage_dtype=self.forward_storage_dtype,
                             fixed=True, precision=getattr(self, 'native_linear_precision', 'ieee'))


@contextmanager
def carrier_linear_modules(model, selected_weights):
    """Temporarily wrap every Linear alias of selected BF16/FP16 matrices."""
    selected = {id(weight) for weight in selected_weights if weight.dtype != torch.float32}
    changes, wrappers = [], {}
    try:
        for name, module in list(model.named_modules(remove_duplicate=False)):
            if not name or not hasattr(module, 'weight') or id(module.weight) not in selected:
                continue
            if type(module) is not nn.Linear:
                raise ValueError('FP32 native coefficient carriers require exact Linear layer matrices or dense expert banks')
            wrapper = wrappers.setdefault(id(module), NativeCarrierLinear(module))
            parent_name, _, child = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            changes.append((parent, child, module))
            parent._modules[child] = wrapper
        yield
    finally:
        for parent, child, original in reversed(changes):
            parent._modules[child] = original
