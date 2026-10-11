"""Reuse frozen vocabulary casts without retaining one table per replay token."""
from __future__ import annotations

import torch
from torch import nn


def readout_weight(module: nn.Module, source: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Keep one disposable cast; trainable tables retain their ordinary autograd path.

    The buffer is nonpersistent, follows module device transfers, and is refreshed
    after checkpoint loads or in-place source updates. Saved backward tensors may
    still own an older cast until their graph finishes, as required by autograd.
    """
    name = '_frozen_readout_cast'
    if name not in module._buffers:
        module.register_buffer(name, None, persistent=False)
    if source.dtype == dtype:
        # Native-precision readouts alternate with FP32 recurrent states. They
        # need no cast and must not evict the cast used by those other states.
        return source
    if (source.requires_grad or source.is_inference()
            or torch.is_inference_mode_enabled()):
        setattr(module, name, None)
        module._frozen_readout_key = None
        return source.to(dtype=dtype)
    key = (id(source), source.data_ptr(), source._version, source.device, source.dtype, dtype)
    cached = getattr(module, name)
    if (getattr(module, '_frozen_readout_key', None) != key or cached is None
            or cached.device != source.device or cached.dtype != dtype):
        cached = source.detach().to(dtype=dtype)
        setattr(module, name, cached)
        module._frozen_readout_key = key
    return cached
