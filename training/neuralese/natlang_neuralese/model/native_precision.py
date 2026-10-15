"""Explicit per-model native GEMM arithmetic; source parameters remain unchanged.

These modes cover fixed native backbone projections and vocabulary readout.
The embedding-space low-rank adapter and shared-state junctions retain FP32.
Stock backbone FFNs, custom routing, attention, state/gradient junctions and
optimizer retain their existing arithmetic. No mode changes torch global flags.
"""
from __future__ import annotations

import torch

NATIVE_LINEAR_PRECISIONS = ('ieee', 'tf32', 'tf32x3', 'bf16', 'bf16x6')
PRODUCTION_NATIVE_LINEAR_PRECISIONS = ('ieee', 'bf16x6')


def set_native_linear_precision(model, mode):
    """Select exact kernel plumbing; caller records any training transition.

    Apply at an update boundary, after restoring optimizer/RNG/exposure. The
    attributes persist through activation-checkpoint recomputation and coexist
    across multiple participants in a shared training process.
    """
    if mode not in NATIVE_LINEAR_PRECISIONS:
        raise ValueError(f'unsupported native linear precision {mode!r}')
    adapter_modules = set(model.adapter.modules())
    for module in model.modules():
        # Custom quantized/adapter module forwards remain their own contracts.
        if type(module) is torch.nn.Linear:
            selected = "ieee" if module in adapter_modules else mode
            module.native_linear_precision = selected
    model.native_linear_precision = mode
    model.port.native_linear_precision = mode
    return model


def same_source_precision_identity(previous, current):
    """Only runtime GEMM arithmetic may change in an ordinary continuation."""
    return ({k: v for k, v in previous.items() if k != 'native_linear_precision'} ==
            {k: v for k, v in current.items() if k != 'native_linear_precision'})
