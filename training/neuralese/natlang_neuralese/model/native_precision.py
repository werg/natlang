"""Explicit per-model native GEMM arithmetic; source parameters remain unchanged.

These modes cover fixed native backbone projections and vocabulary readout.
The embedding-space low-rank adapter and shared-state junctions retain FP32.
Stock backbone FFNs, custom routing, state/gradient junctions and optimizer
retain their existing arithmetic. Attention has its own explicit FP32 setting;
BF16 attention keeps its existing contract. Neither changes torch global flags.
"""
from __future__ import annotations

import torch

NATIVE_LINEAR_PRECISIONS = ('ieee', 'tf32', 'tf32x3', 'bf16', 'bf16x6')
PRODUCTION_NATIVE_LINEAR_PRECISIONS = ('ieee', 'bf16x6')
NATIVE_ATTENTION_PRECISIONS = ('ieee', 'tf32x3')


def set_native_attention_precision(model, mode):
    """Select per-model attention arithmetic without changing weights or globals."""
    if mode not in NATIVE_ATTENTION_PRECISIONS:
        raise ValueError(f'unsupported native attention precision {mode!r}')
    model.native_attention_precision = mode
    model.port.native_attention_precision = mode
    return model


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
    """Only declared native arithmetic may change in an ordinary continuation."""
    arithmetic = {'native_linear_precision', 'native_attention_precision', 'native_shared_execution'}
    return ({k: v for k, v in previous.items() if k not in arithmetic} ==
            {k: v for k, v in current.items() if k not in arithmetic})


def resolve_native_training_precision(previous, requested=None):
    """Diagnostic checkpoint arithmetic needs an explicit production transition."""
    selected = previous if requested is None else requested
    if selected not in PRODUCTION_NATIVE_LINEAR_PRECISIONS:
        raise ValueError('native production training requires ieee or bf16x6 arithmetic; '
                         'explicitly transition a diagnostic checkpoint before training')
    return selected
