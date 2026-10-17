"""Declared FP32 storage for small native norm gains; matrix arithmetic is unchanged."""
from __future__ import annotations

import torch

PROFILE = 'fp32-norms-v1'
_NORMS = {'RMSNorm', 'Lfm2RMSNorm', 'Qwen2RMSNorm', 'Qwen3RMSNorm'}
NORM_EXECUTIONS = ('source-v1', 'stable-v1')


def configure_norm_execution(model, profile=None):
    """Absent legacy declarations retain source constructor/recompute dispatch."""
    selected = 'source-v1' if profile is None else profile
    if selected not in NORM_EXECUTIONS:
        raise ValueError('unsupported native norm execution profile: ' + str(profile))
    from ..maple.model import RMSNorm
    for module in model.port.modules():
        if isinstance(module, RMSNorm):
            module.native_stable_norm = selected == 'stable-v1'
    model.native_norm_execution = selected
    model.port.native_norm_execution = selected
    return selected


def validate_norm_execution(model, config):
    selected = config.get('native_norm_execution', 'source-v1')
    if (selected not in NORM_EXECUTIONS
            or getattr(model, 'native_norm_execution', 'source-v1') != selected
            or getattr(model.port, 'native_norm_execution', 'source-v1') != selected):
        raise ValueError('native norm execution differs from declared profile')
    from ..maple.model import RMSNorm
    if hasattr(model.port, 'modules'):
        for module in model.port.modules():
            if isinstance(module, RMSNorm) and bool(getattr(module, 'native_stable_norm', False)) != (selected == 'stable-v1'):
                raise ValueError('native norm module dispatch differs from declared profile')


def norm_execution_changes_arithmetic(model, previous):
    """Only the existing CUDA compiled Maple fallback differs between profiles."""
    if previous == model.native_norm_execution:
        return False
    import os
    from ..maple.model import RMSNorm
    weight = model.port.embedding_weight
    if not weight.is_cuda or os.environ.get('NATLANG_MAPLE_COMPILE', '1') == '0':
        return False
    if (weight.dtype in (torch.bfloat16, torch.float16)
            and os.environ.get('NATLANG_MAPLE_FUSED_NORM', '1') != '0'):
        return False
    return any(isinstance(module, RMSNorm) for module in model.port.modules())


def _activation_dtype(module, arguments, output):
    # HF RMSNorm multiplies by its gain after casting the normalized input.
    # A FP32 gain must not promote the following BF16 projection's activations.
    return output.to(arguments[0].dtype)


def norm_parameters(model):
    """Exact per-layer norm inventory, excluding the frozen final readout norm."""
    found = {}
    for name, module in model.port.hf.named_modules():
        if not name.startswith('model.layers.') or type(module).__name__ not in _NORMS:
            continue
        weight = getattr(module, 'weight', None)
        if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 1:
            raise ValueError('native norm profile requires a one-dimensional parameter gain: ' + name)
        found[name + '.weight'] = (module, weight)
    if not found:
        raise ValueError('native norm precision profile found no supported layer gains')
    return found


def configure_norm_precision(model, profile):
    """Apply an explicitly declared profile without allocating replacement Parameters."""
    if profile != PROFILE:
        raise ValueError('unsupported native ordinary precision profile: ' + str(profile))
    promoted = {}
    for name, (module, parameter) in norm_parameters(model).items():
        if parameter.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError('native norm profile cannot promote gain dtype: ' + name)
        if parameter.dtype != torch.float32:
            promoted['backbone.' + name] = {'from': str(parameter.dtype), 'to': 'torch.float32'}
            with torch.no_grad():
                parameter.data = parameter.detach().float()
                torch.autograd.graph.increment_version(parameter)
        if type(module).__name__ != 'RMSNorm' and not getattr(module, '_native_norm_output_cast', False):
            module.register_forward_hook(_activation_dtype)
            module._native_norm_output_cast = True
    model.native_ordinary_precision = profile
    return promoted


def upgrade_norm_precision(model, config, profile=PROFILE):
    """Explicit numeric-preserving legacy transition; optimizer restores by promoted names."""
    before = config.get('native_ordinary_precision')
    if before not in (None, profile):
        raise ValueError('native ordinary precision downgrade/replacement is unsupported')
    promoted = configure_norm_precision(model, profile)
    config['native_ordinary_precision'] = profile
    event = {'schema': 'natlang.native-ordinary-precision-upgrade/1', 'profile': profile,
            'previous_profile': before, 'promoted_parameters': promoted,
            'parameters_optimizer_rng_exposure': 'exact-numeric-values-preserved'}
    model._native_ordinary_precision_upgrade = event
    return event


def validate_norm_precision(model, config):
    profile = config.get('native_ordinary_precision')
    if profile != getattr(model, 'native_ordinary_precision', None):
        raise ValueError('native norm storage differs from declared profile')
    if profile is not None:
        if hasattr(model.port, 'hf'):
            dtypes = [p.dtype for _,p in norm_parameters(model).values()]
        else:
            dtypes = getattr(model, '_native_norm_parameter_dtypes', ())
        if profile != PROFILE or not dtypes or any(dtype != torch.float32 for dtype in dtypes):
            raise ValueError('native norm storage violates declared FP32 profile')
