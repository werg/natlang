"""Declared FP32 storage for small native norm gains; matrix arithmetic is unchanged."""
from __future__ import annotations

import torch

PROFILE = 'fp32-norms-v1'
_NORMS = {'RMSNorm', 'Lfm2RMSNorm', 'Qwen2RMSNorm', 'Qwen3RMSNorm'}


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
