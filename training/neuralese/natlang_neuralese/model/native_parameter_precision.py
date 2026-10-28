"""Declared FP32 storage for small native norm gains; matrix arithmetic is unchanged."""
from __future__ import annotations

import torch

PROFILE = 'fp32-norms-v1'
QWEN35_DELTA_PROFILE = 'fp32-norms-qwen35-delta-controls-v1'
# Owner 2026-10-28 (LFM3): no full precision. Every parameter and optimizer state stays BF16, including the learned
# sparse role/reasoning rows and the embedding-space adapter; AdamW-routed parameters are written with stochastic
# rounding (AdamWSR) and Muon matrices through MuonSR. No FP32 masters are built.
BF16_SR_PROFILE = 'bf16-stochastic-rounding-v1'
ORDINARY_PRECISION_PROFILES = (PROFILE, QWEN35_DELTA_PROFILE, BF16_SR_PROFILE)


def profile_dtype(profile):
    """The storage dtype a declared ordinary precision profile requires of its parameters."""
    return torch.bfloat16 if profile == BF16_SR_PROFILE else torch.float32
_NORMS = {'RMSNorm', 'Lfm2RMSNorm', 'Lfm3RMSNorm', 'Qwen2RMSNorm', 'Qwen3RMSNorm',
          'Qwen3_5RMSNorm', 'Qwen3_5RMSNormGated'}
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
        if (not (name.startswith('model.layers.') or name.startswith('model.language_model.layers.'))
                or type(module).__name__ not in _NORMS):
            continue
        weight = getattr(module, 'weight', None)
        if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 1:
            raise ValueError('native norm profile requires a one-dimensional parameter gain: ' + name)
        found[name + '.weight'] = (module, weight)
    if not found:
        raise ValueError('native norm precision profile found no supported layer gains')
    return found


def qwen35_delta_control_parameters(model):
    """FP32 masters for the low-dimensional recurrent dynamics controls."""
    if getattr(model.port, 'architecture', None) != 'qwen3_5':
        return {}
    found = {}
    for name, module in model.port.hf.named_modules():
        if type(module).__name__ != 'Qwen3_5GatedDeltaNet':
            continue
        for suffix in ('A_log', 'dt_bias'):
            parameter = getattr(module, suffix, None)
            if not isinstance(parameter, torch.nn.Parameter) or parameter.ndim != 1:
                raise ValueError('Qwen3.5 ordinary precision profile requires one-dimensional DeltaNet controls: '
                                 + name + '.' + suffix)
            found[name + '.' + suffix] = (module, parameter)
    if not found:
        raise ValueError('Qwen3.5 ordinary precision profile found no DeltaNet controls')
    return found


def ordinary_precision_parameters(model, profile):
    values = norm_parameters(model)
    if profile == QWEN35_DELTA_PROFILE:
        values.update(qwen35_delta_control_parameters(model))
    return values


def ordinary_precision_state_dtypes(state, profile):
    """Precision inventory for metadata-only checkpoint identity validation."""
    selected = []
    for name, value in state.items():
        if not (name.startswith('model.layers.') or name.startswith('model.language_model.layers.')):
            continue
        ordinary_norm = name.endswith('norm.weight') and value.ndim == 1
        delta_control = (profile == QWEN35_DELTA_PROFILE
            and name.endswith(('.linear_attn.A_log', '.linear_attn.dt_bias')) and value.ndim == 1)
        if ordinary_norm or delta_control:
            selected.append(value.dtype)
    return tuple(selected)


def configure_norm_precision(model, profile):
    """Apply an explicitly declared ordinary-parameter profile without replacing Parameters."""
    if profile not in ORDINARY_PRECISION_PROFILES:
        raise ValueError('unsupported native ordinary precision profile: ' + str(profile))
    promoted = {}
    if profile == QWEN35_DELTA_PROFILE and getattr(model.port, 'architecture', None) != 'qwen3_5':
        raise ValueError('Qwen3.5 DeltaNet precision profile requires a Qwen3.5 text backbone')
    if profile == PROFILE and getattr(model.port, 'architecture', None) == 'qwen3_5':
        raise ValueError('Qwen3.5 requires the FP32 DeltaNet controls precision profile')
    if profile == BF16_SR_PROFILE:
        # Nothing is promoted: the declared contract is BF16 storage of every ordinary and learned-row parameter.
        values = [('backbone.' + name, parameter) for name, (_, parameter) in norm_parameters(model).items()]
        values += [(name, parameter) for name, parameter in model.named_parameters()
                   if name.startswith(('input_rows', 'output_rows', 'extra_input_rows', 'extra_output_rows',
                                       'adapter.'))]
        wrong = [name for name, parameter in values if parameter.dtype != torch.bfloat16]
        if wrong:
            raise ValueError('BF16 stochastic-rounding profile requires BF16 parameters: ' + ', '.join(wrong[:4]))
        model.native_ordinary_precision = profile
        return {}
    norms = norm_parameters(model)
    for name, (module, parameter) in ordinary_precision_parameters(model, profile).items():
        if parameter.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError('native norm profile cannot promote gain dtype: ' + name)
        if parameter.dtype != torch.float32:
            promoted['backbone.' + name] = {'from': str(parameter.dtype), 'to': 'torch.float32'}
            with torch.no_grad():
                parameter.data = parameter.detach().float()
                torch.autograd.graph.increment_version(parameter)
        if name in norms and type(module).__name__ != 'RMSNorm' and not getattr(module, '_native_norm_output_cast', False):
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
            try:
                values = ordinary_precision_parameters(model, profile)
            except ValueError as exc:
                raise ValueError('native ordinary precision profile cannot be validated') from exc
            dtypes = [p.dtype for _,p in values.values()]
        else:
            dtypes = getattr(model, '_native_ordinary_parameter_dtypes',
                             getattr(model, '_native_norm_parameter_dtypes', ()))
        if (profile not in ORDINARY_PRECISION_PROFILES or not dtypes
                or any(dtype != profile_dtype(profile) for dtype in dtypes)):
            raise ValueError('native norm storage violates declared ' + profile + ' profile')
