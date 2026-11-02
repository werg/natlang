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


@torch._dynamo.disable
def rowwise_norm_forward(original_forward, x):
    """Keep LFM3 CUDA BF16 normalization independent of invocation packing."""
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.numel() == 0:
        return original_forward(x)
    import torch.nn.functional as F
    rows = x.reshape(-1, x.shape[-1])
    # The custom adjoint's forward and serving run without autograd. Keep
    # only one output allocation there; retaining all pieces before cat
    # otherwise doubles the whole-trajectory output at the allocation peak.
    # The differentiable replay retains its original graph/fan-in ordering.
    differentiable = torch.is_grad_enabled()
    pieces = [] if differentiable else None
    output = None
    for start in range(0, rows.shape[0], 128):
        part = rows[start:start + 128]
        count = part.shape[0]
        if count < 128:
            part = F.pad(part, (0, 0, 0, 128 - count))
        normalized = original_forward(part)[:count]
        if differentiable:
            pieces.append(normalized)
        else:
            if output is None:
                output = normalized.new_empty(rows.shape)
            output[start:start + count].copy_(normalized)
    return (torch.cat(pieces, dim=0) if differentiable else output).reshape(x.shape)



def _batched_lfm3_norm_adjoint(x, weight, gradient, eps):
    """Batch independent 128-row adjoints while retaining BF16 gain fan-in order."""
    import torch.nn.functional as F
    rows = x.reshape(-1, x.shape[-1])
    grads = gradient.reshape_as(rows)
    count = rows.shape[0]
    padding = (-count) % 128
    if padding:
        rows = F.pad(rows, (0, 0, 0, padding))
        grads = F.pad(grads, (0, 0, 0, padding))
    rows = rows.reshape(-1, 128, rows.shape[-1]).detach().requires_grad_(True)
    gains = weight.detach().expand(rows.shape[0], -1).clone().requires_grad_(True)
    with torch.enable_grad():
        value = rows.float()
        normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(rows.dtype)
        output = gains[:, None, :] * normalized
        dx, gain_parts = torch.autograd.grad(output, (rows, gains), grads.reshape_as(rows))
    # Match the original custom adjoint's reverse-creation BF16 accumulation.
    dweight = gain_parts[-1]
    for i in range(gain_parts.shape[0] - 2, -1, -1):
        dweight = dweight + gain_parts[i]
    return dx.reshape(-1, x.shape[-1])[:count].reshape_as(x), dweight


class _RowwiseNormAdjoint(torch.autograd.Function):
    """Replay each norm chunk locally, avoiding full-input slice scatters."""
    @staticmethod
    def forward(ctx, x, weight, original_forward):
        ctx.save_for_backward(x, weight)
        ctx.original_forward = original_forward
        return rowwise_norm_forward(original_forward, x)

    @staticmethod
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        original = ctx.original_forward
        if torch.is_grad_enabled():
            # Preserve the ordinary differentiable path for higher derivatives.
            with torch.enable_grad():
                output = rowwise_norm_forward(original, x)
                targets = tuple(t for t in (x, weight) if t.requires_grad)
                values = iter(torch.autograd.grad(output, targets, gradient,
                                                  create_graph=True, allow_unused=True))
                dx = next(values) if x.requires_grad else None
                dweight = next(values) if weight.requires_grad else None
                return dx, dweight, None
        owner = getattr(original, '__self__', None)
        row_count = x.numel() // x.shape[-1] if x.ndim and x.shape[-1] else 0
        if (type(owner).__name__ == 'Lfm3RMSNorm' and x.is_cuda
                and x.dtype == gradient.dtype == weight.dtype == torch.bfloat16
                and x.requires_grad and weight.requires_grad and 129 <= row_count <= 1024
                and hasattr(owner, 'variance_epsilon')):
            dx, dweight = _batched_lfm3_norm_adjoint(x, weight, gradient, owner.variance_epsilon)
            return dx, dweight, None
        rows = x.reshape(-1, x.shape[-1])
        grads = gradient.reshape(-1, gradient.shape[-1])
        dx = torch.empty_like(rows)
        dweight = None
        # Autograd visits concatenated pieces in reverse creation order.
        for start in reversed(range(0, rows.shape[0], 128)):
            part = rows[start:start + 128].detach().requires_grad_(True)
            incoming = grads[start:start + 128]
            count = part.shape[0]
            with torch.enable_grad():
                if count < 128:
                    import torch.nn.functional as F
                    value = F.pad(part, (0, 0, 0, 128 - count))
                else:
                    value = part
                output = original(value)[:count]
                targets = (part, weight) if weight.requires_grad else (part,)
                derivatives = torch.autograd.grad(output, targets, incoming, allow_unused=True)
            dx[start:start + count].copy_(derivatives[0])
            if weight.requires_grad and derivatives[1] is not None:
                dweight = derivatives[1] if dweight is None else dweight + derivatives[1]
        return dx.reshape(x.shape), dweight, None

def _native_rowwise_forward(module, x):
    """Bound method: cloning a model must rebind both the norm and its original forward."""
    original_forward = module._native_original_forward
    if (torch.is_grad_enabled() and x.is_cuda and x.dtype == torch.bfloat16
            and (x.requires_grad or module.weight.requires_grad)):
        return _RowwiseNormAdjoint.apply(x, module.weight, original_forward)
    return rowwise_norm_forward(original_forward, x)


def configure_norm_execution(model, profile=None):
    """Absent legacy declarations retain source constructor/recompute dispatch."""
    selected = 'source-v1' if profile is None else profile
    if selected not in NORM_EXECUTIONS:
        raise ValueError('unsupported native norm execution profile: ' + str(profile))
    from ..maple.model import RMSNorm
    for module in model.port.modules():
        if isinstance(module, RMSNorm):
            module.native_stable_norm = selected == 'stable-v1'
        if (getattr(getattr(model.port, 'config', None), 'model_type', None) == 'lfm3'
                and type(module).__name__ in {'RMSNorm', 'Lfm2RMSNorm', 'Lfm3RMSNorm'}
                and isinstance(getattr(module, 'weight', None), torch.nn.Parameter)
                and not module.__dict__.get('native_rowwise_norm', False)):
            from types import MethodType
            # Store a bound method instead of a closure over this instance: Python's
            # deepcopy rebinds both methods to the copied module and its weights.
            module._native_original_forward = module.forward
            module.forward = MethodType(_native_rowwise_forward, module)
            module.native_rowwise_norm = True
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
            raise ValueError('native norm storage violates declared '
                             + ('BF16' if profile == BF16_SR_PROFILE else 'FP32') + ' profile')
