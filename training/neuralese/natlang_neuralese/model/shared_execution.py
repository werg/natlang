"""Explicit per-model fixed row arithmetic and shared-state adjoint profile."""
from __future__ import annotations
from types import MethodType
import torch
import torch.nn.functional as F

NATIVE_SHARED_EXECUTIONS=('source-v1','stable-v1')
_NORM_TYPES={'RMSNorm','Lfm2RMSNorm','Qwen2RMSNorm','Qwen3RMSNorm'}


def _fixed_ffn(module,value):
    if hasattr(module,'native_fixed_forward'):
        return module.native_fixed_forward(value)
    if not value.is_cuda:
        return F.linear(value,module.weight,module.bias)
    from ..maple.fused_moe import invariant_linear
    shape=value.shape
    output=invariant_linear(value.reshape(-1,shape[-1]),module.weight,precision='ieee')
    output=output.reshape(*shape[:-1],module.weight.shape[0])
    return output if module.bias is None else output+module.bias


def _fixed_norm(module,value):
    from ..common.native_higher_order import active_reference_arithmetic
    if not value.is_cuda or active_reference_arithmetic() is not None:
        return module._native_shared_source_forward(value)
    from ..maple.norm_kernels import fused_rms_norm
    epsilon=getattr(module,'variance_epsilon',getattr(module,'eps',None))
    if epsilon is None:
        raise ValueError('stable shared norm lacks its source epsilon')
    gain=module.gain()if callable(getattr(module,'gain',None))else module.weight
    return fused_rms_norm(value,gain,epsilon)


def configure_shared_execution(model,profile='source-v1'):
    if profile not in NATIVE_SHARED_EXECUTIONS:
        raise ValueError('unsupported native shared execution profile: '+str(profile))
    for name,module in model.port.named_modules():
        implementation=(_fixed_ffn if type(module)is torch.nn.Linear and '.feed_forward.'in name
            else _fixed_norm if type(module).__name__ in _NORM_TYPES else None)
        if implementation is None:
            continue
        if profile=='stable-v1':
            if not hasattr(module,'_native_shared_source_forward'):
                module._native_shared_source_forward=module.forward
            module.forward=MethodType(implementation,module)
        elif hasattr(module,'_native_shared_source_forward'):
            module.forward=module._native_shared_source_forward
    model.native_shared_execution=profile
    model.port.native_shared_execution=profile
    return profile


def validate_shared_execution(model,config):
    profile=config.get('native_shared_execution','source-v1')
    if (profile not in NATIVE_SHARED_EXECUTIONS
            or getattr(model,'native_shared_execution','source-v1')!=profile
            or getattr(model.port,'native_shared_execution','source-v1')!=profile):
        raise ValueError('native shared execution differs from declared profile')
    for name,module in model.port.named_modules():
        implementation=(_fixed_ffn if type(module)is torch.nn.Linear and '.feed_forward.'in name
            else _fixed_norm if type(module).__name__ in _NORM_TYPES else None)
        if implementation is None:
            continue
        active=getattr(module.forward,'__func__',None)
        if profile=='stable-v1' and (active is not implementation
                or not hasattr(module,'_native_shared_source_forward')):
            raise ValueError('native shared execution module differs from declared profile: '+name)
        if profile=='source-v1' and active in (_fixed_ffn,_fixed_norm):
            raise ValueError('native source execution retains stable module: '+name)


def convolution_state_dtype(port):
    return (torch.float64 if torch.is_grad_enabled()
        and getattr(port,'native_shared_execution','source-v1')=='stable-v1' else torch.float32)
