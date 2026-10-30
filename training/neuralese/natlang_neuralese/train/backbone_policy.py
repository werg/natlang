"""Shared backbone trainable selection and exact state handling."""
from __future__ import annotations

import torch


def resolve_backbone_policy(backbone, policy='auto'):
    """Resolve the architecture-aware trajectory default."""
    if policy=='auto':
        return 'full'
    return policy


def decoder_layer_parameter_prefix(backbone):
    """HF parameter namespace for the text decoder's trainable layers."""
    return ('model.language_model.layers.' if getattr(backbone, 'architecture', None) == 'qwen3_5'
            else 'model.layers.')


def configure_backbone_training(backbone, policy='full', *, rank=16):
    """Freeze the port and enable one declared backbone policy.

    ``full`` means all native transformer layer parameters, matching the
    teacherless text warm-up. Embeddings, tied output weights, and final norms
    remain fixed.
    """
    policy=resolve_backbone_policy(backbone,policy)
    if policy == 'adapters':  # text-warmup's existing spelling
        policy = 'lora'
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    named=[]
    if policy == 'full':
        selected=full_backbone_parameter_names(backbone)
        named=[(name,parameter) for name,parameter in backbone.hf.named_parameters()
               if name in selected]
        for _,parameter in named:
            parameter.requires_grad_(True)
    elif policy == 'lora':
        from .adapters import inject_lora
        inject_lora(backbone,list(range(backbone.num_layers)),rank=rank,alpha=2*rank)
        named=[(name,parameter) for name,parameter in backbone.hf.named_parameters()
               if parameter.requires_grad and 'lora_' in name]
    elif policy == 'latent':
        named=latent_backbone_parameters(backbone)
    else:
        raise ValueError('unknown backbone training policy: '+str(policy))
    if not named:
        raise ValueError('backbone policy has no trainable parameters: '+str(policy))
    return named


def latent_backbone_parameters(backbone):
    """The ``latent`` policy (owner 2026-10-10: QAT woven into training): every native layer weight trains as a BF16
    latent, quantized only by the recipe's precision points (train/quantization.py). On a dense backbone (LFM, Qwen)
    it selects what ``full`` selects, under plain parameter names."""
    from .quantization import plain_parameter_name
    selected=full_backbone_parameter_names(backbone)
    # A precision point wraps a weight as ``parametrizations.weight.original``
    # without replacing the latent. Keep its checkpoint/optimizer identity
    # under the plain name across a BF16 -> QAT objective continuation.
    named=[(plain_parameter_name(name),parameter) for name,parameter in backbone.hf.named_parameters()
           if name in selected]
    for _,parameter in named:
        parameter.requires_grad_(True)
    return named


def plain_named_tensors(module):
    """Named parameters and buffers of ``module``, each parametrized weight also under its plain name (``x.weight``
    for ``x.parametrizations.weight.original``): the names backbone trainables are saved and restored under, so a
    BF16 serving model (no parametrizations, experts as buffers) and a training model agree."""
    from .quantization import plain_parameter_name
    values={}
    for name,tensor in list(module.named_parameters())+list(module.named_buffers()):
        values.setdefault(name,tensor)
        values.setdefault(plain_parameter_name(name),tensor)
    return values


def full_backbone_parameter_names(backbone):
    """Names selected by the native full-layer policy, without changing flags."""
    return {name for name, _ in backbone.hf.named_parameters()
            if name.startswith(decoder_layer_parameter_prefix(backbone))}


def backbone_trainable_state(named):
    """Named values for full, latent or QAT backbone trainables, detached on their device: checkpoint writers copy one
    storage at a time, and a host copy of a 12B model's latents would double it in unified memory."""
    return {name:parameter.detach() for name,parameter in named}


def restore_backbone_trainables(backbone, state, *, expected_names=None):
    """Restore named backbone values and reject omissions or incompatible names."""
    parameters=plain_named_tensors(backbone.hf)
    values=dict(state or {})
    if expected_names is not None and set(values) != set(expected_names):
        raise ValueError('backbone trainable names differ from declared policy')
    with torch.no_grad():
        for name,value in values.items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError('backbone trainable parameter mismatch: '+name)
            parameters[name].copy_(value.to(parameters[name]))
