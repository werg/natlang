"""Shared backbone trainable selection and exact state handling."""
from __future__ import annotations

import torch


def resolve_backbone_policy(backbone, policy='auto'):
    """Resolve the architecture-aware trajectory default."""
    if policy=='auto':
        return 'qat' if getattr(backbone,'ternary',False) else 'full'
    return policy


def configure_backbone_training(backbone, policy='full', *, rank=16):
    """Freeze the port and enable one declared backbone policy.

    ``full`` means all native transformer layer parameters, matching the
    teacherless text warm-up. Embeddings, tied output weights, and final norms
    remain fixed. Ternary Maple uses its explicit ``qat`` policy.
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
    elif policy == 'qat':
        if not getattr(backbone, 'ternary', False):
            raise ValueError('qat is the ternary backbone policy')
        from .adapters import maple_qat_parameters
        named=maple_qat_parameters(backbone)
    else:
        raise ValueError('unknown backbone training policy: '+str(policy))
    if not named:
        raise ValueError('backbone policy has no trainable parameters: '+str(policy))
    return named


def full_backbone_parameter_names(backbone):
    """Names selected by the native full-layer policy, without changing flags."""
    if getattr(backbone, 'ternary', False):
        raise ValueError('ternary backbone has no native full-weight policy')
    return {name for name, _ in backbone.hf.named_parameters()
            if name.startswith('model.layers.')}


def backbone_trainable_state(named):
    """Portable named values for full or QAT backbone trainables."""
    return {name:parameter.detach().cpu() for name,parameter in named}


def restore_backbone_trainables(backbone, state, *, expected_names=None):
    """Restore named backbone values and reject omissions or incompatible names."""
    parameters=dict(backbone.hf.named_parameters())
    values=dict(state or {})
    if expected_names is not None and set(values) != set(expected_names):
        raise ValueError('backbone trainable names differ from declared policy')
    with torch.no_grad():
        for name,value in values.items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError('backbone trainable parameter mismatch: '+name)
            parameters[name].copy_(value.to(parameters[name]))


class shared_parametrized_weights:
    """Build every parametrized backbone weight (Maple QAT: base + LoRA + dense latent, ternarized with learned
    scales) once per forward/backward pass instead of at every use. Without it each attention projection is rebuilt
    for the history and branch streams and again in each checkpoint recompute, each copy with its own backward.

    Weights are built up front, outside checkpointed layers, so a layer's recompute sees the same (cached) tensor as
    its forward. Call the context value after each backward to release the pass's graph and build the next pass's
    weights from the current parameters. A no-op for backbones without parametrizations (LFM, Qwen)."""

    def __init__(self, module):
        from torch.nn.utils import parametrize
        self.parametrize = parametrize
        self.owners = [(m, name) for m in module.modules() if parametrize.is_parametrized(m)
                       for name in m.parametrizations]
        self.context = parametrize.cached() if self.owners else None

    def _build(self):
        for owner, name in self.owners:
            getattr(owner, name)

    def refresh(self):
        if self.context is not None:
            self.parametrize._cache.clear()
            self._build()

    def __enter__(self):
        if self.context is not None:
            self.context.__enter__()
            self._build()
        return self.refresh

    def __exit__(self, *exc):
        if self.context is not None:
            return self.context.__exit__(*exc)
