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
    elif policy == 'latent':
        named=latent_backbone_parameters(backbone)
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


def latent_backbone_parameters(backbone):
    """The ``latent`` policy (owner 2026-10-10: QAT woven into training): every native layer weight trains as a BF16
    latent, quantized only by the recipe's precision points (train/quantization.py). On a dense backbone (LFM, Qwen)
    it selects what ``full`` selects. On a BF16 Maple-family backbone (Mellum loaded with ``precision: bf16``) the
    attention projections, the dense experts (made trainable parameters), the routers and the layer norms; never a
    ternary-deployed checkpoint (its weights are codes, not latents)."""
    if not getattr(backbone, 'ternary', False):
        selected=full_backbone_parameter_names(backbone)
        named=[(name,parameter) for name,parameter in backbone.hf.named_parameters() if name in selected]
    else:
        from ..maple.model import DenseExperts
        if getattr(backbone.hf,'natlang_precision',None)!='bf16':
            raise ValueError('the latent policy needs the BF16 backbone (heads identity precision bf16), '
                             'not ternarized attention')
        for layer in backbone.hf.model.layers:
            experts=layer.mlp.experts
            if not isinstance(experts,DenseExperts):
                raise ValueError('the latent policy needs BF16 (dense) experts')
            experts.make_latent(quantize=False)
        prefixes=('.self_attn.q_proj.weight','.self_attn.k_proj.weight','.self_attn.v_proj.weight',
                  '.self_attn.o_proj.weight','.mlp.experts.gate_up','.mlp.experts.down','.mlp.gate.weight',
                  'input_layernorm.weight','post_attention_layernorm.weight','q_norm.weight','k_norm.weight')
        named=[(name,parameter) for name,parameter in backbone.hf.named_parameters()
               if name.startswith('model.layers.') and name.endswith(prefixes)]
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
    if getattr(backbone, 'ternary', False):
        raise ValueError('ternary backbone has no native full-weight policy')
    return {name for name, _ in backbone.hf.named_parameters()
            if name.startswith('model.layers.')}


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
