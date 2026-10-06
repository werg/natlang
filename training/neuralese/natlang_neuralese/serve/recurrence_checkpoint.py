"""Restore exact serving weights from a complete recurrent training checkpoint."""
import re

import torch

from . import load_engine
from ..model.capacity import set_write_capacity
from ..train.adapters import inject_lora


def load_recurrence_checkpoint(path, *, device='cpu', dtype=None):
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if state.get('schema') != 'natlang.neuralese_recurrence_checkpoint/1':
        raise ValueError('requires complete recurrence checkpoint')
    options = state['identity']['options']
    engine = load_engine(options.get('base'), heads_checkpoint=options.get('heads'), device=device, dtype=dtype)
    engine.heads.set_content_transport(state.get('port_config', {}).get('content_transport', 'learned-residual'))
    engine.heads.load_state_dict(state['heads'])
    capacity = state.get('port_config', {}).get('max_length', engine.heads.max_length)
    set_write_capacity(engine.heads, capacity)
    engine.max_block = capacity
    engine.backbone.ffn_chunk_tokens = options.get('ffn_chunk_tokens', 0)
    if state.get('control_rows') is not None:
        with torch.no_grad():
            engine.backbone.control_rows.copy_(state['control_rows'].to(engine.backbone.control_rows))
    adapters = state.get('lora', {})
    # PEFT LoRA names its matrices `...lora_A.<adapter>.weight`; Maple's ternary QAT adapters are parametrizations,
    # `...parametrizations.weight.0.lora_A`. Both store A as [rank, in].
    ranks = {int(value.shape[0]) for name, value in adapters.items() if re.search(r'\.lora_A(\.|$)', name)}
    layers = sorted({int(match[1]) for name in adapters
                     for match in [re.search(r'model\.layers\.(\d+)\.', name)] if match})
    if len(ranks) > 1:
        raise ValueError('mixed adapter ranks require explicit deployment mapping')
    if adapters and (not layers or not ranks):
        raise ValueError('unrecognized backbone adapter state')
    if layers:
        rank = next(iter(ranks))
        inject_lora(engine.backbone, layers, rank=rank, alpha=2 * rank)
    parameters = dict(engine.backbone.hf.named_parameters())
    with torch.no_grad():
        for name, value in adapters.items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError('backbone adapter mismatch: ' + name)
            parameters[name].copy_(value.to(parameters[name]))
    if state.get('backbone_trainables'):
        if state.get('maple_qat'):
            from ..train.adapters import install_maple_qat
            install_maple_qat(engine.backbone)
        from ..train.backbone_policy import full_backbone_parameter_names, restore_backbone_trainables
        expected=(full_backbone_parameter_names(engine.backbone)
                  if state.get('backbone_training')=='full' else None)
        restore_backbone_trainables(engine.backbone,state['backbone_trainables'],expected_names=expected)
    if not engine.heads.read_markers:
        engine.heads.configure_frozen_reference()
    # Initialization certificates do not qualify this checkpoint's changed weights.
    engine.recurrence_checkpoint = {'step': state['step'], 'runtime_requalified': False}
    return engine, state
