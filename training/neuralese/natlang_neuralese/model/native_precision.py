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
# Production uses tensor-core arithmetic (owner, 2026-10-28): single-pass BF16/TF32 with FP32 accumulation,
# masters and optimizer state. IEEE and the multi-pass emulations remain selectable.
PRODUCTION_NATIVE_LINEAR_PRECISIONS = ('ieee', 'bf16x6', 'tf32', 'bf16')
NATIVE_ATTENTION_PRECISIONS = ('ieee', 'tf32x3', 'tf32')


def transition_gradient_rtol(previous_mode, candidate_mode):
    """Cross-mode gradient bound for an arithmetic transition.

    A move to an FP32-equivalent mode must reproduce the old gradients (1e-4). A deliberate move to single-pass
    tensor-core arithmetic changes rounding by design: old and new gradients agree in direction (measured cosine
    >= 0.9997, relative L2 <= 0.024 on LFM), so the bound is 0.1. Path consistency inside the new arithmetic
    (cached vs reference) keeps its own unchanged check.
    """
    return 0.1 if candidate_mode in ('tf32', 'bf16') and candidate_mode != previous_mode else 1e-4


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


def base_location_transition(previous, current):
    """Separate a file location from the complete byte-pinned HF identity.

    Native checkpoint loading verifies the actual files before this comparison.
    Missing or inconsistent inventories cannot authorize a location change.
    """
    if previous.get('base') == current.get('base'):
        return None
    from ..common.hashing import canonical_json_sha256_hex
    import re
    inventory = current.get('base_inventory')
    if (not isinstance(inventory, dict) or not inventory
            or previous.get('base_inventory') != inventory
            or any(not re.fullmatch('[0-9a-f]{64}', str(value)) for value in inventory.values())):
        raise ValueError('base relocation requires the exact complete input inventory')
    digest = canonical_json_sha256_hex(inventory)
    if any(config.get('base_inventory_sha256') != digest for config in (previous, current)):
        raise ValueError('base relocation inventory digest is inconsistent')
    return {'from': previous.get('base'), 'to': current.get('base'),
            'base_inventory_sha256': digest, 'input_bytes': 'identical'}


def base_restore_transition(previous, current):
    """Validate a checkpoint-backed restore receipt without changing model identity."""
    before=previous.get('base_restore');after=current.get('base_restore')
    if before==after:return None
    from ..common.hashing import canonical_json_sha256_hex
    inventory=current.get('base_inventory')
    if not isinstance(inventory,dict) or not inventory or previous.get('base_inventory')!=inventory:
        raise ValueError('checkpoint restore requires identical complete base inventory')
    digest=canonical_json_sha256_hex(inventory)
    if previous.get('base_inventory_sha256')!=digest or current.get('base_inventory_sha256')!=digest:
        raise ValueError('checkpoint restore inventory digest differs')
    names={'config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer.model',
           'spiece.model','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json',
           'merges.txt','added_tokens.json'}
    expected={k:v for k,v in inventory.items() if k in names}
    def valid(receipt):
        return (isinstance(receipt,dict) and receipt.get('schema')=='natlang.native-config-only-checkpoint-restore/1'
            and receipt.get('mode')=='checkpoint-backbone-state+verified-config-tokenizer'
            and receipt.get('pinned_base_inventory_sha256')==digest and receipt.get('verified_local_files')==expected
            and 'config.json' in expected and any(k.endswith('.safetensors') for k in inventory))
    if not valid(after) or (before is not None and not valid(before)):
        raise ValueError('checkpoint restore receipt is not bound to the pinned base inputs')
    return {'from':before,'to':after,'base_inventory_sha256':digest,
            'input_bytes':'verified config and tokenizer; backbone restored from full checkpoint'}

def same_source_precision_identity(previous, current):
    """Only declared native arithmetic may change in an ordinary continuation."""
    # `precision` is the process's stock arithmetic policy (configure_native_precision), also declared arithmetic.
    arithmetic = {'native_linear_precision', 'native_attention_precision', 'native_shared_execution', 'precision'}
    if previous.get('base_restore') != current.get('base_restore'):
        try:
            base_restore_transition(previous,current)
        except ValueError:
            return False
        arithmetic.add('base_restore')
    if previous.get('base') != current.get('base'):
        try:
            base_location_transition(previous, current)
        except ValueError:
            return False
        arithmetic.add('base')
    return ({k: v for k, v in previous.items() if k not in arithmetic} ==
            {k: v for k, v in current.items() if k not in arithmetic})


def resolve_native_training_precision(previous, requested=None):
    """Diagnostic checkpoint arithmetic needs an explicit production transition."""
    selected = previous if requested is None else requested
    if selected not in PRODUCTION_NATIVE_LINEAR_PRECISIONS:
        raise ValueError('native production training requires ieee, bf16x6, tf32 or bf16 arithmetic; '
                         'explicitly transition a diagnostic checkpoint before training')
    return selected
