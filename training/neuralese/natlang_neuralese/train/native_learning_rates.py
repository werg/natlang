"""Declared native learned-row rates through the existing named optimizer."""
from __future__ import annotations
import math
from .optim_restore import set_learning_rates,_children

LEARNED_ROWS=frozenset('interface.'+name for name in
    ('input_rows','output_rows','extra_input_rows','extra_output_rows'))


def validate_learned_row_rate(rate):
    if rate is not None and (type(rate) not in (int,float) or not math.isfinite(rate) or rate<=0):
        raise ValueError('learned_row_learning_rate must be finite and positive, or omitted to retain saved rates')
    return rate


def apply_native_learning_rates(optimizer,*,backbone_lr,learned_row_lr=None):
    named=getattr(optimizer,'_native_named_parameters',None)
    if named is None:
        raise ValueError('native learning rates require the actual named native optimizer inventory')
    validate_learned_row_rate(learned_row_lr)
    for rate in (backbone_lr,):
        if rate is not None and (type(rate) not in (int,float) or not math.isfinite(rate) or rate<=0):
            raise ValueError('native learning rates must be finite and positive')
    names={id(parameter):name for name,parameter in named}
    current={names[id(parameter)]:group['lr'] for _,child in _children(optimizer)
             for group in child.param_groups for parameter in group['params']}
    rates={name:(learned_row_lr if learned_row_lr is not None else current[name])
           if name in LEARNED_ROWS else backbone_lr for name in current}
    changes=set_learning_rates(optimizer,named,rates)
    optimizer.native_learning_rates={'backbone_lr':backbone_lr,'declared_learned_row_lr':learned_row_lr,
        'effective_learned_rows':{name:rate for name,rate in rates.items() if name in LEARNED_ROWS}}
    return changes
