"""Admitted causal-feature curriculum; cached contexts are not new trajectories."""
import hashlib
import json
from pathlib import Path

from ..serve.chat import render_messages
from .trajectories import crisp_messages, handover_notes


def context_ids(engine, records, pieces, limit, tokens):
    texts = {row['name']: row['text'] for row in map(json.loads, Path(pieces).open())}
    values = {'train': set(), 'test': set()}
    for row in map(json.loads, Path(records).open()):
        split = row.get('split')
        if split not in values or row.get('training_admission', {}).get('approved') is not True:
            continue
        messages = crisp_messages(row['messages'], texts, handover_notes(row))
        prompt = render_messages(messages, row.get('tools'), engine._template, engine.specials)
        if prompt.blocks:
            raise ValueError('crisp bootstrap context unexpectedly contains blocks')
        ids = tuple(token for segment in prompt.segments
                    for token in engine._template_tokens(segment, prompt.escape_nonce))[-tokens:]
        if ids:
            values[split].add(ids)
    overlap = values['train'] & values['test']
    values['test'] -= overlap
    key = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    receipt = {'policy': 'admitted crisp contexts; tail windows only for feedback distillation, not a runtime context limit',
               'excluded_shared_held_windows': len(overlap), 'unique_available': {k: len(v) for k, v in values.items()}}
    return {split: sorted(ids, key=key)[:limit] for split, ids in values.items()}, receipt
