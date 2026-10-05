"""Optimizer and atomic, complete state checkpoints for recurrence training."""
import os
import random
from contextlib import contextmanager
from pathlib import Path

import torch

from .optim import PortMuonAdamW


def trajectory_optimizer(policy, params, lora, heads, *, vocab_size, lr, lora_lr, heads_lr, embedding_ids=()):
    groups = [{'params': list(params.values()), 'lr': lr}]
    if lora:
        groups.append({'params': lora, 'lr': lora_lr})
    if heads:
        groups.append({'params': heads, 'lr': heads_lr})
    if policy == 'adamw':
        return torch.optim.AdamW(groups, weight_decay=0)
    named = [(f'soft.{name}', value) for name, value in params.items()]
    named += [(f'lora_{i}', value) for i, value in enumerate(lora)]
    named += [(f'heads.{i}', value) for i, value in enumerate(heads)]
    optimizer = PortMuonAdamW(named, lr=lr, vocab_size=vocab_size, embedding_ids=embedding_ids)
    rates = {id(q): group['lr'] for group in groups for q in group['params']}
    # Keep each optimizer's partition/schema while preserving the three learning rates.
    for child in [optimizer.muon, optimizer.auxiliary]:
        if child is None:
            continue
        original = dict(child.param_groups[0])
        buckets = {}
        for q in original['params']:
            buckets.setdefault(rates[id(q)], []).append(q)
        first, *rest = buckets.items()
        child.param_groups[0].update(params=first[1], lr=first[0])
        for rate, values in rest:
            child.add_param_group({**original, 'params': values, 'lr': rate})
    optimizer.param_groups = optimizer._groups()
    return optimizer


def atomic_checkpoint(path, state):
    path = Path(path)
    pending = path.with_suffix('.pending')
    with pending.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def validate_resume(state, identity):
    if state.get('schema') != 'natlang.neuralese_recurrence_checkpoint/1':
        raise ValueError('unsupported recurrence checkpoint')
    if state.get('identity') != identity:
        raise ValueError('recurrence inputs or training controls changed')


@contextmanager
def evaluation_state(write_rng, stop_rng, baseline):
    """Periodic probes must not alter the training sampler or stop baseline."""
    python_state, write_state = random.getstate(), write_rng.getstate()
    stop_state, cpu_state = stop_rng.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    saved_baseline = dict(baseline)
    try:
        yield
    finally:
        random.setstate(python_state)
        write_rng.setstate(write_state)
        stop_rng.set_state(stop_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        baseline.clear()
        baseline.update(saved_baseline)


def soft_initialization(path, texts, width):
    """Reuse only parameters whose semantic initialization text is unchanged.

    Unknown new pieces are initialized by the caller. A changed definition under an
    existing parameter name is an error, not silently reinterpreted learned state.
    This is a new training stage; strict optimizer resume remains a separate path.
    """
    import hashlib
    import json
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if 'params' not in state:
        raise ValueError('soft initialization lacks parameters')
    old_texts = state.get('texts')
    if old_texts is None:
        identity = state.get('identity', {})
        piece_path = Path(identity.get('options', {}).get('pieces', ''))
        if not piece_path.is_file():
            raise ValueError('initialization texts unavailable; use a complete export')
        digest = hashlib.sha256(piece_path.read_bytes()).hexdigest()
        if digest != identity.get('files', {}).get(str(piece_path.resolve())):
            raise ValueError('initialization texts changed since checkpoint')
        old_texts = {r['name']: r['text'] for r in map(json.loads, piece_path.open())}
    reused = {}
    for name, rows in state['params'].items():
        if name not in texts:
            continue
        if old_texts.get(name) != texts[name]:
            raise ValueError('soft initialization text changed: ' + name)
        if rows.ndim != 2 or rows.shape[1] != width or not rows.shape[0] or not torch.isfinite(rows).all():
            raise ValueError('invalid soft initialization tensor: ' + name)
        reused[name] = rows
    return reused
