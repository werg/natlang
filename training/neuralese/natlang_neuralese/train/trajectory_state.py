"""Optimizer and atomic, complete state checkpoints for recurrence training."""
import os
import random
from contextlib import contextmanager
from pathlib import Path

import torch

from .optim import PortMuonAdamW


def gradient_norm(parameters):
    """Accumulate the norm in FP64 so large finite BF16/FP32 adjoints do not overflow."""
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    if not gradients:
        return torch.zeros((), dtype=torch.float64)
    norms = torch.stack([torch.linalg.vector_norm(g, dtype=torch.float64) for g in gradients])
    return torch.linalg.vector_norm(norms)


@torch.no_grad()
def clip_finite_gradients(parameters, maximum=1.):
    parameters = list(parameters)
    norm = gradient_norm(parameters)
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite recurrence gradients before optimizer update")
    coefficient = (maximum / (norm + 1e-6)).clamp(max=1.)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient.to(device=parameter.grad.device))
    return norm


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


def validate_continuation(state, identity, *, allowed_changes=()):
    """Explicit new code stage, retaining full optimizer/RNG and fixed inputs.

    The caller records the source checkpoint hash separately. This does not relax
    changed-in-place resume or permit changing the dataset/training controls.
    """
    if state.get('schema') != 'natlang.neuralese_recurrence_checkpoint/1':
        raise ValueError('unsupported recurrence continuation checkpoint')
    old = state.get('identity', {})
    allowed = set(allowed_changes)
    if not allowed <= {'tokens_per_vector', 'writer_text_weight', 'write_depth', 'write_curriculum', 'max_writes'}:
        raise ValueError('unsupported continuation curriculum changes')
    previous, current = dict(old.get('options', {})), dict(identity.get('options', {}))
    previous.setdefault('writer_text_weight', 0.)
    current.setdefault('writer_text_weight', 0.)
    changed = {key for key in previous.keys() | current.keys() if previous.get(key) != current.get(key)}
    if not changed <= allowed or old.get('files') != identity.get('files'):
        raise ValueError('recurrence continuation requires identical inputs and training controls except declared curriculum changes')


def paired_probe_complete(report):
    """All eligible paired readers scored; non-reader held turns are irrelevant."""
    return (report.get('expected_n', 0) >= 2 and report.get('n') == report.get('expected_n')
            and not report.get('write_errors') and not report.get('reader_errors')
            and not report.get('missing_donors'))


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


def soft_initialization(path, texts, width, *, profile='legacy-rms-v1'):
    """Reuse only parameters whose semantic initialization text is unchanged.

    Unknown new pieces are initialized by the caller. A changed definition under an
    existing parameter name is an error, not silently reinterpreted learned state.
    This is a new training stage; strict optimizer resume remains a separate path.
    """
    import hashlib
    import json
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    saved_profile = state.get('port_profile') or state.get('port_config', {}).get('profile', 'legacy-rms-v1')
    if saved_profile != profile:
        raise ValueError('soft initialization belongs to a different port profile; re-encode source texts under the new channel')
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
