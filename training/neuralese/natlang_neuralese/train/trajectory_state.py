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


def trajectory_optimizer(policy, params, lora, heads, *, vocab_size, lr, lora_lr, heads_lr, embedding_ids=(),
                         lora_names=None):
    groups = [{'params': list(params.values()), 'lr': lr}]
    if lora:
        groups.append({'params': lora, 'lr': lora_lr})
    if heads:
        groups.append({'params': heads, 'lr': heads_lr})
    if policy == 'adamw':
        return torch.optim.AdamW(groups, weight_decay=0)
    named = [(f'soft.{name}', value) for name, value in params.items()]
    # Real backbone names (Maple QAT) let Muon take the dense latents and AdamW the scales, routers and norms;
    # positional names keep every adapter tensor on AdamW, as LoRA always was.
    named += ([(f'backbone.{name}', value) for name, value in zip(lora_names, lora)] if lora_names is not None
              else [(f'lora_{i}', value) for i, value in enumerate(lora)])
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


def drop_file_cache(target):
    """Best-effort posix_fadvise(DONTNEED) for a path or open descriptor: drops its clean page cache; pages
    still mapped by live tensors stay resident."""
    try:
        if isinstance(target, int):
            os.posix_fadvise(target, 0, 0, os.POSIX_FADV_DONTNEED)
        else:
            with open(target, 'rb') as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError):
        pass


def atomic_checkpoint(path, state):
    path = Path(path)
    pending = path.with_suffix('.pending')
    try:
        with pending.open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
            # On the GB10 page cache comes out of the same memory CUDA allocates from: a multi-GB checkpoint's
            # clean cache would otherwise sit in MemFree until something reclaims it.
            drop_file_cache(stream.fileno())
        pending.replace(path)
    except BaseException:
        # A failed write (often ENOSPC) leaves an incomplete owned temp file.
        # Remove it so a post-commit recovery save can use the reserved space.
        pending.unlink(missing_ok=True)
        raise


def validate_resume(state, identity):
    if state.get('schema') != 'natlang.neuralese_recurrence_checkpoint/1':
        raise ValueError('unsupported recurrence checkpoint')
    if state.get('identity') != identity:
        raise ValueError('recurrence inputs or training controls changed')


def resumed_initial_rows(state, names, width):
    """Restore original row identities without re-encoding discarded inputs."""
    if state is None:
        return {}
    if set(state.get('init', {})) != set(names) or set(state.get('params', {})) != set(names):
        raise ValueError('recurrence soft-parameter names changed')
    rows = state['init']
    for name, value in rows.items():
        if value.ndim != 2 or not value.shape[0] or value.shape[1] != width or value.shape != state['params'][name].shape:
            raise ValueError('invalid recurrence initialization shape: ' + name)
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('invalid recurrence initialization values: ' + name)
    return rows


def iteration_rng_state(write_rng, stop_rng, baseline, *, cuda=False):
    """A pre-update boundary for replaying an incomplete accumulation step."""
    return {'python': random.getstate(), 'write': write_rng.getstate(),
            'stop': stop_rng.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if cuda else [], 'baseline': dict(baseline)}


def restore_iteration_rng(state, write_rng, stop_rng, baseline):
    random.setstate(state['python'])
    write_rng.setstate(state['write'])
    stop_rng.set_state(state['stop'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    baseline.clear()
    baseline.update(state['baseline'])


def validate_continuation(state, identity, *, allowed_changes=()):
    """Explicit new code stage, retaining full optimizer/RNG and fixed inputs.

    The caller records the source checkpoint hash separately. This does not relax
    changed-in-place resume or permit changing the dataset/training controls.
    """
    if state.get('schema') != 'natlang.neuralese_recurrence_checkpoint/1':
        raise ValueError('unsupported recurrence continuation checkpoint')
    old = state.get('identity', {})
    allowed = set(allowed_changes)
    if not allowed <= {'tokens_per_vector', 'writer_text_weight', 'write_depth', 'write_curriculum', 'max_writes', 'max_write_vectors', 'content_transport', 'content_residual_initialization', 'writer_length_policy', 'writer_supervision', 'stop_supervision', 'steps', 'sketch_gradient', 'member_weight', 'member_tokens', 'member_eval'}:
        raise ValueError('unsupported continuation curriculum changes')
    previous, current = dict(old.get('options', {})), dict(identity.get('options', {}))
    previous.setdefault('stop_supervision', 'generated-length')
    current.setdefault('stop_supervision', 'generated-length')
    previous.setdefault('writer_supervision', 'full-reply')
    current.setdefault('writer_supervision', 'full-reply')
    previous.setdefault('writer_length_policy', 'source-text')
    current.setdefault('writer_length_policy', 'source-text')
    previous.setdefault('content_residual_initialization', 'preserve')
    current.setdefault('content_residual_initialization', 'preserve')
    previous.setdefault('content_transport', 'learned-residual')
    current.setdefault('content_transport', 'learned-residual')
    previous.setdefault('writer_text_weight', 0.)
    current.setdefault('writer_text_weight', 0.)
    for key, default in (('member_weight', 0.), ('member_tokens', 2048), ('member_eval', 4)):
        # The nested-family term (2026-10-08) postdates earlier checkpoints, which trained no member.
        previous.setdefault(key, default)
        current.setdefault(key, default)
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


def compatible_best_evaluation(best, signature):
    """A loss-ranked candidate is comparable only within its declared probe regime."""
    return best if best is not None and best.get('selection_signature') == signature else None


@torch.no_grad()
def initialize_content_residual(heads, optimizer):
    """Activate a previously bypassed residual at zero; reset only its optimizer slots."""
    children = [optimizer.muon, optimizer.auxiliary] if isinstance(optimizer, PortMuonAdamW) else [optimizer]
    reset = []
    for name, parameter in heads.content.proj.named_parameters():
        parameter.zero_()
        for child in children:
            if child is not None:
                child.state.pop(parameter, None)
        reset.append('content.proj.' + name)
    return reset


def weights_digest(backbone,heads):
    import hashlib,json
    digest=hashlib.sha256()
    for section,values in [('backbone',backbone),('heads',heads)]:
        for name,value in sorted(values.items()):
            tensor=value.detach().cpu().contiguous()
            digest.update(json.dumps([section,name,list(tensor.shape),str(tensor.dtype)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
