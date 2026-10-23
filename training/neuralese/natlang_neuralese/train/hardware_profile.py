"""Declared hardware profiles: operational knobs selected by the GPU a stage runs on (plans/COREWEAVE_HARDWARE_OPTIMIZATION.md).

Training math and recipes are one implementation across lines and machines (AGENTS.md: one implementation per
concept). What may differ by hardware is only how a stage computes, never what it optimizes: the profile file
(``training/neuralese/hardware-profiles.json``, schema ``natlang.neuralese-hardware-profiles/1``) maps a device
(compute capability, minimum memory, integrated or discrete) to

- ``stage_parameters``: a subset of :data:`PROFILE_STAGE_PARAMETERS`, all members of the recipe's
  ``OPERATIONAL_STAGE_PARAMETERS`` (a lineage may switch them in place, so a resume on other hardware keeps its
  identity), applied over the recipe's values for the stage kinds the profile names;
- ``environment``: a subset of :data:`PROFILE_ENVIRONMENT`, execution switches that restore an equivalent path
  (allocator, compile caches, compile of numerics-preserving glue);
- ``sandbox``: the CPU and memory request a CoreWeave job of each kind should make on that device (documentation for
  ``scripts/coreweave/jobs``; nothing reads it at training time).

Selection is exact and crisp: the first profile whose ``match`` holds for the detected device, or the one named by
``NATLANG_HARDWARE_PROFILE`` (``none`` disables). A device no profile matches runs the recipe unchanged. The runner
records the selection (profile name, device facts, applied keys with the recipe's previous values) next to the child
command, so a stage report says which knobs the hardware chose.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCHEMA = 'natlang.neuralese-hardware-profiles/1'
# Execution choices only: each is in recipe.OPERATIONAL_STAGE_PARAMETERS and in the trainers' resume-operational sets.
PROFILE_STAGE_PARAMETERS = frozenset({'checkpoint_layers'})
# Environment switches a profile may set: allocator and cache placement, and numerics-preserving compilation.
PROFILE_ENVIRONMENT = frozenset({'PYTORCH_CUDA_ALLOC_CONF', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
                                 'NATLANG_MAPLE_COMPILE', 'CUDA_DEVICE_MAX_CONNECTIONS'})
_MATCH_KEYS = frozenset({'capability', 'min_memory_gb', 'integrated', 'name_contains'})
DEFAULT_PATH = Path(__file__).resolve().parents[2] / 'hardware-profiles.json'


class ProfileError(ValueError):
    pass


def validate(document: dict) -> dict:
    """Return the profiles document, or raise ProfileError naming the first problem."""
    if not isinstance(document, dict) or document.get('schema') != SCHEMA:
        raise ProfileError(f'a hardware profile file is a JSON object with "schema": "{SCHEMA}"')
    profiles = document.get('profiles')
    if not isinstance(profiles, list) or not profiles:
        raise ProfileError('profiles: a nonempty list, matched in order')
    from .recipe import OPERATIONAL_STAGE_PARAMETERS
    names = set()
    for profile in profiles:
        name = profile.get('name') if isinstance(profile, dict) else None
        if not isinstance(name, str) or not name or name in names or name == 'none':
            raise ProfileError('each profile has a unique "name" (not "none")')
        names.add(name)
        match = profile.get('match')
        if not isinstance(match, dict) or not match or set(match) - _MATCH_KEYS:
            raise ProfileError(f'{name}: match uses only {sorted(_MATCH_KEYS)}')
        if 'capability' in match and not (isinstance(match['capability'], list) and len(match['capability']) == 2
                                          and all(isinstance(v, int) for v in match['capability'])):
            raise ProfileError(f'{name}: match.capability is [major, minor]')
        parameters = profile.get('stage_parameters', {})
        if not isinstance(parameters, dict):
            raise ProfileError(f'{name}: stage_parameters maps a stage kind to its operational parameters')
        for kind, values in parameters.items():
            if not isinstance(values, dict):
                raise ProfileError(f'{name}: stage_parameters.{kind} is an object')
            outside = set(values) - PROFILE_STAGE_PARAMETERS
            if outside or set(values) - OPERATIONAL_STAGE_PARAMETERS:
                raise ProfileError(f'{name}: stage_parameters.{kind} may set only {sorted(PROFILE_STAGE_PARAMETERS)} '
                                   f'(operational execution choices); got {sorted(outside)}. Training math stays in '
                                   'the recipe')
        environment = profile.get('environment', {})
        if not isinstance(environment, dict) or set(environment) - PROFILE_ENVIRONMENT or not all(
                isinstance(v, str) for v in environment.values()):
            raise ProfileError(f'{name}: environment may set only {sorted(PROFILE_ENVIRONMENT)}, as strings')
    return document


def load(path: str | Path | None = None) -> dict:
    return validate(json.loads(Path(path or DEFAULT_PATH).read_text()))


def detect(device: str = 'cuda') -> dict | None:
    """Facts of the GPU a stage will run on: name, [major, minor], memory in GB, integrated (unified memory).

    torch when it is importable with CUDA (the trainer's own view), else ``nvidia-smi``; None without a GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            index = torch.device(device).index or 0 if str(device).startswith('cuda') else 0
            props = torch.cuda.get_device_properties(index)
            return {'name': props.name, 'capability': [props.major, props.minor],
                    'memory_gb': round(props.total_memory / 2 ** 30, 1), 'integrated': bool(props.is_integrated)}
    except Exception:
        pass
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
    except Exception:
        return None
    if not out:
        return None
    name, cap, memory = [part.strip() for part in out[0].split(',')[:3]]
    major, minor = (int(v) for v in cap.split('.'))
    try:
        memory_gb = round(float(memory) / 1024, 1)
    except ValueError:  # unified-memory parts report "[N/A]"
        memory_gb = None
    return {'name': name, 'capability': [major, minor], 'memory_gb': memory_gb, 'integrated': memory_gb is None}


def matches(match: dict, device: dict) -> bool:
    if 'capability' in match and list(device.get('capability') or []) != list(match['capability']):
        return False
    if 'integrated' in match and bool(device.get('integrated')) != bool(match['integrated']):
        return False
    if 'min_memory_gb' in match and (device.get('memory_gb') or 0) < match['min_memory_gb']:
        return False
    if 'name_contains' in match and str(match['name_contains']).lower() not in str(device.get('name', '')).lower():
        return False
    return True


def select(document: dict, device: dict | None, override: str | None = None) -> dict | None:
    """The profile for ``device``: ``override`` (a profile name, or "none") wins, else the first match."""
    if override:
        if override == 'none':
            return None
        for profile in document['profiles']:
            if profile['name'] == override:
                return profile
        raise ProfileError(f'NATLANG_HARDWARE_PROFILE={override!r} names no profile; known: '
                           + ', '.join(p['name'] for p in document['profiles']))
    if device is None:
        return None
    return next((p for p in document['profiles'] if matches(p['match'], device)), None)


def apply(profile: dict | None, kind: str, parameters: dict) -> tuple[dict, dict, dict | None]:
    """(parameters for the child command, environment additions, record) for one stage of ``kind``.

    The record names the profile and each applied key with the recipe's value it replaced (absent: ``null``)."""
    if profile is None:
        return dict(parameters), {}, None
    updates = dict(profile.get('stage_parameters', {}).get(kind, {}))
    merged = dict(parameters)
    applied = {}
    for key, value in updates.items():
        if merged.get(key, None) != value:
            applied[key] = {'recipe': parameters.get(key), 'profile': value}
        merged[key] = value
    environment = dict(profile.get('environment', {}))
    record = {'profile': profile['name'], 'applied': applied, 'environment': environment,
              'status': profile.get('status')}
    return merged, environment, record


def for_stage(kind: str, parameters: dict, device: str = 'cuda', path: str | Path | None = None,
              detected: dict | None = None) -> tuple[dict, dict, dict | None]:
    """Detect, select and apply in one call (the recipe runner's entry). A missing profile file applies nothing."""
    override = os.environ.get('NATLANG_HARDWARE_PROFILE')
    source = Path(path or DEFAULT_PATH)
    if override == 'none' or not source.is_file():
        return dict(parameters), {}, None
    document = load(source)
    facts = detected if detected is not None else detect(device)
    merged, environment, record = apply(select(document, facts, override), kind, parameters)
    if record is not None:
        record['device'] = facts
    return merged, environment, record
