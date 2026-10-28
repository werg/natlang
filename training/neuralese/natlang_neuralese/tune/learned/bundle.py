"""The complete adaptation bundle and joint proposal (plan "One complete adaptation bundle").

A bundle maps the program's instruction bindings and skill bindings to native states, and its scope nodes to
composed native adapters together with the mixture encoding that produced them. The two adapter forms must agree by
exact composition; they are not independently editable. Absolute artifacts are authoritative; edits record what
changed from the parent, and unchanged entries are retained by reference.
"""
from __future__ import annotations

import torch

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError

BUNDLE_SCHEMA = 'natlang.joint-adaptation/1'
PROPOSAL_SCHEMA = 'natlang.joint-proposal/1'
IDENTITY_KEYS = ('architecture', 'checkpoint_sha256', 'embedding_weight_sha256', 'tokenizer_fingerprint')


def representation_ref(bank):
    return {'bank': bank.snapshot, 'basis': bank.basis_artifact, 'rank': bank.basis.rank,
            'mixtureFormat': bank.mixture_format.id, 'groups': list(bank.group_labels),
            'keys': [component.key for component in bank.components]}


def make_bundle(*, program, base, bank, parent, instructions, skills, adapters, mixtures, family_gates=None,
                provenance=None):
    if set(adapters) != set(mixtures):
        raise LearnedOptimizerError('every scope adapter needs the mixture encoding that composed it')
    for scope, mixture in mixtures.items():
        if mixture['bank'] != bank.snapshot:
            raise LearnedOptimizerError(f'scope {scope} mixture names another bank snapshot')
    body = {'schema': BUNDLE_SCHEMA, 'program': program, 'base': {key: base[key] for key in IDENTITY_KEYS},
            'representation': representation_ref(bank), 'parent': parent,
            'instructions': dict(sorted(instructions.items())), 'skills': dict(sorted(skills.items())),
            'adapters': dict(sorted(adapters.items())), 'loraMixtures': dict(sorted(mixtures.items())),
            'familyGates': dict(sorted((family_gates or {}).items())), 'provenance': provenance or {}}
    return {**body, 'id': 'nja1:' + canonical_json_sha256_hex(body)}


def verify_bundle(engine, bundle, bank):
    """Exact checks before installation: identity, the bank snapshot, every state and adapter present, and every
    adapter equal to the exact composition of its recorded executed coefficients."""
    body = {key: value for key, value in bundle.items() if key != 'id'}
    if bundle.get('schema') != BUNDLE_SCHEMA or bundle.get('id') != 'nja1:' + canonical_json_sha256_hex(body):
        raise LearnedOptimizerError('bundle content differs from its identity')
    if bundle['base'] != {key: engine.native_identity[key] for key in IDENTITY_KEYS}:
        raise LearnedOptimizerError('bundle base differs from the exact live model')
    if bundle['representation'] != representation_ref(bank):
        raise LearnedOptimizerError('bundle was composed against another representation snapshot')
    for kind in ('instructions', 'skills'):
        for binding, state in bundle[kind].items():
            if engine.store.get_native_state(state) is None:
                raise LearnedOptimizerError(f'{kind} state of {binding} is not restored: {state}')
    for scope, adapter in bundle['adapters'].items():
        mixture = bundle['loraMixtures'][scope]
        slots = [bank.index(component) for component in mixture['componentIds']]
        executed = torch.tensor([[float.fromhex(x) for x in row] for row in mixture['effective']], dtype=torch.float32)
        offset = bank.adapter_cores(engine, mixture['incoming']) if mixture.get('incoming') else None
        expected = bank.adapter_tensors(bank.composed_cores(slots, executed, offset=offset)) if slots or offset else None
        record = engine.store.get_native_tensor_artifact(adapter)
        if record is None:
            raise LearnedOptimizerError(f'adapter of scope {scope} is not restored: {adapter}')
        stored = record.artifact.tensors
        if expected is not None and any(not torch.allclose(stored[name], expected[name].cpu(), rtol=1e-6, atol=1e-7)
                                        for name in expected):
            raise LearnedOptimizerError(f'adapter of scope {scope} differs from the composition of its mixture')
        free = [e and r for e, r in zip(mixture['eligible'], mixture['dropoutMask'])]
        for slot, (probability, available) in enumerate(zip(mixture['distribution'][1:], free)):
            if not available and probability != 0:
                raise LearnedOptimizerError(f'scope {scope} slot {slot} is unavailable but has probability')
            if probability == 0 and any(float.fromhex(x) != 0 for x in mixture['effective'][slot]):
                raise LearnedOptimizerError(f'scope {scope} slot {slot} executes a coefficient its zero probability denies')
    return bundle


def edits(parent, child):
    """Changed entries from ``parent`` to ``child`` (absolute targets; unchanged entries retained by reference)."""
    out = []
    for kind, key in (('instruction', 'instructions'), ('skill', 'skills'), ('adapter', 'adapters')):
        before, after = (parent or {}).get(key, {}), child[key]
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                out.append({'kind': kind, 'target': name, 'before': before.get(name), 'after': after.get(name)})
    return out


def make_proposal(parent, child, behavior):
    body = {'schema': PROPOSAL_SCHEMA, 'parent': None if parent is None else parent['id'], 'next': child['id'],
            'edits': edits(parent, child), 'behavior': behavior}
    return {**body, 'id': 'njp1:' + canonical_json_sha256_hex(body)}


def binding_entries(bundle, manifest, binding_id):
    """Adapter entries of one binding's lexical stack under the bundle: the ancestry root to leaf, each scope's
    composed adapter at its manifest scale (family gates keep their existing composition)."""
    binding = next((item for item in manifest['bindings'] if item['id'] == binding_id), None)
    if binding is None:
        raise LearnedOptimizerError(f'unknown binding {binding_id}')
    scales = manifest.get('scales', {})
    return [{'id': bundle['adapters'][node], 'scale': float(scales.get(node, 1))}
            for node in binding['ancestry'] if node in bundle['adapters'] and scales.get(node, 1) != 0]


def soft_states_for(bundle, binding_id):
    """``{soft key: state id}`` of one binding (keys as `tune.soft.soft_key`)."""
    from ..soft import soft_key
    out = {}
    if binding_id in bundle['instructions']:
        out[soft_key(binding_id, 'instructions', None)] = bundle['instructions'][binding_id]
    for key, state in bundle['skills'].items():
        owner, _, target = key.partition('#')
        if owner == binding_id:
            out[soft_key(binding_id, 'skill', target)] = state
    return out


def skill_key(binding, target):
    return f'{binding}#{target}'


def bundle_slot(soft):
    """``(field, key)`` of a `tune.soft.soft_key` in a bundle: instructions by binding, skills by ``binding#target``."""
    import json
    binding, kind, target = json.loads(soft)
    if kind == 'instructions':
        return 'instructions', binding
    if kind == 'skill':
        return 'skills', skill_key(binding, target)
    raise LearnedOptimizerError(f'unknown soft-state kind {kind}')
