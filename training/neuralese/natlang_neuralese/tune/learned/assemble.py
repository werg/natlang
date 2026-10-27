"""The one joint target record (``natlang.joint-target/1``): a complete tuple's optimizer request, writes, arrays and
target bundle. The same record without labels is the request context of `apply`, `collect` and `propose`.

Inputs come from real records (`program.ProgramContext`): the program's adapter manifest and its training-split rows,
the incoming bundle's soft states and adapters, the bank views bound for this episode, and a target of one tier
(`targets.TIERS`: a free tune of the joint tuner, an in-bank fit, or a verified self-proposal). The evidence view is
the permitted support view (`evidence.build_view`). A target is never recombined from parts of different runs.
"""
from __future__ import annotations

import json

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError
from .evidence import compact_rows, focus_baseline, key_evidence_rule
from .objectives import field_scales
from .request import request_messages

KIND = {'instructions': 'instruction', 'skill': 'skill'}
TARGET_SCHEMA = 'natlang.joint-target/1'
TIERS = ('free-tune', 'in-bank-fit', 'self-proposal')


def store_fitted_states(engine, states, owner=None):
    """Persist directly fitted state tensors as new native states (target-only; their record names the fit as
    their derivation, never a writer capture)."""
    return {key: engine.store.put_native_state(block, owner).id for key, block in states.items()}


def training_target(engine, bank, *, manifest, rows, contract, view, soft_texts, incoming_states, bank_views,
                    target_integers, target_adapters, target_states, target_rho=None, target_delta=None,
                    placeholders, digests=(), focus=None, extra_slots=0, target_factors=None, tier='in-bank-fit',
                    incoming_adapters=None, program=None, lineage=None, key_evidence=None):
    """``soft_texts``: soft key -> its binding text; ``incoming_states``: soft key -> current state id;
    ``incoming_adapters``: scope -> the adapter the mixture revises (slicing the bank basis; absent: zero);
    ``bank_views``: scope -> `views.BankView`; ``target_*``: scope- or key-indexed target tuple, with
    ``target_factors``: scope -> the adapter whose exact effective update the decoded weight loss approaches;
    ``tier``: one of `TIERS`, or None for a request context without labels."""
    if tier is not None and tier not in TIERS:
        raise LearnedOptimizerError(f'unknown target tier {tier}')
    for scope, bank_view in bank_views.items():
        if bank_view.snapshot != bank.snapshot:
            raise LearnedOptimizerError(f'view of {scope} is bound to another bank snapshot')
        if len(target_integers[scope]) != len(bank_view.slots):
            raise LearnedOptimizerError(f'target array of {scope} differs from its view length')
    focus = focus if focus is not None else focus_baseline(view)
    if key_evidence is None:
        key_evidence = key_evidence_rule(view, focus)
    known = {item['id'] for item in view['items']}
    unknown = sorted({item for items in key_evidence.values() for item in items} - known)
    if unknown:
        raise LearnedOptimizerError(f'key evidence {unknown[:3]} is not in the evidence view; name ids of view.items')
    full = {item for items in key_evidence.values() for item in items}
    bindings = sorted({json.loads(key)[0] for key in soft_texts} | {b['id'] for b in manifest['bindings']})
    lines = compact_rows(view, full, focus or bindings)
    bundle_lines = [f'{scope}: {list(bank_views[scope].component_ids).__len__()} components' for scope in sorted(bank_views)]
    current = [(f'{KIND[json.loads(key)[1]]} {json.loads(key)[0]}', incoming_states[key]) for key in sorted(incoming_states)]
    messages = request_messages(contract=contract, bundle_lines=bundle_lines, evidence_lines=lines,
                                current_states=current, digests=digests, focus=focus)
    writes = [{'key': key, 'kind': KIND[json.loads(key)[1]], 'text': soft_texts[key], 'extra_slots': extra_slots,
               'target_state': target_states.get(key)} for key in sorted(soft_texts)]
    arrays = []
    for scope in sorted(bank_views):
        item = {'view': bank_views[scope].as_json(), 'integers': [int(x) for x in target_integers[scope]]}
        if target_rho is not None and scope in target_rho:
            item['rho'] = [float(x) for x in target_rho[scope]]
            item['delta'] = [[float(x) for x in row] for row in target_delta[scope]]
        arrays.append(item)
    scales = field_scales([engine.store.get_native_state(state).block for state in target_states.values()]) if target_states else {}
    scales = {f'{index}.{name}': value for (index, name), value in scales.items()}  # JSON-serializable
    program = program or {'id': manifest['program']['id'], 'buildHash': manifest['program'].get('buildHash')}
    body = {'schema': TARGET_SCHEMA, 'bank': bank.snapshot, 'tier': tier, 'program': program, 'manifest': manifest,
            'rows': rows, 'messages': messages, 'writes': writes,
            'incoming': dict(incoming_states), 'incoming_adapters': dict(sorted((incoming_adapters or {}).items())),
            'arrays': arrays, 'placeholders': placeholders,
            'target_bundle': {'adapters': dict(target_adapters), 'states': dict(target_states)},
            'target_factors': dict(sorted((target_factors or {}).items())),
            'evidence': {'view': view['id'], 'items': [item['id'] for item in view['items']], 'keyEvidence': key_evidence,
                         'focus': focus},
            'lineage': lineage or {}}
    return {**body, 'id': 'njt1:' + canonical_json_sha256_hex(body), 'state_scales': scales}


def require_same_evidence(target, optimizer_input_items):
    """A target may not use support evidence absent from the recorded optimizer input."""
    if list(target['evidence']['items']) != list(optimizer_input_items):
        raise LearnedOptimizerError('the target used support evidence the optimizer input does not hold; label it as a '
                                    'different conditional problem instead')
