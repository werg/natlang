"""Assemble outer-training targets: one complete tuple's optimizer request, writes, arrays and target bundle.

Inputs come from real records: the program's adapter manifest and its training-split rows, the incoming bundle's
soft states and mixtures, the bank views bound for this episode, and a target (an in-bank fit, a projected free tune
or a verified self-proposal). The evidence view is the permitted support view (`evidence.build_view`).
"""
from __future__ import annotations

import json

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError
from .evidence import compact_rows, focus_baseline, key_evidence_rule
from .objectives import field_scales
from .request import request_messages

KIND = {'instructions': 'instruction', 'skill': 'skill'}


def store_fitted_states(engine, states, owner=None):
    """Persist directly fitted state tensors as new native states (target-only; their record names the fit as
    their derivation, never a writer capture)."""
    return {key: engine.store.put_native_state(block, owner).id for key, block in states.items()}


def training_target(engine, bank, *, manifest, rows, contract, view, soft_texts, incoming_states, bank_views,
                    target_integers, target_adapters, target_states, target_rho=None, target_delta=None,
                    placeholders, digests=(), focus=None, extra_slots=0, target_factors=None, tier='in-bank-fit'):
    """``soft_texts``: soft key -> its binding text; ``incoming_states``: soft key -> current state id;
    ``bank_views``: scope -> `views.BankView`; ``target_*``: scope- or key-indexed target tuple."""
    for scope, bank_view in bank_views.items():
        if bank_view.snapshot != bank.snapshot:
            raise LearnedOptimizerError(f'view of {scope} is bound to another bank snapshot')
        if len(target_integers[scope]) != len(bank_view.slots):
            raise LearnedOptimizerError(f'target array of {scope} differs from its view length')
    focus = focus if focus is not None else focus_baseline(view)
    key_evidence = key_evidence_rule(view, focus)
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
    body = {'bank': bank.snapshot, 'tier': tier, 'manifest': manifest, 'rows': rows, 'messages': messages, 'writes': writes,
            'incoming': dict(incoming_states),
            'arrays': arrays, 'placeholders': placeholders,
            'target_bundle': {'adapters': dict(target_adapters), 'states': dict(target_states)},
            'evidence': {'view': view['id'], 'items': [item['id'] for item in view['items']], 'keyEvidence': key_evidence,
                         'focus': focus}}
    target = {**body, 'id': 'nlt1:' + canonical_json_sha256_hex(body), 'state_scales': scales}
    if target_factors:
        target['target_factors'] = target_factors
    return target
