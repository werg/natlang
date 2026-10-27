"""Bank views, base dropout and the weight-array contract (plan "Bank views", "Mixing and base dropout").

Each invocation binds a pinned, ordered view of ``M`` slots per scope, selected from the full bank. Default selection
(`gradient-probe-v1`) keeps every component with nonzero current weight, then ranks the rest by the magnitude of
their first-order support-loss effect; `key-similarity-v1` and learned retrieval plug into the same interface. Slots
are ordered by bank position, so a view is a stable positional mapping recorded like any compacted view.

Availability is ``eligible AND retained``: stable scope/model eligibility and the sampled dropout mask are recorded
separately. A dropout mask is sampled independently of the proposal from a recorded seed and probability and stays
fixed on replay; no inverted-dropout rescaling is applied. Unavailable slots must return exactly zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError

VIEW_SCHEMA = 'natlang.lopt.bank-view/1'
MIXTURE_SCHEMA = 'natlang.lopt.lora-mixture/1'
SELECTION_POLICIES = ('gradient-probe-v1', 'key-similarity-v1', 'full-bank-v1', 'learned-retrieval-v1')


@dataclass(frozen=True)
class BankView:
    snapshot: str
    scope: str
    level: str
    slots: tuple          # bank component indices, ascending
    component_ids: tuple
    eligible: tuple
    retained: tuple       # dropout mask: True = retained
    probe: tuple | None   # small integers beside each slot, or None for forward-only backends
    selection: dict       # policy and its inputs (probe values, scores) for the record
    mask: dict            # {'seed', 'probability', 'policy'} of the dropout draw

    @property
    def available(self):
        return tuple(e and r for e, r in zip(self.eligible, self.retained))

    def as_json(self):
        return {'schema': VIEW_SCHEMA, 'snapshot': self.snapshot, 'scope': self.scope, 'level': self.level,
                'slots': list(self.slots), 'componentIds': list(self.component_ids), 'eligible': list(self.eligible),
                'dropoutMask': list(self.retained), 'available': list(self.available),
                'probe': None if self.probe is None else list(self.probe), 'selection': self.selection, 'mask': self.mask}

    @property
    def id(self):
        return 'nview1:' + canonical_json_sha256_hex(self.as_json())


def select_view(bank, scope, level, size, *, current=None, probe_values=None, key_scores=None,
                policy='gradient-probe-v1', probe_integers=None):
    """A pinned view of at most ``size`` slots. ``current`` maps component IDs to their current weights."""
    if policy not in SELECTION_POLICIES:
        raise LearnedOptimizerError(f'unknown bank-view selection policy {policy}')
    K = len(bank.components)
    current = current or {}
    unknown = set(current) - set(bank.ids)
    if unknown:
        raise LearnedOptimizerError(f'current weights name components outside this bank snapshot: {sorted(unknown)[:3]}')
    in_use = [k for k in range(K) if current.get(bank.ids[k], 0) != 0]
    if len(in_use) > size:
        raise LearnedOptimizerError(f'{len(in_use)} components are in use; a view of {size} slots cannot carry them all')
    rest = [k for k in range(K) if k not in in_use]
    if policy == 'full-bank-v1':
        ranked = rest
    elif policy == 'gradient-probe-v1':
        if probe_values is None or len(probe_values) != K:
            raise LearnedOptimizerError('gradient-probe selection needs one probe value per bank component')
        ranked = sorted(rest, key=lambda k: (-abs(probe_values[k]), k))
    else:
        if key_scores is None or len(key_scores) != K:
            raise LearnedOptimizerError(f'{policy} selection needs one score per bank component')
        ranked = sorted(rest, key=lambda k: (-key_scores[k], k))
    slots = tuple(sorted(in_use + ranked[:size - len(in_use)]))
    eligible = tuple(level in bank.components[k].levels for k in slots)
    probe = None
    if probe_integers is not None:
        if len(probe_integers) != K:
            raise LearnedOptimizerError('probe integers must cover the bank')
        probe = tuple(int(probe_integers[k]) for k in slots)
    selection = {'policy': policy, 'size': size, 'inUse': [bank.ids[k] for k in in_use],
                 **({'probe': [float(probe_values[k]).hex() for k in slots]} if probe_values is not None else {}),
                 **({'scores': [float(key_scores[k]).hex() for k in slots]} if key_scores is not None else {})}
    return BankView(bank.snapshot, scope, level, slots, tuple(bank.ids[k] for k in slots), eligible,
                    (True,) * len(slots), probe, selection, {'policy': 'none'})


def with_dropout(view, probability, seed, *, all_disabled='allow-zero-episode'):
    """Sample a retained mask from ``seed``; in-use components may be dropped too (recovery episodes)."""
    if not 0 <= probability < 1 or not math.isfinite(probability):
        raise LearnedOptimizerError('dropout probability must lie in [0, 1)')
    if all_disabled not in ('allow-zero-episode', 'resample'):
        raise LearnedOptimizerError('the all-disabled treatment is allow-zero-episode or resample')
    generator = torch.Generator().manual_seed(seed)
    attempt = 0
    while True:
        draws = torch.rand(len(view.slots), generator=generator)
        retained = tuple(bool(x >= probability) for x in draws)
        if all_disabled == 'allow-zero-episode' or any(r and e for r, e in zip(retained, view.eligible)) or not any(view.eligible):
            break
        attempt += 1
    mask = {'policy': 'bernoulli-v1', 'seed': seed, 'probability': float(probability).hex(), 'allDisabled': all_disabled,
            'resamples': attempt}
    return BankView(view.snapshot, view.scope, view.level, view.slots, view.component_ids, view.eligible, retained,
                    view.probe, view.selection, mask)


def validate_array(view, bank, values):
    """The returned grid integers for one scope: exact length, bank snapshot, finite grid values and zero at every
    unavailable slot. Returns the integers; a violation is a recorded contract failure."""
    if view.snapshot != bank.snapshot:
        raise LearnedOptimizerError('the weight array was bound to another bank snapshot')
    if tuple(bank.ids[k] for k in view.slots) != view.component_ids:
        raise LearnedOptimizerError('the bank view positions no longer map to the same components')
    if not isinstance(values, (list, tuple)) or len(values) != len(view.slots):
        raise LearnedOptimizerError(f'the weight array must have exactly {len(view.slots)} numbers, one per description')
    out = []
    for slot, (value, free) in enumerate(zip(values, view.available)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value):
            raise LearnedOptimizerError(f'weight {slot} must be a finite grid integer')
        value = int(value)
        if abs(value) > bank.number_format.max_int:
            raise LearnedOptimizerError(f'weight {slot} is outside [-{bank.number_format.max_int}, {bank.number_format.max_int}]')
        if not free and value != 0:
            raise LearnedOptimizerError(f'weight {slot} names an unavailable component; it must be 0')
        out.append(value)
    return out


def view_recall(view, full_coefficients, bank):
    """Share of a full-bank fit's coefficient mass (``|c|`` per component ID) that falls inside the view."""
    total = sum(abs(value) for value in full_coefficients.values())
    if total == 0:
        return 1.
    inside = sum(abs(full_coefficients.get(component, 0.)) for component in view.component_ids)
    return inside / total


def mixture_encoding(view, integers, head_receipt, executed, incoming=None):
    """The bundle's `LoRAMixtureEncoding` for one scope; ``incoming`` names the adapter the mixture revises (its
    cores are added exactly before the mixture; absent: the scope starts at zero)."""
    out = {'schema': MIXTURE_SCHEMA, 'bank': view.snapshot, 'view': view.id, 'componentIds': list(view.component_ids),
           'eligible': list(view.eligible), 'dropoutMask': list(view.retained), 'weights': [int(x) for x in integers],
           'head': head_receipt,
           'effective': [[float(x).hex() for x in row] for row in executed.detach().double().tolist()]}
    if incoming is not None:
        out['incoming'] = incoming
    return out


def current_weights(view, integers):
    return {component: value for component, value in zip(view.component_ids, integers) if value != 0}
