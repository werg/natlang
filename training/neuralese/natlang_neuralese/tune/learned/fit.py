"""In-bank fits and hindsight focus labels (plan "Comparable and plentiful targets" tier 2, "How analysis learns").

With a pinned bank snapshot, fit each scope's coefficients and the program's soft states directly on the support
evidence by gradient through the exact composition. Two phases:

1. per-slot mean strengths ``m`` with the head's bounded centered group modulation, ``c_kg = m_k (1 + delta_kg)``,
   and the soft states, Adam on the support loss (only sign-consistent, executable coefficients are explored);
2. integers frozen at ``quantize(m)``, then refinement and modulation trained through exactly the head's bounded
   functions (`head.executed`) from free per-slot ``u, v``.

The phase-2 result is exactly representable by the token grid plus the coefficient head, with no projection
residual; its ``(n, rho, delta)`` are the mixture-token and head-regression labels. Refit after each bank
publication instead of carrying stale labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import LearnedOptimizerError
from .head import executed, modulation, refinement, shares_from_norms
from .numbers import quantize

FIT_SCHEMA = 'natlang.lopt.in-bank-fit/1'


class SupportSet:
    """Training-split support rows of one program (`tune` SFT row format) and the exact support loss."""

    def __init__(self, engine, manifest, rows):
        from ...data.native_graph_schema import parse_graph
        self.engine, self.manifest = engine, manifest
        self.bindings = {binding['id']: binding for binding in manifest['bindings']}
        fingerprint = engine.model_config['tokenizer_fingerprint']
        self.rows = []
        for row in rows:
            if row['graph'].get('split') != 'train' or row['binding'] not in self.bindings:
                raise LearnedOptimizerError('support rows are training-split rows of known bindings')
            self.rows.append({'binding': row['binding'], 'example': parse_graph(row['graph'], tokenizer_fingerprint=fingerprint),
                              'weight': float(row.get('evidence_weight', 1.)), 'soft': row.get('soft') or {},
                              'occurrence': row.get('occurrence', row['graph']['id'])})
        if not self.rows:
            raise LearnedOptimizerError('an in-bank fit needs support rows')
        counts = {}
        for row in self.rows:
            counts[row['binding']] = counts.get(row['binding'], 0) + 1
        # Mean over bindings of the mean evidence-weighted row loss (the tuner's program objective).
        for row in self.rows:
            row['coefficient'] = row['weight'] / (len(counts) * counts[row['binding']])

    def entries(self, binding_id, adapters):
        scales = self.manifest.get('scales', {})
        return [{'id': adapters[node], 'scale': float(scales.get(node, 1))}
                for node in self.bindings[binding_id]['ancestry'] if node in adapters and scales.get(node, 1) != 0]

    def accumulate(self, adapters, make_leaves, soft_blocks=None, *, backward=True):
        """Run every row under its binding's stack; ``make_leaves()`` returns fresh live ``{adapter id: tensors}``
        (rebuilt per row so each backward has its own composition graph). Returns the weighted loss value."""
        from ...data.native_graph import graph_loss
        from ...model.native_adapters import adapter_state_input_receipt, run_native
        total = 0.
        for row in self.rows:
            entries = self.entries(row['binding'], adapters)
            leaves = make_leaves()
            def execute(row=row):
                blocks, identities = {}, {}
                for name, key in row['soft'].items():
                    if soft_blocks is None or key not in soft_blocks:
                        raise LearnedOptimizerError(f'support row imports soft state {key} without a live value')
                    block, state_id = soft_blocks[key]
                    blocks[name] = block
                    identities[name] = adapter_state_input_receipt(self.engine, block, state_id, target_position_start=0)['source_native_identity']
                loss, _ = graph_loss(self.engine.shared_model, row['example'], checkpoint_layers=False,
                                     external_blocks=blocks or None, external_identity=identities or None)
                if backward and loss.requires_grad:
                    (loss * row['coefficient']).backward()
                return float(loss.detach()) * row['coefficient']
            total += run_native(self.engine, execute, entries, leaves={k: v for k, v in leaves.items() if any(e['id'] == k for e in entries)},
                                gradient_precision='fp32')
        return total


@dataclass
class ScopeArray:
    scope: str
    adapter: str          # placeholder composed adapter whose values the fit overrides
    slots: list           # bank component indices of the view
    available: list
    offset: dict | None = None   # incoming adapter cores (`ComponentBank.incoming`); None: the scope starts at zero


def placeholder_adapters(engine, bank, scopes, owner=None):
    """One zero composed adapter per scope; fits and training substitute live values for its coefficients."""
    out = {}
    for index, scope in enumerate(scopes):
        zero = {name: torch.zeros((*bank.basis.left[name].shape[:-2], bank.basis.rank, bank.basis.rank)) for name in bank.basis.names}
        out[scope] = bank.compose_adapter(engine, zero, owner, {'placeholder': scope, 'ordinal': index}).id
    return out


def fit_in_bank(engine, bank, support, arrays, *, soft_states=None, steps=50, learning_rate=5e-2,
                state_learning_rate=1e-3, refine_steps=30, beta=.5, init=None, train_states=True, fixed_states=()):
    """Fit in the learner's parameterization. ``arrays``: list of `ScopeArray`; ``soft_states``: ``{soft key: state
    id}`` trained jointly when ``train_states`` (except ``fixed_states``, held at their incoming value); ``init``: scope -> (K,) initial mean strengths. Returns labels and
    history.

    Phase 1 trains a per-slot mean strength ``m`` (in grid-free units) and the head's bounded centered modulation
    ``c_kg = m_k (1 + delta_kg)``: only sign-consistent coefficients the grid and head can execute are explored.
    Phase 2 quantizes ``m`` into ``(n, rho)`` and refines ``u, v`` (and the states) through exactly `head.executed`."""
    from ...serve.native_grad import _leaf_block
    G = len(bank.group_labels)
    shares_all = shares_from_norms(bank.group_norms())
    means = {array.scope: torch.zeros(len(array.slots), requires_grad=True) for array in arrays}
    v = {array.scope: torch.zeros(len(array.slots), G, requires_grad=True) for array in arrays}
    if init:
        with torch.no_grad():
            for scope, value in init.items():
                means[scope].copy_(torch.as_tensor(value, dtype=torch.float32))
    masks = {array.scope: torch.tensor(array.available, dtype=torch.float32) for array in arrays}
    states = {}
    for key, state_id in (soft_states or {}).items():
        block, values = _leaf_block(engine.store.get_native_state(state_id).block)
        if not train_states or key in fixed_states:
            for value in values.values():
                value.requires_grad_(False)
        states[key] = (block, state_id, values)
    soft_blocks = {key: (block, state_id) for key, (block, state_id, _) in states.items()}
    adapters = {array.scope: array.adapter for array in arrays}
    def leaves(values):
        return {array.adapter: bank.adapter_tensors(bank.composed_cores(array.slots, values[array.scope], offset=array.offset))
                for array in arrays}
    def continuous():
        return {array.scope: (means[array.scope] * masks[array.scope])[:, None]
                * (1 + modulation(v[array.scope], shares_all[array.slots], beta)) for array in arrays}
    state_params = [value for _, _, values in states.values() for value in values.values() if value.requires_grad]
    groups = [{'params': [*means.values(), *v.values()], 'lr': learning_rate}]
    if state_params:
        groups.append({'params': state_params, 'lr': state_learning_rate})
    optimizer = torch.optim.Adam(groups)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        history.append(support.accumulate(adapters, lambda: leaves(continuous()), soft_blocks))
        optimizer.step()
    labels, u = {}, {}
    for array in arrays:
        integers, rhos, saturated = [], [], 0
        for slot, value in enumerate(means[array.scope].detach().tolist()):
            n, rho, sat = quantize(value, bank.number_format) if array.available[slot] else (0, 0., False)
            integers.append(n); rhos.append(rho); saturated += sat
        u[array.scope] = torch.atanh(torch.tensor(rhos).clamp(-.499, .499) * 2).requires_grad_(True)
        v[array.scope] = v[array.scope].detach().clone().requires_grad_(True)
        lost = sum(1 for n, value, free in zip(integers, means[array.scope].tolist(), array.available) if free and n == 0 and value != 0)
        labels[array.scope] = {'integers': integers, 'shares': shares_all[array.slots], 'saturated': saturated,
                               'roundedToZero': lost}
    exact = lambda: {array.scope: executed(labels[array.scope]['integers'], u[array.scope], v[array.scope],
                                           labels[array.scope]['shares'], step=bank.number_format.step, beta=beta)
                     for array in arrays}
    refine = torch.optim.Adam([*u.values(), *v.values(), *state_params], lr=learning_rate)
    refined = []
    for _ in range(refine_steps):
        refine.zero_grad(set_to_none=True)
        refined.append(support.accumulate(adapters, lambda: leaves(exact()), soft_blocks))
        refine.step()
    with torch.no_grad():
        final = exact()
        fitted_loss = support.accumulate(adapters, lambda: leaves(final), soft_blocks, backward=False)
    out = {}
    for array in arrays:
        scope = array.scope
        shares = labels[scope]['shares']
        out[scope] = {'integers': labels[scope]['integers'], 'rho': refinement(u[scope]).detach(),
                      'delta': modulation(v[scope], shares, beta).detach(), 'u': u[scope].detach(), 'v': v[scope].detach(),
                      'executed': final[scope].detach(), 'saturated': labels[scope]['saturated'],
                      'roundedToZero': labels[scope]['roundedToZero'], 'means': means[scope].detach()}
    new_states = {key: _updated_block(block, values) for key, (block, _, values) in states.items()
                  if key not in fixed_states} if train_states else {}
    return {'schema': FIT_SCHEMA, 'bank': bank.snapshot, 'arrays': out, 'states': new_states,
            'loss': {'continuous': history, 'refined': refined, 'final': fitted_loss}, 'beta': beta}


def _updated_block(block, values):
    from dataclasses import replace
    layers = []
    for index, layer in enumerate(block.layers):
        copied = dict(layer)
        for name in list(copied):
            key = f'layers.{index}.{name}'
            if key in values:
                copied[name] = values[key].detach().clone()
        layers.append(copied)
    return replace(block, layers=tuple(layers))


def focus_labels(fit, bindings, *, full_gain, threshold=.05):
    """Leave-one-binding-out hindsight labels: binding ``b`` is a focus target when restricting the fit to the other
    bindings loses more than ``threshold`` of the full fit's gain. ``fit(excluded) -> gain`` refits on the same support
    evidence with ``excluded`` held at its incoming value."""
    labels = {}
    for binding in bindings:
        gain = fit(binding)
        loss = full_gain - gain
        labels[binding] = {'gainWithout': gain, 'loss': loss,
                           'focus': bool(full_gain > 0 and loss > threshold * full_gain)}
    return labels
