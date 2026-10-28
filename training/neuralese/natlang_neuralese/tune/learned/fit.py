"""In-bank fits and hindsight focus labels (plan "Comparable and plentiful targets" tier 2, "How analysis learns").

With a pinned bank snapshot, fit each scope's component-token distribution, its group gains and the program's soft
states directly on the support evidence by gradient through the exact composition (`mixture`):

1. free logits over [null, view slots] (unavailable slots masked before the softmax) and free gain logits, with
   ``c_kg = gain_max * sigmoid(gain_logit_g) * softmax(logits)_k``; Adam on the support loss;
2. the top-p set of that distribution is fixed and the fit continues on the renormalized truncated law, exactly the
   law the optimizer executes.

The phase-2 result is exactly representable by the component tokens and gains; its truncated distribution and gains are
the KL and gain labels. Refit after each bank publication instead of carrying stale labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import LearnedOptimizerError
from .mixture import coefficients, distribution, gain_bias

FIT_SCHEMA = 'natlang.lopt.in-bank-fit/2'


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
                state_learning_rate=1e-3, refine_steps=30, init=None, train_states=True, fixed_states=(),
                fixed_adapters=None):
    """Fit in the learner's parametrization. ``arrays``: list of `ScopeArray`; ``soft_states``: ``{soft key: state
    id}`` trained jointly when ``train_states`` (except ``fixed_states``, held at their incoming value);
    ``fixed_adapters``: scope -> adapter that runs unchanged in every stack (scopes outside the fit); ``init``: scope ->
    a parent mixture ``{'distribution', 'gain'}`` or (M,) projected coefficients (`targets.project_adapter`).
    Returns labels and history."""
    from ...serve.native_grad import _leaf_block
    fmt = bank.mixture_format
    G = len(bank.group_labels)
    logits, gain_logits, masks = {}, {}, {}
    for array in arrays:
        masks[array.scope] = torch.tensor([True, *array.available])
        logits[array.scope] = torch.zeros(1 + len(array.slots))
        gain_logits[array.scope] = torch.full((G,), gain_bias(fmt))
    for scope, value in (init or {}).items():
        logits[scope], gain_logits[scope] = _initial(value, masks[scope], fmt, G)
    for tensor in (*logits.values(), *gain_logits.values()):
        tensor.requires_grad_(True)
    states = {}
    for key, state_id in (soft_states or {}).items():
        block, values = _leaf_block(engine.store.get_native_state(state_id).block)
        if not train_states or key in fixed_states:
            for value in values.values():
                value.requires_grad_(False)
        states[key] = (block, state_id, values)
    soft_blocks = {key: (block, state_id) for key, (block, state_id, _) in states.items()}
    adapters = {**(fixed_adapters or {}), **{array.scope: array.adapter for array in arrays}}
    keep = {}
    def masked(scope):
        return torch.where(masks[scope], logits[scope], torch.full_like(logits[scope], -torch.inf))
    def law(scope):
        probs = distribution(masked(scope), fmt.top_p, keep[scope])[0] if scope in keep else torch.softmax(masked(scope), -1)
        return probs, fmt.gain_max * torch.sigmoid(gain_logits[scope])
    def executed():
        return {array.scope: coefficients(*law(array.scope)) for array in arrays}
    def leaves(values):
        return {array.adapter: bank.adapter_tensors(bank.composed_cores(array.slots, values[array.scope], offset=array.offset))
                for array in arrays}
    state_params = [value for _, _, values in states.values() for value in values.values() if value.requires_grad]
    def run(count):
        groups = [{'params': [*logits.values(), *gain_logits.values()], 'lr': learning_rate}]
        if state_params:
            groups.append({'params': state_params, 'lr': state_learning_rate})
        optimizer, history = torch.optim.Adam(groups), []
        for _ in range(count):
            optimizer.zero_grad(set_to_none=True)
            history.append(support.accumulate(adapters, lambda: leaves(executed()), soft_blocks))
            optimizer.step()
        return history
    history = run(steps)
    with torch.no_grad():
        for array in arrays:
            keep[array.scope] = distribution(masked(array.scope), fmt.top_p)[1]
    refined = run(refine_steps)
    with torch.no_grad():
        final = executed()
        fitted_loss = support.accumulate(adapters, lambda: leaves(final), soft_blocks, backward=False)
    out = {}
    for array in arrays:
        with torch.no_grad():
            probs, gains = law(array.scope)
        out[array.scope] = {'distribution': probs.detach(), 'gain': gains.detach(), 'kept': keep[array.scope].tolist(),
                            'executed': final[array.scope].detach(), 'support': int(keep[array.scope][1:].sum())}
    new_states = {key: _updated_block(block, values) for key, (block, _, values) in states.items()
                  if key not in fixed_states} if train_states else {}
    return {'schema': FIT_SCHEMA, 'bank': bank.snapshot, 'arrays': out, 'states': new_states,
            'loss': {'continuous': history, 'refined': refined, 'final': fitted_loss}}


def _initial(value, mask, fmt, groups):
    """Free logits and gain logits from a parent mixture or from projected signed coefficients (their positive part:
    the bank absorbs signs; a projection without positive mass starts uniform)."""
    import math
    logit = lambda share: math.log(share / (1 - share))
    if isinstance(value, dict):
        probs = torch.as_tensor(value['distribution'], dtype=torch.float32)
        gains = torch.as_tensor(value['gain'], dtype=torch.float32)
        logits = torch.where(probs > 0, probs.clamp_min(1e-12).log(), torch.full_like(probs, -12.))
        return logits, torch.tensor([logit(min(max(float(g) / fmt.gain_max, 1e-4), 1 - 1e-4)) for g in gains])
    positive = torch.as_tensor(value, dtype=torch.float32).clamp_min(0) * mask[1:]
    total = float(positive.sum())
    if total <= 0:
        return torch.zeros(len(mask)), torch.full((groups,), gain_bias(fmt))
    probs = torch.cat([torch.zeros(1), positive / total])
    share = min(max(total / fmt.gain_max, 1e-4), 1 - 1e-4)
    return (probs + 1e-3).log(), torch.full((groups,), logit(share))


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
