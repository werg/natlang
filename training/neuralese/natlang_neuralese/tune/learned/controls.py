"""Planted controls (plan "Planted controls"): inexpensive fixtures with known answers, run as telemetry.

* planted mixtures: support evidence demonstrates a known component-token distribution;
* perturbation recovery: tuned mixtures with dropped or swapped parts (`targets.perturb_mixture`);
* negative controls: shuffled or irrelevant evidence must not yield query gains; permuting descriptions alone must
  break planted recovery, permuting descriptions together with their tensors and weights must not.

A failed control is a fault to localize against the exact references before any conclusion about the approach.
"""
from __future__ import annotations

import random
from dataclasses import replace

import torch

from . import LearnedOptimizerError


def planted_mixture(view, *, rng_seed, components=2):
    """A known component-token distribution (null first) over a view's available slots."""
    rng = random.Random(rng_seed)
    free = [slot for slot, flag in enumerate(view.available) if flag]
    if len(free) < components:
        raise LearnedOptimizerError('the view has too few available slots for the planted mixture')
    values = [0.] * (1 + len(view.slots))
    weights = [rng.uniform(.5, 1.) for _ in range(components)]
    for slot, weight in zip(rng.sample(free, components), weights):
        values[1 + slot] = weight / sum(weights)
    return values


def planted_rows(engine, adapter_id, prompts, *, binding, prefix='planted'):
    """Support rows labeled by the planted adapter's greedy continuation of each prompt (exact executor behavior)."""
    from ...model.native_adapters import run_native
    from ...model.shared_state import Invocation, compile_graph
    model, rows = engine.shared_model, []
    fingerprint = engine.model_config['tokenizer_fingerprint']
    for index, prompt in enumerate(prompts):
        ids = list(prompt)
        with torch.no_grad():
            for _ in range(3):
                plan = compile_graph([Invocation('x', tuple(ids))])
                logits = run_native(engine, lambda: model.logits(model(plan)['h_final'][0, -1:]), [{'id': adapter_id, 'scale': 1.}])
                ids.append(int(logits.argmax(-1)))
        weights = [0.] * len(prompt) + [1.] * 3
        rows.append({'binding': binding, 'evidence_weight': 1., 'occurrence': f'{prefix}-{index}',
                     'graph': {'schema': 'natlang.neuralese-native-graph/1', 'id': f'{prefix}-{index}', 'split': 'train',
                               'source_groups': [f'{prefix}-group'], 'tokenizer_fingerprint': fingerprint,
                               'nodes': [{'name': 'answer', 'token_ids': ids, 'target_weights': weights}]}})
    return rows


def recovery(predicted, planted, *, threshold=.05):
    """Total-variation distance of a recovered distribution and the share of the planted support it puts at least
    ``threshold`` probability on."""
    if len(predicted) != len(planted):
        raise LearnedOptimizerError('recovered and planted distributions differ in length')
    support = [slot for slot, value in enumerate(planted) if value]
    return {'totalVariation': sum(abs(a - b) for a, b in zip(predicted, planted)) / 2,
            'supportRecall': sum(int(predicted[slot] >= threshold) for slot in support) / max(1, len(support))}


def permute_view(view, permutation, *, aligned):
    """``aligned=True`` permutes descriptions with their components (and so their tensors and weights): a legal new
    view. ``aligned=False`` permutes only the key/description order against unchanged component positions — the
    negative control; it is labeled as such and never used for training."""
    if sorted(permutation) != list(range(len(view.slots))):
        raise LearnedOptimizerError('not a permutation of the view slots')
    if aligned:
        take = lambda values: tuple(values[i] for i in permutation)
        return replace(view, slots=take(view.slots), component_ids=take(view.component_ids), eligible=take(view.eligible),
                       retained=take(view.retained), probe=None if view.probe is None else take(view.probe),
                       selection={**view.selection, 'permutation': list(permutation)})
    return replace(view, selection={**view.selection, 'negativeControl': 'descriptions-only', 'descriptionPermutation': list(permutation)})


def control_keys(bank, view):
    """The keys a request shows for a view: the descriptions-only negative control reorders them."""
    keys = [bank.components[k].key for k in view.slots]
    permutation = view.selection.get('descriptionPermutation')
    return keys if permutation is None else [keys[i] for i in permutation]


def shuffled_evidence(view, rng_seed):
    """Negative control: the same items with verdicts and actions detached from their roots."""
    rng = random.Random(rng_seed)
    items = [dict(item) for item in view['items']]
    attempts = [item for item in items if item['kind'] == 'attempt']
    verdicts = [(item['passed'], item['failingBinding'], item['errorKind']) for item in attempts]
    rng.shuffle(verdicts)
    for item, (passed, binding, kind) in zip(attempts, verdicts):
        item.update(passed=passed, failingBinding=binding, errorKind=kind)
    return {**view, 'items': items, 'control': 'shuffled-evidence'}
