"""Tuning the meta-learner itself: exact tuning examples for the learned optimizer's own decision stages.

The joint learned optimizer's policy (`applications/learned-optimizer`) is an ordinary Natlang program, so it is tuned
like any other program: `natlang tune` with SFT/RL on its stages, and also `natlang tune --learned` once its own
episodes are in the corpus. What it lacks is examples with exact expected values. Every adaptation episode supplies
them, because each decision can be scored exactly against the services after the fact:

* ``adapt/chooseOperator`` -- counterfactual: from the episode's actual state, every operator (learned proposal,
  in-bank refinement, retrieval) runs on the same parent and is scored on the support evidence; the expected operator
  is the one with the lowest support loss when it beats the incumbent, else ``keep``.
* ``adapt/shouldStop`` -- hindsight over a full-budget episode: stop exactly when no later step lowers the best
  support loss.
* ``adapt/chooseFocus`` -- leave-one-binding-out hindsight (`fit.focus_labels`): a binding is in focus when fitting
  the program with that binding's own scope and soft states held at their incoming value loses a material share of
  the full fit's gain.

Each example also carries its exact ``reference`` measurements, which the program's verifier
(`applications/learned-optimizer/tune.reward.mjs`) turns into graded rewards toward the global target -- the
adaptation's support-loss improvement: the share of the best improvement an operator achieves, the share of the
episode's gain an early stop loses, and the F1 of a focus. Teacher demonstrations still need a full score.

The episode follows the counterfactual best operator, so its states are those of a good policy. Every example of one
program carries the program's group: a program's episodes stay on one side of the meta-tuning split, and programs
on the optimizer's evaluation side never supply meta-tuning examples. Support evidence only: no query or
confirmation outcome enters a label.
"""
from __future__ import annotations

import json

from . import LearnedOptimizerError

OPERATORS = ('learned-proposal', 'local-refinement', 'retrieval')
META_SCHEMA = 'natlang.lopt.meta-example/1'


def summary(bundle, loss, actor):
    return None if bundle is None else {'id': bundle['id'], 'supportLoss': loss, 'parent': bundle.get('parent'), 'actor': actor}


def operator_outcomes(context, parent, *, step, refine_steps):
    """Every operator from the same parent, scored exactly: ``{operator: (bundle, support loss)}``."""
    out = {}
    bundle, _, _ = context.propose(parent, sample=step > 0, seed=step)
    out['learned-proposal'] = (bundle, context.evaluate(bundle))
    bundle = context.refine(parent, steps=refine_steps)
    out['local-refinement'] = (bundle, context.evaluate(bundle))
    bundle, loss = context.retrieve(parent)
    out['retrieval'] = (bundle, loss)
    return out


def best_operator(outcomes, incumbent_loss):
    operator = min(OPERATORS, key=lambda name: outcomes[name][1])
    return operator if outcomes[operator][1] < incumbent_loss else 'keep'


def stop_labels(best_losses):
    """``best_losses[t]``: best support loss after step t. Stop at t when no later step improves on it."""
    return [all(later >= best_losses[t] for later in best_losses[t + 1:]) for t in range(len(best_losses))]


def focus_label(context, *, threshold=.05, limit=3, fit=None):
    """Hindsight focus of the context's program at its incoming bundle (sorted binding names, at most ``limit``)."""
    from .fit import fit_in_bank, focus_labels
    incoming = {key: record['state'] for key, record in context.soft.items()}
    incoming.update(getattr(context, 'incoming_states', {}) or {})
    adapters = getattr(context, 'incoming_adapters', {}) or {}
    base_loss = context.evaluate(None)
    nodes = {binding['id']: binding['ancestry'] for binding in context.manifest['bindings']}
    def gain(excluded=None):
        own = set()
        if excluded is not None:
            shared = {node for other, ancestry in nodes.items() if other != excluded for node in ancestry}
            own = {node for node in nodes[excluded] if node not in shared}
        arrays = [array for array in context.pc.scope_arrays(adapters) if array.scope not in own]
        fixed = {key for key in incoming if json.loads(key)[0] == excluded}
        result = fit_in_bank(context.engine, context.bank, context.support, arrays, soft_states=incoming,
                             fixed_states=fixed, **(fit or {'steps': 10, 'refine_steps': 4}))
        return base_loss - result['loss']['final']
    full = gain()
    labels = focus_labels(gain, sorted(nodes), full_gain=full, threshold=threshold)
    chosen = sorted((name for name, label in labels.items() if label['focus']), key=lambda name: -labels[name]['loss'])[:limit]
    return sorted(chosen), {'fullGain': full, 'labels': labels}


def episode_examples(context, *, steps, group, refine_steps=10, focus=True, fit=None):
    """Exact meta-tuning examples from one full-budget, counterfactual-best episode on ``context`` (an
    `apply.Context`). Returns ``(examples, receipt)``."""
    if steps < 1:
        raise LearnedOptimizerError('a meta episode needs at least one step')
    examples, receipt = [], {'schema': META_SCHEMA, 'program': context.pc.program, 'steps': []}
    view = context.evidence
    chosen, focus_receipt = focus_label(context, fit=fit) if focus else (None, None)
    if focus:
        examples.append({'function': 'adapt/chooseFocus.nl', 'input': {'view': view, 'history': []}, 'expected': chosen,
                         'reference': {'focus': chosen}, 'group': group})
        receipt['focus'] = focus_receipt
    plan = {'operations': {binding['id']: ('weights' if chosen and binding['id'] in chosen else 'keep')
                           for binding in context.manifest['bindings']}, 'reason': 'hindsight focus'}
    incoming_loss = context.evaluate(None)
    current, current_loss, history, best_losses, states = None, incoming_loss, [], [], []
    for step in range(steps):
        remaining = steps - step
        outcomes = operator_outcomes(context, current, step=step, refine_steps=refine_steps)
        operator = best_operator(outcomes, current_loss)
        examples.append({'function': 'adapt/chooseOperator.nl', 'expected': operator, 'group': group,
                         'input': {'history': list(history), 'plan': plan, 'remainingSteps': remaining},
                         'reference': {'losses': {name: loss for name, (_, loss) in outcomes.items()}, 'incumbentLoss': current_loss}})
        receipt['steps'].append({'step': step, 'losses': {name: loss for name, (_, loss) in outcomes.items()},
                                 'incumbentLoss': current_loss, 'operator': operator})
        candidate, loss = outcomes[operator] if operator != 'keep' else (None, None)
        accepted = candidate is not None and loss < current_loss
        if accepted:
            current, current_loss = candidate, loss
        history.append({'step': step, 'operator': operator, 'bundle': candidate['id'] if candidate else None,
                        'supportLoss': loss, 'accepted': accepted, 'reason': plan['reason']})
        best_losses.append(current_loss)
        states.append({'current': summary(current, current_loss, current['provenance'].get('actor', '') if current else ''),
                       'incomingLoss': incoming_loss, 'remainingSteps': remaining - 1, 'history': list(history),
                       'stopped': False, 'stopReason': ''})
    total = incoming_loss - best_losses[-1]
    for state, stop, best in zip(states, stop_labels(best_losses), best_losses):
        remaining = (best - best_losses[-1]) / total if total > 0 else 0.
        examples.append({'function': 'adapt/shouldStop.nl', 'input': {'state': state}, 'expected': stop, 'group': group,
                         'reference': {'stop': stop, 'remainingShare': remaining}})
    receipt['bestLosses'] = best_losses
    return examples, receipt


def write_examples(path, examples):
    """Append-only JSONL in the tuner's ``tune.examples.jsonl`` row format."""
    from pathlib import Path
    with Path(path).open('a') as stream:
        for row in examples:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
