"""Joint target records, tuner export, episode multiplication and exposure accounting (plan "Turning existing training
into optimizer training data", "Comparable and plentiful targets", "Many problems from each program").

Three tiers share one schema: ``free-tune`` (the joint SFT/RL tuner from the common incoming bundle; supplies bank
growth and the full-tune ceiling), ``in-bank-fit`` (behaviorally fitted in the learner's own parameterization;
mixture-token labels), ``self-proposal`` (the learner's verified proposals with their refinements). A target is a
complete tuple: never a recombination of parts from different runs. The target generator may not use support evidence
absent from the recorded optimizer input.
"""
from __future__ import annotations

import json
import random

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError

TARGET_SCHEMA = 'natlang.joint-target/1'
TIERS = ('free-tune', 'in-bank-fit', 'self-proposal')
EXPOSURE_SCHEMA = 'natlang.lopt.exposure/1'


def target_record(*, tier, program, before, after, evidence, objective, actors, representation, mixtures,
                  residual=None, outcomes=None, lineage=None, trajectories=None, compute=None, binding_family=None):
    """``evidence``: the evidence view ID and the ordered support item IDs exactly as the optimizer input holds them.
    ``mixtures``: scope -> {componentIds, available, integers, rho, delta, fittingRule, numberFormat}."""
    if tier not in TIERS:
        raise LearnedOptimizerError(f'unknown target tier {tier}')
    if not isinstance(evidence, dict) or not evidence.get('view') or not isinstance(evidence.get('items'), list):
        raise LearnedOptimizerError('a target binds its exact evidence view and ordered support items')
    for scope, mixture in mixtures.items():
        if len(mixture['integers']) != len(mixture['componentIds']) or any(
                not free and n for n, free in zip(mixture['integers'], mixture['available'])):
            raise LearnedOptimizerError(f'target mixture of {scope} is misaligned or nonzero at unavailable slots')
    body = {'schema': TARGET_SCHEMA, 'tier': tier, 'program': program, 'bindingFamily': binding_family or program,
            'before': before, 'after': after, 'evidence': evidence, 'objective': objective, 'actors': actors,
            'representation': representation, 'mixtures': mixtures, 'residual': residual or {},
            'outcomes': outcomes or {}, 'lineage': lineage or {}, 'trajectories': trajectories or [],
            'compute': compute or {}}
    return {**body, 'id': 'njt1:' + canonical_json_sha256_hex(body)}


def require_same_evidence(target, optimizer_input_items):
    if list(target['evidence']['items']) != list(optimizer_input_items):
        raise LearnedOptimizerError('the target used support evidence the optimizer input does not hold; label it as a '
                                    'different conditional problem instead')


def node_effective_updates(trainer, node):
    """``{m: (B, A)}`` of one scope node's effective residual (scaling folded once), from the live tuner state."""
    artifact = trainer.state['node_artifacts'][node]
    values = trainer.engine.store.get_native_tensor_artifact(artifact).artifact.tensors
    return trainer._node_factors(artifact, values)


def tuner_boundary(trainer, *, before_state, label):
    """A free-tune before/after pair at a real committed update boundary of `tune.trainer.JointTrainer`: node and
    soft-state versions, the step and exposure, so incremental revisions and final residuals are both recorded."""
    after = trainer.state
    changed_nodes = sorted(node for node, artifact in after['node_artifacts'].items()
                           if before_state['node_artifacts'].get(node) != artifact)
    changed_soft = sorted(key for key, state in after.get('soft_states', {}).items()
                          if before_state.get('soft_states', {}).get(key) != state)
    return {'label': label, 'kind': 'incremental' if before_state['step'] else 'final-residual',
            'stepBefore': before_state['step'], 'stepAfter': after['step'], 'inputSha256': after['input_sha256'],
            'nodesBefore': {node: before_state['node_artifacts'][node] for node in changed_nodes},
            'nodesAfter': {node: after['node_artifacts'][node] for node in changed_nodes},
            'softBefore': {key: before_state.get('soft_states', {}).get(key) for key in changed_soft},
            'softAfter': {key: after['soft_states'][key] for key in changed_soft},
            'optimizerStates': {node: after['optimizer_states'].get(node) for node in changed_nodes},
            'exposure': dict(after['exposure'])}


def project_tune(bank, trainer, nodes):
    """Project each tuned node's effective update into the bank: basis coordinates, mixture fit and residual energies
    (weight-space projection initializes and grows the bank; it is not a mixture-token label)."""
    out = {}
    for node in nodes:
        factors = node_effective_updates(trainer, node)
        factors = {name: value for name, value in factors.items() if name in bank.basis.left}
        cores, energy = bank.project_factors(factors)
        coefficients, residual, target = bank.fit_mixture(cores) if bank.components else (None, None, None)
        observed = sum(bank.q[name] * total for name, (total, _) in energy.items())
        captured = sum(bank.q[name] * inside for name, (_, inside) in energy.items())
        out[node] = {'cores': cores, 'basisCapture': captured / observed if observed else None,
                     'mixture': None if coefficients is None else coefficients.tolist(),
                     'bankResidualShare': (residual / target) if target else None}
    return out


# -- episode multiplication -------------------------------------------------------------------------------------
def episode_variants(problem, *, rng_seed, subsets=4, include_empty=True, starts=('base', 'tuned', 'dropped', 'swapped'),
                     other_bundles=()):
    """Episodes from one problem: support subsets from one world to the full pool, an evidence-free episode, and
    starting bundles (base, partly tuned, another program's bundle, dropped/swapped parts). Every variant of one
    program keeps the program's split."""
    rng = random.Random(rng_seed)
    items = list(problem['support'])
    sizes = sorted({1, len(items), *[rng.randint(1, max(1, len(items))) for _ in range(max(0, subsets - 2))]}) if items else []
    episodes = []
    for size in sizes:
        subset = sorted(rng.sample(items, size))
        for start in starts:
            episodes.append({'program': problem['program'], 'bindingFamily': problem.get('bindingFamily', problem['program']),
                             'family': problem.get('family', problem['program']), 'split': problem['split'],
                             'support': subset, 'start': start, 'seed': rng.randrange(1 << 31)})
    if include_empty:
        episodes.append({'program': problem['program'], 'bindingFamily': problem.get('bindingFamily', problem['program']),
                         'family': problem.get('family', problem['program']), 'split': problem['split'], 'support': [],
                         'start': 'tuned', 'seed': rng.randrange(1 << 31)})
    for other in other_bundles:
        episodes.append({'program': problem['program'], 'bindingFamily': problem.get('bindingFamily', problem['program']),
                         'family': problem.get('family', problem['program']), 'split': problem['split'],
                         'support': items, 'start': {'otherBundle': other}, 'seed': rng.randrange(1 << 31)})
    return episodes


def perturb_mixture(integers, available, *, mode, rng_seed):
    """Drop or swap parts of a tuned mixture (recovery episodes); unavailable slots stay zero."""
    rng = random.Random(rng_seed)
    values = list(integers)
    used = [slot for slot, value in enumerate(values) if value]
    if mode == 'dropped' and used:
        values[rng.choice(used)] = 0
    elif mode == 'swapped':
        free = [slot for slot, flag in enumerate(available) if flag]
        if len(free) >= 2:
            a, b = rng.sample(free, 2)
            values[a], values[b] = values[b], values[a]
    elif mode not in ('dropped', 'swapped'):
        raise LearnedOptimizerError('perturbation mode is dropped or swapped')
    return values


# -- exposure ---------------------------------------------------------------------------------------------------
class Exposure:
    """Prepared, admitted and actually exposed counts, kept separate (a readback or queued job is not exposure)."""

    def __init__(self, state=None):
        self.state = state or {'schema': EXPOSURE_SCHEMA, 'prepared': {}, 'admitted': {}, 'exposed': {}, 'updates': 0,
                               'contrastGroups': 0, 'noContrastGroups': 0, 'outcomes': 0}

    def prepare(self, target):
        self.state['prepared'][target['id']] = target['program']

    def admit(self, target):
        if target['id'] not in self.state['prepared']:
            raise LearnedOptimizerError('a target is admitted only after preparation')
        self.state['admitted'][target['id']] = target['program']

    def expose(self, target_id):
        if target_id not in self.state['admitted']:
            raise LearnedOptimizerError('only admitted targets can be exposed')
        self.state['exposed'][target_id] = self.state['exposed'].get(target_id, 0) + 1

    def summary(self):
        admitted = self.state['admitted']
        return {'prepared': len(self.state['prepared']), 'admitted': len(admitted),
                'exposedDistinct': len(self.state['exposed']), 'exposures': sum(self.state['exposed'].values()),
                'programsAdmitted': len(set(admitted.values())), 'updates': self.state['updates'],
                'contrastGroups': self.state['contrastGroups'], 'noContrastGroups': self.state['noContrastGroups']}
