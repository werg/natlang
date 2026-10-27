"""Joint target records, tuner export, episode multiplication and exposure accounting (plan "Turning existing training
into optimizer training data", "Comparable and plentiful targets", "Many problems from each program").

Three tiers share one schema: ``free-tune`` (the joint SFT/RL tuner from the common incoming bundle; supplies bank
growth and the full-tune ceiling), ``in-bank-fit`` (behaviorally fitted in the learner's own parameterization;
mixture-token labels), ``self-proposal`` (the learner's verified proposals with their refinements). A target is a
complete tuple: never a recombination of parts from different runs. The target generator may not use support evidence
absent from the recorded optimizer input.
"""
from __future__ import annotations

import random

from . import LearnedOptimizerError

EXPOSURE_SCHEMA = 'natlang.lopt.exposure/1'


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


def project_adapter(bank, engine, adapter):
    """Basis coordinates of any adapter's exact effective update, with the share of its relative energy the bank
    basis captures and (with components) the least-squares mixture and its residual share. Weight-space projection
    initializes fits and grows the bank; it is never a mixture-token label. An adapter slicing the bank basis is
    captured exactly."""
    from ..trainer import adapter_factors
    factors = {name: value for name, value in adapter_factors(engine, adapter).items() if name in bank.basis.left}
    cores, energy = bank.project_factors(factors)
    coefficients, residual, target = bank.fit_mixture(cores) if bank.components else (None, None, None)
    observed = sum(bank.q[name] * total for name, (total, _) in energy.items())
    captured = sum(bank.q[name] * inside for name, (_, inside) in energy.items())
    return {'cores': cores, 'basisCapture': captured / observed if observed else None,
            'mixture': None if coefficients is None else coefficients.tolist(),
            'bankResidualShare': (residual / target) if target else None}


# -- target builders over `program.ProgramContext` ----------------------------------------------------------------
def fit_labels(pc, *, incoming_states, incoming_adapters=None, init=None, fit=None, train_states=True):
    """An in-bank fit of the context's support rows from the incoming bundle: mixture-token and head labels and the
    fitted states, composed into stored target adapters."""
    from .assemble import store_fitted_states
    from .fit import fit_in_bank
    arrays = pc.scope_arrays(incoming_adapters)
    result = fit_in_bank(pc.engine, pc.bank, pc.support, arrays, soft_states=incoming_states, init=init,
                         train_states=train_states, **(fit or {}))
    states = store_fitted_states(pc.engine, result['states'])
    adapters = {array.scope: pc.bank.compose_adapter(pc.engine, pc.bank.composed_cores(
        array.slots, result['arrays'][array.scope]['executed'], offset=array.offset)).id for array in arrays}
    labels = {'integers': {s: result['arrays'][s]['integers'] for s in pc.scopes}, 'adapters': adapters, 'states': states,
              'rho': {s: result['arrays'][s]['rho'].tolist() for s in pc.scopes},
              'delta': {s: result['arrays'][s]['delta'].tolist() for s in pc.scopes}}
    report = {'loss': result['loss'], 'saturated': {s: result['arrays'][s]['saturated'] for s in pc.scopes},
              'roundedToZero': {s: result['arrays'][s]['roundedToZero'] for s in pc.scopes}}
    return labels, report


def in_bank_target(pc, *, fit=None, lineage=None):
    """Tier 2 from the program's own incoming bundle (initial soft states, zero adapters)."""
    incoming = {key: record['state'] for key, record in pc.soft.items()}
    labels, report = fit_labels(pc, incoming_states=incoming, fit=fit)
    target = pc.request(tier='in-bank-fit', labels=labels, lineage={'fit': report, **(lineage or {})})
    pc.persist([*labels['adapters'].values(), *labels['states'].values()])
    return target


def free_tune_targets(pc, *, fit=None, incremental=True, lineage=None, incremental_max=None):
    """Tier 1 from the context's tuner build: the final residual from the common incoming bundle and, when the fit
    sliced this bank's basis and recorded its update boundaries, each committed update as a revision of the partly
    tuned bundle before it. The tuned bundle is the behavior target (distillation, soft states and exact decoded
    weights); mixture labels are an in-bank fit of the same support rows started from its weight projection.
    ``incremental_max`` keeps that many evenly spaced update boundaries (each is one more labeled fit)."""
    from .program import read_build
    if pc.build is None:
        raise LearnedOptimizerError('free-tune targets need the program context of a tuner build')
    build = read_build(pc.build)
    state = build['state']
    initial = {key: record['state'] for key, record in pc.soft.items()}
    source = {'build': str(pc.build), 'inputSha256': state['input_sha256'], 'step': state['step'],
              'exposure': state.get('exposure', {}), **(lineage or {})}
    episodes = [{'kind': 'final-residual', 'incomingStates': initial, 'incomingAdapters': {},
                 'targetStates': {**initial, **state.get('soft_states', {})}, 'targetAdapters': dict(state['node_artifacts'])}]
    if incremental and build['boundaries'] and _slices_bank(pc, state['node_artifacts'].values()):
        nodes, soft = {node: history[0] for node, history in state['node_history'].items()}, dict(initial)
        for boundary in build['boundaries']:
            if boundary['inputSha256'] != state['input_sha256']:
                continue  # a boundary of another fit of this build directory is not this fit's revision
            before_nodes = {**nodes, **boundary['nodesBefore']}
            before_soft = {**soft, **{k: v for k, v in boundary['softBefore'].items() if v}}
            after_nodes, after_soft = {**before_nodes, **boundary['nodesAfter']}, {**before_soft, **boundary['softAfter']}
            if boundary['stepBefore']:
                episodes.append({'kind': 'incremental', 'boundary': [boundary['stepBefore'], boundary['stepAfter']],
                                 'incomingStates': before_soft, 'incomingAdapters': before_nodes,
                                 'targetStates': after_soft, 'targetAdapters': after_nodes})
            nodes, soft = after_nodes, after_soft
    if incremental_max is not None and len(episodes) - 1 > incremental_max:
        revisions = episodes[1:]
        keep = sorted({round(i * (len(revisions) - 1) / max(1, incremental_max - 1)) for i in range(incremental_max)})
        episodes = [episodes[0], *(revisions[i] for i in keep)]
    out = []
    for episode in episodes:
        incoming_adapters = {s: a for s, a in episode['incomingAdapters'].items() if s in pc.scopes}
        target_adapters = {s: a for s, a in episode['targetAdapters'].items() if s in pc.scopes}
        init = {}
        for scope in pc.scopes:
            if scope in target_adapters and pc.bank.components:
                projected = project_adapter(pc.bank, pc.engine, target_adapters[scope])
                previous = project_adapter(pc.bank, pc.engine, incoming_adapters[scope])['mixture'] if scope in incoming_adapters else None
                mixture = projected['mixture'] if previous is None else [a - b for a, b in zip(projected['mixture'], previous)]
                init[scope] = [mixture[slot] if free else 0. for slot, free in zip(pc.views[scope].slots, pc.views[scope].available)]
        labels, report = fit_labels(pc, incoming_states=episode['incomingStates'], incoming_adapters=incoming_adapters,
                                    init=init or None, fit=fit)
        # The free tune itself is the behavior and weight target; the in-bank fit supplies the mixture tokens.
        labels.update(adapters=target_adapters, states=episode['targetStates'], factors=target_adapters)
        out.append(pc.request(tier='free-tune', incoming_states=episode['incomingStates'], incoming_adapters=incoming_adapters,
                              labels=labels, lineage={**source, 'episode': {k: v for k, v in episode.items() if k == 'kind' or k == 'boundary'},
                                                      'labelFit': report}))
    return out


def _slices_bank(pc, adapters):
    try:
        for adapter in adapters:
            pc.bank.adapter_cores(pc.engine, adapter)
    except LearnedOptimizerError:
        return False
    return True


# -- episode multiplication -------------------------------------------------------------------------------------
def episode_variants(problem, *, rng_seed, subsets=4, include_empty=True, starts=('base', 'tuned', 'dropped', 'swapped'),
                     other_bundles=(), max_size=None):
    """Episodes from one problem: support subsets from one world to the full pool, an evidence-free episode, and
    starting bundles (base, partly tuned, another program's bundle, dropped/swapped parts). Every variant of one
    program keeps the program's split."""
    rng = random.Random(rng_seed)
    items = list(problem['support'])
    largest = min(len(items), max_size or len(items))
    sizes = sorted({1, largest, *[rng.randint(1, max(1, largest)) for _ in range(max(0, subsets - 2))]}) if items else []
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
