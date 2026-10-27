"""The ceiling ladder, evaluation power and program-count accounting (plan "The ceiling ladder", "Evaluation power",
"Many problems from each program").

Rungs, measured per development problem on the same query worlds and limits:

1. incoming bundle; 2. retrieval / gradient-free bank mixing; 3. learned proposal (one step, and after its episode);
4. in-bank fit on the support evidence; 5. free joint tune on the support evidence;
6. in-bank fit on the development query worlds themselves (diagnostic only, never reported, never on confirmation).

Gaps: amortization = 4 - 3 (optimizer and its data); representation = 5 - 4 (bank, rank, state capacity);
evidence = 6 - 4 (what the support leaves undetermined); headroom = 5 - 1 (program, oracle, base).
"""
from __future__ import annotations

import math

from . import LearnedOptimizerError

RUNGS = ('incoming', 'retrieval', 'learned_one_step', 'learned_episode', 'in_bank_support', 'free_tune', 'in_bank_query')
GAP_ACTIONS = {'amortization': 'more programs and optimizer training',
               'representation': 'bank growth, rank or refitting',
               'evidence': 'better evidence selection, quantity or analysis',
               'headroom': 'different problems or a stronger base'}


def ladder(measurements, *, role):
    """``measurements``: rung -> quality on the shared query worlds. Returns the rungs, gaps and the largest gap."""
    if role == 'confirmation' and 'in_bank_query' in measurements:
        raise LearnedOptimizerError('the query-fitted ceiling is never computed on confirmation worlds')
    unknown = set(measurements) - set(RUNGS)
    if unknown:
        raise LearnedOptimizerError(f'unknown ladder rungs {sorted(unknown)}')
    m = measurements
    learned = m.get('learned_episode', m.get('learned_one_step'))
    gaps = {}
    if 'in_bank_support' in m and learned is not None:
        gaps['amortization'] = m['in_bank_support'] - learned
    if 'free_tune' in m and 'in_bank_support' in m:
        gaps['representation'] = m['free_tune'] - m['in_bank_support']
    if 'in_bank_query' in m and 'in_bank_support' in m:
        gaps['evidence'] = m['in_bank_query'] - m['in_bank_support']
    if 'free_tune' in m and 'incoming' in m:
        gaps['headroom'] = m['free_tune'] - m['incoming']
    largest = max(gaps, key=lambda name: gaps[name]) if gaps else None
    reported = {rung: value for rung, value in m.items() if rung != 'in_bank_query'}
    return {'role': role, 'rungs': reported, 'diagnostic': {'in_bank_query': m.get('in_bank_query')} if 'in_bank_query' in m else {},
            'gaps': gaps, 'largest': largest, 'action': GAP_ACTIONS.get(largest)}


def aggregate(ladders):
    """Median gaps across problems and the recovered-gain fraction (in-bank / free-tune gain, the bank sizing rule)."""
    def median(values):
        values = sorted(values)
        if not values:
            return None
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    gaps = {name: median([item['gaps'][name] for item in ladders if name in item['gaps']]) for name in GAP_ACTIONS}
    recovered = []
    for item in ladders:
        rungs = item['rungs']
        if {'free_tune', 'incoming', 'in_bank_support'} <= set(rungs) and rungs['free_tune'] > rungs['incoming']:
            recovered.append((rungs['in_bank_support'] - rungs['incoming']) / (rungs['free_tune'] - rungs['incoming']))
    return {'problems': len(ladders), 'medianGaps': gaps, 'medianRecoveredGain': median(recovered),
            'recoveryTarget': .75}


def warm_start_effect(work_from_proposal, work_from_incoming):
    """Free-tune work to reach the tuner's selected development quality from the learned proposal vs the incoming
    bundle (steps or compute; same unit)."""
    if work_from_incoming <= 0:
        raise LearnedOptimizerError('work from the incoming bundle must be positive')
    return {'fromProposal': work_from_proposal, 'fromIncoming': work_from_incoming,
            'ratio': work_from_proposal / work_from_incoming}


def paired_worlds_needed(effect=.1, discordant=.29, alpha_z=1.96, power_z=.84, design_effect=1.):
    """Independent worlds for a paired binary comparison: ``(z_a + z_b)^2 * p_disc / effect^2 * design effect``."""
    if effect <= 0 or not 0 < discordant <= 1:
        raise LearnedOptimizerError('effect must be positive and the discordant share in (0, 1]')
    return math.ceil((alpha_z + power_z) ** 2 * discordant / effect ** 2 * design_effect)


def design_effect(cluster_size, intraclass_correlation):
    return 1 + (cluster_size - 1) * intraclass_correlation


def problem_counts(episodes):
    """Independent problems are programs and binding families, not episodes or checkpoints."""
    programs = {episode['program'] for episode in episodes}
    families = {(episode['program'], episode.get('bindingFamily') or episode['program']) for episode in episodes}
    return {'episodes': len(episodes), 'programs': len(programs), 'bindingFamilies': len(families),
            'programFamilies': len({episode.get('family', episode['program']) for episode in episodes})}


def family_split(families, seed=0, shares=(.7, .15, .15)):
    """Assign whole program families to train/development/confirmation before any collection (~70/15/15)."""
    import hashlib
    if abs(sum(shares) - 1) > 1e-9:
        raise LearnedOptimizerError('split shares must sum to one')
    out = {}
    for family in sorted(set(families)):
        draw = int(hashlib.sha256(f'{seed}:{family}'.encode()).hexdigest()[:12], 16) / 16 ** 12
        out[family] = 'train' if draw < shares[0] else 'development' if draw < shares[0] + shares[1] else 'confirmation'
    return out
