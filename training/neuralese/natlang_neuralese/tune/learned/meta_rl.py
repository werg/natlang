"""Reward learning for the complete adapted program (plan "Meta-RL", "Dense and contrastive reward").

Reward is the oracle's paired improvement over the incoming bundle on query worlds the proposal never saw, minus
declared costs. The shaping term (mean log-likelihood change of query actions the task contract makes uniquely
correct) is reported as its own component and enters the objective only under its declared weight.

Estimators:

* `filtered-bc-v1`: imitate proposals whose paired reward clears a threshold, with their in-bank refinements.
* `expected-score-v1` (`trainer.OuterTrainer.policy_step`): the optimizer's action space is finite (the component
  tokens of each scope, null included), so the expected score ``sum_k p_k R_k`` over the scored kept tokens is
  differentiated exactly (`mixture.expected_score_loss`); ``p`` is recomputed under current parameters from the same
  context, view and mask. No score-function sampling and no importance ratio.

Leave-one-out advantages over a context's proposals (never the scored proposal's own reward) are kept as the recorded
contrast signal (`advantage_snr`).
"""
from __future__ import annotations

import math

import torch

from . import LearnedOptimizerError

ESTIMATORS = ('filtered-bc-v1', 'expected-score-v1')


def paired_reward(quality_new, quality_before, *, adaptation_cost=0., execution_cost=0., lambda_adapt=0.,
                  lambda_run=0., shaping=None, lambda_shape=0.):
    """Components kept separately visible; ``total`` is what the estimator uses."""
    for value in (quality_new, quality_before, adaptation_cost, execution_cost):
        if not math.isfinite(value):
            raise LearnedOptimizerError('reward components must be finite')
    task = quality_new - quality_before
    total = task - lambda_adapt * adaptation_cost - lambda_run * execution_cost
    if shaping is not None:
        total += lambda_shape * shaping
    return {'task': task, 'adaptationCost': adaptation_cost, 'executionCost': execution_cost,
            'shaping': shaping, 'weights': {'adapt': lambda_adapt, 'run': lambda_run, 'shape': lambda_shape}, 'total': total}


def shaping_term(new_logps, before_logps, unique):
    """Mean change in log-likelihood of uniquely correct query actions (a boolean mask excludes actions with valid
    alternatives, under the reward-validity rule)."""
    mask = torch.as_tensor(unique, dtype=torch.bool)
    if not bool(mask.any()):
        return None
    return float((torch.as_tensor(new_logps)[mask] - torch.as_tensor(before_logps)[mask]).mean())


def leave_one_out_advantages(rewards):
    """Advantages against the mean of the other proposals of the same context; a context with one proposal or no
    spread contributes none (recorded honestly, never invented variance)."""
    if len(rewards) < 2:
        return [0.] * len(rewards), {'proposals': len(rewards), 'contrast': False}
    total = sum(rewards)
    advantages = [reward - (total - reward) / (len(rewards) - 1) for reward in rewards]
    spread = max(rewards) - min(rewards)
    return advantages, {'proposals': len(rewards), 'contrast': spread > 0, 'spread': spread}


def advantage_snr(groups):
    """Signal-to-noise of contrasting groups' advantages: ``|mean group spread| / std`` (reported per round)."""
    spreads = [info['spread'] for _, info in groups if info.get('contrast')]
    if len(spreads) < 2:
        return {'groups': len(groups), 'contrasting': len(spreads), 'snr': 0.}
    mean = sum(spreads) / len(spreads)
    std = math.sqrt(sum((s - mean) ** 2 for s in spreads) / (len(spreads) - 1))
    return {'groups': len(groups), 'contrasting': len(spreads), 'snr': mean / std if std > 0 else math.inf}


def filtered_bc(proposals, threshold):
    """Tier-3 targets: proposals whose paired reward clears ``threshold``; each keeps its own complete tuple."""
    if not math.isfinite(threshold):
        raise LearnedOptimizerError('the filtered-BC threshold must be finite')
    return [proposal for proposal in proposals if proposal['reward']['total'] >= threshold]


def returns_to_go(gains):
    """Stepwise credit for a multi-step episode from per-step paired gains, without double counting: each step's
    return is the sum of its own and later gains (the terminal total equals the sum of the gains)."""
    out, running = [], 0.
    for gain in reversed(gains):
        running += gain
        out.append(running)
    return out[::-1]


def check_replay(record, current):
    """Replay compatibility: same view, mask, tokenizer and native layout; otherwise recollect."""
    for key in ('view', 'mask', 'tokens', 'tokenizer', 'layout', 'graph'):
        if record.get(key) != current.get(key):
            raise LearnedOptimizerError(f'replay differs in {key}; this rollout needs fresh collection')
    return True
