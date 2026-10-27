"""Reward learning for the complete adapted program (plan "Meta-RL", "Dense and contrastive reward").

Reward is the oracle's paired improvement over the incoming bundle on query worlds the proposal never saw, minus
declared costs. The shaping term (mean log-likelihood change of query actions the task contract makes uniquely
correct) is reported as its own component and enters the objective only under its declared weight.

Estimators:

* `filtered-bc-v1` (default start): imitate proposals whose paired reward clears a threshold, with their in-bank
  refinements. Low variance; no score function.
* `joint-ratio-v1`: ``r = pi(z|x)/pi_old(z|x) * p_exec(tau|A(x,z))/p_exec(tau|A_old(x,z))``. The executor factor is
  recomputed under the bundle regenerated from the same recorded optimizer actions (same tokens, mask and view; the
  coefficient head, keys and bank at current parameters). The clipped surrogate is the shared
  `rl.policy_math.clipped_token_loss` applied to the summed joint log-ratio.

The baseline depends only on the incoming context: a leave-one-out mean over the other proposals of the same
context (never the scored proposal's own reward).
"""
from __future__ import annotations

import math

import torch

from ...rl.policy_math import clipped_token_loss
from . import LearnedOptimizerError

ESTIMATORS = ('filtered-bc-v1', 'joint-ratio-v1')


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
    """Signal-to-noise of contrasting groups' advantages: ``|mean group spread| / std``; the schedule switches from
    filtered BC to the joint-ratio estimator as this measured signal supports it."""
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


def joint_log_ratio(optimizer_logps, old_optimizer_logps, executor_logps, old_executor_logps):
    """Summed log-ratio of every trained stochastic action: the optimizer's free tokens (numbers included) and the
    executor's actions under the regenerated bundle. Forced tokens are excluded by their callers (no sampling score)."""
    for a, b in ((optimizer_logps, old_optimizer_logps), (executor_logps, old_executor_logps)):
        if a.shape != b.shape:
            raise LearnedOptimizerError('replayed actions differ from the recorded ones; recollect instead of reusing')
    return (optimizer_logps.float() - old_optimizer_logps.float()).sum() + (executor_logps.float() - old_executor_logps.float()).sum()


def joint_ratio_loss(log_ratio, advantage, *, clip_low=.2, clip_high=.28):
    """The shared clipped surrogate applied once to the whole joint action (`joint-ratio-v1`)."""
    return clipped_token_loss(log_ratio.reshape(1), torch.zeros(1, device=log_ratio.device), advantage,
                              clip_low=clip_low, clip_high=clip_high)[0]


def score_function_loss(optimizer_logps, executor_logps, advantage):
    """On-policy ``-(R - b)(log pi(z|x) + log p_exec(tau|A_theta(x,z)))``; the advantage is detached by construction."""
    return -float(advantage) * (optimizer_logps.float().sum() + executor_logps.float().sum())


def returns_to_go(gains):
    """Stepwise credit for a multi-step episode from per-step paired gains, without double counting: each step's
    return is the sum of its own and later gains (the terminal total equals the sum of the gains)."""
    out, running = [], 0.
    for gain in reversed(gains):
        running += gain
        out.append(running)
    return out[::-1]


def check_replay(record, current):
    """Replay compatibility: same view, mask, action spellings, tokenizer and native layout; otherwise recollect."""
    for key in ('view', 'mask', 'tokens', 'tokenizer', 'layout', 'graph'):
        if record.get(key) != current.get(key):
            raise LearnedOptimizerError(f'replay differs in {key}; this rollout needs fresh collection')
    return True
