"""Decision readouts as RL actions over their finite option set.

A ``decide`` readout scores every option s_o (the served sequence log-probability of the closed continuation) and
the runtime answers from softmax(s). During RL collection the readout *samples* its answer from
q = softmax(s / T) (``natlang.decision-sample/1``, recorded in the decision capture) instead of taking the argmax.

Why sampling at collection rather than an exact expected-score gradient over all options: the episode reward depends
on the whole execution that follows the chosen option, and the counterfactual outcome of the other options is
unknown, so E_pi[R] over options cannot be evaluated exactly from one rollout. Argmax collection would make every
member of a group take the same option for the same prompt (zero variance, no learning signal for the choice).
Sampling gives an unbiased score-function gradient, and because the option set is finite and fully scored, the
behaviour law is known exactly: the score-centered estimator runs with H = all options (no tail, exact centering).

The current policy is pi_theta(o) = softmax(s_theta / T)_o with s_theta recomputed in the same packed graph forward
as every other action, so gradients reach the prompt, its imports and their in-graph producers. Losses:

* score-centered/1 (and the JOLT+ outcome term): -A U with U from ``policy_math.score_centered_surrogate`` over the
  option distribution;
* clipped GRPO: ``policy_math.clipped_token_loss`` on log pi(chosen) against the recorded log q(chosen);
* RFT: -log pi(chosen);
* JOLT+ privileged term: beta KL(pi || sg pi~), pi~ the same readout in the privileged twin graph.

An argmax readout (no ``decision_sample``) has no behaviour density: it stays context in the episode graph.
"""
from __future__ import annotations

import math

import torch

SAMPLE_SCHEMA = 'natlang.decision-sample/1'


def sample_receipt(log_probs, *, chosen, temperature, seed=None):
    """The collector's exact sampling receipt (what ``decision_sample`` holds)."""
    return {'schema': SAMPLE_SCHEMA, 'log_probs': [float(value) for value in log_probs], 'chosen': chosen,
            'temperature': float(temperature), **({'seed': seed} if seed is not None else {})}


def behavior_log_probs(log_probs, temperature):
    scores = torch.tensor(log_probs, dtype=torch.float64) / temperature
    return [float(value) for value in torch.log_softmax(scores, 0)]


def validate_decision_sample(capture):
    """Validate a decision capture's sampling receipt; returns continuations, chosen, T and behaviour log q."""
    receipt = capture.get('decision_sample')
    continuations = capture.get('continuations')
    if (not isinstance(receipt, dict) or receipt.get('schema') != SAMPLE_SCHEMA
            or set(receipt) - {'schema', 'log_probs', 'chosen', 'temperature', 'seed'}):
        raise ValueError('decision sample receipt must be natlang.decision-sample/1')
    if (not isinstance(continuations, list) or len(continuations) < 2
            or any(not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids)
                   for ids in continuations)
            or len({tuple(ids) for ids in continuations}) != len(continuations)):
        raise ValueError('sampled decision needs at least two distinct nonempty option continuations')
    log_probs, chosen, temperature = receipt.get('log_probs'), receipt.get('chosen'), receipt.get('temperature')
    if (not isinstance(log_probs, list) or len(log_probs) != len(continuations)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in log_probs)):
        raise ValueError('decision sample needs one finite served score per option')
    if type(chosen) is not int or not 0 <= chosen < len(continuations):
        raise ValueError('decision sample chosen option is out of range')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('decision sample temperature must be finite and positive')
    return {'continuations': [list(ids) for ids in continuations], 'chosen': chosen,
            'temperature': float(temperature), 'behavior_logps': behavior_log_probs(log_probs, temperature)}


def _shifted(options, offset, key='query'):
    return [[{**action, 'query': action[key] + offset} for action in option] for option in options]


def decision_log_policy(model, hidden, action, *, offset=0, chunk_tokens=128, queries=None):
    """log pi_theta over the decision's options (temperature-scaled softmax of the option scores)."""
    from ..train.decision_supervision import option_scores
    options = _shifted(action['options'], offset) if queries is None else queries
    scores = option_scores(model, hidden, options, chunk_tokens=chunk_tokens)
    return torch.log_softmax(scores.float() / action['temperature'], 0)


def decision_policy_terms(model, hidden, actions, *, advantage, mode, policy_loss, offset=0, chunk_tokens=128,
                          twin=None, beta=0.):
    """Per-decision loss terms [n] (same sign convention as the token terms) and summed KL telemetry.

    ``twin``: per action, its options' twin-view queries (JOLT+ privileged twin) or None."""
    from .policy_math import clipped_token_loss, score_centered_surrogate
    terms, kl_sum = [], 0.
    for index, action in enumerate(actions):
        log_pi = decision_log_policy(model, hidden, action, offset=offset, chunk_tokens=chunk_tokens)
        chosen = action['chosen']
        behavior = torch.tensor(action['behavior_logps'], dtype=torch.float32, device=hidden.device)
        if mode == 'rft':
            term = -log_pi[chosen]
        elif policy_loss is None:
            term = clipped_token_loss(log_pi[chosen][None], behavior[chosen][None], advantage)[0]
        else:
            outcome = policy_loss['outcome'] if policy_loss['schema'] == 'jolt-plus/1' else policy_loss
            ids = torch.arange(len(behavior), device=hidden.device)[None]
            surrogate, _ = score_centered_surrogate(log_pi[None], torch.tensor([chosen], device=hidden.device),
                behavior[chosen][None], ids, behavior[None], importance=outcome['importance'], eps=outcome['eps'],
                masked=False)
            term = -float(advantage) * surrogate[0]
            queries = (twin or [None] * len(actions))[index]
            if policy_loss['schema'] == 'jolt-plus/1' and queries is not None and beta:
                from .jolt import reverse_kl
                with torch.no_grad():
                    twin_log = decision_log_policy(model, hidden.detach(), action, queries=_shifted(queries, offset),
                                                   chunk_tokens=chunk_tokens)
                kl = reverse_kl(log_pi[None], twin_log[None])[0]
                term = term + beta * kl
                kl_sum += float(kl.detach())
        terms.append(term)
    if not terms:
        return hidden.new_zeros((0,), dtype=torch.float32), 0.
    return torch.stack(terms), kl_sum
