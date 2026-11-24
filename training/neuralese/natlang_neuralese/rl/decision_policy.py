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
    alternatives = capture.get('continuation_alternatives')
    if alternatives is not None and (
            not isinstance(alternatives, list) or len(alternatives) != len(continuations)
            or any(not isinstance(alts, list) or any(not isinstance(ids, list) or not ids
                                                     or any(type(t) is not int or t < 0 for t in ids) for ids in alts)
                   for alts in alternatives)):
        raise ValueError('continuation_alternatives holds one list of nonempty id sequences per option')
    # a split-tokenized readout: the served score of an option is the logsumexp over its canonical continuation and
    # these alternatives (serve.native_learning.decision_sequences); the learner scores the same sequences
    return {'continuations': [list(ids) for ids in continuations], 'chosen': chosen,
            'alternatives': [[list(ids) for ids in alts] for alts in alternatives or [[] for _ in continuations]],
            'temperature': float(temperature), 'behavior_logps': behavior_log_probs(log_probs, temperature)}


def remap_decision(action, rows):
    """``action`` with its option (and split-tokenization alternative, and twin alternative) queries mapped through
    ``rows`` (flat -> packed physical rows)."""
    remap = lambda options: [[{**item, 'query': rows[item['query']]} for item in option] for option in options]
    return {**action, 'options': remap(action['options']),
            **({'alternatives': [remap(alts) for alts in action['alternatives']]} if 'alternatives' in action else {}),
            **({'twin_alternatives': [remap(alts) for alts in action['twin_alternatives']]}
               if 'twin_alternatives' in action else {})}


def decision_log_policy(model, hidden, action, *, chunk_tokens=128, queries=None, alternative_queries=None):
    """log pi_theta over the decision's options (temperature-scaled softmax of the option scores); option queries
    are rows of ``hidden`` (the caller maps original-token coordinates through its packed physical-row map,
    ``remap_decision``). A split-tokenized readout's option score is the logsumexp over its canonical continuation and
    its ``alternatives`` (``alternative_queries`` with ``queries``), as the engines served it."""
    from ..common.decision_objective import option_sequence_scores, sequence_owner
    from ..train.decision_supervision import option_scores
    options = action['options'] if queries is None else queries
    alternatives = (action.get('alternatives') if queries is None else alternative_queries) or ()
    if any(alternatives):
        sequences = option_scores(model, hidden, [*options, *(a for alts in alternatives for a in alts)],
                                  chunk_tokens=chunk_tokens)
        scores = option_sequence_scores(sequences, sequence_owner(len(options), alternatives), len(options))
    else:
        scores = option_scores(model, hidden, options, chunk_tokens=chunk_tokens)
    return torch.log_softmax(scores.float() / action['temperature'], 0)


def decision_policy_terms(model, hidden, actions, *, advantage, mode, policy_loss, chunk_tokens=128,
                          twin=None, beta=0., student=True):
    """Per-decision loss terms [n] (same sign convention as the token terms) and summed KL telemetry.

    ``twin``: per action, its options' twin-view queries (JOLT+ privileged twin) or None; ``student``: the episode's
    JOLT+ role is student (its twin term takes the declaration's ``student_kl`` direction)."""
    from .policy_math import clipped_token_loss, score_centered_surrogate
    terms, kl_sum = [], 0.
    for index, action in enumerate(actions):
        log_pi = decision_log_policy(model, hidden, action, chunk_tokens=chunk_tokens)
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
                from .jolt import twin_kl
                with torch.no_grad():
                    twin_log = decision_log_policy(model, hidden.detach(), action, queries=queries,
                                                   alternative_queries=action.get('twin_alternatives'),
                                                   chunk_tokens=chunk_tokens)
                # a student's twin term in the declared direction (jolt-plus/1 student_kl); teachers keep reverse
                direction = policy_loss.get('student_kl', 'reverse') if student else 'reverse'
                kl = twin_kl(direction)(log_pi[None], twin_log[None])[0]
                term = term + beta * kl
                kl_sum += float(kl.detach())
        terms.append(term)
    if not terms:
        return hidden.new_zeros((0,), dtype=torch.float32), 0.
    return torch.stack(terms), kl_sum
