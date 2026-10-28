"""Decision-readout supervision: the choice of a natlang ``decide`` call is the model's own output.

A decision is scored, not generated: the runtime reads each option as the complete closed assistant continuation
and returns softmax over the option sequence log-probabilities (``serve.native_learning._score_rendered``). The
objective supervises exactly that readout distribution with the log score over the option set:

    s_o = sum_t log p(c_{o,t} | prompt, c_{o,<t})        (end of message included, as served)
    L   = -sum_o y_o log softmax(s)_o                     (y: one-hot exact/verified label, or its distribution)

Why the option-normalized CE and not the per-option token CE of the correct continuation: the runtime's answer and
its reported confidence are softmax(s); the option CE is the proper log score of that actual output. Its gradient
sum_o (softmax(s)_o - y_o) grad s_o raises the correct option *and* lowers the competing ones, and is invariant to
what the readout ignores (the shared absolute likelihood of all replies, e.g. a common end-of-message token or a
longer JSON spelling). The token CE of the correct continuation alone cannot lower a wrongly preferred option and
spends capacity on the absolute likelihood of an answer format the decision never generates. A decision counts as
one invocation of the own-output objective (mass 1, like an invocation's decisions), and gradients flow through the
prompt, its imported Neuralese spans and their in-graph producers in the same packed graph forward.
"""
from __future__ import annotations

import torch

from ..common.decision_objective import decision_loss
from ..data.decision_graph import option_actions

RULE = 'logLoss'


def option_scores(model, hidden, options, *, chunk_tokens=128, readout=None):
    """Sequence log-probabilities of each option's continuation (actions in ``hidden`` coordinates)."""
    from ..rl.native_replay import token_action_logps
    flat = [action for option in options for action in option]
    if not flat or any(not option for option in options):
        raise ValueError('every decision option needs its own continuation tokens')
    logps = token_action_logps(model, hidden, flat, chunk_tokens=chunk_tokens, readout=readout)
    return torch.stack([part.sum() for part in logps.split([len(option) for option in options])])


def physical_options(example, decision, physical_rows=None):
    """Option actions of ``decision`` in packed physical coordinates (``physical_rows``: the graph's flat -> physical
    row map after prefix sharing; ``None``: the example's own flat coordinates)."""
    options = option_actions(example, decision)
    if physical_rows is None:
        return options
    return [[{**action, 'query': physical_rows[action['query']]} for action in option] for option in options]


def decision_terms(model, hidden, example, *, physical_rows=None, chunk_tokens=128):
    """(sum of per-decision option CE, number of decisions, device count of decisions whose argmax is the target's).

    No host synchronization: the trainer resolves its telemetry in one transfer."""
    total = hidden.new_zeros((), dtype=torch.float32)
    correct = hidden.new_zeros((), dtype=torch.float32)
    for decision in example.decisions:
        scores = option_scores(model, hidden, physical_options(example, decision, physical_rows),
                               chunk_tokens=chunk_tokens)
        total = total + decision_loss(scores, rule=RULE, probabilities=list(decision.target))
        best = max(range(len(decision.target)), key=decision.target.__getitem__)
        correct = correct + (scores.detach().argmax() == best).float()
    return total, len(example.decisions), correct
