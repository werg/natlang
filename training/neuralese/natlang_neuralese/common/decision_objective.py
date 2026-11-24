"""Exact finite decision proper scores shared by service objective backends."""
import math


def decision_loss(scores, *, rule, probabilities=None, rewards=None):
    """One finite-decision equation shared by native and legacy readouts."""
    if rule == 'expectedReward':
        if not isinstance(rewards, list) or len(rewards) != scores.numel() or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in rewards):
            raise ValueError('expectedReward needs finite rewards, one per option')
        from .stable_softmax import softmax
        return -(softmax(scores, 0).to(scores.dtype) * scores.new_tensor(rewards)).sum()
    if not isinstance(probabilities, list) or len(probabilities) != scores.numel() or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in probabilities
            ) or sum(probabilities) <= 0:
        raise ValueError('decision needs nonnegative finite target probabilities with positive total mass')
    return _proper_decision_score(scores, scores.new_tensor(probabilities), rule)


def decision_loss_to_goal(scores, goal, *, rule):
    """Same proper score with an already computed, stopped teacher target."""
    import torch
    if (not torch.is_tensor(goal) or goal.shape != scores.shape or not torch.isfinite(goal).all()
            or (goal < 0).any() or not (goal.sum() > 0)):
        raise ValueError('decision teacher needs finite nonnegative aligned probabilities with positive mass')
    return _proper_decision_score(scores,goal,rule)


def _proper_decision_score(scores,goal,rule):
    # Static public probabilities are already validated on CPU; do not add GPU
    # synchronization to that existing path for the dynamic-teacher mechanism.
    from . import stable_softmax
    goal = goal / goal.sum()
    if rule == 'logLoss':
        if goal.requires_grad or stable_softmax.active_policy() != stable_softmax.POLICY:
            return -(goal * scores.log_softmax(0)).sum()
        # Soft-target CE with the cancellation-free gradient (teacher goals near agreement).
        return stable_softmax.soft_cross_entropy(scores[None], goal.log()[None])[0].to(scores.dtype)
    logp = stable_softmax.log_softmax(scores, 0).to(scores.dtype)
    if rule == 'brier':
        return ((logp.exp() - goal) ** 2).sum()
    if rule == 'rps':
        return ((logp.exp().cumsum(0) - goal.cumsum(0))[:-1] ** 2).sum() / max(1, scores.numel()-1)
    raise ValueError('decision rule must be logLoss, brier, rps or expectedReward')



def option_sequence_scores(sequence_scores, owner, options):
    """Per option the logsumexp of its sequences' log-probabilities (a split-tokenized readout scores an option as all
    the token sequences that write it, ``serve.native_learning.decision_sequences``). ``sequence_scores`` [n] (torch),
    ``owner`` the option index of each sequence, ``options`` the number of options. An option with one sequence keeps
    its score exactly."""
    import torch
    owner = torch.as_tensor(list(owner), dtype=torch.long, device=sequence_scores.device)
    if owner.numel() == options and bool((owner == torch.arange(options, device=owner.device)).all()):
        return sequence_scores
    high = torch.full((options,), float('-inf'), dtype=sequence_scores.dtype, device=sequence_scores.device)
    high = high.scatter_reduce(0, owner, sequence_scores.detach(), 'amax')
    mass = torch.zeros_like(high).index_add(0, owner, (sequence_scores - high[owner]).exp())
    return high + mass.log()


def sequence_owner(options, alternatives):
    """The option index of each sequence of ``options`` canonical continuations followed by their ``alternatives``
    (per option a list), the order the scorers and learners lay them out."""
    return list(range(options)) + [index for index, alts in enumerate(alternatives or ()) for _ in alts]
