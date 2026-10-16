"""Exact finite decision proper scores shared by service objective backends."""
import math


def decision_loss(scores, *, rule, probabilities=None, rewards=None):
    """One finite-decision equation shared by native and legacy readouts."""
    if rule == 'expectedReward':
        if not isinstance(rewards, list) or len(rewards) != scores.numel() or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in rewards):
            raise ValueError('expectedReward needs finite rewards, one per option')
        return -(scores.softmax(0) * scores.new_tensor(rewards)).sum()
    if not isinstance(probabilities, list) or len(probabilities) != scores.numel() or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in probabilities
            ) or sum(probabilities) <= 0:
        raise ValueError('decision needs nonnegative finite target probabilities with positive total mass')
    logp = scores.log_softmax(0)
    goal = scores.new_tensor(probabilities)
    goal = goal / goal.sum()
    if rule == 'logLoss':
        return -(goal * logp).sum()
    if rule == 'brier':
        return ((logp.exp() - goal) ** 2).sum()
    if rule == 'rps':
        return ((logp.exp().cumsum(0) - goal.cumsum(0))[:-1] ** 2).sum() / max(1, scores.numel()-1)
    raise ValueError('decision rule must be logLoss, brier, rps or expectedReward')

