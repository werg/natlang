"""Pure group statistics for scored native task rollouts."""
from __future__ import annotations

import math


def group_advantages(rewards: list[float | None], eps: float = 1e-6) -> tuple[list[float | None], dict]:
    """Group-relative advantages of one task's rollouts; None for unscored rollouts. A group whose scored rewards
    have no spread gets all-zero advantages."""
    scored = [r for r in rewards if r is not None]
    if len(scored) < 2:
        return [None if r is None else 0.0 for r in rewards], {"scored": len(scored), "spread": False}
    mean = sum(scored) / len(scored)
    std = math.sqrt(sum((r - mean) ** 2 for r in scored) / len(scored))
    if std < eps:
        return [None if r is None else 0.0 for r in rewards], {"scored": len(scored), "spread": False, "mean": mean}
    return [None if r is None else (r - mean) / (std + eps) for r in rewards], \
        {"scored": len(scored), "spread": True, "mean": mean, "std": std}
