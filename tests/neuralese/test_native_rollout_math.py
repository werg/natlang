"""Pure reward normalization shared by whole-graph native RL."""
import math

from natlang_neuralese.rl.rollout_math import group_advantages


def test_group_advantages_are_relative_and_skip_unscored_and_flat_groups():
    values, stats = group_advantages([1.0, 0.0, None, 1.0])
    assert values[2] is None and stats["spread"]
    assert math.isclose(sum(v for v in values if v is not None), 0.0, abs_tol=1e-9)
    assert values[0] > 0 > values[1]
    flat, stats = group_advantages([0.5, 0.5, None])
    assert flat == [0.0, 0.0, None] and not stats["spread"]
    single, stats = group_advantages([None, 1.0])
    assert single == [None, 0.0] and not stats["spread"]
