"""S7 GRPO pieces that need no model: advantages, the clipped surrogate's coefficient, KL shaping, replay terms."""
import math

from natlang_neuralese.rl.grpo import clipped_weight, group_advantages, run_round, shaped_reward, turn_terms, usable


def test_group_advantages_are_relative_and_skip_unscored_and_flat_groups():
    values, stats = group_advantages([1.0, 0.0, None, 1.0])
    assert values[2] is None and stats["spread"]
    assert math.isclose(sum(v for v in values if v is not None), 0.0, abs_tol=1e-9)
    assert values[0] > 0 > values[1]
    flat, stats = group_advantages([0.5, 0.5, None])
    assert flat == [0.0, 0.0, None] and not stats["spread"]
    single, stats = group_advantages([None, 1.0])
    assert single == [None, 0.0] and not stats["spread"]


def test_clipped_weight_follows_the_ppo_surrogate():
    assert clipped_weight(1.0, 1.0, 0.2) == (1.0, False)
    assert clipped_weight(1.0, 1.3, 0.2) == (0.0, True)       # good action already made more likely: held
    assert clipped_weight(1.0, 0.7, 0.2) == (0.7, False)      # good action made less likely: still pushed up
    assert clipped_weight(-1.0, 0.7, 0.2) == (0.0, True)      # bad action already made less likely: held
    assert clipped_weight(-1.0, 1.3, 0.2) == (-1.3, False)


def test_kl_shaping_penalises_drift_from_the_reference():
    assert shaped_reward(1.0, -10.0, -10.0, 0.1) == 1.0
    assert shaped_reward(1.0, -8.0, -10.0, 0.1) < 1.0


def test_turns_become_weighted_loglikelihood_terms_and_unscored_rollouts_are_unusable():
    rollout = {"reward": {"quality": 1.0}, "turns": [
        {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}], "target": {"role": "assistant", "content": "x"}},
        {"messages": [{"role": "user", "content": "go"}], "tools": None, "target": {"role": "assistant", "content": "y"}}]}
    terms = turn_terms(rollout, 0.5)
    assert [t["kind"] for t in terms] == ["logLikelihood"] * 2 and all(t["weight"] == 0.5 for t in terms)
    assert "tools" in terms[0] and "tools" not in terms[1]
    assert usable(rollout) and not usable({**rollout, "reward": None}) and not usable({**rollout, "turns": []})


class FakeServer:
    """A server whose policy log-probability of every turn is −1 (adapter) or −1.5 (reference); records requests."""

    def __init__(self):
        self.grads, self.optims = [], []

    def sequence_logps(self, rollouts, adapters, chunk=8):
        return [(-1.0 if adapters else -1.5) * len(r["turns"]) for r in rollouts]

    def post(self, path, body):
        if path == "/v1/neuralese/grad":
            self.grads.append(body)
            return {"loss": sum(t["weight"] for t in body["terms"]), "terms": [], "gradients": {body["arguments"][0]: "g1"}}
        self.optims.append(body)
        return {"params": ["a1"], "state": {"step": 1}}


def test_a_round_weights_turns_by_group_advantage_and_steps_the_adapter():
    turn = {"messages": [{"role": "user", "content": "q"}], "target": {"role": "assistant", "content": "a"}}
    rollouts = [{"group": "t1", "reward": {"quality": q}, "turns": [turn]} for q in (1.0, 0.0)] + \
        [{"group": "t2", "reward": {"quality": 1.0}, "turns": [turn]} for _ in range(2)] + \
        [{"group": "t1", "reward": None, "turns": [turn]}]
    server = FakeServer()
    result = run_round(server, rollouts, "a0", {}, lr=1e-3, beta=0.0, clip=0.2, steps=1)
    report = result["report"]
    assert result["adapter"] == "a1" and report["usable"] == 4 and report["unscored"] == 1
    assert report["groups"] == 2 and report["groups_with_spread"] == 1
    weights = sorted(t["weight"] for t in server.grads[0]["terms"])
    assert len(weights) == 2 and weights[0] < 0 < weights[1]   # only the spread group's two rollouts carry gradient
    assert server.optims[0]["params"] == ["a0"] and server.optims[0]["optimizer"] == "adam"
