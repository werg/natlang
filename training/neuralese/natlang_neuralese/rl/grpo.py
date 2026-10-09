"""Group-relative policy optimisation over recorded natlang rollouts (S7 §4.1, §5.4), served end to end over HTTP.

Rollouts come from ts-host/scripts/rl/rollout-episodes.mjs (natlang.rollout/1): every model turn of a natlang
program run, recorded from the wire, and the episode scorer's reward. The policy is an adapter block on a model
server that serves `/v1/neuralese/grad` and `/v1/neuralese/optim` (the reference server), so the same code trains
any backbone that server loads. One round:

1. Behaviour log-probabilities: each rollout's turns scored as `logLikelihood` terms under the adapter the rollouts
   were sampled with (a grad request without arguments; at sampling temperature 1 this is the sampling policy, up to
   guided-generation retries, which the clipped ratio absorbs).
2. Reference log-probabilities: the same turns without the adapter (the frozen pre-RL model). The reward is shaped
   by a sequence-level KL estimate, r − β·(log π − log π_ref) (S7 §4.1, KL on the policy's own tokens).
3. Group-relative advantages: per task, (r − mean) / (std + ε) over the scored rollouts; groups without spread give
   no gradient and are logged for the curriculum; unscored rollouts (infrastructure failures) are excluded, never
   rewarded 0.
4. Updates: for each step, the current log-probabilities give sequence importance ratios ρ = exp(log π − log π_b); the
   clipped surrogate's gradient is A·ρ·∇log π where the unclipped term is active and 0 where it is clipped, so each
   turn of rollout i becomes a `logLikelihood` term (loss −log π) with weight A_i·ρ_i/N. The adapter's gradient goes
   through `/v1/neuralese/optim` (Adam); the new adapter block is the next round's policy.

    python -m natlang_neuralese.rl.grpo round --server URL --rollouts R.jsonl --adapter ID [--state S.json] --out DIR
    python -m natlang_neuralese.rl.grpo init --server URL --kind xs --rank 8 --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

SCHEMA_ROUND = "natlang.grpo-round/1"


# Pure parts (tested without a model) ----------------------------------------------------------------------------
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


def clipped_weight(advantage: float, ratio: float, clip: float) -> tuple[float, bool]:
    """Gradient coefficient of the PPO surrogate min(ρA, clip(ρ)A) with respect to log π: A·ρ where the unclipped
    term is the active one, 0 where clipping holds it (ρ above 1+clip for A > 0, below 1−clip for A < 0)."""
    if (advantage > 0 and ratio > 1 + clip) or (advantage < 0 and ratio < 1 - clip):
        return 0.0, True
    return advantage * ratio, False


def shaped_reward(quality: float, logp: float, logp_ref: float, beta: float) -> float:
    """Reward with the sequence-level KL penalty β·(log π − log π_ref) on the rollout's own tokens."""
    return quality - beta * (logp - logp_ref)


def turn_terms(rollout: dict, weight: float = 1.0) -> list[dict]:
    """The rollout's model turns as `logLikelihood` replay terms (loss −log π of the recorded assistant message)."""
    terms = []
    for turn in rollout.get("turns") or []:
        term = {"kind": "logLikelihood", "messages": turn["messages"], "target": turn["target"], "weight": weight}
        if turn.get("tools"):
            term["tools"] = turn["tools"]
        terms.append(term)
    return terms


def usable(rollout: dict) -> bool:
    return rollout.get("reward") is not None and bool(rollout.get("turns"))


# Server client ---------------------------------------------------------------------------------------------------
class Server:
    def __init__(self, base: str, timeout: float = 3600):
        self.base, self.timeout = base.rstrip("/"), timeout

    def post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def sequence_logps(self, rollouts: list[dict], adapters: list | None, chunk: int = 8) -> list[float]:
        """Σ over each rollout's turns of log π(turn) under `adapters` (None: the base model)."""
        out = []
        for rollout in rollouts:
            terms, total = turn_terms(rollout), 0.0
            for start in range(0, len(terms), chunk):
                body = {"arguments": [], "terms": terms[start:start + chunk], **({"adapters": adapters} if adapters else {})}
                total -= sum(self.post("/v1/neuralese/grad", body)["terms"])
            out.append(total)
        return out


def load_rollouts(path: str | Path, round_: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in rows if round_ is None or r.get("round") == round_]


def run_round(server: Server, rollouts: list[dict], adapter: str, state: dict, *, lr: float, beta: float, clip: float,
              steps: int, scale: float = 1.0) -> dict:
    """One GRPO round on `rollouts` (sampled with `adapter`). Returns the new adapter, optimiser state and a report."""
    started = time.time()
    policy = [{"id": adapter, "scale": scale}]
    kept = [r for r in rollouts if usable(r)]
    report = {"schema": SCHEMA_ROUND, "adapter_in": adapter, "rollouts": len(rollouts), "usable": len(kept),
              "unscored": sum(1 for r in rollouts if r.get("reward") is None), "beta": beta, "clip": clip, "lr": lr}
    if not kept:
        return {"adapter": adapter, "state": state, "report": {**report, "skipped": "no usable rollouts"}}
    behaviour = server.sequence_logps(kept, policy)
    reference = server.sequence_logps(kept, None)
    shaped = [shaped_reward(r["reward"]["quality"], lp, ref, beta) for r, lp, ref in zip(kept, behaviour, reference)]
    groups: dict[str, list[int]] = {}
    for index, rollout in enumerate(kept):
        groups.setdefault(rollout["group"], []).append(index)
    advantages = [0.0] * len(kept)
    spread = 0
    for members in groups.values():
        values, stats = group_advantages([shaped[i] for i in members])
        spread += stats["spread"]
        for i, value in zip(members, values):
            advantages[i] = value or 0.0
    report.update({"groups": len(groups), "groups_with_spread": spread,
                   "reward_mean": sum(r["reward"]["quality"] for r in kept) / len(kept),
                   "kl_estimate_mean": sum(b - r for b, r in zip(behaviour, reference)) / len(kept),
                   "steps": []})
    current_adapter, current_state = adapter, dict(state)
    active = [i for i, a in enumerate(advantages) if a != 0.0]
    for step in range(steps):
        if not active:
            report["steps"].append({"step": step, "skipped": "no advantage"})
            break
        now = behaviour if step == 0 else server.sequence_logps([kept[i] for i in range(len(kept))],
                                                                [{"id": current_adapter, "scale": scale}])
        terms, clipped, ratios = [], 0, []
        for i in active:
            ratio = math.exp(max(-20.0, min(20.0, now[i] - behaviour[i])))
            weight, was_clipped = clipped_weight(advantages[i], ratio, clip)
            ratios.append(ratio)
            clipped += was_clipped
            if weight:
                terms += turn_terms(kept[i], weight / len(active))
        if not terms:
            report["steps"].append({"step": step, "skipped": "all clipped", "clipped": clipped})
            break
        grad = server.post("/v1/neuralese/grad", {"arguments": [current_adapter], "terms": terms,
                                                  "adapters": [{"id": current_adapter, "scale": scale}]})
        stepped = server.post("/v1/neuralese/optim", {"optimizer": "adam", "hyper": {"lr": lr}, "params": [current_adapter],
                                                      "grads": [grad["gradients"][current_adapter]], "state": current_state})
        current_adapter, current_state = stepped["params"][0], stepped["state"]
        report["steps"].append({"step": step, "surrogate_loss": grad["loss"], "terms": len(terms), "clipped": clipped,
                                "ratio_min": min(ratios), "ratio_max": max(ratios), "adapter": current_adapter})
    report.update({"adapter_out": current_adapter, "seconds": time.time() - started})
    return {"adapter": current_adapter, "state": current_state, "report": report}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init", help="a zero adapter on the server: the policy before RL equals the reference")
    i.add_argument("--server", required=True)
    i.add_argument("--kind", default="xs")
    i.add_argument("--rank", type=int, default=8)
    i.add_argument("--out", required=True)
    r = sub.add_parser("round")
    r.add_argument("--server", required=True)
    r.add_argument("--rollouts", required=True)
    r.add_argument("--round", type=int, default=None)
    r.add_argument("--adapter", required=True, help="adapter block the rollouts were sampled with")
    r.add_argument("--state", default=None, help="optimiser state JSON from the previous round")
    r.add_argument("--out", required=True)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--beta", type=float, default=0.02)
    r.add_argument("--clip", type=float, default=0.2)
    r.add_argument("--steps", type=int, default=1)
    a = p.parse_args(argv)
    server = Server(a.server)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.command == "init":
        block = server.post("/v1/neuralese/adapters", {"kind": a.kind, "rank": a.rank})
        (out / "adapter.json").write_text(json.dumps({"adapter": block["id"], "state": {}, "init": block}, indent=1) + "\n")
        print(json.dumps({"adapter": block["id"]}))
        return
    state = json.loads(Path(a.state).read_text())["state"] if a.state else {}
    result = run_round(server, load_rollouts(a.rollouts, a.round), a.adapter, state, lr=a.lr, beta=a.beta, clip=a.clip,
                       steps=a.steps)
    (out / "adapter.json").write_text(json.dumps({"adapter": result["adapter"], "state": result["state"]}, indent=1) + "\n")
    with (out / "rounds.jsonl").open("a") as log:
        log.write(json.dumps({**result["report"], "round": a.round}) + "\n")
    print(json.dumps(result["report"]))


if __name__ == "__main__":
    main()
