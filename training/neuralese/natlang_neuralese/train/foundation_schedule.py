"""Pure-Python plateau-gated projection-first foundation schedule.

The shallow and full-depth feedback projections bootstrap against fixed gold
embedding targets. Full-stack adaptation starts only after both held metrics
plateau; the plateau records its errors and trend but is not a qualification.
"""
from __future__ import annotations

import copy
import math


class ProjectionFirstSchedule:
    """Track held projection errors and return the schedule for the next update.

    ``observe`` is called once per held evaluation with relative-MSE values for
    both named projections. Plateau requires the configured minimum number of
    observations and ``patience`` observations since that head's last
    significant relative improvement. Once both have plateaued, the next
    update can adapt the transformer; its LR and sequence-pass count ramp by
    subsequent held evaluations. Projection LR scale remains one throughout.

    The schedule never imposes a maximum-step promotion. If either projection
    keeps improving, the caller remains in projection-only bootstrap.
    """

    SCHEMA = "natlang.projection-first-schedule/2"

    def __init__(self, *, heads=("shallow", "full_depth"), min_evals=2,
                 patience=3, min_relative_improvement=0.01,
                 backbone_ramp_evals=4, pass_ramp_evals=2, max_sequence_passes=3):
        heads = tuple(heads)
        if not heads or len(set(heads)) != len(heads) or any(not isinstance(h, str) or not h for h in heads):
            raise ValueError("heads must be unique nonempty names")
        if min_evals < 1 or patience < 1 or backbone_ramp_evals < 1 or pass_ramp_evals < 1:
            raise ValueError("evaluation and ramp counts must be positive")
        if type(max_sequence_passes) is not int or max_sequence_passes < 3:
            raise ValueError("max_sequence_passes must be an integer of at least 3")
        if not math.isfinite(min_relative_improvement) or min_relative_improvement < 0:
            raise ValueError("minimum relative improvement must be finite and nonnegative")
        self.config = {
            "heads": list(heads), "min_evals": int(min_evals), "patience": int(patience),
            "min_relative_improvement": float(min_relative_improvement),
            "backbone_ramp_evals": int(backbone_ramp_evals), "pass_ramp_evals": int(pass_ramp_evals),
            "max_sequence_passes": max_sequence_passes,
        }
        self.eval_count = 0
        self.head_state = {head: {"best": None, "last_significant_eval": 0,
                                  "history": [], "plateau": None} for head in heads}
        self.adaptation_started_eval = None
        self.depth_ramp_origin_eval = None

    @property
    def plateau_reached(self):
        return self.adaptation_started_eval is not None

    def _recent_slope(self, history):
        values = history[-(self.config["patience"] + 1):]
        if len(values) < 2:
            return 0.0
        x_mean = (len(values) - 1) / 2
        y_mean = sum(values) / len(values)
        denominator = sum((i - x_mean) ** 2 for i in range(len(values)))
        return sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values)) / denominator

    def observe(self, held_relative_mse):
        """Observe one held evaluation and return schedule controls for next update."""
        if not isinstance(held_relative_mse, dict) or set(held_relative_mse) != set(self.config["heads"]):
            raise ValueError(f"held_relative_mse must contain exactly {self.config['heads']}")
        errors = {}
        for head in self.config["heads"]:
            value = held_relative_mse[head]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid held relative-MSE for {head}")
            errors[head] = float(value)

        self.eval_count += 1
        for head, value in errors.items():
            state = self.head_state[head]
            state["history"].append(value)
            best = state["best"]
            if best is None:
                state["best"] = value
                state["last_significant_eval"] = self.eval_count
            elif (self.adaptation_started_eval is None and best > 0 and value < best and
                  (best - value) / best >= self.config["min_relative_improvement"]):
                state["best"] = value
                state["last_significant_eval"] = self.eval_count
                state["plateau"] = None
            elif self.adaptation_started_eval is None and best == 0 and value < best:
                # Nonnegative errors cannot improve on zero, retained for clarity.
                state["best"] = value
                state["last_significant_eval"] = self.eval_count
                state["plateau"] = None
            stale = self.eval_count - state["last_significant_eval"]
            if (self.adaptation_started_eval is None and state["plateau"] is None
                    and self.eval_count >= self.config["min_evals"]
                    and stale >= self.config["patience"]):
                state["plateau"] = {
                    "eval": self.eval_count,
                    "error": value,
                    "slope": self._recent_slope(state["history"]),
                    "best_error": state["best"],
                    "stale_evals": stale,
                }

        if self.adaptation_started_eval is None and all(
                self.head_state[h]["plateau"] is not None for h in self.config["heads"]):
            self.adaptation_started_eval = self.eval_count

        return self.controls()

    def controls(self):
        """Read current controls without counting an additional observation."""
        adapting = self.adaptation_started_eval is not None
        adaptation_evals = (max(0, self.eval_count - self.adaptation_started_eval + 1)
                            if adapting else 0)
        lr_scale = (min(1.0, adaptation_evals / self.config["backbone_ramp_evals"])
                    if adapting else 0.0)
        passes = 1
        if adapting:
            base_passes = 1 + min(2, (adaptation_evals - 1) // self.config["pass_ramp_evals"])
            passes = base_passes
            if self.config["max_sequence_passes"] > 3 and base_passes >= 3:
                if self.depth_ramp_origin_eval is None:
                    self.depth_ramp_origin_eval = 1 + 2 * self.config["pass_ramp_evals"]
                extension_evals = max(0, adaptation_evals - self.depth_ramp_origin_eval)
                passes = min(self.config["max_sequence_passes"],
                             3 + extension_evals // self.config["pass_ramp_evals"])
        return {
            "eval": self.eval_count,
            "phase": "whole_transformer_adaptation" if adapting else "projection_only",
            "plateau_reached": adapting,
            "plateaus": {h: copy.deepcopy(self.head_state[h]["plateau"]) for h in self.config["heads"]},
            "backbone_lr_scale": lr_scale,
            "projection_lr_scale": 1.0,
            "sequence_passes": passes,
            "target_sequence_passes": self.config["max_sequence_passes"],
            "adaptation_eval": adaptation_evals,
        }

    def state_dict(self):
        """Return JSON-compatible state for exact schedule continuation."""
        return copy.deepcopy({
            "schema": self.SCHEMA,
            "config": self.config,
            "eval_count": self.eval_count,
            "head_state": self.head_state,
            "adaptation_started_eval": self.adaptation_started_eval,
            "depth_ramp_origin_eval": self.depth_ramp_origin_eval,
        })

    def load_state_dict(self, state):
        """Restore schedule counters after constructing with the same config."""
        if not isinstance(state, dict) or state.get("schema") not in {
                "natlang.projection-first-schedule/1", self.SCHEMA}:
            raise ValueError("invalid projection-first schedule state")
        saved_config = dict(state.get("config") or {})
        old_maximum = saved_config.pop("max_sequence_passes", 3)
        current_config = dict(self.config)
        current_maximum = current_config.pop("max_sequence_passes")
        if (saved_config != current_config or type(old_maximum) is not int or
                old_maximum < 3 or current_maximum < old_maximum):
            raise ValueError("projection-first schedule configuration changed")
        if not isinstance(state.get("eval_count"), int) or state["eval_count"] < 0:
            raise ValueError("invalid projection-first evaluation count")
        if not isinstance(state.get("head_state"), dict) or set(state["head_state"]) != set(self.config["heads"]):
            raise ValueError("invalid projection-first head state")
        count = state["eval_count"]
        for head in self.config["heads"]:
            item = state["head_state"][head]
            history = item.get("history")
            last = item.get("last_significant_eval")
            best = item.get("best")
            if (not isinstance(history, list) or len(history) != count or
                    not isinstance(last, int) or last < 0 or last > count or
                    (best is not None and (not isinstance(best, (int, float)) or not math.isfinite(best) or best < 0))):
                raise ValueError(f"invalid projection-first state for {head}")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in history):
                raise ValueError(f"invalid projection-first history for {head}")
            plateau = item.get("plateau")
            if plateau is not None and (not isinstance(plateau, dict) or plateau.get("eval", 0) > count):
                raise ValueError(f"invalid projection-first plateau for {head}")
        started = state.get("adaptation_started_eval")
        if started is not None and (not isinstance(started, int) or started < 1 or started > count):
            raise ValueError("invalid projection-first transition count")
        self.eval_count = count
        self.head_state = copy.deepcopy(state["head_state"])
        self.adaptation_started_eval = started
        saved_origin = state.get("depth_ramp_origin_eval")
        if saved_origin is not None and (type(saved_origin) is not int or saved_origin < 1):
            raise ValueError("invalid sequence-depth ramp cursor")
        adaptation_evals = (max(0, count - started + 1) if started is not None else 0)
        if saved_origin is not None and saved_origin > max(
                adaptation_evals, 1 + 2 * self.config["pass_ramp_evals"]):
            raise ValueError("sequence-depth ramp cursor is ahead of restored schedule")
        if old_maximum < current_maximum:
            if saved_origin is None:
                saved_origin = max(adaptation_evals, 1 + 2 * self.config["pass_ramp_evals"])
        self.depth_ramp_origin_eval = saved_origin


class RolloutStage:
    """Deeper sketch-rollout training after whole-transformer adaptation has started.

    With ``sketch_first`` the stage begins with only the shallow sketch map trainable, everything else frozen,
    at ``start_passes`` sequence passes, and deepens by one pass whenever the held CE delta of the trained
    sketch-history passes (1 .. depth-1) stops improving by ``min_relative_improvement`` for ``patience``
    observations (after at least ``min_evals`` at that depth). A plateau at the target depth ``passes``
    unfreezes the whole stack at that depth. Each depth starts its own plateau record, since its metric covers
    one more pass. Plateau is not qualification. Without ``sketch_first`` the whole stack trains at ``passes``.

    With ``converge_ratio`` the ramp also deepens as soon as the parallel rollout is settling: the deepest
    trained pass's held CE delta is within ``converge_ratio`` times the pass before it (and at least
    ``min_evals`` observations were made at this depth). The plateau rule remains the fallback.
    """

    SCHEMA = "natlang.sketch-rollout-stage/2"

    def __init__(self, *, passes, start_passes=None, sketch_first=True, min_evals=2, patience=3,
                 min_relative_improvement=0.01, converge_ratio=None, metric="pooled"):
        start_passes = passes if start_passes is None or not sketch_first else min(int(start_passes), int(passes))
        if passes < 2 or start_passes < 2:
            raise ValueError("a sketch rollout needs at least two sequence passes")
        if min_evals < 1 or patience < 1 or not math.isfinite(min_relative_improvement) or min_relative_improvement < 0:
            raise ValueError("positive evaluation counts and a finite nonnegative improvement are required")
        if converge_ratio is not None and (not math.isfinite(converge_ratio) or converge_ratio < 1):
            raise ValueError("converge_ratio must be a finite ratio of at least 1")
        self.config = {"passes": int(passes), "start_passes": int(start_passes), "sketch_first": bool(sketch_first),
                       "converge_ratio": None if converge_ratio is None else float(converge_ratio),
                       "metric": str(metric),
                       "min_evals": int(min_evals), "patience": int(patience),
                       "min_relative_improvement": float(min_relative_improvement)}
        self.phase = "sketch_only" if sketch_first else "whole_stack"
        self.depth = int(start_passes)
        self.history = []
        self._reset_depth()
        self.unfrozen_at_eval = None if sketch_first else 0
        self.deepened = []

    def _reset_depth(self):
        self.depth_observations = 0
        self.best = None
        self.last_significant = 0

    def observe(self, rollout_ce_delta, pass_ce_deltas=None):
        """Observe one held evaluation's CE delta over the trained sketch-history passes (1 .. depth-1).

        ``pass_ce_deltas`` maps pass index to that pass's held CE delta, for the convergence rule."""
        if isinstance(rollout_ce_delta, bool) or not isinstance(rollout_ce_delta, (int, float)) or not math.isfinite(rollout_ce_delta):
            raise ValueError("invalid rollout CE delta")
        value = float(rollout_ce_delta)
        self.history.append({"depth": self.depth, "phase": self.phase, "ce_delta": value,
                             **({"pass_ce_deltas": {str(k): float(v) for k, v in pass_ce_deltas.items()}}
                                if pass_ce_deltas else {})})
        if self.phase != "sketch_only":
            return self.controls()
        self.depth_observations += 1
        count = self.depth_observations
        if self.best is None or (value < self.best and (self.best - value) / max(abs(self.best), 1e-12)
                                 >= self.config["min_relative_improvement"]):
            self.best = value if self.best is None else min(self.best, value)
            self.last_significant = count
        elif value < self.best:
            self.best = value
        converged = False
        ratio = self.config.get("converge_ratio")
        deepest, previous = self.depth - 1, self.depth - 2
        if ratio is not None and pass_ce_deltas and previous >= 1 and count >= self.config["min_evals"]:
            last, before = pass_ce_deltas.get(deepest), pass_ce_deltas.get(previous)
            if last is not None and before is not None and math.isfinite(last) and math.isfinite(before):
                converged = last <= ratio * max(before, 0.0) + 1e-12
        plateau = count >= self.config["min_evals"] and count - self.last_significant >= self.config["patience"]
        if converged and self.depth < self.config["passes"]:
            self.deepened.append({"observation": len(self.history), "from": self.depth, "to": self.depth + 1,
                                  "reason": "converged", "deepest_ce_delta": pass_ce_deltas[deepest],
                                  "previous_ce_delta": pass_ce_deltas[previous]})
            self.depth += 1
            self._reset_depth()
        elif plateau:
            if self.depth < self.config["passes"]:
                self.deepened.append({"observation": len(self.history), "from": self.depth, "to": self.depth + 1,
                                      "reason": "plateau", "plateau_ce_delta": value})
                self.depth += 1
                self._reset_depth()
            else:
                self.phase = "whole_stack"
                self.unfrozen_at_eval = len(self.history)
        return self.controls()

    def controls(self):
        return {"phase": self.phase, "passes": self.depth, "target_passes": self.config["passes"],
                "sketch_only": self.phase == "sketch_only", "observations": len(self.history),
                "depth_observations": self.depth_observations, "best_ce_delta": self.best,
                "last_significant_observation": self.last_significant, "deepened": copy.deepcopy(self.deepened),
                "unfrozen_at_observation": self.unfrozen_at_eval}

    def state_dict(self):
        return copy.deepcopy({"schema": self.SCHEMA, "config": self.config, "phase": self.phase, "depth": self.depth,
                              "history": self.history, "depth_observations": self.depth_observations,
                              "best": self.best, "last_significant": self.last_significant,
                              "unfrozen_at_eval": self.unfrozen_at_eval, "deepened": self.deepened})

    def load_state_dict(self, state):
        if state["config"]["passes"] != self.config["passes"] or state["config"]["sketch_first"] != self.config["sketch_first"]:
            raise ValueError("sketch rollout depth or order differs from the saved stage")
        if state.get("schema") == "natlang.sketch-rollout-stage/1":
            # A fixed-depth stage from before the depth ramp: keep its phase; a sketch-only one restarts the ramp.
            self.history = [{"depth": state["config"]["passes"], "phase": "sketch_only", "ce_delta": v}
                            for v in state["history"]]
            if state["phase"] != "sketch_only":
                self.phase, self.depth, self.unfrozen_at_eval = state["phase"], self.config["passes"], state["unfrozen_at_eval"]
            return
        if state.get("schema") != self.SCHEMA:
            raise ValueError("unsupported sketch rollout stage state")
        self.phase = state["phase"]
        self.depth = state["depth"]
        self.history = list(state["history"])
        self.depth_observations = state["depth_observations"]
        self.best = state["best"]
        self.last_significant = state["last_significant"]
        self.unfrozen_at_eval = state["unfrozen_at_eval"]
        self.deepened = list(state["deepened"])
        if state["config"].get("metric", "pooled") != self.config["metric"]:
            # The held metric changed definition (e.g. system prompt masked): keep depth and phase, but this
            # depth's plateau record restarts on the new scale.
            self._reset_depth()
