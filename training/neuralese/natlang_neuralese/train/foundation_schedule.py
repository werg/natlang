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

        controls = self.controls()
        if (self.config["max_sequence_passes"] > 3 and self.depth_ramp_origin_eval is None
                and controls["adaptation_eval"] >= 1 + 2 * self.config["pass_ramp_evals"]):
            self.depth_ramp_origin_eval = 1 + 2 * self.config["pass_ramp_evals"]
            controls = self.controls()
        return controls

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
                ramp_origin = (self.depth_ramp_origin_eval if self.depth_ramp_origin_eval is not None
                               else 1 + 2 * self.config["pass_ramp_evals"])
                extension_evals = max(0, adaptation_evals - ramp_origin)
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
            ramp_evals = self.config["pass_ramp_evals"]
            minimum_origin = 1 + 2 * ramp_evals
            if saved_origin is None:
                saved_origin = max(adaptation_evals, minimum_origin)
            old_actual_depth = min(old_maximum, 3 + max(0, adaptation_evals - saved_origin) // ramp_evals)
            if adaptation_evals and old_actual_depth >= old_maximum:
                saved_origin = adaptation_evals - (old_actual_depth - 3) * ramp_evals
        self.depth_ramp_origin_eval = saved_origin
