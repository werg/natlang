"""Finite natlang-led scenario exploration with an independent graph oracle."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from applications.experiment_lab import digest
from natlang.codebase import load_function
from natlang.host import load
from natlang.invocation import RunOptions, SeedPolicy
from natlang.runtime import Runtime
from natlang.trace import TraceRecorder
from natlang.values import dump


ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "codebases/dependency_plan/plan.nl"
SELECT = ROOT / "codebases/test_explorer/select.nl"
ASSESS = ROOT / "codebases/test_explorer/assess.nl"


@dataclass(frozen=True)
class GraphCase:
    id: str
    group: str
    description: str
    tasks: tuple[dict, ...]
    split: str = "train"

    def descriptor(self):
        return {"id": self.id, "group": self.group, "description": self.description}


def validate_tasks(tasks: list[dict]) -> None:
    ids = [t["id"] for t in tasks]
    if (len(tasks) > 16 or len(set(ids)) != len(ids) or any(not x for x in ids) or
            any(not isinstance(t["needs"], list) or not isinstance(t["description"], str)
                or len(set(t["needs"])) != len(t["needs"]) for t in tasks)):
        raise ValueError("invalid dependency-plan case")


def graph_violations(tasks: list[dict], state: dict) -> list[str]:
    """Check safety and required completion independently of semantic priority."""
    by_id = {t["id"]: t for t in tasks}
    order = state["order"]
    errors = []
    if state["tasks"] != tasks:
        errors.append("source-mutated")
    if len(order) != len(set(order)):
        errors.append("duplicate")
    if any(x not in by_id for x in order):
        errors.append("invented")
    for index, task_id in enumerate(order):
        if task_id in by_id and any(need not in order[:index] for need in by_id[task_id]["needs"]):
            errors.append("dependency-order")
            break
    remaining = set(by_id) - set(order)
    if not remaining and not state["finished"]:
        errors.append("unfinished-state")
    if state["finished"] and any(set(by_id[x]["needs"]).issubset(set(order)) for x in remaining):
        errors.append("premature-finish")
    if set(state["blocked"]) != remaining:
        errors.append("blocked-set")
    return sorted(set(errors))


class Explorer:
    def __init__(self, *, analyst_factory: Callable, target_factory: Callable,
                 model_id: str, analyst_seed: int, target_seed: int,
                 trace_dir: Path | None = None, backend=None):
        self.analyst_factory, self.target_factory = analyst_factory, target_factory
        self.model_id, self.analyst_seed, self.target_seed = model_id, analyst_seed, target_seed
        self.trace_dir, self.backend = trace_dir, backend
        self.source_revision = digest(load_function(PLAN).to_inline())

    def _run(self, path: Path, inputs: dict, *, seed: int, agent_factory: Callable, label: str):
        options = RunOptions(seed=SeedPolicy("derived", seed))
        trace_path = None
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = self.trace_dir / f"{label}-{options.run_id}.jsonl"
        recorder = TraceRecorder({"run_id": options.run_id, "source_sha256": digest(load_function(path).to_inline()),
                                  "model": self.model_id, "seed_policy": vars(options.seed),
                                  "phase": label}, trace_path)
        try:
            outcome, value = Runtime(agent_factory, options=options, trace_sink=recorder).run_root(
                load(path, copy.deepcopy(inputs)))
        finally:
            recorder.close()
        return outcome.kind, dump(value) if outcome.kind == "done" else None, {
            "sha256": digest(recorder.events), "path": str(trace_path) if trace_path else None,
            "events": recorder.events if trace_path is None else None}

    def _probe(self, case_id: str, tasks: list[dict], attempt: int):
        if digest(load_function(PLAN).to_inline()) != self.source_revision:
            raise ValueError("target source changed during exploration")
        seed = SeedPolicy("derived", self.target_seed).seed(case_id, attempt, "target")
        try:
            if self.backend is not None:
                status, state, trace = self.backend(copy.deepcopy(tasks), seed)
            else:
                status, state, trace = self._run(PLAN, {"tasks": tasks}, seed=seed,
                                                 agent_factory=self.target_factory,
                                                 label=f"target-{case_id}-{attempt}")
            detail = ""
        except Exception as exc:
            status, state = "execution-error", None
            detail = f"{type(exc).__name__}: {exc}"
            trace = {"sha256": digest(detail), "path": None, "events": None}
        try:
            violations = graph_violations(tasks, state) if status == "done" else []
        except (KeyError, TypeError, ValueError) as exc:
            status, violations = "invalid-output", []
            detail = f"{type(exc).__name__}: {exc}"
        return {"status": "violated" if violations else status, "state": state,
                "violations": violations, "trace": trace, "model_seed": seed, "detail": detail}

    def _shrink(self, case: GraphCase, original: dict) -> tuple[list[dict], list[dict]]:
        tasks = list(copy.deepcopy(case.tasks))
        history = []
        target_codes = set(original["violations"])
        if not target_codes:
            return tasks, history
        changed = True
        while changed:
            changed = False
            for index in range(len(tasks)):
                candidate = tasks[:index] + tasks[index + 1:]
                observation = self._probe(case.id, candidate, 1)
                history.append({"candidate_sha256": digest(candidate),
                                "ids": [t["id"] for t in candidate],
                                "status": observation["status"],
                                "violations": observation["violations"],
                                "trace_sha256": observation["trace"]["sha256"]})
                if target_codes.intersection(observation["violations"]):
                    tasks = candidate
                    changed = True
                    break
        return tasks, history

    def run(self, question: str, cases: list[GraphCase], *, budget: int, split: str = "train") -> dict:
        if budget < 1 or len({c.id for c in cases}) != len(cases):
            raise ValueError("positive budget and unique case IDs required")
        offered = {c.id: c for c in cases if c.split == split}
        if not offered:
            raise ValueError("no offered cases")
        for case in offered.values():
            validate_tasks(list(case.tasks))
        status, selection, selection_trace = self._run(
            SELECT, {"question": question, "cases": [c.descriptor() for c in offered.values()],
                     "budget": budget}, seed=self.analyst_seed, agent_factory=self.analyst_factory,
            label="select")
        if status != "done":
            raise ValueError("natlang did not select cases")
        ids = selection["ids"]
        if not ids or len(ids) > budget or len(set(ids)) != len(ids) or any(x not in offered for x in ids):
            raise ValueError("invalid natlang case selection")
        observations = []
        for case_id in ids:
            case = offered[case_id]
            trial = self._probe(case_id, list(case.tasks), 1)
            minimized, shrink_history = self._shrink(case, trial) if trial["violations"] else ([], [])
            observations.append({"id": case_id, "source_revision": self.source_revision,
                                 "status": trial["status"], "violations": trial["violations"],
                                 "minimized": [t["id"] for t in minimized],
                                 "trace_sha256": trial["trace"]["sha256"],
                                 "input_sha256": digest(list(case.tasks)),
                                 "model_seed": trial["model_seed"], "state": trial["state"],
                                 "detail": trial["detail"]})
            observations[-1]["shrink_history"] = shrink_history
        status, assessment, assessment_trace = self._run(
            ASSESS, {"question": question, "observations": [
                {k: row[k] for k in ("id", "source_revision", "status", "violations",
                                       "minimized", "trace_sha256", "detail")} for row in observations]},
            seed=self.analyst_seed, agent_factory=self.analyst_factory, label="assess")
        if status != "done":
            raise ValueError("natlang did not assess observations")
        return {"schema": "test-explorer/v1", "question": question,
                "source_revision": self.source_revision, "model_id": self.model_id,
                "analyst_seed": self.analyst_seed, "target_seed": self.target_seed,
                "selection": selection, "observations": observations, "assessment": assessment,
                "traces": {"selection": selection_trace, "assessment": assessment_trace}}
