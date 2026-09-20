"""Finite experiment host with explicit natlang design and interpretation phases."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from natlang.host import load
from natlang.invocation import RunOptions, SeedPolicy
from natlang.runtime import Runtime
from natlang.trace import TraceRecorder
from natlang.values import dump


ROOT = Path(__file__).resolve().parent.parent
DESIGN = ROOT / "codebases/experiment_lab/design.nl"
REPORT = ROOT / "codebases/experiment_lab/report.nl"
STATUSES = frozenset({"done", "quiesced", "exception", "missing"})
CONFLICTS = frozenset({"merged", "unresolved", "rejected", "unknown"})
QUALITIES = frozenset({"pass", "fail", "pending"})


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class LabCase:
    id: str
    group: str
    description: str
    expected: str
    split: str
    payload: dict
    input_sha256: str

    @classmethod
    def from_row(cls, row: dict) -> "LabCase":
        if digest(row["inputs"]) != row["input_sha256"]:
            raise ValueError(f"case {row['case_id']} has changed inputs")
        return cls(row["case_id"], row["group"], row["rubric"], row["expected"],
                   row["split"], copy.deepcopy(row), row["input_sha256"])

    def descriptor(self) -> dict:
        return {"id": self.id, "group": self.group, "description": self.description,
                "expected": self.expected}


@dataclass(frozen=True)
class Candidate:
    id: str
    source_revision: str
    model_id: str
    model_seed: int
    strategy: str = "history"

    def descriptor(self) -> dict:
        return {"id": self.id, "source_revision": self.source_revision,
                "model_id": self.model_id, "model_seed": self.model_seed}


class TrialBackend(Protocol):
    def run(self, candidate: Candidate, case: LabCase, attempt: int, *,
            model_seed: int, world_seed: int) -> dict: ...


def _exact_metrics(candidates: list[Candidate], trials: list[dict]) -> list[dict]:
    metrics = []
    for candidate in candidates:
        runs = [trial for trial in trials if trial["candidate"] == candidate.id]
        by_case = {}
        for trial in runs:
            by_case.setdefault(trial["case_id"], []).append(trial)
        compared = agreed = 0
        for group in by_case.values():
            completed = [trial for trial in group if trial["status"] == "done" and trial["value_digest"]]
            for trial in completed[1:]:
                compared += 1
                agreed += trial["value_digest"] == completed[0]["value_digest"]
        metrics.append({"candidate": candidate.id, "planned": len(runs),
                        "done": sum(t["status"] == "done" for t in runs),
                        "failed": sum(t["status"] in ("quiesced", "exception") for t in runs),
                        "missing": sum(t["status"] == "missing" for t in runs),
                        "provenance_ok": sum(t["status"] == "done" and t["provenance"] for t in runs),
                        "reviewed_pass": sum(t["quality"] == "pass" for t in runs),
                        "reviewed_fail": sum(t["quality"] == "fail" for t in runs),
                        "review_pending": sum(t["quality"] == "pending" for t in runs),
                        "repeats_compared": compared, "repeats_agree": agreed})
    return metrics


class ExperimentLab:
    def __init__(self, *, agent_factory: Callable, backend: TrialBackend, world_seed: int,
                 analyst_seed: int, analyst_model_id: str, trace_dir: Path | None = None):
        self.agent_factory = agent_factory
        self.backend = backend
        self.world_seed = world_seed
        self.analyst_seed = analyst_seed
        self.analyst_model_id = analyst_model_id
        self.trace_dir = trace_dir

    def _natlang(self, entry: Path, inputs: dict) -> tuple[dict, dict]:
        from natlang.codebase import load_function
        source_revision = digest(load_function(entry).to_inline())
        options = RunOptions(seed=SeedPolicy("derived", self.analyst_seed))
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.trace_dir / f"{entry.stem}-{options.run_id}.jsonl" if self.trace_dir else None
        recorder = TraceRecorder({"run_id": options.run_id, "source_sha256": source_revision,
                                  "model": self.analyst_model_id, "seed_policy": vars(options.seed),
                                  "phase": entry.stem}, trace_path)
        try:
            outcome, value = Runtime(self.agent_factory, options=options,
                                     trace_sink=recorder).run_root(load(entry, inputs))
        finally:
            recorder.close()
        if outcome.kind != "done":
            raise ValueError(f"{entry.stem} did not complete: {outcome.detail}")
        return dump(value), {"sha256": digest(recorder.events), "path": str(trace_path) if trace_path else None,
                             "events": recorder.events if trace_path is None else None}

    def run(self, question: str, cases: list[LabCase], candidates: list[Candidate], *,
            budget: int, repeats: int = 1, split: str = "train",
            journal_path: Path | None = None) -> dict:
        if budget < 1 or repeats < 1 or repeats > 8 or not candidates:
            raise ValueError("positive case budget, one to eight repeats and candidates are required")
        if len({c.id for c in cases}) != len(cases) or len({c.id for c in candidates}) != len(candidates):
            raise ValueError("case and candidate IDs must be unique")
        if any(not c.id or not c.source_revision or not c.model_id for c in candidates):
            raise ValueError("candidates need identity, source revision and model identity")
        offered = [case for case in cases if case.split == split]
        if not offered:
            raise ValueError(f"no cases in split {split}")
        by_id = {case.id: case for case in offered}
        plan, design_trace = self._natlang(DESIGN, {"question": question,
                                                   "cases": [case.descriptor() for case in offered],
                                                   "budget": budget})
        selected = plan["selected"]
        if (not selected or len(selected) > budget or len(selected) != len(set(selected)) or
                any(case_id not in by_id for case_id in selected)):
            raise ValueError("natlang selected an invalid or unoffered experiment case")
        manifest = {"schema": "experiment-lab-journal/v1", "kind": "plan", "question": question,
                    "split": split, "plan": plan, "selected": selected, "repeats": repeats,
                    "candidates": [c.descriptor() for c in candidates],
                    "cases": [{"id": by_id[case_id].id, "input_sha256": by_id[case_id].input_sha256}
                              for case_id in selected], "world_root_seed": self.world_seed,
                    "analyst_seed": self.analyst_seed, "design_trace_sha256": design_trace["sha256"]}
        if journal_path is not None:
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            with journal_path.open("x") as stream:
                stream.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        trials = []
        observations = []
        for case_id in selected:
            case = by_id[case_id]
            for attempt in range(1, repeats + 1):
                world_seed = SeedPolicy("derived", self.world_seed).seed(case_id, attempt, "world")
                for candidate in candidates:
                    model_seed = SeedPolicy("derived", candidate.model_seed).seed(case_id, attempt, "trial-root")
                    try:
                        result = self.backend.run(candidate, copy.deepcopy(case), attempt,
                                                  model_seed=model_seed, world_seed=world_seed)
                        if result.get("source_revision") != candidate.source_revision:
                            raise ValueError("candidate source revision changed during the experiment")
                        trial = {key: result[key] for key in ("status", "conflict", "provenance",
                                                              "quality", "value_digest", "trace_id")}
                        if (trial["status"] not in STATUSES or trial["conflict"] not in CONFLICTS or
                                trial["quality"] not in QUALITIES or not isinstance(trial["provenance"], bool) or
                                (trial["status"] != "done" and trial["quality"] != "pending") or
                                (trial["quality"] != "pending" and not result.get("review_ref"))):
                            raise ValueError("trial backend returned an invalid observation")
                        observations.append({"candidate": candidate.id, "case_id": case_id,
                                             "attempt": attempt, "model_seed": model_seed,
                                             "world_seed": world_seed, "source_revision": candidate.source_revision,
                                             "detail": result.get("detail", ""),
                                             "review_ref": result.get("review_ref", "")})
                    except Exception as exc:
                        trial = {"status": "exception", "conflict": "unknown", "provenance": False,
                                 "quality": "pending", "value_digest": "", "trace_id": ""}
                        observations.append({"candidate": candidate.id, "case_id": case_id,
                                             "attempt": attempt, "model_seed": model_seed,
                                             "world_seed": world_seed, "source_revision": candidate.source_revision,
                                             "error": {"type": type(exc).__name__, "message": str(exc)}})
                    trials.append({"candidate": candidate.id, "case_id": case_id,
                                   "attempt": attempt, **trial})
                    if journal_path is not None:
                        with journal_path.open("a") as stream:
                            stream.write(json.dumps({"kind": "trial", "trial": trials[-1],
                                                     "observation": observations[-1]}, ensure_ascii=False) + "\n")
        report, report_trace = self._natlang(REPORT, {"question": question, "plan": plan,
                                                    "candidates": [c.descriptor() for c in candidates],
                                                    "trials": trials})
        if report["trials"] != trials or report["plan"] != plan:
            raise ValueError("natlang report changed the frozen experiment plan or observations")
        if report["metrics"] != _exact_metrics(candidates, trials):
            raise ValueError("natlang report changed the exact experiment metrics")
        if journal_path is not None:
            with journal_path.open("a") as stream:
                stream.write(json.dumps({"kind": "complete", "report_sha256": digest(report),
                                         "report_trace_sha256": report_trace["sha256"]}) + "\n")
        return {"schema": "experiment-lab-run/v1", "question": question, "split": split,
                "cases": [{"id": case.id, "group": case.group, "input_sha256": case.input_sha256}
                          for case in (by_id[case_id] for case_id in selected)],
                "candidates": [c.descriptor() for c in candidates], "world_root_seed": self.world_seed,
                "analyst_seed": self.analyst_seed, "analyst_model_id": self.analyst_model_id,
                "traces": {"design": design_trace, "report": report_trace},
                "journal": str(journal_path) if journal_path else None,
                "repeats": repeats, "report": report, "observations": observations}


def inspect_journal(path: Path) -> dict:
    """Report unobserved planned slots from a partial append-only run; never retry effects."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or rows[0].get("kind") != "plan" or rows[0].get("schema") != "experiment-lab-journal/v1":
        raise ValueError("invalid experiment journal")
    manifest = rows[0]
    planned = {(candidate["id"], case_id, attempt)
               for case_id in manifest["selected"]
               for attempt in range(1, manifest["repeats"] + 1)
               for candidate in manifest["candidates"]}
    observed = {}
    for row in rows[1:]:
        if row["kind"] == "trial":
            trial = row["trial"]
            key = (trial["candidate"], trial["case_id"], trial["attempt"])
            if key not in planned or key in observed:
                raise ValueError("journal contains an unplanned or repeated trial")
            observed[key] = trial
        elif row["kind"] != "complete":
            raise ValueError("unknown experiment journal event")
    complete = rows[-1].get("kind") == "complete"
    if complete and len(observed) != len(planned):
        raise ValueError("journal completion marker has unobserved trials")
    return {"complete": complete,
            "planned": len(planned), "observed": len(observed),
            "missing": [{"candidate": c, "case_id": case, "attempt": attempt}
                        for c, case, attempt in sorted(planned - observed.keys())],
            "manifest": manifest}


def load_merge_cases(path: Path | None = None) -> list[LabCase]:
    path = path or ROOT / "codebases/semantic_merge/scenarios/cases.jsonl"
    return [LabCase.from_row(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def merge_source_revision(program: str, strategy: str) -> str:
    from natlang.codebase import load_function
    paths = [ROOT / "codebases/semantic_merge" / f"{program}.nl"]
    if strategy == "incremental":
        if program != "merge_history":
            raise ValueError("incremental semantic reduction is implemented for documents only")
        paths.append(ROOT / "codebases/semantic_merge/apply_update.nl")
    elif strategy != "history":
        raise ValueError(f"unknown merge strategy {strategy}")
    return digest([load_function(path).to_inline() for path in paths])


class MergeTrialBackend:
    """Run actual natlang semantic-merge programmes with a fresh runtime per step."""

    def __init__(self, *, agent_factory: Callable, trace_dir: Path):
        self.agent_factory = agent_factory
        self.trace_dir = trace_dir

    def _step(self, entry: Path, inputs: dict, *, seed: int, world_seed: int,
              candidate: Candidate, case: LabCase, attempt: int, ordinal: int) -> tuple[str, dict | None, str]:
        from natlang.codebase import load_function
        source_revision = digest(load_function(entry).to_inline())
        root = load(entry, copy.deepcopy(inputs))
        options = RunOptions(seed=SeedPolicy("derived", seed), world_seed=world_seed)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.trace_dir / (f"{case.id.replace(':', '-')}-{candidate.id}-{attempt}-{ordinal}-"
                                       f"{options.run_id}.jsonl")
        recorder = TraceRecorder({"run_id": options.run_id, "source_sha256": source_revision,
                                  "input_sha256": digest(inputs), "case_id": case.id,
                                  "candidate": candidate.id, "model": candidate.model_id,
                                  "seed_policy": vars(options.seed), "world_seed": world_seed,
                                  "attempt": attempt, "ordinal": ordinal}, trace_path)
        runtime = Runtime(self.agent_factory, options=options, trace_sink=recorder, trace_path=trace_path)
        outcome, value = runtime.run_root(root)
        return outcome.kind, dump(value) if outcome.kind == "done" else None, digest(recorder.events)

    def run(self, candidate: Candidate, case: LabCase, attempt: int, *,
            model_seed: int, world_seed: int) -> dict:
        payload = copy.deepcopy(case.payload)
        program = payload["program"]
        source_revision = merge_source_revision(program, candidate.strategy)
        if source_revision != candidate.source_revision:
            raise ValueError("candidate source revision changed before execution")
        inputs = payload["inputs"]
        traces = []
        if candidate.strategy == "history":
            entry = ROOT / "codebases/semantic_merge" / f"{program}.nl"
            status, value, trace = self._step(entry, inputs, seed=model_seed, world_seed=world_seed,
                                              candidate=candidate, case=case, attempt=attempt, ordinal=0)
            traces.append(trace)
        else:
            from natlang.codebase import load_function
            from natlang.host import instantiate
            from natlang.types import TypeEnv
            from natlang.values import coerce
            prepare_fn = load_function(ROOT / "codebases/semantic_merge/merge_history.nl").codebase["prepare"]
            prepare_root = instantiate(prepare_fn)
            env = prepare_root.env(TypeEnv())
            for name in ("base", "updates"):
                prepare_root.in_[name] = coerce(inputs[name], prepare_root.type.params.get(name)[0], env,
                                                yaml=False, path=f"args/{name}")
            prepared_out, prepared = Runtime(None).run_root(prepare_root)
            if prepared_out.kind != "done" or not prepared["valid"]:
                raise ValueError("incremental input delivery cannot be normalized")
            ordered = prepared["updates"]
            history = ROOT / "codebases/semantic_merge/merge_history.nl"
            bootstrap = {"base": inputs["base"], "updates": [], "policy": inputs["policy"]}
            first_seed = SeedPolicy("derived", model_seed).seed(case.id, 0, "incremental-step")
            status, value, trace = self._step(history, bootstrap, seed=first_seed,
                                              world_seed=world_seed, candidate=candidate,
                                              case=case, attempt=attempt, ordinal=0)
            traces.append(trace)
            for ordinal, update in enumerate(ordered, 1):
                if status != "done":
                    break
                entry = ROOT / "codebases/semantic_merge/apply_update.nl"
                step_seed = SeedPolicy("derived", model_seed).seed(case.id, ordinal, "incremental-step")
                status, value, trace = self._step(entry, {"base": inputs["base"], "current": value,
                                                   "update": update, "policy": inputs["policy"]},
                                                  seed=step_seed, world_seed=world_seed,
                                                  candidate=candidate, case=case, attempt=attempt,
                                                  ordinal=ordinal)
                traces.append(trace)
        if status not in ("done", "quiesced"):
            raise ValueError(f"unsupported natlang trial outcome {status}")
        ids = {update["id"] for update in inputs["updates"]}
        claims = ([*value["applied"], *(id for alt in value["alternatives"] for id in alt["update_ids"])]
                  if value else [])
        provenance = (len(claims) == len(ids) and len(set(claims)) == len(ids) and set(claims) == ids)
        conflict = (value["status"] if program == "merge_history" else
                    ("unresolved" if value["alternatives"] else "merged")) if value else "unknown"
        return {"source_revision": source_revision, "status": status, "conflict": conflict,
                "provenance": provenance, "quality": "pending",
                "value_digest": digest(value) if value else "", "trace_id": digest(traces),
                "detail": {"trace_hashes": traces, "matches_expected_conflict": conflict == case.expected}}
