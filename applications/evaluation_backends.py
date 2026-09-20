"""Independent application trials for the shared natlang experiment lab."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable

from applications.data_migration import Export, MigrationStudio
from applications.experiment_lab import Candidate, LabCase, digest
from applications.test_explorer import Explorer, GraphCase
from applications.type_studio import TypeStudio
from natlang.codebase import load_function


ROOT = Path(__file__).resolve().parent.parent


def evaluation_harness_revision() -> str:
    paths = ["natlang/invocation.py", "natlang/runtime.py", "natlang/surface.py",
             "natlang/tool_agent.py", "natlang/decoder.py", "natlang/trace.py",
             "natlang/prompts/tools_teacher_compact.md",
             "applications/teacher.py", "applications/experiment_lab.py",
             "applications/evaluation_backends.py", "scripts/run_application_evals.py"]
    return digest({path: (ROOT / path).read_text() for path in paths})


def application_revision(suite: str) -> str:
    entries = {
        "types": (["codebases/type_studio/infer.nl", "codebases/type_studio/check.nl"],
                  "applications/type_studio.py"),
        "tests": (["codebases/test_explorer/select.nl", "codebases/test_explorer/assess.nl",
                   "codebases/dependency_plan/plan.nl"], "applications/test_explorer.py"),
        "migration": (["codebases/data_migration/map.nl", "codebases/data_migration/decide.nl"],
                      "applications/data_migration.py"),
    }
    if suite not in entries:
        raise ValueError(f"unknown evaluation suite {suite}")
    codebases, host = entries[suite]
    return digest({"codebases": [load_function(ROOT / path).to_inline() for path in codebases],
                   "host": (ROOT / host).read_text()})


def load_application_cases(path: Path | None = None) -> list[LabCase]:
    path = path or ROOT / "codebases/experiment_lab/scenarios/applications.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cases = []
    for row in rows:
        payload = dict(row["payload"])
        if row["suite"] == "types":
            payload["source_sha256"] = digest(load_function(ROOT / payload["source"]).to_inline())
        if row["suite"] == "migration" and "source_file" in payload:
            source = json.loads((ROOT / payload.pop("source_file")).read_text())
            payload["exports"] = source["exports"]
        cases.append(LabCase.from_payload(id=row["id"], group=row["group"],
                                          description=row["description"], expected=row["expected"],
                                          split=row["split"], payload={"suite": row["suite"], **payload}))
    return cases


class _BaseBackend:
    suite: str

    def __init__(self, *, agent_factory: Callable, trace_dir: Path):
        self.agent_factory = agent_factory
        self.trace_dir = Path(trace_dir)

    def _dir(self, candidate: Candidate, case: LabCase, attempt: int) -> Path:
        target = self.trace_dir / self.suite / f"{digest([candidate.id, case.id, attempt])[:20]}"
        target.mkdir(parents=True, exist_ok=False)
        return target

    def _check_source(self, candidate: Candidate):
        revision = application_revision(self.suite)
        if candidate.source_revision != revision:
            raise ValueError(f"{self.suite} source revision changed")
        return revision


class TypeTrialBackend(_BaseBackend):
    suite = "types"

    def run(self, candidate: Candidate, case: LabCase, attempt: int, *, model_seed: int,
            world_seed: int) -> dict:
        revision = self._check_source(candidate)
        path = self._dir(candidate, case, attempt)
        payload = case.payload
        source = ROOT / payload["source"]
        if digest(load_function(source).to_inline()) != payload["source_sha256"]:
            raise ValueError("type evaluation source changed after case selection")
        operation = payload.get("operation", "infer")
        studio = TypeStudio.from_file(source, payload["target"], **payload.get("evidence", {}))
        result = studio.run(operation, agent_factory=self.agent_factory,
                            model_id=candidate.model_id, root_seed=model_seed,
                            trace_path=path / "type.trace.jsonl")
        value = result["value"]
        gold = payload.get("gold_signature")
        if result["outcome"] == "done" and gold is not None and operation == "infer":
            proposed = value["candidate"]
            accepted = (proposed["args"] == gold["args"] and proposed["returns"] == gold["returns"] and
                        value["fit"]["parseable"] and value["fit"]["obligations_ok"])
            quality, label = ("pass", "reference-compatible") if accepted else ("fail", "reference-mismatch")
        else:
            quality, label = "pending", result["outcome"]
        return {"source_revision": revision, "status": result["outcome"], "label": label,
                "provenance": bool(result["trace_sha256"] and Path(result["trace_path"]).is_file()),
                "quality": quality, "review_ref": f"fixture:{case.id}" if quality != "pending" else "",
                "value_digest": digest(value) if value is not None else "",
                "trace_id": result["trace_sha256"],
                "detail": {"source_snapshot": studio.snapshot_sha256, "trace": result["trace_path"],
                           "outcome_detail": result["detail"], "proposed": value}}


class TestExplorerTrialBackend(_BaseBackend):
    suite = "tests"

    def __init__(self, *, agent_factory: Callable, trace_dir: Path, target_backend=None):
        super().__init__(agent_factory=agent_factory, trace_dir=trace_dir)
        self.target_backend = target_backend

    def run(self, candidate: Candidate, case: LabCase, attempt: int, *, model_seed: int,
            world_seed: int) -> dict:
        revision = self._check_source(candidate)
        path = self._dir(candidate, case, attempt)
        graph = GraphCase(case.id, case.group, case.description,
                          tuple(case.payload["tasks"]), case.split)
        result = Explorer(analyst_factory=self.agent_factory, target_factory=self.agent_factory,
                          model_id=candidate.model_id, analyst_seed=model_seed,
                          target_seed=model_seed, trace_dir=path,
                          backend=self.target_backend).run(
            "Check the dependency-plan graph contract", [graph], budget=1, split=case.split)
        observation = result["observations"][0]
        status = observation["status"]
        completed = status in ("done", "violated")
        quality = ("pass" if status == "done" and not observation["violations"] else
                   "fail" if completed else "pending")
        if not result["assessment_check"]["ok"]:
            quality = "fail"
        trial_status = "done" if completed else "quiesced"
        trace_id = digest([result["traces"], observation["trace_sha256"]])
        return {"source_revision": revision, "status": trial_status,
                "label": status if result["assessment_check"]["ok"] else "assessment-mismatch",
                "provenance": bool(observation["trace_sha256"] and
                                   result["traces"]["selection"]["sha256"] and
                                   result["traces"]["assessment"]["sha256"]),
                "quality": quality, "review_ref": "graph-oracle/v1" if quality != "pending" else "",
                "value_digest": digest(observation["state"]) if observation["state"] is not None else "",
                "trace_id": trace_id,
                "detail": {"observation": observation, "assessment": result["assessment"],
                           "assessment_check": result["assessment_check"],
                           "trace_dir": str(path)}}


class MigrationTrialBackend(_BaseBackend):
    suite = "migration"

    def run(self, candidate: Candidate, case: LabCase, attempt: int, *, model_seed: int,
            world_seed: int) -> dict:
        revision = self._check_source(candidate)
        path = self._dir(candidate, case, attempt)
        exports = [Export(row["source"], tuple(row["customers"]), tuple(row["orders"]))
                   for row in case.payload["exports"]]
        studio = MigrationStudio(path / "target.sqlite", agent_factory=self.agent_factory,
                                 model_id=candidate.model_id, root_seed=model_seed,
                                 trace_dir=path / "traces")
        preview = studio.preview(exports)
        applied = studio.apply(preview, exports)
        with sqlite3.connect(studio.path) as db:
            observed = {"customers": db.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
                        "order_total_cents": db.execute("SELECT COALESCE(SUM(cents),0) FROM orders").fetchone()[0],
                        "lineage_fields": db.execute("SELECT COUNT(*) FROM lineage").fetchone()[0],
                        "review": len(applied["review"])}
        expected = case.payload["expected_target"]
        quality = "pass" if observed == expected else "fail"
        trace_id = digest({key: value["sha256"] for key, value in preview["traces"].items()})
        artifact = path / "result.json"
        artifact.write_text(json.dumps({"preview": preview, "applied": applied, "observed": observed},
                                       ensure_ascii=False, indent=2) + "\n")
        return {"source_revision": revision, "status": "done",
                "label": "reference-compatible" if quality == "pass" else "reference-mismatch",
                "provenance": all(Path(t["path"]).is_file() for t in preview["traces"].values()),
                "quality": quality, "review_ref": f"fixture:{case.id}",
                "value_digest": digest(observed), "trace_id": trace_id,
                "detail": {"observed": observed, "expected": expected, "artifact": str(artifact)}}


BACKENDS = {"types": TypeTrialBackend, "tests": TestExplorerTrialBackend,
            "migration": MigrationTrialBackend}
