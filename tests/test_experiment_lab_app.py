from applications.experiment_lab import (Candidate, ExperimentLab, MergeTrialBackend,
                                         inspect_journal, load_merge_cases, merge_source_revision)
from natlang.surface import ToolSurface
from natlang.types import format_type


SURFACE = ToolSurface()


class Interpreter:
    def __init__(self, lam, selected):
        self.lam, self.selected = lam, selected

    def run(self, session):
        def do(name, args):
            result = SURFACE.apply(session, name, args)
            assert result.kind not in ("rejected", "refused", "error", "quiesced"), (name, args, result.text)
            return result

        def call(function, to, **inputs):
            return do("call", {"function": function, "to": to, "inputs": inputs})

        name = self.lam.fn_name
        if name == "design":
            do("write", {"path": "return", "type": format_type(self.lam.type.returns),
                         "value": {"selected": self.selected, "reason": "Paired conflict examples."}})
        elif name == "report":
            call("summarize", "let/metrics", candidates="args/candidates", trials="args/trials")
            call("interpret", "let/analysis", question="args/question", plan="args/plan",
                 metrics="let/metrics", trials="args/trials")
            call("assemble", "return", plan="args/plan", metrics="let/metrics",
                 trials="args/trials", analysis="let/analysis")
        else:
            do("write", {"path": "return", "type": format_type(self.lam.type.returns),
                         "value": {"interpretation": "Counts are exact; semantic quality awaits review.",
                                   "unknowns": ["Whether either candidate preserved intent."],
                                   "followups": ["Review the captured merge content."]}})
        assert session.finish()


class Backend:
    def __init__(self):
        self.calls = []

    def run(self, candidate, case, attempt, *, model_seed, world_seed):
        self.calls.append((candidate.id, case.id, attempt, model_seed, world_seed))
        case.payload["inputs"]["policy"] = "MUTATED BY TRIAL"
        if candidate.id == "incremental" and attempt == 2:
            raise RuntimeError("worker lost")
        return {"source_revision": candidate.source_revision, "status": "done",
                "label": "unresolved", "provenance": True, "quality": "pending",
                "value_digest": f"stable-{candidate.id}-{case.id}", "trace_id": f"trace-{len(self.calls)}"}


def test_lab_accounts_for_every_trial_and_keeps_seed_namespaces_stable(tmp_path):
    cases = [case for case in load_merge_cases() if case.group == "document-contradiction"]
    selected = [case.id for case in cases[:2]]
    backend = Backend()
    candidates = [Candidate("history", "source-h", "model-v1", 43),
                  Candidate("incremental", "source-i", "model-v1", 43, "incremental")]
    lab = ExperimentLab(agent_factory=lambda lam: Interpreter(lam, selected), backend=backend,
                        world_seed=711, analyst_seed=29, analyst_model_id="scripted-analyst")
    journal = tmp_path / "run.jsonl"
    result = lab.run("Are the two strategies repeatable?", cases, candidates, budget=2,
                     repeats=2, journal_path=journal)
    report = result["report"]
    assert len(report["trials"]) == 8
    history, incremental = report["metrics"]
    assert (history["planned"], history["done"], history["repeats_compared"],
            history["repeats_agree"]) == (4, 4, 2, 2)
    assert (incremental["planned"], incremental["failed"], incremental["review_pending"]) == (4, 2, 4)
    assert all(t["quality"] == "pending" for t in report["trials"])
    assert result["traces"]["design"]["events"] and result["traces"]["report"]["events"]
    assert backend.calls[0][3] == backend.calls[1][3]
    assert backend.calls[0][4] == backend.calls[1][4]
    assert cases[0].payload["inputs"]["policy"] != "MUTATED BY TRIAL"
    assert inspect_journal(journal)["complete"] and inspect_journal(journal)["missing"] == []
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join(lines[:4]) + "\n")
    partial = inspect_journal(journal)
    assert not partial["complete"] and partial["observed"] == 1 and len(partial["missing"]) == 7
    assert len(partial["in_progress"]) == 1


def test_lab_rejects_unoffered_or_oversized_natlang_plan():
    cases = [case for case in load_merge_cases() if case.group == "document-contradiction"]
    lab = ExperimentLab(agent_factory=lambda lam: Interpreter(lam, ["unknown-case"]),
                        backend=Backend(), world_seed=1, analyst_seed=1, analyst_model_id="scripted")
    import pytest
    with pytest.raises(ValueError, match="invalid or unoffered"):
        lab.run("Question", cases, [Candidate("h", "source", "model", 1)], budget=1)


class MergeInterpreter:
    def __init__(self, lam):
        self.lam = lam

    def run(self, session):
        def do(name, args):
            result = SURFACE.apply(session, name, args)
            assert result.kind not in ("rejected", "refused", "error", "quiesced"), (name, args, result.text)
            return result

        def call(function, to, **inputs):
            return do("call", {"function": function, "to": to, "inputs": inputs})

        name = self.lam.fn_name
        if name == "merge_history":
            call("prepare", "let/prepared", base="args/base", updates="args/updates")
            prepared = do("read", {"path": "let/prepared"}).value
            if not prepared["updates"]:
                call("unchanged", "return", base="args/base", prepared="let/prepared")
            else:
                call("interpret_history", "let/draft", base="args/base",
                     updates="let/prepared/updates", policy="args/policy")
                call("finish", "return", base="args/base", prepared="let/prepared", draft="let/draft")
        elif name == "apply_update":
            call("prepare_step", "let/step", base="args/base", current="args/current", update="args/update")
            call("interpret_update", "let/draft", base="args/base", current="args/current",
                 update="args/update", policy="args/policy")
            call("finish_step", "return", current="args/current", update="args/update",
                 step="let/step", draft="let/draft")
        else:
            ids = ([u["id"] for u in self.lam.in_["updates"]] if name == "interpret_history" else
                   [u["id"] for u in self.lam.in_["current"]["updates"]] + [self.lam.in_["update"]["id"]])
            do("write", {"path": "return", "type": format_type(self.lam.type.returns),
                         "value": {"text": "Both opening hours are present.", "applied": ids,
                                   "alternatives": [], "explanation": "Both changes are compatible."}})
        assert session.finish()


def test_merge_backend_runs_both_real_natlang_reduction_styles(tmp_path):
    case = next(c for c in load_merge_cases() if c.id == "document-disjoint:reversed")
    backend = MergeTrialBackend(agent_factory=lambda lam: MergeInterpreter(lam), trace_dir=tmp_path)
    for strategy in ("history", "incremental"):
        candidate = Candidate(strategy, merge_source_revision("merge_history", strategy),
                              "scripted", 43, strategy)
        result = backend.run(candidate, case, 1, model_seed=99, world_seed=77)
        assert result["status"] == "done" and result["conflict"] == "merged"
        assert result["provenance"] and result["quality"] == "pending" and result["trace_id"]
    assert len(list(tmp_path.glob("*.jsonl"))) == 4
