from applications.evaluation_backends import (BACKENDS, application_revision,
                                               load_application_cases)
from applications.experiment_lab import Candidate, ExperimentLab
from natlang.surface import ToolSurface
from natlang.types import format_type


SURFACE = ToolSurface()


class Agent:
    def __init__(self, lam):
        self.lam = lam

    def run(self, session):
        def do(name, args):
            result = SURFACE.apply(session, name, args)
            assert result.kind not in ("rejected", "refused", "error", "quiesced"), (name, result.text)
            return result

        def call(fn, to, **inputs):
            do("call", {"function": fn, "to": to, "inputs": inputs})

        def write(value):
            do("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": value})

        name = self.lam.fn_name
        if name == "design":
            write({"selected": [self.lam.in_["cases"][0]["id"]], "reason": "First offered case."})
        elif name == "report":
            call("summarize", "let/metrics", candidates="args/candidates", trials="args/trials")
            call("interpret", "let/analysis", question="args/question", plan="args/plan",
                 metrics="let/metrics", trials="args/trials")
            call("assemble", "return", plan="args/plan", metrics="let/metrics",
                 trials="args/trials", analysis="let/analysis")
        elif name == "interpret":
            write({"interpretation": "One independently checked trial.", "unknowns": [], "followups": []})
        elif name == "infer":
            call("propose", "let/candidate", target="args/target", context="args/context")
            call("check_candidate", "let/fit", target="args/target", context="args/context",
                 candidate="let/candidate")
            call("finalize", "return", candidate="let/candidate", fit="let/fit")
        elif name == "propose":
            write({"args": {"ready": "Task[]"}, "returns": "Text", "effects": [],
                   "reason": "Ready tasks are listed.", "alternatives": []})
        elif name == "select":
            write({"ids": [self.lam.in_["cases"][0]["id"]], "reason": "Check graph."})
        elif name == "assess":
            observations = self.lam.in_["observations"]
            write({"confirmed_ids": [x["id"] for x in observations if x["status"] == "violated"],
                   "unknown_ids": [x["id"] for x in observations if x["status"] not in ("done", "violated")],
                   "findings": [], "unknowns": [], "followups": []})
        elif name == "map":
            if self.lam.in_["source"] == "legacy":
                write({"customer_id": "cid", "email": "mail", "name": "person",
                       "order_id": "oid", "order_customer": "buyer", "amount": "price",
                       "unit": "unit", "reason": "Legacy columns."})
            else:
                write({"customer_id": "id", "email": "email", "name": "name",
                       "order_id": "id", "order_customer": "customer", "amount": "amount",
                       "unit": "unit", "reason": "Direct columns."})
        elif name == "decide":
            known = {row["email"]: row["name"] for row in self.lam.in_["existing"]}
            decisions = []
            for row in self.lam.in_["customers"]:
                email = row["email"]
                action = ("review" if email in known and known[email] != row["name"] else
                          "merge" if email in known else "new")
                decisions.append({"source_key": row["source_key"], "action": action,
                                  "target_email": email if action != "review" else "",
                                  "reason": "Exact email and name comparison."})
                if action == "new":
                    known[email] = row["name"]
            write(decisions)
        else:
            raise AssertionError(name)
        assert session.finish()


def case_for(suite, split="train"):
    return next(case for case in load_application_cases()
                if case.payload["suite"] == suite and case.split == split)


def candidate(suite):
    return Candidate(suite, application_revision(suite), "scripted", 27, suite)


def test_type_trial_runs_through_generic_lab_and_reference_rubric(tmp_path):
    case = case_for("types")
    factory = lambda lam: Agent(lam)
    lab = ExperimentLab(agent_factory=factory,
                        backend=BACKENDS["types"](agent_factory=factory, trace_dir=tmp_path / "traces"),
                        world_seed=2, analyst_seed=3, analyst_model_id="scripted",
                        trace_dir=tmp_path / "traces")
    result = lab.run("Does the signature match?", [case], [candidate("types")],
                     budget=1, journal_path=tmp_path / "journal.jsonl")
    trial = result["report"]["trials"][0]
    assert trial["quality"] == "pass" and trial["label"] == "reference-compatible"
    assert result["report"]["metrics"][0]["reviewed_pass"] == 1


def test_graph_and_migration_backends_use_independent_rubrics(tmp_path):
    factory = lambda lam: Agent(lam)
    def graph_backend(tasks, seed):
        state = {"tasks": tasks, "order": ["a", "b", "c"],
                 "blocked": [], "finished": True}
        return "done", state, {"sha256": "scripted-target-trace", "events": []}

    graph = BACKENDS["tests"](agent_factory=factory, trace_dir=tmp_path / "graph",
                               target_backend=graph_backend)
    graph_case = case_for("tests")
    observed = graph.run(candidate("tests"), graph_case, 1, model_seed=42, world_seed=7)
    assert observed["quality"] == "pass" and observed["review_ref"] == "graph-oracle/v1"

    migration = BACKENDS["migration"](agent_factory=factory, trace_dir=tmp_path / "migration")
    for split, expected_review in (("train", 0), ("eval", 2)):
        case = case_for("migration", split)
        result = migration.run(candidate("migration"), case, 1, model_seed=42, world_seed=7)
        assert result["quality"] == "pass" and result["detail"]["observed"]["review"] == expected_review
