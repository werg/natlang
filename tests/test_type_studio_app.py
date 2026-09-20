import pytest
import json
from pathlib import Path

from applications.type_studio import TypeStudio
from natlang.codebase import from_definitions
from natlang.surface import ToolSurface
from natlang.types import format_type


SURFACE = ToolSurface()


def studio():
    graph = from_definitions({
        "normalize": {"args": {"raw": "Text"}, "returns": "Text",
                      "instructions": "Normalize a customer identifier."},
        "caller": {"args": {"value": "Text"}, "returns": "Text",
                   "instructions": "Call normalize with value."},
        "optional": {"args": {"prefix?": "Text", "value": "Text"},
                     "returns": "Text", "instructions": "Combine inputs."},
    }, "normalize")
    return TypeStudio.from_graph(
        graph, "normalize",
        obligations=[{"kind": "argument", "parameter": "raw", "type": "Text", "source": "caller:raw"},
                     {"kind": "return", "parameter": "", "type": "Text", "source": "caller:return"}],
        witnesses=[{"id": "run-1", "callee": "optional", "arg_types": {"value": "Num"},
                    "source": "trace:run-1"}])


def candidate(raw="Text", returns="Text", alternatives=None):
    return {"args": {"raw": raw}, "returns": returns, "effects": [],
            "reason": "Body and caller agree.", "alternatives": alternatives or []}


class Interpreter:
    def __init__(self, lam, proposed=None, claims=None):
        self.lam, self.proposed, self.claims = lam, proposed, claims

    def run(self, session):
        def do(name, args):
            result = SURFACE.apply(session, name, args)
            assert result.kind not in ("rejected", "refused", "error", "quiesced"), (name, result.text)
            return result

        def call(fn, to, **inputs):
            do("call", {"function": fn, "to": to, "inputs": inputs})

        name = self.lam.fn_name
        if name == "infer":
            call("propose", "let/candidate", target="args/target", context="args/context")
            call("check_candidate", "let/fit", target="args/target", context="args/context",
                 candidate="let/candidate")
            call("finalize", "return", candidate="let/candidate", fit="let/fit")
        elif name == "check":
            call("identify_calls", "let/claims", target="args/target", context="args/context")
            call("check_calls", "let/diagnostics", context="args/context", claims="let/claims")
            call("assemble", "return", claims="let/claims", diagnostics="let/diagnostics")
        else:
            value = self.proposed if name == "propose" else self.claims
            do("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": value})
        assert session.finish()


def test_inference_checks_frozen_evidence_and_runs_through_natlang(tmp_path):
    s = studio()
    fit = s.check_candidate(s.target, s.context, candidate())
    assert fit["parseable"] and fit["obligations_ok"] and fit["checked"] == 2
    assert not s.check_candidate(s.target, s.context, candidate("Num"))["obligations_ok"]
    assert not s.check_candidate(s.target, s.context, candidate("MissingType"))["parseable"]
    with pytest.raises(ValueError, match="frozen source"):
        s.check_candidate({**s.target, "body": "changed"}, s.context, candidate())
    trace = tmp_path / "type-trace.jsonl"
    result = s.run("infer", agent_factory=lambda lam: Interpreter(lam, candidate()),
                   model_id="scripted", root_seed=19, trace_path=trace)
    assert result["outcome"] == "done", result
    assert result["value"]["status"] == "consistent"
    assert result["value"]["fit"]["checked"] == 2
    assert trace.read_text().count("\n") > 1


def test_call_checker_separates_witnessed_errors_from_hypotheses():
    s = studio()
    claims = [
        {"callee": "optional", "arg_types": {"value": "Num"},
         "evidence_id": "run-1", "rationale": "observed"},
        {"callee": "optional", "arg_types": {"value": "Num"},
         "evidence_id": "", "rationale": "possible"},
        {"callee": "optional", "arg_types": {"value": "Text"},
         "evidence_id": "run-1", "rationale": "forged"},
    ]
    result = s.run("check", agent_factory=lambda lam: Interpreter(lam, claims=claims),
                   model_id="scripted", root_seed=20)
    assert result["outcome"] == "done", result
    levels = [d["level"] for d in result["value"]["diagnostics"]]
    assert levels == ["exact", "hypothesis", "unknown"]


def test_file_source_view_masks_target_signature():
    s = TypeStudio.from_file("codebases/spell_arena/cast.nl", "cast")
    assert s.target["name"] == "cast"
    assert "returns" not in s.target and "args" not in s.target
    assert s.target["revision"] and s.snapshot_sha256


def test_optional_parameter_marker_is_preserved_in_inference():
    graph = from_definitions({"optional": {"args": {"prefix?": "Text", "value": "Text"},
                                           "returns": "Text", "instructions": "Combine inputs."}},
                             "optional")
    s = TypeStudio.from_graph(graph, "optional")
    assert s.target["parameters"] == ["prefix?", "value"]
    correct = {"args": {"prefix?": "Text", "value": "Text"}, "returns": "Text", "effects": [],
               "reason": "Optional prefix.", "alternatives": []}
    assert s.check_candidate(s.target, s.context, correct)["obligations_ok"]
    wrong = {**correct, "args": {"prefix": "Text", "value": "Text"}}
    assert not s.check_candidate(s.target, s.context, wrong)["obligations_ok"]


def test_grouped_type_fixtures_have_independent_exact_outcomes():
    cases = [json.loads(line) for line in Path(
        "codebases/type_studio/scenarios/cases.jsonl").read_text().splitlines()]
    assert {c["split"] for c in cases} == {"train", "eval"}
    groups = {}
    for case in cases:
        groups.setdefault(case["group"], set()).add(case["split"])
        fit = studio().check_candidate(studio().target, studio().context, case["candidate"])
        observed = ("invalid" if not fit["parseable"] or not fit["obligations_ok"] else
                    "compatible-uncertain" if case["candidate"]["alternatives"] else "compatible")
        assert observed == case["expected"], case["id"]
    assert all(len(splits) == 1 for splits in groups.values())
