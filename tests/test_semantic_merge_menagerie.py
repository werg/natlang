"""Seven typed merge programmes, shared exact guards, and grouped scenario expansion."""
from pathlib import Path

import pytest

from natlang.codebase import load_function
from natlang.decoder import ChatTurn
from natlang.host import load
from natlang.runtime import Runtime
from natlang.surface import ToolSurface
from natlang.types import format_type
from natlang.values import dump
from scripts.generate_semantic_merge_cases import generate
from scripts.collect_semantic_merge_teacher import collect


ROOT = Path(__file__).resolve().parent.parent
SURFACE = ToolSurface()
STATES = {
    "counter-independent": {"revision": 2, "value": 17, "unit": "visitors"},
    "set-independent": {"revision": 2, "members": ["red", "blue", "green"]},
    "map-independent": {"revision": 2, "fields": {"owner": "Bo", "region": "east"}},
    "list-move-edit": {"revision": 2, "items": [
        {"id": "safety", "text": "Safety and preparation"}, {"id": "intro", "text": "Introduction"}]},
    "tree-move-rename": {"revision": 2, "nodes": [
        {"id": "root", "parent": "", "title": "Guide"},
        {"id": "api", "parent": "root", "title": "API"},
        {"id": "auth", "parent": "root", "title": "Authentication"}]},
    "graph-independent": {"revision": 2, "nodes": [
        {"id": "a", "label": "API"}, {"id": "b", "label": "Database"}, {"id": "c", "label": "Cache"}],
        "edges": [{"from": "a", "relation": "depends_on", "to": "b"},
                  {"from": "a", "relation": "reads_from", "to": "c"}]},
    "schedule-independent": {"revision": 2, "events": [
        {"id": "demo", "title": "Demo", "start": "2026-10-01T10:00:00Z",
         "end": "2026-10-01T11:00:00Z", "place": "Room C"},
        {"id": "review", "title": "Sprint Review", "start": "2026-10-02T10:00:00Z",
         "end": "2026-10-02T11:00:00Z", "place": "Room B"}]},
    "permissions-independent": {"revision": 2, "rules": [
        {"subject": "editor", "resource": "draft", "action": "read", "decision": "allow"},
        {"subject": "guest", "resource": "page", "action": "publish", "decision": "deny"}]},
    "scene-move-recolor": {"revision": 2, "objects": [
        {"id": "orb", "label": "Orb", "x": 3, "y": 1, "color": "gold"}]},
}


class Interpreter:
    def __init__(self, lam, state):
        self.lam, self.state = lam, state

    def run(self, session):
        def do(name, args):
            result = SURFACE.apply(session, name, args)
            assert result.kind not in ("rejected", "refused", "error"), (name, args, result.text)
            return result

        def call(function, to, **inputs):
            return do("call", {"function": function, "to": to, "inputs": inputs})

        if self.lam.fn_name == "interpret":
            draft = {"state": self.state, "applied": [u["id"] for u in self.lam.in_["updates"]],
                     "alternatives": [], "explanation": "The compatible changes are combined."}
            do("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": draft})
        else:
            call("prepare_envelope", "let/prepared", revision="args/base/revision", updates="args/updates")
            prepared = do("read", {"path": "let/prepared"}).value
            assert prepared["valid"]
            call("interpret", "let/draft", base="args/base", updates="let/prepared/updates", policy="args/policy")
            call("validate_claims", "let/valid", updates="let/prepared/updates",
                 applied="let/draft/applied", alternatives="let/draft/alternatives")
            assert do("read", {"path": "let/valid"}).value
            call("check_state", "let/shape_ok", state="let/draft/state")
            if not do("read", {"path": "let/shape_ok"}).value:
                return do("report_error", {"message": "Invalid proposed state."}).text
            do("write", {"path": "return", "type": format_type(self.lam.type.returns),
                         "source": "let/draft"})
        assert session.finish()


@pytest.mark.parametrize("group", sorted(STATES))
def test_typed_merge_programme_accepts_an_independent_semantic_draft(group):
    row = next(row for row in generate() if row["group"] == group and row["delivery"] == "reversed")
    entry = ROOT / "codebases/semantic_merge" / f"{row['program']}.nl"
    load_function(entry)
    rt = Runtime(lambda lam: Interpreter(lam, STATES[group]))
    outcome, value = rt.run_root(load(entry, row["inputs"]))
    assert outcome.kind == "done", outcome.detail
    assert dump(value)["state"] == STATES[group]


def test_graph_rejects_dangling_edge_even_when_all_updates_are_claimed():
    row = next(row for row in generate() if row["group"] == "graph-independent" and
               row["delivery"] == "canonical")
    invalid = {**STATES["graph-independent"], "edges": [
        {"from": "a", "relation": "depends_on", "to": "missing"}]}
    entry = ROOT / "codebases/semantic_merge/merge_graph.nl"
    outcome, _ = Runtime(lambda lam: Interpreter(lam, invalid)).run_root(load(entry, row["inputs"]))
    assert outcome.kind == "quiesced" and "Invalid proposed state" in outcome.detail


def test_scenarios_expand_in_groups_without_turning_agreement_into_an_oracle():
    rows = generate()
    assert len(rows) == 80 and len({r["group"] for r in rows}) == 20
    assert {r["delivery"] for r in rows} == {"canonical", "reversed", "redelivery-first", "redelivery-last"}
    for group in {r["group"] for r in rows}:
        variants = [r for r in rows if r["group"] == group]
        assert len(variants) == 4 and len({r["input_sha256"] for r in variants}) >= 2
        assert len({r["rubric"] for r in variants}) == 1
        assert len({r["split"] for r in variants}) == 1
        assert all(r["root_seed"] == 43 for r in variants)


def test_teacher_collector_keeps_incomplete_run_out_of_training_admission(tmp_path):
    class EmptyDriver:
        def chat(self, messages, tools, *, temperature, seed, max_tokens):
            return ChatTurn(calls=[], text="", completion_tokens=1)

    case = next(row for row in generate() if row["case_id"] == "map-independent:canonical")
    path = tmp_path / "trace.jsonl"
    row = collect(case, EmptyDriver(), model_id="scripted-empty", trace_path=path,
                  max_turns=3, max_tokens=100, max_seconds=10)
    assert path.is_file() and row["case_id"] == case["case_id"]
    assert not row["checks"]["ready_for_semantic_review"]
    assert row["semantic_review"] == "pending"
