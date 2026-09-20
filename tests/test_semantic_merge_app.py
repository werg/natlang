"""A scripted semantic leaf exercises the actual natlang call tree and exact provenance gate."""
from pathlib import Path

from natlang.host import load
from natlang.invocation import RunOptions, SeedPolicy
from natlang.runtime import Runtime
from natlang.surface import ToolSurface
from natlang.types import format_type
from natlang.values import dump


ENTRY = Path(__file__).resolve().parent.parent / "codebases/semantic_merge/merge_history.nl"
SURFACE = ToolSurface()
BASE = {"revision": 7, "text": "The garden is open on Sunday."}


class Interpreter:
    def __init__(self, lam, draft, presentations):
        self.lam, self.draft, self.presentations = lam, draft, presentations

    def run(self, session):
        def do(name, args):
            result = SURFACE.apply(session, name, args)
            assert result.kind not in ("rejected", "refused", "error", "quiesced"), (name, args, result.text)
            return result

        def call(function, to, **inputs):
            return do("call", {"function": function, "to": to, "inputs": inputs})

        if self.lam.fn_name == "interpret_history":
            self.presentations.append([u["id"] for u in self.lam.in_["updates"]])
            do("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": self.draft})
        else:
            call("prepare", "let/prepared", base="args/base", updates="args/updates")
            prepared = do("read", {"path": "let/prepared"}).value
            if not prepared["valid"]:
                call("reject", "return", base="args/base", prepared="let/prepared")
            elif not prepared["updates"]:
                call("unchanged", "return", base="args/base", prepared="let/prepared")
            else:
                call("interpret_history", "let/draft", base="args/base", updates="let/prepared/updates",
                     policy="args/policy")
                call("finish", "return", base="args/base", prepared="let/prepared", draft="let/draft")
        assert session.finish()


def run(updates, draft):
    presentations = []
    options = RunOptions(seed=SeedPolicy(mode="derived", root=43))
    rt = Runtime(lambda lam: Interpreter(lam, draft, presentations), options=options)
    outcome, value = rt.run_root(load(ENTRY, {"base": BASE, "updates": updates,
                                              "policy": "Preserve opening hours unless an author clearly changes them."}))
    assert outcome.kind == "done", outcome.detail
    return dump(value), presentations


def test_history_presentation_is_stable_and_duplicate_delivery_is_removed():
    a = {"id": "a", "parents": [], "base_revision": 7, "author": "Ada", "text": "Open at 10:00."}
    b = {"id": "b", "parents": ["a"], "base_revision": 7, "author": "Bo", "text": "Close at 17:00."}
    draft = {"text": "The garden is open Sunday, 10:00–17:00.", "applied": ["a", "b"],
             "alternatives": [], "explanation": "The two changes set different ends of the opening interval."}
    first, p1 = run([b, a, a], draft)
    second, p2 = run([a, b], draft)
    assert first == second and p1 == p2 == [["a", "b"]]
    assert first["status"] == "merged" and first["applied"] == ["a", "b"]


def test_unresolved_meaning_and_incomplete_draft_keep_sources_visible():
    a = {"id": "a", "parents": [], "base_revision": 7, "author": "Ada", "text": "Open at 10:00."}
    b = {"id": "b", "parents": [], "base_revision": 7, "author": "Bo", "text": "Keep closed Sunday."}
    unresolved = {"text": "The garden is open Sunday at 10:00.", "applied": ["a"],
                  "alternatives": [{"update_ids": ["b"], "proposal": "Closed Sunday.",
                                    "reason": "Contradicts the opening request."}], "explanation": "Ask the editors."}
    result, _ = run([b, a], unresolved)
    assert result["status"] == "unresolved" and result["alternatives"][0]["update_ids"] == ["b"]
    rejected, _ = run([b, a], {**unresolved, "alternatives": []})
    assert rejected["status"] == "rejected" and rejected["text"] == BASE["text"]
    assert {u["id"] for u in rejected["updates"]} == {"a", "b"}
    assert {alt["update_ids"][0] for alt in rejected["alternatives"]} == {"a", "b"}


def test_conflicting_transport_id_is_rejected_before_semantic_interpretation():
    a = {"id": "a", "parents": [], "base_revision": 7, "author": "Ada", "text": "Open at 10:00."}
    result, presentations = run([a, {**a, "text": "Close all day."}],
                                {"text": "", "applied": [], "alternatives": [], "explanation": ""})
    assert result["status"] == "rejected" and "Conflicting deliveries" in result["explanation"]
    assert presentations == []


class IncrementalInterpreter:
    def __init__(self, lam, draft, calls):
        self.lam, self.draft, self.calls = lam, draft, calls

    def run(self, session):
        def do(name, args):
            result = SURFACE.apply(session, name, args)
            assert result.kind not in ("rejected", "refused", "error", "quiesced"), (name, args, result.text)
            return result

        def call(function, to, **inputs):
            return do("call", {"function": function, "to": to, "inputs": inputs})

        if self.lam.fn_name == "interpret_update":
            self.calls.append(self.lam.in_["update"]["id"])
            do("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": self.draft})
        else:
            call("prepare_step", "let/step", base="args/base", current="args/current", update="args/update")
            kind = do("read", {"path": "let/step/kind"}).value
            if kind == "duplicate":
                do("write", {"path": "return", "type": format_type(self.lam.type.returns),
                             "source": "args/current"})
            elif kind == "invalid":
                call("reject_step", "return", current="args/current", update="args/update", step="let/step")
            else:
                call("interpret_update", "let/draft", base="args/base", current="args/current",
                     update="args/update", policy="args/policy")
                call("finish_step", "return", current="args/current", update="args/update",
                     step="let/step", draft="let/draft")
        assert session.finish()


def test_incremental_reduction_preserves_prior_updates_and_detects_redelivery():
    a = {"id": "a", "parents": [], "base_revision": 7, "author": "Ada", "text": "Open at 10:00."}
    b = {"id": "b", "parents": ["a"], "base_revision": 7, "author": "Bo", "text": "Close at 17:00."}
    current, _ = run([a], {"text": "The garden opens Sunday at 10:00.", "applied": ["a"],
                           "alternatives": [], "explanation": "Opening time added."})
    draft = {"text": "The garden is open Sunday, 10:00–17:00.", "applied": ["a", "b"],
             "alternatives": [], "explanation": "Closing time added."}
    entry = ENTRY.parent / "apply_update.nl"

    def step(update, proposal):
        calls = []
        rt = Runtime(lambda lam: IncrementalInterpreter(lam, proposal, calls))
        outcome, value = rt.run_root(load(entry, {"base": BASE, "current": current, "update": update,
                                                 "policy": "Preserve compatible opening-hour details."}))
        assert outcome.kind == "done", outcome.detail
        return dump(value), calls

    result, calls = step(b, draft)
    assert result["status"] == "merged" and result["applied"] == ["a", "b"] and calls == ["b"]
    duplicate, calls = step(a, draft)
    assert duplicate == current and calls == []
    conflict, calls = step({**a, "text": "Close all day."}, draft)
    assert conflict["status"] == "rejected" and calls == []
    incomplete, calls = step(b, {**draft, "applied": ["a"]})
    assert incomplete["status"] == "rejected" and incomplete["text"] == current["text"]
    assert incomplete["alternatives"][-1]["update_ids"] == ["b"]
