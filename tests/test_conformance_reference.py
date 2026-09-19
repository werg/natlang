"""Run every migrated conformance program (those with a `reference:` block) through the real harness,
driven by a small scripted agent that follows the reference instead of a model, and grade the outcome
the way `natlang/checks.py` does.

Reference block conventions (see conformance/README.md):
  - keyed by function name (the root lambda's `function:`, or a code-base function's name).
  - a root/non-leaf function (one whose pseudocode calls others) has `calls:`: an ordered list of
    `{tool: <name>, args: {...}}`. `tool: glue` is shorthand for `run_code` followed by a `write` of its
    result to `path` with `type`.
  - a natural-language leaf has `answer`, `answer_by` (keyed on the value of its `item` parameter if it
    has one, else its first parameter; the key is the value itself for a Text argument, or
    `json.dumps(value, sort_keys=True)` otherwise), or `blocker`.
  - an `answer_by`/`answer` entry may instead be `{blocker: "...", then: <value>}`: the first time this
    leaf instance runs it reports that blocker; if the same pending node is resumed, it writes `then`.
  - a leaf may give `variants: [{if_body_contains: "...", answer: ...}, ..., {answer: ...}]`, matched in
    order against the instance's (possibly edited) instructions text, for copy-edit-call scenarios.
  - `known_issue: "..."` marks a program whose reference cannot be driven through the current harness;
    its test is skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from natlang import gbnf
from natlang.checks import grade
from natlang.gen.policy import native_text
from natlang.host import load
from natlang.native import call_grammar
from natlang.runtime import Runtime, Session
from natlang.surface import ToolSurface
from natlang.types import TypeEnv, format_type
from natlang.values import dump, load_program

ROOT = Path(__file__).resolve().parent.parent
PROGRAMS = ROOT / "conformance" / "programs"
S = ToolSurface()


def _key(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


class ScriptedRef:
    """Drives one episode from its `reference:` entry (see module docstring)."""

    def __init__(self, lam, spec_by_fn, attempts):
        self.lam, self.spec_by_fn, self.attempts = lam, spec_by_fn, attempts

    def do(self, session, name, args):
        assert gbnf.accepts(call_grammar(S.tools(session)), native_text([(name, args)])), (name, args)
        r = S.apply(session, name, args)
        assert r.kind not in ("rejected", "refused", "error"), f"{name}({args}): {r.text}"
        return r

    def _leaf_key(self):
        if "item" in self.lam.in_:
            return _key(self.lam.in_["item"])
        return _key(next(iter(self.lam.in_.values()), None))

    def run(self, session):
        fn = self.lam.fn_name
        spec = self.spec_by_fn.get(fn)
        assert spec is not None, f"no reference entry for function {fn!r}"
        d = lambda n, a: self.do(session, n, a)

        if "calls" in spec:
            for call in spec["calls"]:
                tool = call["tool"]
                if tool == "glue":
                    result = d("run_code", {"code": call["code"]}).value
                    d("write", {"path": call["path"], "type": call["type"], "value": result})
                else:
                    r = d(tool, call.get("args", {}))
                    if tool == "report_blocker":
                        return r.text
            assert session.finish()
            return None

        if "variants" in spec:
            entry = next((v for v in spec["variants"] if "if_body_contains" not in v
                         or v["if_body_contains"] in self.lam.body), spec["variants"][-1])
            return self._answer(session, d, entry)

        if "blocker" in spec:
            r = d("report_blocker", {"missing": spec["blocker"]})
            return r.text

        if "answer" in spec:
            return self._answer(session, d, spec)

        if "answer_by" in spec:
            key = self._leaf_key()
            assert key in spec["answer_by"], f"{fn}: no answer_by entry for {key!r}"
            return self._answer(session, d, {"answer": spec["answer_by"][key]})

        raise AssertionError(f"reference entry for {fn} has none of calls/answer/answer_by/blocker/variants")

    def _answer(self, session, d, entry):
        value = entry["answer"]
        if isinstance(value, dict) and set(value) >= {"blocker"}:
            seen = self.attempts.setdefault(id(self.lam), False)
            if not seen:
                self.attempts[id(self.lam)] = True
                r = d("report_blocker", {"missing": value["blocker"]})
                return r.text
            value = value["then"]
        rtype = format_type(self.lam.type.returns)
        d("write", {"path": "return", "type": rtype, "value": value})
        assert session.finish()
        return None


def _load_doc(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _load_root(doc: dict, path: Path):
    if doc.get("program_file"):
        return load(path.parent / doc["program_file"], doc.get("inputs") or {})
    if "program" in doc:
        return load(path, doc.get("inputs") or {})
    return load_program(doc["start_state"])


PROGRAM_FILES = sorted(PROGRAMS.glob("*.yaml"))
WITH_REFERENCE = [p for p in PROGRAM_FILES if "reference" in (yaml.safe_load(p.read_text()) or {})]


@pytest.mark.parametrize("path", WITH_REFERENCE, ids=lambda p: p.stem)
def test_reference_reaches_a_passing_verdict(path):
    doc = _load_doc(path)
    reference = doc["reference"]
    if "known_issue" in reference:
        pytest.skip(reference["known_issue"])

    root = _load_root(doc, path)
    attempts: dict = {}
    rt = Runtime(lambda lam: ScriptedRef(lam, reference, attempts))
    out, value = rt.run_root(root)

    kind = out.kind
    note = out.detail if kind == "quiesced" else ""
    graded_value = dump(value) if kind == "done" else None
    verdict, details = grade(doc["expect"], kind, graded_value, note, judge=None)
    assert verdict in ("yes", "?"), f"{path.name}: verdict {verdict}: {details}"
