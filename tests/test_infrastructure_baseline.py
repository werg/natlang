"""Compact semantic baseline for infrastructure extraction."""
import json
from pathlib import Path

import pytest

from natlang.runtime import Runtime
from natlang.runtime import Session
from natlang.surface import ToolSurface
from natlang.types import TypeEnv
from natlang.values import dump, load_program
from test_conformance_reference import ScriptedRef, _load_doc, _load_root

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / "conformance/infrastructure_baseline.json").read_text())


@pytest.mark.parametrize("case", BASELINE["fixtures"], ids=lambda c: c["file"])
def test_compatibility_fixture(case):
    path = ROOT / "conformance/programs" / case["file"]
    doc = _load_doc(path)
    attempts = {}
    rt = Runtime(lambda lam: ScriptedRef(lam, doc["reference"], attempts))
    out, value = rt.run_root(_load_root(doc, path))
    assert out.kind == case["outcome"]
    if "value" in case:
        assert dump(value) == case["value"]
    assert rt.emitted == case["effects"]


def test_optional_missing_null_and_empty_are_three_distinct_observations():
    def opening(args):
        root = load_program({"$lambda": {"type": "Lambda<{ document?: Text | Null }, Text>",
                                         "instructions": "Return the document.", "args": args}})
        return ToolSurface().opening_read(Session(Runtime(None), root, TypeEnv()))[2]
    assert "not supplied" in opening({})
    assert "read-only): null" in opening({"document": None})
    assert 'read-only): ""' in opening({"document": ""})


@pytest.mark.parametrize("case", BASELINE["action_fixtures"], ids=lambda c: c["file"])
def test_action_outcomes_are_frozen(case):
    path = ROOT / "conformance/programs" / case["file"]
    session = Session(Runtime(None), _load_root(_load_doc(path), path), TypeEnv())
    for call in case["calls"]:
        result = session.apply(call["name"], call["arguments"])
        assert (result.kind, result.codes) == (call["outcome"], call["codes"])
    assert session.finish() is case["finish"]
