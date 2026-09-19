"""Compact semantic baseline for infrastructure extraction."""
import json
from pathlib import Path

import pytest

from natlang.runtime import Runtime
from natlang.values import dump
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
