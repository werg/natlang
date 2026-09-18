"""Run conformance/harness/*.yaml against the runtime. No model involved."""
import copy
from pathlib import Path

import pytest
import yaml

from natlang.agents import StubAgent
from natlang.nodes import Lambda
from natlang.runtime import Runtime, Session
from natlang.types import TypeEnv
from natlang.values import dump, load_program

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "conformance" / "harness").glob("*.yaml"))


def _read(session, path):
    from natlang.paths import parse_path
    from natlang.refs import resolve

    p = parse_path(path)
    if p.meta:
        return session._meta(p, resolve(session.lam, session.outer_env, p))
    return dump(resolve(session.lam, session.outer_env, p).get())


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_harness_script(script):
    doc = yaml.safe_load(script.read_text())
    failures = []
    rt = Runtime(lambda lam: StubAgent(doc.get("stub_agent") or {}, lam, failures))
    root = load_program(doc["setup"])
    assert isinstance(root, Lambda)
    session = Session(rt, root, TypeEnv())

    for i, step in enumerate(doc["script"]):
        exp = step["expect"]
        before = copy.deepcopy(dump(root))
        episodes_before = rt.episodes_started
        r = session.act(step["action"])
        where = f"step {i} ({step['action'].splitlines()[0]}): {r.text!r}"

        assert r.kind == exp["result"], where
        for code in exp.get("codes", []):
            assert code in r.codes, where
        for path in exp.get("paths", []):
            assert path in [d.path for d in r.diags], where
        if exp.get("tree_changed") is False:
            assert dump(root) == before, where
        if "problems" in exp:
            assert f"{exp['problems']['holes']} holes" in r.text, where
        if "value" in exp:
            assert dump(r.value) == exp["value"], where
        if "journal_entries" in exp:
            assert len(root.journal) == exp["journal_entries"], where
        if "note_contains" in exp:
            assert exp["note_contains"] in r.text, where
        if "reduced" in exp:
            assert exp["reduced"] in r.text, where
        if "episodes_started" in exp:
            assert rt.episodes_started - episodes_before == exp["episodes_started"], where
        reads = exp.get("read") or []
        for rd in [reads] if isinstance(reads, dict) else reads:
            assert _read(session, rd["path"]) == rd["value"], f"{where} read {rd['path']}"
    assert not failures, failures
