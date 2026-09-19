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


CHILD = "set return : Lambda<{}, Bool>\ninstructions: Decide."


def _root():
    return load_program({"$lambda": {"type": "Lambda<{}, Bool>", "instructions": "Decide."}})


def test_identical_child_is_refused():
    """Delegating the same task to a child cannot make progress; the harness refuses to start it."""
    class Delegator:
        def run(self, session):
            session.act(CHILD)
            self.result = session.act("reduce return")
            return "gave up"

    agents = []
    rt = Runtime(lambda lam: agents.append(Delegator()) or agents[-1])
    out, _ = rt.run_root(_root())
    assert rt.episodes_started == 1                      # the identical child never ran
    assert "identical to a lambda already being reduced" in agents[0].result.text


def test_reopen_needs_a_child_origin():
    """A value the agent wrote itself cannot be reopened; it can simply be set again."""
    results = []

    class SelfReopener:
        def run(self, session):
            session.act("set return : Bool\ntrue")
            results.append(session.act("reopen return\nTry again."))
            return "done trying"

    Runtime(lambda lam: SelfReopener()).run_root(_root())
    assert results[0].kind == "rejected" and "no-origin" in results[0].codes


def test_pending_nesting_is_bounded():
    results = []

    class Nester:
        def run(self, session):
            path = "return"
            for _ in range(9):
                results.append(session.act(f"set {path} : Lambda<{{}}, Bool>\ninstructions: Decide differently {len(results)}."))
                path += "/return"
            return "stop"

    Runtime(lambda lam: Nester()).run_root(_root())
    kinds = [r.kind for r in results]
    assert kinds[:6] == ["ok"] * 6 and "too-deep" in results[6].codes


def test_run_budgets_bound_the_number_of_episodes():
    class Spawner:
        n = 0

        def run(self, session):
            Spawner.n += 1
            session.act(f"set return : Lambda<{{}}, Bool>\ninstructions: Variant {Spawner.n}.")
            session.act("reduce return")
            return "gave up"

    rt = Runtime(lambda lam: Spawner(), max_episodes=40, max_depth=5)
    out, _ = rt.run_root(_root())
    assert out.kind == "quiesced" and rt.episodes_started <= 5
