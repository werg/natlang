from applications.test_explorer import Explorer, GraphCase, graph_violations
from natlang.surface import ToolSurface
from natlang.types import format_type


SURFACE = ToolSurface()


class Analyst:
    def __init__(self, lam):
        self.lam = lam

    def run(self, session):
        if self.lam.fn_name == "select":
            value = {"ids": [self.lam.in_["cases"][0]["id"]], "reason": "Probe a dependency edge."}
        else:
            value = {"findings": ["Dependency order violated."], "unknowns": [], "followups": []}
        result = SURFACE.apply(session, "write", {"path": "return",
                                                 "type": format_type(self.lam.type.returns), "value": value})
        assert result.kind not in ("rejected", "refused", "error"), result.text
        assert session.finish()


def test_graph_oracle_sees_dependency_and_completion_errors():
    tasks = [{"id": "a", "needs": [], "description": "first"},
             {"id": "b", "needs": ["a"], "description": "second"}]
    good = {"tasks": tasks, "order": ["a", "b"], "blocked": [], "finished": True}
    assert graph_violations(tasks, good) == []
    bad = {**good, "order": ["b", "a"]}
    assert graph_violations(tasks, bad) == ["dependency-order"]
    assert "premature-finish" in graph_violations(tasks, {**good, "order": [], "blocked": ["a", "b"]})


def test_explorer_selects_runs_and_shrinks_against_same_oracle(tmp_path):
    tasks = ({"id": "a", "needs": [], "description": "first"},
             {"id": "b", "needs": ["a"], "description": "second"},
             {"id": "c", "needs": [], "description": "unrelated"})
    calls = []

    def backend(current, seed):
        calls.append(([t["id"] for t in current], seed))
        ids = [t["id"] for t in current]
        order = [x for x in ("b", "a", "c") if x in ids]
        state = {"tasks": current, "order": order, "blocked": [], "finished": True}
        return "done", state, {"sha256": "trace", "events": []}

    explorer = Explorer(analyst_factory=lambda lam: Analyst(lam),
                        target_factory=lambda lam: None, model_id="scripted",
                        analyst_seed=5, target_seed=8, trace_dir=tmp_path,
                        backend=backend)
    result = explorer.run("Does the graph order hold?", [GraphCase(
        "case-a", "ordering", "An edge and unrelated task", tasks)], budget=1)
    observation = result["observations"][0]
    assert observation["status"] == "violated"
    assert observation["violations"] == ["dependency-order"]
    assert observation["minimized"] == ["b"]
    assert all(seed == calls[0][1] for _, seed in calls)
    assert result["traces"]["selection"]["path"] and result["assessment"]["findings"]


def test_real_planner_runtime_handles_empty_graph():
    class EmptyPlanner:
        def __init__(self, lam):
            self.lam = lam

        def run(self, session):
            assert self.lam.fn_name == "plan"
            result = SURFACE.apply(session, "call", {"function": "prepare", "to": "return",
                                                     "inputs": {"tasks": "args/tasks"}})
            assert result.kind not in ("rejected", "refused", "error"), result.text
            assert session.finish()

    explorer = Explorer(analyst_factory=lambda lam: Analyst(lam),
                        target_factory=lambda lam: EmptyPlanner(lam),
                        model_id="scripted", analyst_seed=2, target_seed=3)
    result = explorer.run("Is empty graph complete?", [GraphCase(
        "empty", "boundary", "No tasks", ())], budget=1)
    observation = result["observations"][0]
    assert observation["status"] == "done" and observation["violations"] == []
