from natlang.invocation import RunOptions
from natlang.meta import SourceWorkspace, meta_capabilities
from natlang.runtime import Runtime
from natlang.values import load_program


def _workspace():
    return SourceWorkspace({"cell": {"args": {"n": "Num"}, "returns": "Num",
                                     "code": "return args.n + 1;"}}, "cell")


def test_checked_meta_operations_and_edited_cell_revision():
    workspace = _workspace()
    assert workspace.describe()["signature"] == "cell(n: Num) -> Num"
    assert workspace.type_check("Num", "Num")["fits"] is True
    edited = workspace.edited("cell", {"args": {"n": "Num"}, "returns": "Num",
                                       "code": "return args.n + 2;"})
    assert edited.graph.revision != workspace.graph.revision
    old = workspace.invoke("cell", {"n": 2}, agent_factory=None, options=RunOptions())
    new = edited.invoke("cell", {"n": 2}, agent_factory=None, options=RunOptions())
    assert (old.outcome, old.value, new.value) == ("done", 3, 4)
    assert old.trace[0]["parent_call_id"] is None


def test_eval_can_use_selected_meta_capability_with_parent_link():
    workspace = _workspace()
    options = RunOptions()
    caps = meta_capabilities(workspace, agent_factory=None, options=options)
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "effects": ["meta.invoke"],
        "code": "const child = fx.meta.invoke({name: 'cell', inputs: {n: 5}, parent_call_id: 'root@1'}); "
                "return child.value;"}})
    out, value = Runtime(None, capabilities={"meta.invoke": caps["meta.invoke"]}).run_root(root)
    assert (out.kind, value) == ("done", 6)


def test_child_run_cannot_reset_parent_episode_budget():
    workspace = SourceWorkspace({"cell": {"args": {}, "returns": "Num",
                                       "instructions": "Return one."}}, "cell")
    options = RunOptions(max_episodes=1)
    parent = Runtime(None, options=options)
    assert parent._budget.reserve()
    parent.episodes_started = 1
    child = workspace.invoke("cell", {}, agent_factory=None, options=options,
                             max_episodes=1, parent_runtime=parent,
                             parent_call_id="$root@1")
    assert child.outcome == "quiesced" and parent._budget.used == 1
    assert child.trace[0]["parent_call_id"] == "$root@1"


def test_child_logical_identity_includes_parent_call_for_seed_isolation():
    workspace = SourceWorkspace({"cell": {"args": {}, "returns": "Num",
                                       "instructions": "Return one."}}, "cell")
    class Agent:
        def run(self, session):
            session.apply("write", {"path": "return", "value": 1})
            session.finish()
    options = RunOptions(max_episodes=5)
    parent = Runtime(None, options=options)
    first = workspace.invoke("cell", {}, agent_factory=lambda lam: Agent(), options=options,
                             parent_runtime=parent, parent_call_id="parent-a@1")
    second = workspace.invoke("cell", {}, agent_factory=lambda lam: Agent(), options=options,
                              parent_runtime=parent, parent_call_id="parent-b@1")
    one = next(e for e in first.trace if e["kind"] == "invocation" and e["phase"] == "start")
    two = next(e for e in second.trace if e["kind"] == "invocation" and e["phase"] == "start")
    assert one["call_id"] == "parent-a@1/child@1"
    assert two["call_id"] == "parent-b@1/child@1"
    assert parent.episodes_started == 2
