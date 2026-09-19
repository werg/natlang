from natlang.runtime import Runtime, Session
from natlang.surface import ToolSurface
from natlang.surface_projection import project_legacy_turn
from natlang.types import TypeEnv
from natlang.values import load_program


class StubEngine:
    name = "stub-v1"

    def __init__(self):
        self.calls = []

    def run(self, request, effect):
        self.calls.append(request)
        return 7


def test_new_surface_requires_an_available_engine():
    stub = StubEngine()
    rt = Runtime(None, executors={"quickjs-isolated": StubEngine(), "stub-v1": stub}, engine_selection=True)
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "instructions": "Find seven."}})
    session = Session(rt, root, TypeEnv())
    tool = next(t for t in ToolSurface().tools(session) if t["function"]["name"] == "run_code")
    params = tool["function"]["parameters"]
    assert params["required"] == ["code", "engine"]
    assert params["properties"]["engine"]["enum"] == ["quickjs-isolated", "stub-v1"]
    assert session.apply("run_code", {"code": "7"}).kind == "rejected"
    assert session.apply("run_code", {"code": "7", "engine": "missing"}).kind == "error"
    answer = session.apply("run_code", {"code": "7", "engine": "stub-v1"})
    assert answer.kind == "ok" and answer.value == 7 and len(stub.calls) == 1


def test_authored_crisp_engine_is_bound_at_load_time():
    stub = StubEngine()
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "engine": "stub-v1", "code": "return 7;"}})
    out, value = Runtime(None, executors={"stub-v1": stub}, engine_selection=True).run_root(root)
    assert (out.kind, value) == ("done", 7)
    out, _ = Runtime(None, engine_selection=True).run_root(load_program({"$lambda": {
        "type": "Lambda<{}, Num>", "engine": "stub-v1", "code": "return 7;"}}))
    assert out.kind == "quiesced" and "unavailable" in out.detail


def test_legacy_projection_preserves_original():
    row = {"target": {"function": {"name": "run_code", "arguments": '{"code":"2+5"}'}}}
    projected = project_legacy_turn(row)
    assert row["target"]["function"]["arguments"] == '{"code":"2+5"}'
    assert '"engine": "quickjs-isolated"' in projected["target"]["function"]["arguments"]
    assert projected["surface_projection"]["to"] == "tools-v3"
