import pytest

from natlang.runtime import Runtime
from natlang.scenario import ScenarioContract, admit
from natlang.trace import TraceReader
from natlang.values import load_program


def test_admission_requires_exact_action_destination_and_order(tmp_path):
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "instructions": "Write seven."}})
    class Agent:
        def run(self, session):
            session.apply("write", {"path": "return", "value": 7})
            session.finish()
    path = tmp_path / "trace.jsonl"
    out, value = Runtime(lambda lam: Agent(), trace_path=path).run_root(root)
    reader = TraceReader.open(path)
    assert admit(reader, ScenarioContract("done", 7, required_actions=(
        {"name": "write", "arguments": {"path": "return", "value": 7}},))) ["admitted"]
    with pytest.raises(ValueError, match="required ordered action"):
        admit(reader, ScenarioContract("done", 7, required_actions=(
            {"name": "write", "arguments": {"path": "return/size"}},)))


def test_replay_observations_never_reexecutes_effect(tmp_path):
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "effects": ["out.emit"],
        "code": "fx.out.emit({id: 'first'}); throw new Error('failed');"}})
    path = tmp_path / "trace.jsonl"
    rt = Runtime(None, trace_path=path)
    out, _ = rt.run_root(root)
    reader = TraceReader.open(path)
    replay = reader.replay_observations()
    assert replay["outcome"] == "quiesced"
    assert rt.emitted == [{"id": "first"}]
    assert admit(reader, ScenarioContract("quiesced", effects=(("out.emit", [{"id": "first"}]),)))["admitted"]
    assert rt.emitted == [{"id": "first"}]
