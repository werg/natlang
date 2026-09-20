from natlang.runtime import Runtime, Session
from natlang.trace import TraceReader
from natlang.types import TypeEnv
from natlang.values import dump_state, load_program


def test_trace_reconstructs_state_and_excludes_rejected_action(tmp_path):
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "instructions": "Write two."}})
    def agent(lam):
        class Driver:
            def run(self, session):
                assert session.apply("write", {"path": "return", "value": "wrong"}).kind == "rejected"
                assert session.apply("write", {"path": "return", "value": 2}).kind == "ok"
                assert session.finish()
                return None
        return Driver()

    path = tmp_path / "trace.jsonl"
    rt = Runtime(agent, trace_path=path)
    out, value = rt.run_root(root)
    reader = TraceReader.open(path)
    assert (out.kind, value) == ("done", 2)
    assert reader.final_state() == dump_state(value)
    assert reader.reconstruct() == dump_state(value)
    assert reader.of_kind("reduction")
    assert [e["outcome"] for e in reader.of_kind("action")] == ["rejected", "ok"]
    assert reader.of_kind("state")[1]["value"]["$lambda"].get("return") is None
    assert reader.coverage()["native_effects_replayable"] is False
    assert reader.manifest["run_id"] == rt.options.run_id


def test_effect_before_failure_is_ordered_and_reader_has_no_effects(tmp_path):
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "effects": ["out.emit"],
                                      "code": "fx.out.emit({id: 'first'}); throw new Error('failed');"}})
    path = tmp_path / "trace.jsonl"
    rt = Runtime(None, trace_path=path)
    out, _ = rt.run_root(root)
    assert out.kind == "quiesced" and rt.emitted == [{"id": "first"}]
    reader = TraceReader.open(path)
    effects = reader.of_kind("effect")
    assert [e["phase"] for e in effects] == ["requested", "completed"]
    assert reader.of_kind("eval")[-1]["phase"] == "failed"
    assert reader.reconstruct() == reader.final_state()
    assert rt.emitted == [{"id": "first"}]  # opening the trace did not execute it again
