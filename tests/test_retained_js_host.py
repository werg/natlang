import time

from hosts.retained_js import RetainedJSExecutor
from natlang.runtime import Runtime, Session
from natlang.types import TypeEnv
from natlang.values import load_program


def _session(engine):
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "instructions": "Inspect host data."}})
    return Session(Runtime(None, executors={engine.name: engine}, engine_selection=True), root, TypeEnv())


def test_retained_buffer_and_process_survive_between_evals(tmp_path):
    file = tmp_path / "frame.bin"
    file.write_bytes(b"abc\x00def")
    with RetainedJSExecutor() as engine:
        session = _session(engine)
        first = session.apply("run_code", {"engine": engine.name,
            "code": f"host.readBytes({str(file)!r})"})
        assert first.kind == "ok" and first.value.startswith("buffer-")
        second = session.apply("run_code", {"engine": engine.name,
            "code": f"host.buffer({first.value!r}).length"})
        assert second.kind == "ok" and second.value == 7
        job = session.apply("run_code", {"engine": engine.name,
            "code": "host.start(['node','-e','setTimeout(() => process.stdout.write(\"built\"), 20)'])"})
        assert job.kind == "ok" and job.value.startswith("job-")
        for _ in range(50):
            result = session.apply("run_code", {"engine": engine.name,
                "code": f"host.poll({job.value!r})"})
            if result.value["status"] == "finished":
                break
            time.sleep(0.01)
        assert result.value["status"] == "finished" and result.value["stdout"] == "built"
        assert [e["operation"] for e in engine.drain_events()] == ["file.readBytes", "process.start"]
    assert engine.process.poll() is not None


def test_native_result_cannot_cross_typed_boundary():
    with RetainedJSExecutor() as engine:
        session = _session(engine)
        result = session.apply("run_code", {"engine": engine.name, "code": "Buffer.from('abc')"})
        assert result.kind == "error"


def test_authored_function_uses_retained_engine(tmp_path):
    file = tmp_path / "name.txt"
    file.write_text("Dana")
    root = load_program({"$lambda": {"type": "Lambda<{}, Text>", "engine": "node-retained",
                                      "code": f"return host.readText({str(file)!r});"}})
    with RetainedJSExecutor() as engine:
        out, value = Runtime(None, executors={engine.name: engine}, engine_selection=True).run_root(root)
        assert (out.kind, value) == ("done", "Dana")
