import time

from hosts.job_stream import JobEventSource
from hosts.retained_js import RetainedJSExecutor
from natlang.execution import CrispRequest
from natlang.runtime import Runtime
from natlang.streams import StreamBuffer
from natlang.values import load_program


def test_process_completion_arrives_after_user_event_with_stable_identity():
    with RetainedJSExecutor() as engine:
        job_id = engine.run(CrispRequest(
            "host.start(['node','-e','setTimeout(() => process.stdout.write(\"ok\"), 25)'])",
            {}, False, "start"), lambda *_: None)
        source = JobEventSource(engine)
        source.watch(job_id)
        source.put_user("change-1")
        root = load_program({"$fold": {"type": "Fold<{ kind: Text, id: Text }, Text[]>",
            "init": [], "step": {"$lambda": {
                "type": "Lambda<{ acc: Text[], item: { kind: Text, id: Text } }, Text[]>",
                "code": "return [...args.acc, args.item.kind + ':' + args.item.id];"}}}})
        root.over = StreamBuffer(source)
        runtime = Runtime(None)
        out, _ = runtime.run_root(root)
        assert out.kind == "waiting" and root.acc == ["user:change-1"]
        for _ in range(50):
            out, _ = runtime.run_root(root)
            if root.over.position == 2:
                break
            time.sleep(0.01)
        assert root.acc == ["user:change-1", f"job_complete:{job_id}"]
        source.close()
        out, value = runtime.run_root(root)
        assert out.kind == "done" and value == root.acc
