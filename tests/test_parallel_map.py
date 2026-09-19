import threading
import time

from natlang.runtime import Runtime
from natlang.trace import TraceReader, TraceRecorder
from natlang.values import load_program


def program(items):
    return load_program({"$map": {"type": "Map<Num, Num>", "over": items,
        "fn": {"$lambda": {"type": "Lambda<{ item: Num }, Num>",
                           "instructions": "Double the item."}}}})


def test_parallel_map_preserves_slots_attempts_and_bounded_admission():
    lock = threading.Lock()
    active = peak = 0
    def factory(lam):
        class Agent:
            def run(self, session):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(0.005 * (6 - lam.in_["item"] % 6))
                    if lam.in_["item"] == 2 and lam.attempts == 1:
                        return "retry item two"
                    session.apply("write", {"path": "return", "value": lam.in_["item"] * 2})
                    session.finish()
                finally:
                    with lock:
                        active -= 1
        return Agent()

    recorder = TraceRecorder({"run_id": "parallel-fixture", "source_sha256": "fixture"})
    root = program(list(range(8)))
    rt = Runtime(factory, map_workers=3, parallel_model_safe=True, trace_sink=recorder)
    out, _ = rt.run_root(root)
    assert out.kind == "quiesced" and 1 < peak <= 3
    assert root.slots[0] == 0 and root.slots[2].attempts == 1
    out, value = rt.run_root(root)
    assert (out.kind, value) == ("done", [i * 2 for i in range(8)])
    assert rt.episodes_started == 9
    slots = TraceReader(recorder.events).of_kind("map_slot")
    assert {event["slot"] for event in slots} == set(range(8))


def test_parallel_map_defaults_to_serial_without_driver_attestation():
    root = load_program({"$map": {"type": "Map<Num, Num>", "over": [1, 2, 3],
        "fn": {"$lambda": {"type": "Lambda<{ item: Num }, Num>", "code": "return args.item*2;"}}}})
    out, value = Runtime(None, map_workers=4).run_root(root)
    assert (out.kind, value) == ("done", [2, 4, 6])


def test_parallel_map_shares_run_episode_budget():
    def factory(lam):
        class Agent:
            def run(self, session):
                session.apply("write", {"path": "return", "value": lam.in_["item"]})
                session.finish()
        return Agent()
    rt = Runtime(factory, max_episodes=2, map_workers=3, parallel_model_safe=True)
    out, _ = rt.run_root(program([1, 2, 3, 4]))
    assert out.kind == "quiesced" and rt.episodes_started == 2
