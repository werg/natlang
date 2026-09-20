from natlang.decoder import ChatTurn, LlamaServerDecoder
from natlang.invocation import Invocation, ModelSettings, RunOptions, SeedPolicy
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.trace import TraceRecorder
from natlang.values import load_program


class RecordedTurns:
    def __init__(self):
        self.seeds, self.requests = [], []

    def chat(self, messages, tools, *, temperature, seed, max_tokens):
        self.seeds.append(seed)
        self.requests.append((temperature, max_tokens))
        if len(self.seeds) == 1:
            return ChatTurn(calls=[("write", {"path": "return", "value": True})], completion_tokens=1)
        return ChatTurn(text="done", completion_tokens=1)


def test_seed_derivation_vectors_and_attempts():
    policy = SeedPolicy("derived", 43)
    assert policy.seed("", 1, "model-turn", 0) == 1079124865
    assert policy.seed("return/0", 1, "model-turn", 0) == 365401298
    assert policy.seed("return/0", 2, "model-turn", 0) == 1411459069
    assert policy.seed("return/0", 1, "review", 1) == 1880645260
    assert SeedPolicy().seed("any", 9, "review", 14) == 0
    assert SeedPolicy("backend").seed("any", 1, "model-turn") is None
    assert Invocation("run-a", "return/0", 2).call_id == Invocation("run-b", "return/0", 2).call_id


def test_tool_turn_seed_is_scoped_to_logical_invocation():
    def run(run_id):
        driver = RecordedTurns()
        root = load_program({"$lambda": {"type": "Lambda<{}, Bool>", "instructions": "Return true."}})
        opts = RunOptions(seed=SeedPolicy("derived", 43), run_id=run_id)
        outcome, value = Runtime(lambda lam: ToolAgent(driver), options=opts).run_root(root)
        assert (outcome.kind, value) == ("done", True)
        return driver.seeds

    assert run("one") == run("two") == [1079124865, SeedPolicy("derived", 43).seed("", 1, "model-turn", 1)]


def test_decoder_deadline_context_restores_without_mutating_backend():
    decoder = LlamaServerDecoder(timeout=120)
    decoder.deadline = None
    with decoder.request_scope(deadline=0):
        try:
            decoder.request_timeout()
        except TimeoutError:
            pass
        else:
            raise AssertionError("expired request was accepted")
    assert decoder.deadline is None and decoder.request_timeout() == 120


def test_run_owned_model_settings_and_separate_world_randomness():
    driver = RecordedTurns()
    root = load_program({"$lambda": {"type": "Lambda<{}, Bool>", "instructions": "Return true."}})
    options = RunOptions(seed=SeedPolicy("derived", 43), world_seed=43,
                         model=ModelSettings(temperature=0.7, max_tokens=20, turn_tokens=5))
    outcome, value = Runtime(lambda lam: ToolAgent(driver), options=options).run_root(root)
    assert (outcome.kind, value) == ("done", True)
    assert driver.requests == [(0.7, 5), (0.7, 5)]
    assert options.world_rng("fixture").random() == options.world_rng("fixture").random()
    assert options.world_rng("other").random() != options.world_rng("fixture").random()
    assert options.seed.backend_range.endswith("2147483647)")


def test_unbounded_defaults_still_observe_each_model_request():
    settings = ModelSettings()
    assert settings.max_turns is settings.max_tokens is settings.max_seconds is None
    driver = RecordedTurns()
    root = load_program({"$lambda": {"type": "Lambda<{}, Bool>", "instructions": "Return true."}})
    trace = TraceRecorder({"run_id": "unbounded", "source_sha256": "fixture"})
    outcome, value = Runtime(lambda lam: ToolAgent(driver), trace_sink=trace).run_root(root)
    assert (outcome.kind, value) == ("done", True)
    requests = [row for row in trace.events if row["kind"] == "model_request"]
    assert [row["phase"] for row in requests] == ["start", "end", "start", "end"]
    assert all(row["observed_at"] and row["elapsed_ms"] >= 0 for row in trace.events)
    assert driver.requests == [(0.2, None), (0.2, None)]
