from natlang.decoder import ChatTurn
from scripts.probe_engine_surface import run_profile


class Driver:
    def __init__(self, explicit):
        self.explicit = explicit
        self.turn = 0
        self.usage = {"turns": 0, "completion_tokens": 0}

    def chat(self, messages, tools, *, temperature, seed, max_tokens):
        self.turn += 1
        self.usage["turns"] += 1
        self.usage["completion_tokens"] += 1
        if self.turn == 1:
            schema = next(t for t in tools if t["function"]["name"] == "run_code")["function"]["parameters"]
            assert ("engine" in schema["required"]) == self.explicit
            args = {"code": "sum(args.values)"}
            if self.explicit:
                args["engine"] = "quickjs-isolated"
            return ChatTurn(calls=[("run_code", args)], completion_tokens=1)
        if self.turn == 2:
            return ChatTurn(calls=[("write", {"path": "return", "type": "Num", "value": 20})],
                            completion_tokens=1)
        return ChatTurn(text="done", completion_tokens=1)


def test_paired_probe_records_surface_and_outcomes():
    old = run_profile(Driver(False), explicit_engine=False, seed=43, model_id="recorded")
    new = run_profile(Driver(True), explicit_engine=True, seed=43, model_id="recorded")
    assert old["correct"] and new["correct"]
    assert old["turns"] == new["turns"] == 3
    assert not old["false_success"] and not new["wrong_engine"]
    assert old["trace"][0]["tool_schema"] == "tools-v2"
    assert new["trace"][0]["tool_schema"] == "tools-v3"
