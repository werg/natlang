import json
import threading

from applications.teacher import _RecordingToolAgent
from natlang.decoder import ChatTurn
from natlang.runtime import Runtime
from natlang.values import load_program


class Decoder:
    def __init__(self):
        self.turns = 0

    def chat(self, messages, tools, *, temperature, seed, max_tokens):
        self.turns += 1
        if self.turns == 1:
            return ChatTurn(calls=[("write", {"path": "return", "value": True})],
                            completion_tokens=3, raw_response={"id": "first"})
        return ChatTurn(text="done", completion_tokens=2, raw_response={"id": "second"})


def test_raw_teacher_turns_are_written_after_episode(tmp_path):
    path = tmp_path / "turns.jsonl"
    root = load_program({"$lambda": {"type": "Lambda<{}, Bool>", "instructions": "Return true."}})
    decoder = Decoder()
    outcome, value = Runtime(lambda lam: _RecordingToolAgent(
        decoder, turns_path=path, turns_lock=threading.Lock())).run_root(root)
    assert (outcome.kind, value) == ("done", True)
    turns = [json.loads(line) for line in path.read_text().splitlines()]
    assert [turn["response"]["id"] for turn in turns] == ["first", "second"]
    assert turns[0]["executions"][0]["kind"] == "ok"
