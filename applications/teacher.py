"""Current Bonsai teacher/server adapter shared by application CLIs."""
from __future__ import annotations

from pathlib import Path
import json
import threading

from natlang.decoder import LlamaServerDecoder
from natlang.tool_agent import ToolAgent


ROOT = Path(__file__).resolve().parent.parent
PROMPT = (ROOT / "natlang/prompts/tools_teacher_compact.md").read_text()


class _RecordingToolAgent(ToolAgent):
    def __init__(self, *args, turns_path: Path | None = None, turns_lock=None, **kwargs):
        self._captured_turns = []
        self._turns_path, self._turns_lock = turns_path, turns_lock
        super().__init__(*args, teacher_turns=self._captured_turns, **kwargs)

    def run(self, session):
        try:
            return super().run(session)
        finally:
            if self._turns_path is not None:
                with self._turns_lock:
                    with self._turns_path.open("a", encoding="utf-8") as stream:
                        for turn in self._captured_turns:
                            stream.write(json.dumps(turn, ensure_ascii=False) + "\n")


def teacher_factory(server: str, *, turns_path: Path | None = None,
                    validation_feedback: str = "caller"):
    decoder = LlamaServerDecoder(
        server,
        chat_extra={"thinking_budget_tokens": 256, "top_p": 0.95, "top_k": 20,
                    "chat_template_kwargs": {"reasoning_effort": "low"}},
        tool_aliases={"call": "call_function"},
        typed_alternatives=True)
    lock = threading.Lock()
    if turns_path is not None:
        turns_path = Path(turns_path)
        turns_path.parent.mkdir(parents=True, exist_ok=True)
        turns_path.touch(exist_ok=False)
    return lambda lam: _RecordingToolAgent(decoder, system_prompt=PROMPT, temperature=0,
                                           validation_feedback=validation_feedback, turns_path=turns_path,
                                           turns_lock=lock)
