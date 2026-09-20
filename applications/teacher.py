"""Current Bonsai teacher/server adapter shared by application CLIs."""
from __future__ import annotations

from pathlib import Path

from natlang.decoder import LlamaServerDecoder
from natlang.tool_agent import ToolAgent


ROOT = Path(__file__).resolve().parent.parent
PROMPT = (ROOT / "natlang/prompts/tools_teacher_compact.md").read_text()


def teacher_factory(server: str):
    decoder = LlamaServerDecoder(
        server, timeout=900,
        chat_extra={"thinking_budget_tokens": 256, "top_p": 0.95, "top_k": 20,
                    "chat_template_kwargs": {"reasoning_effort": "low"}},
        tool_aliases={"call": "call_function"}, json_text_values=True)
    return lambda lam: ToolAgent(decoder, system_prompt=PROMPT, temperature=0,
                                 validation_feedback="caller")
