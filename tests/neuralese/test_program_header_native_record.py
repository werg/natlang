"""program_header_native_record: system soft pieces -> program header, tools dropped, other messages unchanged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "neuralese"))
from natlang_neuralese.data.prompt_rendering import program_header_native_record  # noqa: E402

PIECES = [{"id": "g", "scope": "generic", "text": "You are a natlang runtime. Use eval."},
          {"id": "p", "scope": "program", "text": "This call is a directory reducer: fold the files."}]


def record(system_parts):
    return {"id": "r", "messages": [{"role": "system", "content": system_parts},
                                    {"role": "user", "content": [{"type": "read", "name": "result:x", "source": "v"}]}],
            "tools": [{"type": "function", "function": {"name": "eval"}}], "target": {"role": "assistant", "content": "ok"}}


def test_generic_only_drops_system_turn_and_tools():
    out = program_header_native_record(record([{"type": "soft", "name": "s"}]), {"s": PIECES[0]["text"]}, PIECES)
    assert out["messages"][0]["role"] == "user" and out["tools"] == []
    assert out["system_prompt_mode"] == "none" and out["prompt_rendering"] == "program-header"
    assert out["messages"][0]["content"][0]["type"] == "read"


def test_program_paragraph_kept_as_plain_header():
    texts = {"s": PIECES[0]["text"], "t": "\n\n" + PIECES[1]["text"]}
    out = program_header_native_record(record([{"type": "soft", "name": "s"}, {"type": "soft", "name": "t"}]), texts, PIECES)
    assert out["messages"][0] == {"role": "system", "content": PIECES[1]["text"]}
    assert out["system_prompt_mode"] == "program-own" and out["target"] == {"role": "assistant", "content": "ok"}
