import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/neuralese"))

from natlang_neuralese.data.text_corpus import _hydrate_tool_argument_blocks
from natlang_neuralese.train.trajectories import render


def test_tool_argument_neuralese_body_is_json_escaped_and_rejoined_exactly():
    body = 'quoted "value"; backslash \\; first line\nsecond line'
    messages = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "save",
                "arguments": [
                    {"type": "text", "text": '{"value":"'},
                    {"type": "neuralese", "id": "block-1"},
                    {"type": "text", "text": '"}'},
                ],
            },
        }],
    }]

    hydrated, used = _hydrate_tool_argument_blocks(messages, {"block-1": body})
    rendered = render(hydrated, lambda name: {"type": "text", "text": name}, {})
    arguments = rendered[0]["tool_calls"][0]["function"]["arguments"]

    assert used == ["block-1"]
    assert json.loads(arguments) == {"value": body}
    escaped_body = json.dumps(body, ensure_ascii=False)[1:-1]
    assert arguments == '{"value":"' + escaped_body + '"}'
