"""Serving mode is content-bound to the same rendering policy as native text SFT."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
from natlang_neuralese.data.prompt_rendering import (  # noqa: E402
    apply_prompt_rendering,
    checkpoint_prompt_rendering,
    prompt_rendering_identity,
)
from natlang_neuralese.serve.engine import Engine  # noqa: E402


def test_serving_program_header_uses_native_renderer_and_contract_identity():
    pieces = [
        {"scope": "generic", "text": "General runtime policy.\n\nThis call has one binding."},
        {"scope": "program", "text": "This call has one binding."},
    ]
    messages = [{"role": "system", "content": "General runtime policy.\n\nThis call has one binding."},
                {"role": "user", "content": "Do the task."}]
    rendered, tools = apply_prompt_rendering(messages, [{"type": "function"}], "program-header", pieces)
    assert rendered == [{"role": "system", "content": "This call has one binding."},
                        messages[1]]
    assert tools is None
    assert apply_prompt_rendering(messages, [{"type": "function"}], "full", pieces) == (
        messages, [{"type": "function"}])
    identity = prompt_rendering_identity("program-header")
    assert identity["mode"] == "program-header"
    assert len(identity["identity_sha256"]) == 64
    assert {"data/prompt_rendering.py", "serve/engine.py", "serve/chat.py"} == set(
        identity["implementation_sha256"])


def test_engine_request_rendering_is_the_training_transform():
    class Tokenizer:
        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            system = "|".join(message["content"] for message in messages if message["role"] == "system")
            user = "|".join(message["content"] for message in messages if message["role"] == "user")
            return system + "\\n" + user + "\\n" + ("SCHEMA" if tools else "NO-SCHEMA")

    engine = Engine.__new__(Engine)
    engine.prompt_rendering = "program-header"
    engine.tokenizer = Tokenizer()
    engine._specials = ()
    engine.store = type("Store", (), {"get": lambda self, _block: None})()
    fixture = json.loads((ROOT / "training/prompt-pieces/parity-fixture.jsonl").read_text().splitlines()[0])
    messages = [{"role": "system", "content": fixture["text"]},
                {"role": "user", "content": "Do the task."}]
    expected_header = fixture["program_header"]
    rendered = engine.render_request(messages, [{"type": "function", "name": "hidden schema"}])
    text = "".join(segment for segment in rendered.segments if isinstance(segment, str))
    assert expected_header in text
    assert fixture["text"] != expected_header
    assert fixture["text"] not in text
    assert "hidden schema" not in text
    assert text.endswith("NO-SCHEMA")


def test_checkpoint_mode_requires_exact_homogeneous_text_input(tmp_path):
    data = tmp_path / "text-data.jsonl"
    raw = (json.dumps({"prompt_rendering": "program-header", "text": "x"}) + "\n" +
           json.dumps({"prompt_rendering": "program-header", "text": "y"}) + "\n").encode()
    data.write_bytes(raw)
    identity = {"options": {"text_data": str(data)},
                "inputs": {str(data.resolve()): hashlib.sha256(raw).hexdigest()}}
    mode, proof = checkpoint_prompt_rendering(identity)
    assert mode == "program-header"
    assert proof["row_count"] == 2
    assert proof["mode_counts"] == {"program-header": 2}
    assert proof["input_sha256"] == hashlib.sha256(raw).hexdigest()

    data.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="changed since training"):
        checkpoint_prompt_rendering(identity)


def test_checkpoint_mode_requires_explicit_policy_for_mixed_rendering_rows(tmp_path):
    data = tmp_path / "mixed.jsonl"
    raw = (json.dumps({"prompt_rendering": "program-header"}) + "\n" +
           json.dumps({"prompt_rendering": "full"}) + "\n").encode()
    data.write_bytes(raw)
    identity = {"options": {"text_data": str(data)},
                "inputs": {str(data.resolve()): hashlib.sha256(raw).hexdigest()}}
    with pytest.raises(ValueError, match="mixes prompt rendering modes"):
        checkpoint_prompt_rendering(identity)


def test_explicit_policy_preserves_mixed_source_counts_and_undeclared_rows(tmp_path):
    data = tmp_path / "mixed-policy.jsonl"
    rows = [
        {"prompt_rendering": "program-header", "system_prompt_mode": "program-header", "text": "header"},
        {"system_prompt_mode": None, "text": "List of tools: [{\"name\":\"read\"}]"},
    ]
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    data.write_bytes(raw)
    identity = {"options": {"text_data": str(data)},
                "inputs": {str(data.resolve()): hashlib.sha256(raw).hexdigest()}}

    mode, proof = checkpoint_prompt_rendering(identity, serving_mode="program-header")
    assert mode == "program-header"
    assert proof["mode_selection"] == "explicit-serving-policy-on-mixed-training-input"
    assert proof["row_count"] == 2
    assert proof["mode_counts"] == {"program-header": 1, "full": 1}
    assert proof["declared_mode_counts"] == {"program-header": 1}
    assert proof["undeclared_mode_rows"] == 1
    assert proof["system_prompt_mode_counts"] == {"program-header": 1, "absent": 1}
    assert proof["rows_with_tool_schema_marker"] == 1
    assert proof["undeclared_rows_with_tool_schema_marker"] == 1
    assert proof["input_sha256"] == hashlib.sha256(raw).hexdigest()


def test_explicit_policy_must_match_homogeneous_training_rows(tmp_path):
    data = tmp_path / "header-only.jsonl"
    raw = (json.dumps({"prompt_rendering": "program-header"}) + "\n").encode()
    data.write_bytes(raw)
    identity = {"options": {"text_data": str(data)},
                "inputs": {str(data.resolve()): hashlib.sha256(raw).hexdigest()}}
    with pytest.raises(ValueError, match="differs from homogeneous"):
        checkpoint_prompt_rendering(identity, serving_mode="full")
