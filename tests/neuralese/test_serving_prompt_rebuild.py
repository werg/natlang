"""The current ts-host serving prompt rebuilt for recorded tool-loop calls (data/serving_prompt.py)."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
import natlang_neuralese.data.serving_prompt as serving_prompt  # noqa: E402
from natlang_neuralese.data.serving_prompt import load_surface, serving_messages  # noqa: E402

SURFACE = load_surface()
CURRENT = SURFACE["_current"]


def historical(logical):
    return next(p["text"] for p in SURFACE["_bank"] if p["historical"] and p["logical"] == logical)


def call(system, tools=None, extra=()):
    return [{"role": "system", "content": system},
            {"role": "user", "content": "You are inside this call: f(): number"}, *extra], tools


def test_collector_and_runtime_prompts_replace_recorded_runtime_prose(monkeypatch):
    monkeypatch.setattr(serving_prompt, "RETURN_SCHEMAS", lambda user: ({"type": "number"}, None))
    old = historical("interpreter")
    guidance = "\n\nProgram guidance: prefer the newest record."
    system = old + historical("generation-guidance") + guidance
    for entry in ("collector", "runtime"):
        messages, tools, report = serving_messages(*call(system), entry=entry, surface=SURFACE)
        assert report["mode"] == f"serving-{entry}"
        assert messages[0]["content"] == SURFACE["bases"][entry]["default"] + guidance
        assert [t["function"]["name"] for t in tools] == ["eval", "read_page", "read_code", "compact_history",
                                                         "return_result"]
        assert report["return_schema"] == "derived"
        assert tools[-1]["function"]["parameters"]["properties"]["value"]["type"] == "number"
        assert messages[1:] == call(system)[0][1:]


def test_underivable_return_schema_keeps_the_recording(monkeypatch):
    monkeypatch.setattr(serving_prompt, "RETURN_SCHEMAS", lambda user: (None, "unknown type"))
    recorded = call(CURRENT["interpreter"]["text"])
    messages, tools, report = serving_messages(*recorded, surface=SURFACE)
    assert report["mode"] == "recorded" and report["reason"] == "return schema unavailable"
    assert (messages, tools) == recorded


def test_return_schemas_bridge_matches_ts_host(tmp_path):
    import os
    dist = os.environ.get("NATLANG_TS_DIST") or str(ROOT / "ts-host" / "dist")
    agent = Path(dist) / "native" / "agent.js"
    import shutil
    if not shutil.which("node") or not agent.exists() or "export function schemaOf" not in agent.read_text():
        import pytest
        pytest.skip("needs a ts-host build that exports schemaOf")
    bridge = serving_prompt._ReturnSchemas()
    user = 'You are inside this call: route(ticket: Ticket): Destination\ntype Destination = "a" | "b";\n\nInstructions:\nx'
    assert bridge(user) == ({"enum": ["a", "b"]}, None)
    assert bridge("You are inside this call: f(): { n: number, tags: string[] }") == ({
        "type": "object", "properties": {"n": {"type": "number"}, "tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["n", "tags"], "additionalProperties": False}, None)


def test_surface_follows_depth_limit_own_code_and_folder(monkeypatch):
    monkeypatch.setattr(serving_prompt, "RETURN_SCHEMAS", lambda user: ({"type": "number"}, None))
    system = CURRENT["interpreter/depth-limit"]["text"] + CURRENT["function-tools"]["text"] + \
        CURRENT["directory-reducer/editor/read-only/depth-limit"]["text"]
    messages, tools, report = serving_messages(*call(system), entry="runtime", surface=SURFACE)
    assert report["call_surface"] == {"depthLimited": True, "ownCode": True, "folder": "read", "fileTools": "editor"}
    assert messages[0]["content"] == system
    names = [t["function"]["name"] for t in tools]
    assert {"edit_code", "diff_code", "read_code"} <= set(names) and "write_file" not in names
    assert tools[names.index("read_code")] == SURFACE["tools"]["depth_limited"]["read_code"]


def test_return_result_keeps_the_recorded_declared_type():
    recorded = [{"type": "function", "function": {"name": "return_result", "description": "old",
                 "parameters": {"type": "object", "properties": {
                     "value": {"type": "number", "description": "old value"}}}}}]
    _, tools, report = serving_messages(*call(CURRENT["interpreter"]["text"], recorded), surface=SURFACE)
    value = tools[-1]["function"]["parameters"]["properties"]["value"]
    assert report["return_schema"] == "recorded"
    assert value["type"] == "number"
    assert value["description"] == SURFACE["tools"]["default"]["return_result"]["function"]["parameters"][
        "properties"]["value"]["description"]
    assert list(value) == ["type", "description"]


def test_other_calls_keep_their_recording(monkeypatch):
    monkeypatch.setattr(serving_prompt, "RETURN_SCHEMAS", lambda user: ({"type": "number"}, None))
    harness = call("You are a coding agent.", [{"type": "function", "function": {"name": "bash"}}])
    assert serving_messages(*harness, surface=SURFACE)[2]["reason"] == "not a tool-loop call"
    assert serving_messages(*harness, surface=SURFACE)[:2] == harness
    unknown = call(CURRENT["interpreter"]["text"], None, [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_function"}}]}])
    messages, tools, report = serving_messages(*unknown, surface=SURFACE)
    assert report["reason"] == "calls tools the surface lacks" and messages == unknown[0]


def test_zero_context_segmented_weights_keep_only_targets():
    import torch
    from natlang_neuralese.train.text_supervision import segmented_position_weights
    span = torch.ones((1, 12))
    weights = segmented_position_weights(span, [(3, 5), (8, 10)], context_weight=0.0, feedback_weight=0.0)[0]
    assert all(weights[i] == 0 for i in (0, 1, 2, 5, 6, 7, 10, 11))
    assert all(weights[i] > 0 for i in (3, 4, 8, 9))
    assert math.isclose(float(weights.sum()), 12.0, rel_tol=1e-6)
