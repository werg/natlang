import pytest
import json

from natlang.terminal import reply_only_sample
from scripts.normalize_terminal_done import convert_file


def test_legacy_terminal_target_becomes_reply_without_losing_line_marks():
    old_prompt = ("When the result is ready and all numbered lines are closed, use done to finish\n"
                  "successfully, or reply briefly. Error and blocker reports are only for failures.")
    row = {"skill": "done", "messages": [{"role": "system", "content": old_prompt}],
           "tools": [{"function": {"name": "write"}}, {"function": {"name": "done"}}],
           "target": {"role": "assistant", "content": "", "tool_calls": [
               {"function": {"name": "done", "arguments": "{}"}}]}}
    converted = reply_only_sample(row)
    assert converted["skill"] == "reply"
    assert converted["target"] == {"role": "assistant", "content": "Done."}
    assert [tool["function"]["name"] for tool in converted["tools"]] == ["write"]
    assert "use done to finish" not in converted["messages"][0]["content"]
    assert row["target"]["tool_calls"][0]["function"]["name"] == "done"

    row["skill"] = "write"
    row["target"] = {"role": "assistant", "tool_calls": [
        {"function": {"name": "write", "arguments": '{"path":"return","done":1}'}}]}
    converted = reply_only_sample(row)
    assert converted["target"] == row["target"]
    assert '"done":1' in converted["target"]["tool_calls"][0]["function"]["arguments"]


def test_mixed_terminal_batch_is_not_silently_changed():
    row = {"skill": "write+done", "messages": [], "tools": [],
           "target": {"role": "assistant", "tool_calls": [
               {"function": {"name": "write", "arguments": "{}"}},
               {"function": {"name": "done", "arguments": "{}"}}]}}
    with pytest.raises(ValueError, match="mixed terminal"):
        reply_only_sample(row)


def test_streaming_normalizer_writes_reply_only_rows(tmp_path):
    src, dst = tmp_path / 'old.jsonl', tmp_path / 'new.jsonl'
    src.write_text(json.dumps({"skill": "done", "messages": [],
        "tools": [{"function": {"name": "done"}}],
        "target": {"role": "assistant", "tool_calls": [
            {"function": {"name": "done", "arguments": "{}"}}]}}) + '\n')
    assert convert_file(src, dst) == (1, 1)
    row = json.loads(dst.read_text())
    assert row['skill'] == 'reply' and row['tools'] == []
