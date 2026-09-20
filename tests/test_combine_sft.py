import json

import pytest

from scripts.combine_sft import combine


def test_bundle_checks_renderer_and_unique_ids(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    def write(path, row_id, template):
        path.write_text(json.dumps({"id": row_id, "completion": "<think>why</think>yes"}) + "\n")
        path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps({
            "renderer": {"template_sha256": template, "end_token": "<|im_end|>",
                         "terminal_tool_policy": "empty-success-turn-v2",
                         "teacher_reasoning_policy": "render-if-present-v1"}}))
    write(first, "one", "same")
    write(second, "two", "same")
    destination = tmp_path / "all.jsonl"
    result = combine(destination, [first, second])
    assert result["pairs"] == result["reasoning_pairs"] == 2
    assert [json.loads(line)["id"] for line in destination.read_text().splitlines()] == ["one", "two"]

    write(second, "one", "same")
    with pytest.raises(ValueError, match="duplicate SFT id"):
        combine(tmp_path / "duplicates.jsonl", [first, second])
    write(second, "two", "different")
    with pytest.raises(ValueError, match="incompatible SFT renderer"):
        combine(tmp_path / "different.jsonl", [first, second])
