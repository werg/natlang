import json
import hashlib

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


def test_bundle_retains_tokenizer_binding_and_rejects_tampering(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps({'id': 'one', 'completion': 'done'}) + '\n')
    renderer = {'model': 'tokenizer', 'revision': 'pinned', 'template_sha256': 'a',
                'tokenizer_fingerprint_sha256': 'b', 'local_tokenizer_artifacts_sha256': 'c', 'end_token': '</s>'}
    manifest = {'version': 'natlang.sft.native/1', 'renderer': renderer,
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    source.with_suffix('.jsonl.manifest.json').write_text(json.dumps(manifest))
    result = combine(tmp_path / 'combined.jsonl', [source])
    assert result['renderer'] == renderer
    assert result['version'] == 'natlang.sft.native/1' and result['rows'] == 1
    source.write_text(json.dumps({'id': 'changed', 'completion': 'done'}) + '\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        combine(tmp_path / 'tampered.jsonl', [source])
