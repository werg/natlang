import json
from pathlib import Path

import pytest

from scripts.render_training_corpus import render_corpus, render_turn


class MockTokenizer:
    chat_template = "mock-native-template-v1"
    eos_token = "<eos>"
    name_or_path = "mock/model"

    def __init__(self):
        self.calls = 0

    def get_vocab(self):
        return {"mock": 0, "<eos>": 1}

    def apply_chat_template(self, messages, *, tools=None, tokenize=False, add_generation_prompt=False):
        self.calls += 1
        assert tokenize is False
        body = "".join(f"<{message['role']}>{message.get('content', '')}<eos>" for message in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


class BadPrefixTokenizer(MockTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        value = super().apply_chat_template(messages, **kwargs)
        return value.replace("<user>", "<human>") if len(messages) > 2 else value


class NoEndTokenizer(MockTokenizer):
    eos_token = None


def test_render_native_turn_preserves_provenance_and_uses_assistant_generation_prefix():
    row = {"id": "turn-1", "program_id": "program-1", "source_groups": ["source-group"],
           "split": "train", "family": "fixture", "skill": "write", "quality": {"score": 0.9},
           "messages": [{"role": "user", "content": "Say hi"}], "tools": [],
           "target": {"role": "assistant", "content": "Hello"},
           "training_admission": {"approved": True}}
    pair = render_turn(row, MockTokenizer(), "<eos>")
    assert pair["prompt"] == "<user>Say hi<eos><assistant>"
    assert pair["completion"] == "Hello<eos>"
    for key in ("source_groups", "split", "family", "skill", "quality", "training_admission"):
        assert pair[key] == row[key]


def test_render_decodes_tool_arguments_for_templates_without_changing_ir():
    class MappingTokenizer(MockTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            for message in messages:
                for call in message.get('tool_calls', []):
                    assert call['function']['arguments'] == {'code': '1 + 1'}
            return super().apply_chat_template(messages, **kwargs)

    raw_call = {'id': 'source-call', 'type': 'function',
                'function': {'name': 'eval', 'arguments': '{"code":"1 + 1"}'}}
    row = {'id': 'tool', 'messages': [{'role': 'user', 'content': 'Compute.'},
                                     {'role': 'assistant', 'content': '', 'tool_calls': [raw_call]},
                                     {'role': 'tool', 'tool_call_id': 'source-call', 'content': '2'}],
           'tools': [], 'target': {'role': 'assistant', 'content': '', 'tool_calls': [raw_call]},
           'training_admission': {'approved': True}}
    assert render_turn(row, MappingTokenizer(), '<eos>') is not None
    assert raw_call['function']['arguments'] == '{"code":"1 + 1"}'


def test_render_preserves_teacher_reasoning_for_thinking_only_templates():
    class ThinkingTokenizer(MockTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            formatted = [{**message, 'content': (f"<think>{message['thinking']}</think>"
                         if message.get('thinking') else '') + message.get('content', '')}
                         for message in messages]
            return super().apply_chat_template(formatted, **kwargs)

    row = {'id': 'thinking', 'messages': [{'role': 'user', 'content': 'Compute.'}],
           'tools': [], 'target': {'role': 'assistant', 'content': 'Done.'},
           'teacher_reasoning': 'The computation is complete.',
           'training_admission': {'approved': True}}
    pair = render_turn(row, ThinkingTokenizer(), '<eos>')
    assert '<think>The computation is complete.</think>' in pair['completion']


def test_target_prefix_and_end_token_are_required():
    row = {"id": "bad", "messages": [{"role": "user", "content": "x"}], "tools": [],
           "target": {"role": "assistant", "content": "answer"},
           "training_admission": {"approved": True}}
    with pytest.raises(ValueError, match="changed the assistant prefix"):
        render_turn(row, BadPrefixTokenizer(), "<eos>")
    with pytest.raises(ValueError, match="end token is absent"):
        render_turn(row, MockTokenizer(), "<missing>")
    row['target']['content'] = 'hello<eos>silently lost tail'
    with pytest.raises(ValueError, match='refusing silent truncation'):
        render_turn(row, MockTokenizer(), '<eos>')


def test_code_sft_inputs_render_offline_and_preserve_source_quality_and_split(tmp_path):
    source, output = tmp_path / "code.jsonl", tmp_path / "rendered.jsonl"
    rows = [
        {"kind": "code_sft", "id": "code-1", "program_id": "group-1", "source_groups": ["group-1"],
         "split": "test", "quality": "syntax-checked", "syntax_checked": True,
         "prompt": "Write a function", "completion": "function answer() {}", "source": {"name": "mock"}},
        {"id": "rejected", "messages": [{"role": "user", "content": "skip"}], "tools": [],
         "target": {"role": "assistant", "content": "skip"}, "training_admission": {"approved": False}},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rc = render_corpus([source], output, model="fixture", tokenizer=MockTokenizer(), chunk_rows=1)
    assert rc == 0
    rendered = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rendered) == 1
    assert rendered[0]["prompt"] == "<user>Write a function<eos><assistant>"
    assert rendered[0]["completion"] == "function answer() {}<eos>"
    assert rendered[0]["split"] == "test"
    assert rendered[0]["quality"] == "syntax-checked"
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["renderer"]["template_sha256"]
    assert manifest["renderer"]["end_token"] == "<eos>"


def test_committed_chunks_resume_after_stop_and_corruption_is_rejected(tmp_path):
    source, output = tmp_path / "turns.jsonl", tmp_path / "resume.jsonl"
    rows = [{"id": str(i), "messages": [{"role": "user", "content": str(i)}], "tools": [],
             "target": {"role": "assistant", "content": f"ok-{i}"}, "training_admission": {"approved": True}}
            for i in range(3)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tokenizer = MockTokenizer()
    assert render_corpus([source], output, model="fixture", tokenizer=tokenizer, chunk_rows=1,
                         should_stop=lambda: True) == 75
    assert tokenizer.calls == 2
    assert not output.exists()
    assert render_corpus([source], output, model="fixture", tokenizer=tokenizer, chunk_rows=1) == 0
    assert tokenizer.calls == 6  # the first committed row was loaded from cache, not rendered again
    assert len(output.read_text().splitlines()) == 3
    chunk_manifest = json.loads((output.with_name(output.name + ".cache") / "manifest.json").read_text())
    first_chunk = chunk_manifest["chunks"][0]
    assert (first_chunk["input_row_start"], first_chunk["input_row_end"]) == (0, 1)
    assert first_chunk["input_sha256"] and first_chunk["sha256"]
    cache = output.with_name(output.name + ".cache") / "chunk-00000000.jsonl"
    cache.write_text(cache.read_text() + " ")
    with pytest.raises(ValueError, match="chunk hash mismatch"):
        render_corpus([source], output, model="fixture", tokenizer=MockTokenizer(), chunk_rows=1)


def test_changed_input_or_revision_cannot_reuse_existing_cache(tmp_path):
    source, output = tmp_path / "one.jsonl", tmp_path / "out.jsonl"
    row = {"id": "one", "messages": [{"role": "user", "content": "x"}], "tools": [],
           "target": {"role": "assistant", "content": "y"}}
    source.write_text(json.dumps(row) + "\n")
    render_corpus([source], output, model="fixture", tokenizer=MockTokenizer())
    with pytest.raises(ValueError, match="cache identity differs"):
        render_corpus([source], output, model="fixture", revision="a" * 40, tokenizer=MockTokenizer())
    with pytest.raises(ValueError, match="immutable 40-character"):
        render_corpus([source], tmp_path / "other.jsonl", model="fixture", revision="main", tokenizer=MockTokenizer())


def test_tokenizer_vocab_and_local_artifact_fingerprint_are_part_of_identity(tmp_path):
    source, output = tmp_path / "one.jsonl", tmp_path / "fingerprinted.jsonl"
    source.write_text(json.dumps({"id": "one", "messages": [{"role": "user", "content": "x"}],
                                 "target": {"role": "assistant", "content": "y"}}) + "\n")
    render_corpus([source], output, model="fixture", tokenizer=MockTokenizer())

    class DifferentVocab(MockTokenizer):
        def get_vocab(self):
            return {"different": 0}

    with pytest.raises(ValueError, match="identity differs"):
        render_corpus([source], output, model="fixture", tokenizer=DifferentVocab())

    local = tmp_path / "local-model"
    local.mkdir()
    (local / "tokenizer.json").write_text('{"version":"one"}')
    local_output = tmp_path / "local-output.jsonl"
    render_corpus([source], local_output, model=str(local), tokenizer=MockTokenizer())
    (local / "tokenizer.json").write_text('{"version":"two"}')
    with pytest.raises(ValueError, match="identity differs"):
        render_corpus([source], local_output, model=str(local), tokenizer=MockTokenizer())


def test_native_turn_requires_explicit_approval():
    row = {"id": "missing-approval", "messages": [{"role": "user", "content": "x"}], "tools": [],
           "target": {"role": "assistant", "content": "y"}}
    assert render_turn(row, MockTokenizer(), "<eos>") is None


def test_final_data_without_manifest_is_recovered_only_when_cache_matches(tmp_path):
    source, output = tmp_path / "one.jsonl", tmp_path / "orphan.jsonl"
    source.write_text(json.dumps({"id": "one", "messages": [{"role": "user", "content": "x"}],
                                 "target": {"role": "assistant", "content": "y"},
                                 "training_admission": {"approved": True}}) + "\n")
    render_corpus([source], output, model="fixture", tokenizer=MockTokenizer())
    expected = output.read_bytes()
    output.with_suffix(".jsonl.manifest.json").unlink()
    assert render_corpus([source], output, model="fixture", tokenizer=MockTokenizer()) == 0
    assert output.read_bytes() == expected
    assert output.with_suffix(".jsonl.manifest.json").exists()
    output.with_suffix(".jsonl.manifest.json").unlink()
    output.write_text("user content that must not be replaced\n")
    with pytest.raises(ValueError, match="differs from verified shard cache"):
        render_corpus([source], output, model="fixture", tokenizer=MockTokenizer())


def test_invalid_targets_get_a_resumable_rejection_ledger(tmp_path):
    source, output = tmp_path / 'input', tmp_path / 'output'
    rows = [{'id': 'reserved', 'messages': [{'role': 'user', 'content': 'x'}],
             'target': {'role': 'assistant', 'content': 'bad<eos>tail'}, 'training_admission': {'approved': True}},
            {'id': 'valid', 'messages': [{'role': 'user', 'content': 'x'}],
             'target': {'role': 'assistant', 'content': 'ok'}, 'training_admission': {'approved': True}},
            {'id': 'denied-code', 'kind': 'code_sft', 'syntax_checked': True,
             'prompt': 'x', 'completion': 'let x = 1;', 'training_admission': {'approved': False}}]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert render_corpus([source], output, model='fixture', tokenizer=MockTokenizer(),
                         chunk_rows=1, should_stop=lambda: True) == 75
    assert render_corpus([source], output, model='fixture', tokenizer=MockTokenizer(), chunk_rows=1) == 0
    assert [json.loads(line)['id'] for line in output.read_text().splitlines()] == ['valid']
    ledger = output.with_name(output.name + '.rejected.jsonl')
    rejected = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r['reason'] for r in rejected] == ['invalid_training_view', 'not_explicitly_admitted']
    assert 'silent truncation' in rejected[0]['detail']
    assert render_corpus([source], output, model='fixture', tokenizer=MockTokenizer(), chunk_rows=1) == 0
    ledger.write_text('tamper')
    with pytest.raises(ValueError, match='rejection ledger differs'):
        render_corpus([source], output, model='fixture', tokenizer=MockTokenizer(), chunk_rows=1)
