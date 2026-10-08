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
        return value.replace("<user>", "<human>") if not kwargs.get("add_generation_prompt") else value


class NoEndTokenizer(MockTokenizer):
    eos_token = None


class DualEndTokenizer(MockTokenizer):
    eos_token = "</s>"
    all_special_tokens = ["</s>", "<|im_end|>"]

    def apply_chat_template(self, messages, **kwargs):
        return super().apply_chat_template(messages, **kwargs).replace("<eos>", "<|im_end|>")


def test_chat_terminator_is_resolved_from_template_instead_of_generic_eos(tmp_path):
    from scripts.render_training_corpus import _tokenizer_info

    tokenizer = DualEndTokenizer()
    _, renderer = _tokenizer_info(tokenizer, "mock-dual-eos", None)
    assert renderer["end_token"] == "<|im_end|>"
    with pytest.raises(ValueError, match="closed assistant template"):
        _tokenizer_info(tokenizer, "mock-dual-eos", None, end_token="</s>")
    source = tmp_path / "turns.jsonl"
    source.write_text(json.dumps({"id": "dual-end", "messages": [{"role": "user", "content": "Hello"}],
        "target": {"role": "assistant", "content": "Hi"}, "training_admission": {"approved": True}}) + "\n")
    output = tmp_path / "rendered.jsonl"
    render_corpus([source], output, model="mock-dual-eos", tokenizer=tokenizer)
    assert json.loads(output.read_text())["completion"] == "Hi<|im_end|>"
    assert json.loads(output.with_suffix(".jsonl.manifest.json").read_text())["rows"] == 1


def test_added_chat_terminator_does_not_require_special_flag():
    from scripts.render_training_corpus import _tokenizer_info

    class AddedEndTokenizer(DualEndTokenizer):
        all_special_tokens = ["</s>"]

        def get_added_vocab(self):
            return {"<|im_end|>": 130073}

    assert _tokenizer_info(AddedEndTokenizer(), "mock-added-eos", None)[1]["end_token"] == "<|im_end|>"


def test_all_invalid_rendering_fails_and_preserves_evidence_on_resume(tmp_path):
    source, output = tmp_path / "invalid.jsonl", tmp_path / "rendered.jsonl"
    source.write_text(json.dumps({"id": "bad", "messages": [{"role": "user", "content": "Hi"}],
        "target": {"role": "assistant", "content": "lost<eos>tail"},
        "training_admission": {"approved": True}}) + "\n")
    for _ in range(2):
        with pytest.raises(ValueError, match="no training decisions could be rendered"):
            render_corpus([source], output, model="mock", tokenizer=MockTokenizer())
    assert output.read_text() == ""
    assert output.with_suffix(".jsonl.manifest.json").exists()
    assert json.loads(output.with_name(output.name + ".rejected.jsonl").read_text())["reason"] == "invalid_training_view"


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


def test_streaming_render_matches_batch_output_and_resumes_from_cached_chunks(tmp_path):
    rows = [
        {"id": f"turn-{i}", "program_id": "p", "source_groups": ["src"], "split": "train",
         "family": "fixture", "messages": [{"role": "user", "content": f"Say {i}"}], "tools": [],
         "target": {"role": "assistant", "content": f"Answer {i}"},
         "training_admission": {"approved": True}}
        for i in range(5)
    ]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    batch, streaming = tmp_path / "batch.jsonl", tmp_path / "streaming.jsonl"
    assert render_corpus([source], batch, model="fixture", tokenizer=MockTokenizer(), chunk_rows=2) == 0
    tokenizer = MockTokenizer()
    assert render_corpus([source], streaming, model="fixture", tokenizer=tokenizer,
                         chunk_rows=2, streaming=True) == 0
    assert streaming.read_bytes() == batch.read_bytes()
    assert streaming.with_name(streaming.name + ".rejected.jsonl").read_bytes() == batch.with_name(batch.name + ".rejected.jsonl").read_bytes()
    calls = tokenizer.calls
    assert render_corpus([source], streaming, model="fixture", tokenizer=tokenizer,
                         chunk_rows=2, streaming=True) == 0
    assert tokenizer.calls == calls


def test_streaming_render_interrupt_preserves_chunks_and_resumes_only_missing_work(tmp_path):
    rows = [{"id": f"turn-{i}", "messages": [{"role": "user", "content": str(i)}], "tools": [],
             "target": {"role": "assistant", "content": str(i)},
             "training_admission": {"approved": True}} for i in range(5)]
    source, output = tmp_path / "input.jsonl", tmp_path / "resumable.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tokenizer = MockTokenizer()
    assert render_corpus([source], output, model="fixture", tokenizer=tokenizer,
                         chunk_rows=2, streaming=True, should_stop=lambda: True) == 75
    cache_manifest = json.loads((output.with_name(output.name + ".cache") / "manifest.json").read_text())
    assert len(cache_manifest["chunks"]) == 1
    calls_after_first = tokenizer.calls
    assert render_corpus([source], output, model="fixture", tokenizer=tokenizer,
                         chunk_rows=2, streaming=True) == 0
    assert tokenizer.calls > calls_after_first
    assert len([line for line in output.read_text().splitlines() if line]) == 5


def test_streaming_all_invalid_render_preserves_rejection_and_empty_manifest(tmp_path):
    source, output = tmp_path / "invalid.jsonl", tmp_path / "streaming.jsonl"
    source.write_text(json.dumps({"id": "bad", "messages": [], "target": {"role": "assistant", "content": "x<eos>"},
                                  "training_admission": {"approved": True}}) + "\n")
    with pytest.raises(ValueError, match="no training decisions could be rendered"):
        render_corpus([source], output, model="fixture", tokenizer=MockTokenizer(), streaming=True)
    assert output.exists() and output.read_bytes() == b""
    assert output.with_suffix(".jsonl.manifest.json").exists()
    assert json.loads(output.with_name(output.name + ".rejected.jsonl").read_text())["reason"] == "invalid_training_view"


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


def test_typed_result_write_receipts_hydrate_only_hash_bound_same_split_context():
    import hashlib
    from training.neuralese.natlang_neuralese.data.text_corpus import (
        _attested_neuralese_message_bodies, _soft_writer_sources,
    )

    block = 'nz1_' + 'a' * 32
    body = 'the exact emitted note'
    body_sha = hashlib.sha256(body.encode()).hexdigest()
    receipts = [{
                'schema': 'natlang.typed-result-write/1', 'invocation_id': 'writer-call',
                'source_row_sha256': 'row-sha', 'writer_node': 'writer-call#8', 'block_id': block,
                'source_kind': 'typed-text-result-field', 'source': 'return_result',
                'result_type': 'Neuralese<string>', 'result_path': ['return', 'notes', 0],
                'body_sha256': body_sha, 'body_source': body,
                'body_source_basis': 'exact-raw-model-result-field',
            }, {
                'schema': 'natlang.typed-result-write/1', 'invocation_id': 'writer-call',
                'source_row_sha256': 'row-sha', 'writer_node': 'writer-call#9', 'block_id': block,
                'source_kind': 'typed-text-result-field', 'source': 'return_result',
                'result_type': 'Neuralese<string>', 'result_path': ['return', 'other'],
                'body_sha256': body_sha, 'body_source': body,
                'body_source_basis': 'exact-raw-model-result-field',
            }]
    writer = {
        'id': 'writer-decision', 'split': 'train', 'source_groups': ['case-group'],
        'source_ref': {'trajectory_id': 'trajectory-1', 'source_row_sha256': 'row-sha',
                       'invocation_id': 'writer-call'},
        'training_admission': {'approved': True}, 'decision': {'training_approved': True, 'failed_action': False,
            'assistant': {'calls': [{'outcome': {'typed_result_writes': receipts}}]}},
    }
    reader = {
        'id': 'reader-decision', 'split': 'train', 'source_groups': ['case-group'],
        'source_ref': {'trajectory_id': 'trajectory-1', 'source_row_sha256': 'row-sha',
                       'invocation_id': 'reader-call'},
        'messages': [{'role': 'user', 'content': [{'type': 'neuralese', 'id': block}]}],
    }
    writers = _soft_writer_sources([writer, reader], {'writer-decision': 'decision-sha'})
    assert len(writers[block]) == 1, 'two real writes of identical content remain two occurrences on one body source'
    assert {item['writer_write_node'] for item in writers[block][0]['write_occurrences']} == {
        'writer-call#8', 'writer-call#9'}
    hydrated, evidence = _attested_neuralese_message_bodies(
        reader, writers, split='train', source_groups=['case-group'])
    assert hydrated[block] == f'<|neuralese|>{body}<|/neuralese|>'
    assert evidence[0]['writer_record_id'] == 'writer-decision'

    with pytest.raises(ValueError, match='no unique same-split'):
        _attested_neuralese_message_bodies(reader, writers, split='test', source_groups=['case-group'])


def test_eval_result_write_body_requires_same_run_reader_attestation():
    import hashlib
    from training.neuralese.natlang_neuralese.data.text_corpus import (
        _attested_neuralese_message_bodies, _soft_writer_sources,
    )

    block, body = 'nz1_' + 'b' * 32, 'computed from the eval result'
    body_sha = hashlib.sha256(body.encode()).hexdigest()
    receipt = {
                'schema': 'natlang.typed-result-write/1', 'invocation_id': 'eval-call',
                'source_row_sha256': 'row-2', 'writer_node': 'eval-call#8', 'block_id': block,
                'source_kind': 'typed-text-result-field', 'source': 'eval-finish',
                'result_type': 'Neuralese<string>', 'result_path': ['return', 'note'],
                'body_sha256': body_sha, 'body_source_basis': 'authenticated-final-host-output-reference',
    }
    writer = {'id': 'eval-writer', 'split': 'train', 'source_groups': ['same'],
        'source_ref': {'trajectory_id': 'trajectory-2', 'source_row_sha256': 'row-2', 'invocation_id': 'eval-call'},
        'training_admission': {'approved': True}, 'decision': {'training_approved': True, 'failed_action': False,
            'assistant': {'calls': [{'outcome': {'typed_result_writes': [receipt]}}]}}}
    reader = {'id': 'eval-reader', 'split': 'train', 'source_groups': ['same'],
        'source_ref': {'trajectory_id': 'trajectory-2', 'source_row_sha256': 'row-2', 'invocation_id': 'reader-call',
            'provider_expanded_read_contexts': [{
                'origin': 'same-run-producer',
                'producer_write': {'kind': 'block_write', 'call_id': 'eval-call', 'node': 'eval-call#8',
                                   'block': block, 'text_body_sha256': body_sha},
                'block': {'id': block, 'type': 'Neuralese<string>', 'body': body, 'body_sha256': body_sha},
            }]},
        'messages': [{'role': 'user', 'content': [{'type': 'neuralese', 'id': block}]}]}
    writers = _soft_writer_sources([writer, reader], {'eval-writer': 'writer-sha'})
    assert writers[block][0]['body'] == body
    assert writers[block][0]['source_kind'] == 'same-run-provider-expanded-writer-context'
    assert writers[block][0]['writer_write_node'] == 'eval-call#8'
    hydrated, _ = _attested_neuralese_message_bodies(reader, writers, split='train', source_groups=['same'])
    assert hydrated[block] == f'<|neuralese|>{body}<|/neuralese|>'

    wrong = {**reader, 'source_ref': {**reader['source_ref'], 'provider_expanded_read_contexts': [{
        **reader['source_ref']['provider_expanded_read_contexts'][0],
        'producer_write': {**reader['source_ref']['provider_expanded_read_contexts'][0]['producer_write'],
                           'node': 'different-writer#2'},
    }]}}
    wrong_writers = _soft_writer_sources([writer, wrong], {'eval-writer': 'writer-sha'})
    with pytest.raises(ValueError, match='no unique same-split'):
        _attested_neuralese_message_bodies(wrong, wrong_writers, split='train', source_groups=['same'])


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
    assert tokenizer.calls == 3  # one probe of the generation prompt (cached on the tokenizer) and two rows
    assert not output.exists()
    assert render_corpus([source], output, model="fixture", tokenizer=tokenizer, chunk_rows=1) == 0
    assert tokenizer.calls == 7  # the first committed row was loaded from cache, not rendered again
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


def test_minicpm_closed_target_preserves_reasoning_and_native_tool_arguments():
    jinja2 = pytest.importorskip("jinja2")
    template = Path("models/templates/Sharp-MiniCPM5-2B.jinja").read_text()
    env = jinja2.Environment()
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=kwargs.get("ensure_ascii", False))

    class MiniCPMTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return env.from_string(template).render(messages=messages, bos_token="<s>", **kwargs)

    row = {"id": "minicpm", "messages": [{"role": "user", "content": "Compute."}],
           "tools": [], "target": {"role": "assistant", "content": "",
           "tool_calls": [{"type": "function", "function": {"name": "eval", "arguments": '{"code":"1+1"}'}}]},
           "teacher_reasoning": "One plus one is two.", "teacher_reasoning_trained": False,
           "training_admission": {"approved": True}}
    pair = render_turn(row, MiniCPMTokenizer(), "<|im_end|>")
    assert pair["prompt"].endswith("<think>\n")
    assert "One plus one is two." in pair["completion"]
    assert '<function name="eval"><param name="code">1+1</param></function>' in pair["completion"]
    assert pair["completion"].endswith("<|im_end|>")
    assert pair["completion_masked"] == pair["completion"].index("</think>") + len("</think>")


def test_reasoning_history_stays_identical_when_a_new_assistant_target_is_appended():
    class ForgetfulTokenizer(MockTokenizer):
        chat_template = 'set preserve_thinking = preserve_thinking | default(false)'

        def apply_chat_template(self, messages, *, preserve_thinking=False, **kwargs):
            assert preserve_thinking is True
            rendered = []
            for message in messages:
                text = message.get('content', '') + message.get('thinking', '')
                rendered.append({**message, 'content': text})
            return super().apply_chat_template(rendered, **kwargs)

    row = {'id': 'reasoning-history', 'messages': [
        {'role': 'user', 'content': 'Repair.'},
        {'role': 'assistant', 'content': 'Inspect.', 'reasoning_content': 'Earlier diagnosis.'},
        {'role': 'tool', 'content': 'Observed failure.'}],
        'target': {'role': 'assistant', 'content': 'Correct.'},
        'training_admission': {'approved': True}}
    pair = render_turn(row, ForgetfulTokenizer(), '<eos>')
    assert 'Earlier diagnosis.' in pair['prompt']
    assert pair['completion'] == 'Correct.<eos>'


class ThinkOpeningTokenizer(MockTokenizer):
    """A reasoning template (like Maple's): the generation prompt opens a think block, an unreasoned turn has none."""

    def apply_chat_template(self, messages, **kwargs):
        value = super().apply_chat_template(messages, **kwargs)
        return value + "<think>\n" if kwargs.get("add_generation_prompt") else value


def test_unreasoned_turn_gets_an_empty_think_block_when_the_template_opens_one():
    row = {"id": "turn-think", "messages": [{"role": "user", "content": "Say hi"}], "tools": [],
           "target": {"role": "assistant", "content": "Hello"}, "training_admission": {"approved": True}}
    pair = render_turn(row, ThinkOpeningTokenizer(), "<eos>")
    assert pair["prompt"] == "<user>Say hi<eos><assistant><think>\n"
    assert pair["completion"] == "\n</think>\n\nHello<eos>"
    # A template that does not open a think block is unaffected, and a real prefix change is still refused.
    assert render_turn(row, MockTokenizer(), "<eos>")["completion"] == "Hello<eos>"
    with pytest.raises(ValueError, match="changed the assistant prefix"):
        render_turn(row, BadPrefixTokenizer(), "<eos>")
