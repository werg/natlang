"""Neuralese-output spans of a direct row (scripts/neuralese_data/neuralese_spans.py): the mixed run's sidecar."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('neuralese_spans', ROOT / 'scripts/neuralese_data/neuralese_spans.py')
spans_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spans_module)


class CharTokenizer:
    """One token per character (ids are code points): offsets are exact and re-tokenising reproduces the ids."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids):
        return ''.join(chr(i) for i in ids)


def row(prompt, target):
    text = prompt + target
    return {'id': 'r', 'text': text, 'token_ids': [ord(c) for c in text], 'supervised_suffix_start': len(prompt)}


def test_a_written_value_is_found_in_the_supervised_part_not_in_the_prompt():
    value = 'the answer'
    r = row(f'user: earlier you said {value}\n', f'assistant: return_result("{value}")')
    spans, missing = spans_module.spans_for_row(r, [('result:a', value, True)], CharTokenizer())
    start = len(r['text']) - len(f'{value}")')
    assert missing == 0 and spans == [[start, start + len(value), 'result:a', True]]


def test_a_json_escaped_value_is_found_and_an_absent_one_is_counted():
    value = 'say "hi"\nthen go'
    escaped = 'say \\"hi\\"\\nthen go'
    r = row('user: x\n', f'assistant: {{"value": "{escaped}"}}')
    spans, missing = spans_module.spans_for_row(
        r, [('result:a', value, False), ('result:b', 'not there', True), ('result:c', None, True)], CharTokenizer())
    begin = r['text'].index(escaped)
    assert spans == [[begin, begin + len(escaped), 'result:a', False]] and missing == 2


def test_compact_handover_json_maps_to_the_actual_spaced_tool_value():
    value = '{"need":1,"timing":1}'
    rendered = '{"need": 1, "timing": 1}'
    r = row('user: x\n', f"[return_result(status='success', value={rendered})]")
    spans, missing = spans_module.spans_for_row(r, [('result:a', value, True)], CharTokenizer())
    begin = r['text'].index(rendered)
    assert missing == 0 and spans == [[begin, begin + len(rendered), 'result:a', True]]


def test_json_spacing_forms_preserve_nested_values_and_escape_them():
    import json
    value = '{"label":"café","items":[1,true,{"text":"a\\nb"}]}'
    rendered = json.dumps(json.loads(value), ensure_ascii=False)
    escaped = json.dumps(rendered, ensure_ascii=False)[1:-1]
    assert rendered in spans_module.value_forms(value)
    assert escaped in spans_module.value_forms(value)
    assert json.dumps(json.loads(value), ensure_ascii=True) in spans_module.value_forms(value)


def test_a_value_mentioned_in_leading_reasoning_is_not_the_written_output():
    r = row('user: x\n', '<think>The result should be 42.</think>return_result(value=42)')
    spans, missing = spans_module.spans_for_row(r, [('result:a', '42', True)], CharTokenizer())
    begin = r['text'].rindex('42')
    assert missing == 0 and spans == [[begin, begin + 2, 'result:a', True]]
    r = row('user: x\n', '<think>The result might be 42, still deciding')
    assert spans_module.spans_for_row(r, [('result:a', '42', True)], CharTokenizer()) == ([], 1)


def test_the_mixed_loader_rejects_a_changed_pinned_span_sidecar(tmp_path):
    import hashlib
    import json
    import pytest
    from natlang_neuralese.train.direct_sft import neuralese_spans
    path = tmp_path / 'spans.jsonl'
    path.write_text(json.dumps({'id': 'r', 'spans': [[1, 3, 'a', True], [4, 5, 'b', False]]}) + '\n')
    binding = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert neuralese_spans({'neuralese_spans': binding}, tmp_path) == ({'r': [[1, 3]]}, 1)
    path.write_text(json.dumps({'id': 'r', 'spans': [[7, 9, 'a', True]]}) + '\n')
    with pytest.raises(ValueError, match='span hash mismatch'):
        neuralese_spans({'neuralese_spans': binding}, tmp_path)


def test_native_return_values_do_not_mask_status_words_identifiers_or_partial_numbers():
    target = "return_result(status='success', value='success')"
    r = row('user: x\n', target)
    spans, missing = spans_module.spans_for_row(r, [('result:a', 'success', True)], CharTokenizer())
    begin = r['text'].rindex('success')
    assert missing == 0 and spans == [[begin, begin + 7, 'result:a', True]]
    for target in ["return_result(status='success', value=done1)", 'return_result(value=10)']:
        assert spans_module.spans_for_row(row('', target), [('result:a', '1', True)], CharTokenizer()) == ([], 1)


def test_native_literal_offsets_cover_unicode_and_multiline_json_boolean_values():
    target = 'return_result(status="é", value={\n"flag": true, "text": "café"\n})'
    r = row('', target)
    value = '{"text":"café","flag":true}'
    spans, missing = spans_module.spans_for_row(r, [('result:a', value, True)], CharTokenizer())
    begin, end = target.index('{'), target.rindex('}') + 1
    assert missing == 0 and spans == [[begin, end, 'result:a', True]]


def test_single_quoted_native_strings_with_apostrophes_and_newlines_map_to_gold_content():
    value = "Karl Jaspers' theory\nAmerica'en"
    rendered = "'Karl Jaspers\\' theory\\nAmerica\\'en'"
    target = "return_result(status='success', value=" + rendered + ')'
    spans, missing = spans_module.spans_for_row(row('', target), [('result:a', value, True)], CharTokenizer())
    begin = target.index(rendered) + 1
    assert missing == 0 and spans == [[begin, begin + len(rendered) - 2, 'result:a', True]]


def test_a_quoted_example_call_is_not_a_native_return_value():
    text = 'eval(code="return_result(value=42)")'
    assert spans_module.return_value_literals(text, 0) is None


def test_mellum_tool_call_value_location_is_exact_and_not_the_status_or_a_partial_number():
    import json
    call = {'name': 'return_result', 'arguments': {'status': 'success', 'value': 'success'}}
    target = '<tool_call>\n' + json.dumps(call) + '\n</tool_call>'
    spans, missing = spans_module.spans_for_row(row('', target), [('result:a', 'success', True)], CharTokenizer())
    begin = target.rindex('success')
    assert missing == 0 and spans == [[begin, begin + 7, 'result:a', True]]
    call['arguments']['value'] = 10
    target = '<tool_call>' + json.dumps(call) + '</tool_call>'
    assert spans_module.spans_for_row(row('', target), [('result:a', '1', True)], CharTokenizer()) == ([], 1)


def test_mellum_tool_call_unicode_nested_value_and_escaped_string_offsets():
    import json
    for value in [{'é': [True, {'x': 'café'}]}, 'say "hi"\nthen go']:
        call = {'name': 'return_result', 'arguments': {'status': 'success', 'value': value}}
        target = '<tool_call>' + json.dumps(call, ensure_ascii=False) + '</tool_call>'
        expected = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        spans, missing = spans_module.spans_for_row(row('', target), [('result:a', expected, True)], CharTokenizer())
        rendered = json.dumps(value, ensure_ascii=False)
        begin = target.index(rendered)
        end = begin + len(rendered)
        if isinstance(value, str):
            begin, end = begin + 1, end - 1
        assert missing == 0 and spans == [[begin, end, 'result:a', True]]


def test_cohort_write_cache_is_invalidated_by_records_edges_and_extractor_code(tmp_path, monkeypatch):
    import json
    from natlang_neuralese.train import trajectories as T
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'hashes.jsonl'))
    monkeypatch.setattr(T, 'target_writes', lambda record: {'answer'})
    monkeypatch.setattr(T, 'handover_notes', lambda record: {'answer': record['value']})
    checkout = tmp_path / 'checkout'
    code = checkout / 'training/neuralese/natlang_neuralese/policy.py'
    code.parent.mkdir(parents=True)
    code.write_text('extractor = 1\n')
    monkeypatch.setattr(spans_module, 'ROOT', checkout)
    records = tmp_path / 'records.jsonl'
    edges = tmp_path / 'edges.jsonl'
    records.write_text(json.dumps({'id': 'r', 'value': 'first'}) + '\n')
    edges.write_text(json.dumps({'writer_record_id': 'r', 'write_name': 'answer'}) + '\n')
    expected = {'r': [('answer', 'first', True)]}
    assert spans_module.cohort_writes(tmp_path) == expected
    with monkeypatch.context() as patch:
        patch.setattr(T, 'target_writes', lambda record: (_ for _ in ()).throw(AssertionError('cache missed')))
        assert spans_module.cohort_writes(tmp_path) == expected
    records.write_text(json.dumps({'id': 'r', 'value': 'second'}) + '\n')
    assert spans_module.cohort_writes(tmp_path) == {'r': [('answer', 'second', True)]}
    edges.write_text('')
    assert spans_module.cohort_writes(tmp_path) == {'r': [('answer', 'second', False)]}
    with monkeypatch.context() as patch:
        code.write_text('extractor = 2\n')
        patch.setattr(T, 'handover_notes', lambda record: {'answer': 'new extractor'})
        assert spans_module.cohort_writes(tmp_path) == {'r': [('answer', 'new extractor', False)]}


def test_canonical_nested_write_path_masks_only_its_field_even_when_another_value_is_equal():
    import json
    value = {'answer': 'same', 'facts': 'same'}
    for target in ["return_result(status='success', value=" + json.dumps(value) + ')',
                   '<tool_call>' + json.dumps({'name': 'return_result',
                                              'arguments': {'status': 'success', 'value': value}}) + '</tool_call>']:
        site = {'tool': 'return_result', 'value_path': ['value', 'facts'], 'source_kind': 'json-write-marker'}
        spans, missing = spans_module.spans_for_row(row('', target), [('fact', 'same', True, site)], CharTokenizer())
        begin = target.rindex('same')
        assert missing == 0 and spans == [[begin, begin + 4, 'fact', True]]
        site = {**site, 'value_path': ['value', 'missing']}
        assert spans_module.spans_for_row(row('', target), [('fact', 'same', True, site)], CharTokenizer()) == ([], 1)


def test_nested_array_field_paths_preserve_exact_unicode_literal_offsets():
    import json
    value = {'items': [{'facts': 'wrong'}, {'facts': 'café'}]}
    for target in ["return_result(value=" + json.dumps(value, ensure_ascii=False) + ')',
                   '<tool_call>' + json.dumps({'name': 'return_result', 'arguments': {'value': value}},
                                              ensure_ascii=False) + '</tool_call>']:
        site = {'tool': 'return_result', 'value_path': ['value', 'items', 1, 'facts']}
        spans, missing = spans_module.spans_for_row(row('', target), [('fact', 'café', True, site)], CharTokenizer())
        begin = target.index('café')
        assert missing == 0 and spans == [[begin, begin + 4, 'fact', True]]


def test_real_cohort_write_join_retains_the_shared_extractor_value_path(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'hashes.jsonl'))
    args = {'status': 'success', 'value': {'answer': 'same', 'facts': {
        '$write': {'name': 'fact', 'type': 'Neuralese<string>', 'source': 'same'}}}}
    record = {'id': 'r', 'messages': [], 'target': {'tool_calls': [
        {'function': {'name': 'return_result', 'arguments': json.dumps(args)}}]}}
    (tmp_path / 'records.jsonl').write_text(json.dumps(record) + '\n')
    (tmp_path / 'edges.jsonl').write_text(json.dumps({'writer_record_id': 'r', 'write_name': 'fact'}) + '\n')
    assert spans_module.cohort_writes(tmp_path) == {'r': [('fact', 'same', True, {
        'tool': 'return_result', 'value_path': ['value', 'facts'], 'source_kind': 'json-write-marker'})]}


def test_inline_code_span_uses_attested_location_through_native_and_json_escaping():
    import json
    code = '// repeated: package\'s café\nconst judgment = nl<boolean>`Check package\'s café`(input);'
    body = "Check package's café"
    start = code.index(body)
    site = {'tool': 'eval', 'value_path': ['code'], 'source_kind': 'inline-code-sidecar',
            'code': code, 'source_span': [start, start + len(body)]}
    single = code.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    targets = ["eval(code='" + single + "', finish=True)",
               '<tool_call>' + json.dumps({'name': 'eval', 'arguments': {'code': code}}, ensure_ascii=False) + '</tool_call>']
    for target in targets:
        spans, missing = spans_module.spans_for_row(row('', target), [('instruction', body, True, site)], CharTokenizer())
        assert missing == 0 and len(spans) == 1
        begin, end = spans[0][:2]
        assert target[begin:end].startswith('Check package') and target[begin:end].endswith('café')
    invalid = {**site, 'code': code + ' changed'}
    assert spans_module.spans_for_row(row('', targets[0]), [('instruction', body, True, invalid)], CharTokenizer()) == ([], 1)


def test_json_escaped_surrogate_pair_has_one_decoded_character_offset():
    import json
    value = 'x😀\ny'
    rendered = json.dumps(value, ensure_ascii=True)
    bounds = spans_module.quoted_content_offsets(rendered, value)
    assert len(bounds) == len(value)
    assert rendered[bounds[1][0]:bounds[1][1]] == '\\ud83d\\ude00'
    assert rendered[bounds[2][0]:bounds[2][1]] == '\\n'


def test_compact_history_note_uses_declared_tool_and_argument_path():
    import json
    note = "Customer's request\nKeep the original question."
    escaped = note.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    site = {'tool': 'compact_history', 'value_path': ['note'], 'source_kind': 'json-write-marker'}
    targets = ["compact_history(note='" + escaped + "')",
               '<tool_call>' + json.dumps({'name': 'compact_history', 'arguments': {'note': note}}) + '</tool_call>']
    for target in targets:
        spans, missing = spans_module.spans_for_row(row('', target), [('note', note, True, site)], CharTokenizer())
        assert missing == 0 and len(spans) == 1
        begin, end = spans[0][:2]
        assert target[begin:end].startswith('Customer') and target[begin:end].endswith('question.')
