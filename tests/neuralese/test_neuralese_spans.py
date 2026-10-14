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
