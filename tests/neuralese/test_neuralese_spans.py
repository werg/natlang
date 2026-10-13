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
