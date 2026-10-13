"""T_self (train/self_text.py): the producer's own free-running text stops at the native value boundary and replaces
the gold value where the consumer's crisp rendering reads it."""

import json

import torch

from natlang_neuralese.train.self_text import (decode_value, generate_span, string_end, substitute_sources,
                                               value_end, value_source)
from natlang_neuralese.train.trajectories import crisp_messages, handover_notes


def lfm_native(value):
    """The LFM2.5 template's argument spelling (format_arg_value): quoted, escaped strings; JSON containers."""
    if isinstance(value, str):
        return "'" + value.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '\\r') + "'"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def string_native(value):
    return lfm_native(value)[1:-1]


class Scripted:
    """A stub readout that emits a fixed token sequence over a toy vocabulary."""

    def __init__(self, vocab, script):
        self.vocab, self.script, self.fed = vocab, [vocab.index(t) for t in script], []

    def next_logits(self):
        logits = torch.full((len(self.vocab),), -10.0)
        logits[self.script[len(self.fed)]] = 10.0
        return logits

    def advance(self, token):
        self.fed.append(token)

    def decode(self, ids):
        return ''.join(self.vocab[i] for i in ids)


VOCAB = ['<eos>', 'Mira', ' signed', ' it', "\\'", "s", "')]", ')]', '{"a"', ': 1', ', "b": [2', ']}', ' 12', '3', '.5',
         "'x", "\\\\", "'"]


def run(script, end_of, cap=50, **kw):
    model = Scripted(VOCAB, script)
    return generate_span(model.next_logits, model.advance, model.decode, stop_ids={0}, end_of=end_of, cap=cap, **kw), model


def test_string_boundary_is_the_first_unescaped_closing_quote():
    assert string_end("it\\'s done')]", "'") == 10
    assert string_end("a\\\\')", "'") == 3  # an escaped backslash does not escape the quote
    assert string_end("it\\'s", "'") is None
    got, model = run(['Mira', ' signed', ' it', "\\'", 's', "')]", '<eos>'], lambda t, f: string_end(t, "'"))
    assert got['span'] == "Mira signed it\\'s" and got['end'] == 'boundary' and not got['cap_hit']
    assert len(model.fed) == 5  # the boundary token is never fed back
    source, exact = value_source(got['span'], 'string', "'", string_native, 'gold')
    assert source == "Mira signed it's" and exact


def test_json_boundary_needs_a_complete_value():
    assert value_end('{"a": 1, "b": [2') is None
    assert value_end('{"a": 1, "b": [2]})]') == len('{"a": 1, "b": [2]}')
    assert value_end(' 12') is None and value_end(' 12', final=True) == 3
    assert value_end(' 12.') is None and value_end(' 12.5)]') == 5
    assert value_end("'x')]") == 3 and value_end('True)') == 4
    got, _ = run(['{"a"', ': 1', ', "b": [2', ']}', ')]'], value_end)
    assert got['span'] == '{"a": 1, "b": [2]}' and got['end'] == 'boundary'
    source, exact = value_source(got['span'], 'unknown', "'", lfm_native, '{"a":0,"b":[]}')
    assert source == '{"a":1,"b":[2]}' and exact  # in the gold source's own (compact) spelling
    got, _ = run([' 12', '3', '.5', ')]'], value_end)
    assert got['span'] == ' 123.5'
    got, _ = run([' 12', '<eos>'], value_end)
    assert got['span'] == ' 12' and got['end'] == 'boundary'
    assert decode_value("'x'", 'unknown', "'", lfm_native) == ('x', True)


def test_cap_and_stop_token_are_recorded():
    got, _ = run(['Mira', ' signed', ' it', ' it'], lambda t, f: string_end(t, "'"), cap=3)
    assert got['cap_hit'] and got['end'] == 'cap' and got['span'] == 'Mira signed it'
    got, _ = run(['Mira', '<eos>'], lambda t, f: string_end(t, "'"))
    assert got['end'] == 'stop_token' and not got['cap_hit'] and got['span'] == 'Mira'
    source, exact = value_source('{"a": [1', 'unknown', "'", lfm_native, '{}')
    assert source == '{"a": [1' and not exact  # unparsable: the raw text is what the reader sees


def test_sampling_is_reproducible_by_seed():
    logits = torch.zeros(len(VOCAB))
    logits[0] = -1e9

    def sample(seed):
        g = torch.Generator().manual_seed(seed)
        return generate_span(lambda: logits, lambda t: None, lambda ids: '|'.join(map(str, ids)), stop_ids={0},
                             end_of=lambda t, f: None, cap=12, temperature=0.6, generator=g)['ids']
    assert sample(5) == sample(5) and sample(5) != sample(6)


def test_substitution_replaces_the_value_where_the_gold_crisp_rendering_reads_it():
    gold, own = 'Line 2 is a contractual fee.', 'Line 2 is a fee.'
    record = {'id': 'caller', 'messages': [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Decide.'}]},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'h', 'type': 'function', 'function': {
            'name': 'return_result', 'arguments': json.dumps({'value': {'$write': {
                'name': 'result:child', 'type': 'Neuralese<string>', 'source': gold}}})}}]},
        {'role': 'tool', 'tool_call_id': 'e', 'content': [{'type': 'text', 'text': 'console:\n'},
                                                          {'type': 'read', 'name': 'result:child', 'source': gold},
                                                          {'type': 'read', 'name': 'result:other', 'source': 'keep'}]},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'f', 'type': 'function', 'function': {
            'name': 'eval', 'arguments': [{'type': 'text', 'text': '{"code": "'},
                                          {'type': 'read', 'name': 'result:child', 'source': gold},
                                          {'type': 'text', 'text': '"}'}]}}]}]}
    shown = substitute_sources(record, {'result:child': own})
    assert record['messages'][2]['content'][1]['source'] == gold  # the input record is unchanged
    assert handover_notes(shown)['result:child'] == own and handover_notes(shown)['result:other'] == 'keep'
    rendered = json.dumps(crisp_messages(shown['messages'], {}, handover_notes(shown)))
    assert gold not in rendered and rendered.count(own) == 3 and 'keep' in rendered
    assert substitute_sources(record, {}) is record
