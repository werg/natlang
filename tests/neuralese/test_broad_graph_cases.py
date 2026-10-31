"""Broad Neuralese graph case adapters: verbatim views, code-computed references, split closure."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))

import broad_graph_cases as b  # noqa: E402


def test_sentence_spans_are_verbatim_and_skip_abbreviations():
    text = 'Dr. Smith arrived in the U.S. in 1990. He left in 1995! Then he wrote "Home." It sold well.'
    spans = b.sentence_spans(text)
    sentences = [text[a:c] for a, c in spans]
    assert sentences == ['Dr. Smith arrived in the U.S. in 1990.', 'He left in 1995!', 'Then he wrote "Home."',
                         'It sold well.']
    assert all(text[a:c] in text for a, c in spans)


def test_greedy_oracle_selects_overlapping_sentences_in_order():
    sentences = ['The council approved the new budget on Monday night.', 'Weather was mild across the region today.',
                 'The budget raises school funding by ten percent next year.']
    picked = b.greedy_oracle(sentences, 'The council approved a budget that raises school funding by ten percent.')
    assert picked and picked == sorted(picked) and 1 not in picked
    assert b.greedy_oracle(sentences, 'Xylophones zigzag quietly.') == []


def test_sql_identifiers_ignore_string_literals_and_keep_quoted_names():
    found = b._sql_identifiers('SELECT `Home Town`, T1.name FROM people AS T1 WHERE T1.city = \'Paris people\'')
    assert {'home town', 'name', 'people', 't1', 'city'} <= found
    assert 'paris' not in found


def _case(source, split, groups, view='A sentence that is long enough.'):
    docs = [{'title': 'x', 'text': f'# x\n{source}{groups}', 'view': view},
            {'title': 'y', 'text': f'# y\n{source}{groups} other', 'view': 'No relevant facts.'}]
    return b.make_case(source=source, dataset=source, upstream_id=str(groups), upstream_split=split, split=split,
                       groups=groups, kind='evidence', request_text='Who?', answer='Someone', documents=docs,
                       license={'spdx': 'MIT'}, revision='r', files=[], adaptation='test', domain='test')


def test_make_case_declares_distinct_functions_and_document_groups():
    case = _case('hotpotqa', 'train', ['hotpotqa-q:1'])
    assert case['schema'] == b.SCHEMA
    names = {case['view']['name'], case['consumer']['name'], case['root']['name'], case['request']['name']}
    assert len(names) == 4
    assert all(doc['group'] in case['groups'] for doc in case['documents'])
    assert case['documents'][1]['view_origin'] == 'none-line'
    assert case['view']['name'] in case['root']['instructions_root']


def test_close_moves_whole_component_to_test():
    a = _case('hotpotqa', 'train', ['wiki:A'])
    c = _case('squad', 'train', ['wiki:A', 'wiki:B'])
    t = _case('2wiki', 'test', ['wiki:B'])
    other = _case('xlam', 'train', ['tool-schema:z'])
    assert b.close([a, c, t, other]) == 2
    assert (a['split'], c['split'], t['split'], other['split']) == ('test', 'test', 'test', 'train')


def test_kinds_keep_none_lines_above_edge_minimum():
    for kind in b.KINDS.values():
        assert len(kind['none']) >= 16
        assert all(name.isidentifier() for name, _, _ in kind['views'])


def test_plain_child_result_becomes_a_producer_span_of_the_complete_call():
    sys.path.insert(0, str(ROOT))
    from scripts.neuralese_data.native_chain_render import _plain_result_span
    body = 'Arthur\'s Magazine (1844-1846) was an American literary periodical.'
    writer = {'target': {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't', 'type': 'function',
              'function': {'name': 'return_result', 'arguments': json.dumps({'status': 'success', 'value': body})}}]}}
    occurrence = {'writer_source_kind': 'Neuralese<string>', 'payload_sha256': 'f' * 64,
                  'writer_json_pointer': '/target/tool_calls/0/function/arguments<json>/value'}
    span = _plain_result_span(writer, occurrence, body)
    assert span == {'call_index': 0, 'argument_path': ['value'], 'body': body, 'payload_sha256': 'f' * 64,
                    'value_type': 'string'}
    assert _plain_result_span(writer, occurrence, body + ' changed') is None
    assert _plain_result_span(writer, {**occurrence, 'writer_source_kind': 'system-prompt'}, body) is None
