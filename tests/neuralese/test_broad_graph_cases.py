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


def test_close_keeps_capped_held_cases_and_drops_related_train():
    a = _case('hotpotqa', 'train', ['wiki:A'])
    c = _case('squad', 'train', ['wiki:A', 'wiki:B'])
    t = _case('2wiki', 'test', ['wiki:B'])
    t2 = _case('2wiki', 'test', ['wiki:C'])
    other = _case('xlam', 'train', ['tool-schema:z'])
    kept, dropped_train, dropped_test = b.close([a, c, t, t2, other], test_cap=1)
    ids = {case['id'] for case in kept}
    assert dropped_test == 1 and len([k for k in kept if k['split'] == 'test']) == 1
    held = next(k for k in kept if k['split'] == 'test')
    assert all(not set(k['groups']) & set(held['groups']) for k in kept if k['split'] == 'train')
    assert other['id'] in ids and a['id'] in ids


def test_kinds_keep_none_lines_above_edge_minimum():
    for kind in b.KINDS.values():
        assert len(kind['none']) >= 16
        assert all(name.isidentifier() for name, _, _ in kind['views'])


def test_plain_child_result_becomes_a_producer_span_of_the_complete_call():
    import hashlib
    sys.path.insert(0, str(ROOT))
    from scripts.neuralese_data.native_chain_render import _plain_result_span
    body = 'Arthur\'s Magazine (1844-1846) was an American literary periodical.'
    writer = {'target': {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't', 'type': 'function',
              'function': {'name': 'return_result', 'arguments': json.dumps({'status': 'success', 'value': body})}}]}}
    occurrence = {'writer_source_kind': 'Neuralese<string>', 'payload_sha256': hashlib.sha256(body.encode()).hexdigest(),
                  'writer_json_pointer': '/target/tool_calls/0/function/arguments<json>/value'}
    span = _plain_result_span(writer, occurrence, body)
    assert span == {'call_index': 0, 'argument_path': ['value'], 'body': body, 'payload_sha256': occurrence['payload_sha256'],
                    'value_type': 'string'}
    assert _plain_result_span(writer, occurrence, body + ' changed') is None
    assert _plain_result_span(writer, {**occurrence, 'writer_source_kind': 'system-prompt'}, body) is None


def test_typed_text_result_is_marked_only_inside_exact_successful_return_result_value():
    sys.path.insert(0, str(ROOT))
    from scripts.neuralese_data.native_chain_render import _plain_result_span
    body = 'silver maple'
    arguments = json.dumps({'status': 'success', 'value': body}, separators=(',', ':'))
    writer = {'target': {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't', 'type': 'function',
              'function': {'name': 'return_result', 'arguments': arguments}}]}}
    occurrence = {'writer_source_kind': 'typed-text-result', 'payload_sha256': __import__('hashlib').sha256(body.encode()).hexdigest(),
                  'writer_json_pointer': '/target/tool_calls/0/function/arguments<json>/value',
                  'write_name': 'soft-state:nz1_' + 'a' * 52}
    assert _plain_result_span(writer, occurrence, body) == {
        'call_index': 0, 'argument_path': ['value'], 'body': body,
        'payload_sha256': occurrence['payload_sha256'], 'value_type': 'string'}
    assert _plain_result_span(writer, {**occurrence, 'write_name': 'ordinary-result'}, body) is None
    assert _plain_result_span(writer, {**occurrence, 'writer_json_pointer': '/target/tool_calls/0/function/arguments<json>/status'}, body) is None
    assert _plain_result_span(writer, {**occurrence, 'payload_sha256': '0' * 64}, body) is None
    wrong_tool = json.loads(json.dumps(writer))
    wrong_tool['target']['tool_calls'][0]['function']['name'] = 'eval'
    assert _plain_result_span(wrong_tool, occurrence, body) is None
    failed_result = json.loads(json.dumps(writer))
    failed_result['target']['tool_calls'][0]['function']['arguments'] = json.dumps({'status': 'error', 'value': body})
    assert _plain_result_span(failed_result, occurrence, body) is None


def test_out_of_band_inline_receipt_binds_to_exact_original_eval_code_span():
    import hashlib
    sys.path.insert(0, str(ROOT))
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import inline_instruction_target_span
    from natlang_neuralese.train.inline_instructions import INLINE_CODE_SCHEMA
    body = 'Rate ticket urgency from its reported impact.'
    name = 'inline-site:example'
    code = 'const level = await nl`' + body + '`(ticket);\nreturn level;'
    start = code.index(body)
    sidecar = {'schema': INLINE_CODE_SCHEMA, 'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'parts': [{'type': 'text', 'text': 'const level = await nl`'},
            {'$write': {'name': name, 'type': 'Neuralese<string>', 'source_kind': 'inline-instruction', 'source': body}},
            {'type': 'text', 'text': '`(ticket);\nreturn level;'}],
        'sites': [{'name': name, 'actual_tool_call_id': 'provider-call-1', 'definition_id': 'nl:example',
            'code_span': {'start': start - 1, 'end': start + len(body) + 1}, 'plan': {'parameters': [],
            'returns': {'text': 'string', 'natlang': 'string', 'aliases': {}}, 'captures': [], 'explicit_captures': False}}]}
    writer = {'target': {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'teacher_call_1', 'type': 'function',
            'function': {'name': 'eval', 'arguments': json.dumps({'code': code}, separators=(',', ':'))}}]},
        'native_source_conversion': {'source_receipts': [{'schema': 'natlang.native-inline-instruction-source-receipt/1',
            'call_id': 'teacher_call_1', 'sidecar': sidecar}]}}
    occurrence = {'writer_source_kind': 'inline-instruction', 'writer_json_pointer':
        '/source_receipts/0/sidecar/parts/1/$write', 'write_name': name,
        'payload_sha256': hashlib.sha256(body.encode()).hexdigest()}
    span = inline_instruction_target_span(writer, occurrence, body)
    assert span == {'call_index': 0, 'argument_path': ['code'], 'body': body,
        'payload_sha256': occurrence['payload_sha256'], 'value_type': 'inline-string',
        'source_start': start, 'source_end': start + len(body), 'sidecar_code_sha256': sidecar['code_sha256'],
        'site_name': name, 'site_actual_tool_call_id': 'provider-call-1',
        'source_receipt_sha256': hashlib.sha256(json.dumps(writer['native_source_conversion']['source_receipts'][0],
            ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'source_receipt_pointer': occurrence['writer_json_pointer']}
    for mutate in (
        lambda row: row['target']['tool_calls'][0]['function'].__setitem__('arguments', json.dumps({'code': code + ' ' })),
        lambda row: row['native_source_conversion']['source_receipts'][0].__setitem__('call_id', 'wrong-call'),
        lambda row: row['native_source_conversion']['source_receipts'][0]['sidecar']['parts'][1]['$write'].__setitem__('source', 'wrong'),
    ):
        forged = json.loads(json.dumps(writer)); mutate(forged)
        assert inline_instruction_target_span(forged, occurrence, body) is None
    assert inline_instruction_target_span(writer, {**occurrence, 'writer_json_pointer': '/source_receipts/0/sidecar/parts/2/$write'}, body) is None


def test_typed_string_producer_binds_to_hash_anchored_nested_return_value():
    import hashlib
    sys.path.insert(0, str(ROOT))
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import structured_string_result_span
    body = 'restaurant_suggestion'
    target = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call-1', 'type': 'function',
        'function': {'name': 'return_result', 'arguments': json.dumps(
            {'status': 'success', 'value': body}, separators=(',', ':'))}}]}
    target_sha = hashlib.sha256(json.dumps(target, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    writer = {'id': 'writer', 'target': target, 'source_ref': {'native_target_sha256': target_sha},
        'native_source_conversion': {'source_target_present': True, 'source_target': target,
                                     'source_target_sha256': target_sha}}
    occurrence = {'writer_source_kind': 'Neuralese<string>',
        'writer_json_pointer': '/target/tool_calls/0/function/arguments<json>/value',
        'payload_sha256': hashlib.sha256(body.encode()).hexdigest()}
    assert structured_string_result_span(writer, occurrence, body) == {
        'call_index': 0, 'argument_path': ['value'], 'body': body,
        'payload_sha256': occurrence['payload_sha256'], 'value_type': 'string',
        'source_target_pointer': occurrence['writer_json_pointer']}
    for invalid in ({**occurrence, 'writer_json_pointer': '/target/tool_calls/0/other/function/arguments<json>/value'},
                    {**occurrence, 'writer_json_pointer': '/target/tool_calls/0/function/arguments<json>/status'}):
        try:
            structured_string_result_span(writer, invalid, body)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid typed-result path must fail closed')
    changed = json.loads(json.dumps(writer))
    changed['target']['tool_calls'][0]['function']['arguments'] = json.dumps(
        {'status': 'success', 'value': 'different'})
    try:
        structured_string_result_span(changed, occurrence, body)
    except ValueError:
        pass
    else:
        raise AssertionError('changed target string must fail closed')
    changed = json.loads(json.dumps(writer))
    changed['source_ref']['native_target_sha256'] = '0' * 64
    try:
        structured_string_result_span(changed, occurrence, body)
    except ValueError:
        pass
    else:
        raise AssertionError('changed target binding must fail closed')


def test_inline_receipt_derivation_keeps_original_eval_target_and_records_span(monkeypatch):
    import hashlib
    sys.path.insert(0, str(ROOT))
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import pair_derivations
    from natlang_neuralese.train.inline_instructions import INLINE_CODE_SCHEMA
    body = 'Classify this ticket by its reported impact.'
    name = 'inline-site:pair-test'
    code = 'const level = await nl`' + body + '`(ticket);\nreturn level;'
    start = code.index(body)
    sidecar = {'schema': INLINE_CODE_SCHEMA, 'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'parts': [{'type': 'text', 'text': 'const level = await nl`'},
            {'$write': {'name': name, 'type': 'Neuralese<string>', 'source_kind': 'inline-instruction', 'source': body}},
            {'type': 'text', 'text': '`(ticket);\nreturn level;'}],
        'sites': [{'name': name, 'actual_tool_call_id': 'provider-call-1', 'definition_id': 'nl:test',
            'code_span': {'start': start - 1, 'end': start + len(body) + 1}, 'plan': {'parameters': [],
            'returns': {'text': 'string', 'natlang': 'string', 'aliases': {}}, 'captures': [], 'explicit_captures': False}}]}
    writer_target = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'teacher_call_2', 'type': 'function',
        'function': {'name': 'eval', 'arguments': json.dumps({'code': code}, separators=(',', ':'))}}]}
    writer = {'id': 'writer', 'target': writer_target,
        'native_source_conversion': {'source_receipts': [{'schema': 'natlang.native-inline-instruction-source-receipt/1',
            'call_id': 'teacher_call_2', 'sidecar': sidecar}]}}
    reader = {'id': 'reader', 'messages': [{'role': 'user', 'content': [{'type': 'read', 'name': name, 'source': body}]}]}
    monkeypatch.setattr('scripts.neuralese_data.build_native_semantic_pairs_e2e.authenticated_context_writer_sources',
                        lambda records: ({}, {}))
    monkeypatch.setattr('scripts.neuralese_data.build_native_semantic_pairs_e2e.authenticated_crisp_context_messages',
                        lambda *args, **kwargs: [])
    occurrence = {'writer_source_kind': 'inline-instruction', 'writer_json_pointer':
        '/source_receipts/0/sidecar/parts/1/$write', 'writer_occurrence_id': 'writer-occurrence',
        'reader_occurrence_ids': ['reader-occurrence'], 'reader_json_pointers': ['/messages/0/content/0'],
        'write_name': name, 'payload_sha256': hashlib.sha256(body.encode()).hexdigest()}
    derivations = pair_derivations(writer, reader, [], tokenizer=object(), payload=body,
        occurrence=occurrence, role_markers=['<|producer|>', '</|producer|>', '<|read|>', '</|read|>'],
        tokenizer_sha256='test', source_records=[writer, reader])
    assert derivations['writer']['target'] == writer_target
    span = derivations['writer']['producer_span']
    assert span['value_type'] == 'inline-string'
    assert span['source_start'] == start and span['source_end'] == start + len(body)
    assert span['source_receipt_pointer'] == occurrence['writer_json_pointer']
