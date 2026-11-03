"""Inline-instruction edges are soft-instruction context, never communication (S2, 2026-11-03)."""
import hashlib
import json

import pytest

from natlang_neuralese.data.native_graph_schema import (
    COMMUNICATION_EDGE, INSTRUCTION_EDGE, communication_classification, communication_payload_digests,
    edge_kind_of, indexed_graph_metadata, parse_graph)
from natlang_neuralese.train.native_continuation import (
    DEFAULT_MAX_READERS_PER_PAYLOAD, cap_readers_per_payload, communication_minimum_quotas)
from scripts.neuralese_data import native_chain_assembly as assembly

FP = 'fp'


def graph(imports, *, name='g', producer=(7, 8, 9, 10)):
    """Producer n0 exports tokens [1:3]; a consumer imports it at each declared import."""
    return {'schema': 'natlang.neuralese-native-graph/1', 'id': name, 'split': 'train', 'source_groups': ['w'],
            'tokenizer_fingerprint': FP, 'nodes': [
                {'name': 'n0', 'token_ids': list(producer), 'target_weights': [0, 0, 0, 0], 'producing': [[1, 3]]},
                {'name': 'n1', 'token_ids': [3, 4, 5, 6, 11, 12], 'target_weights': [0, 0, 0, 0, 1, 1], 'imports': imports}]}


COMMUNICATION = {'at': 1, 'source': 'n0', 'start': 1, 'stop': 3}
INSTRUCTION = {**COMMUNICATION, 'instruction': True}


def test_instruction_import_is_not_communication_but_a_value_import_is():
    value = parse_graph(graph([COMMUNICATION]), tokenizer_fingerprint=FP)
    inline = parse_graph(graph([INSTRUCTION]), tokenizer_fingerprint=FP)
    assert communication_classification(value) == {'communication': True, 'producer_consumer': True,
                                                   'external_state_consumption': False}
    assert communication_classification(inline) == {'communication': False, 'producer_consumer': False,
                                                    'external_state_consumption': False}
    # The graph still executes with its import: the plan is the same, only the accounting differs.
    assert inline.plan.token_count == value.plan.token_count
    assert any(node.imports for node in inline.plan.invocations)
    indexed = indexed_graph_metadata(graph([INSTRUCTION]), tokenizer_fingerprint=FP)
    assert communication_classification(indexed)['communication'] is False


def test_mixed_graph_with_a_genuine_value_edge_remains_communication():
    mixed = graph([{**INSTRUCTION, 'at': 1}, {**COMMUNICATION, 'at': 3}])
    assert communication_classification(parse_graph(mixed, tokenizer_fingerprint=FP))['communication'] is True


def test_import_instruction_flag_is_strict():
    with pytest.raises(ValueError, match='instruction'):
        parse_graph(graph([{**COMMUNICATION, 'instruction': False}]), tokenizer_fingerprint=FP)


def test_payload_digest_identifies_identical_payloads_and_skips_instructions():
    a = parse_graph(graph([COMMUNICATION], name='a'), tokenizer_fingerprint=FP)
    b = parse_graph(graph([COMMUNICATION], name='b', producer=(99, 8, 9, 10)), tokenizer_fingerprint=FP)
    other = parse_graph(graph([COMMUNICATION], name='c', producer=(7, 8, 55, 10)), tokenizer_fingerprint=FP)
    assert communication_payload_digests(a) == communication_payload_digests(b)  # same exported tokens
    assert communication_payload_digests(a) != communication_payload_digests(other)
    assert communication_payload_digests(parse_graph(graph([INSTRUCTION]), tokenizer_fingerprint=FP)) is None


def test_edge_kind_derives_from_writer_kind_and_fails_closed_on_contradiction():
    assert edge_kind_of('inline-instruction') == INSTRUCTION_EDGE
    assert edge_kind_of('typed-text-result') == COMMUNICATION_EDGE
    assert edge_kind_of(None) == COMMUNICATION_EDGE
    assert edge_kind_of('inline-instruction', 'instruction') == INSTRUCTION_EDGE
    with pytest.raises(ValueError, match='contradicts'):
        edge_kind_of('typed-text-result', 'instruction')
    with pytest.raises(ValueError, match='contradicts'):
        edge_kind_of('inline-instruction', 'communication')
    with pytest.raises(ValueError, match='unknown'):
        edge_kind_of(None, 'banana')


def assemble_pair(monkeypatch, writer_source_kind, declared=None):
    class Tokenizer:
        def decode(self, ids, **_): return ''.join({8: 'x'}[x] for x in ids)
    monkeypatch.setattr(assembly, 'writer_payload', lambda writer, name, sha: 'x')
    monkeypatch.setattr(assembly, 'source_occurrences', lambda *args, **kw: {
        'reader_occurrence_ids': ['r'], 'reader_json_pointers': ['/messages/0'], 'writer_occurrence_id': 'w',
        'writer_source_kind': writer_source_kind})
    chain = {'source': 's', 'trajectory': 't', 'split': 'train', 'records': ['a', 'b'], 'edges': 1}
    records = {x: {'id': x, 'split': 'train', 'source_groups': ['g'], 'target': {'content': x},
                   'source_ref': {'trajectory_id': 't'}} for x in 'ab'}
    edge = {'source': 's', 'trajectory': 't', 'split': 'train', 'writer_split': 'train', 'writer_record_id': 'a',
            'reader_record_id': 'b', 'write_name': 'first', 'body_sha256': hashlib.sha256(b'x').hexdigest(),
            **({'edge_kind': declared} if declared else {})}
    sha = hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    rendered = {
        'a': {'token_ids': [1, 10, 8, 11, 2], 'target_weights': [0, 1, 1, 1, 1], 'producing': [[2, 3]],
              'producer_bindings': [{'edge_sha256': sha, 'writer_occurrence_id': 'w', 'start': 2, 'stop': 3}]},
        'b': {'token_ids': [1, 12, 8, 13, 2], 'target_weights': [0, 1, 1, 1, 1],
              'reads': [{'start': 2, 'stop': 3, 'source_record_id': 'a', 'source_start': 2, 'source_stop': 3,
                         'edge_sha256': sha, 'occurrence_id': 'r', 'source_pointer': '/messages/0'}]}}
    digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                                     separators=(',', ':')).encode()).hexdigest()
    for rid, row in rendered.items():
        row['source_record_sha256'] = digest(records[rid])
        row['source_target_sha256'] = digest(records[rid]['target'])
        row['supervised_suffix_start'] = 1
    return assembly.assemble(chain, records, [edge], rendered, tokenizer=Tokenizer(), tokenizer_fingerprint=FP,
                             role_ids=(10, 11, 12, 13), source_binding={'records_sha256': 'r'})


def test_chain_assembly_tags_inline_instruction_edges_and_keeps_value_edges_communication(monkeypatch):
    inline = assemble_pair(monkeypatch, 'inline-instruction')
    assert inline['nodes'][1]['imports'][0]['instruction'] is True
    assert communication_classification(parse_graph(inline, tokenizer_fingerprint=FP))['communication'] is False
    value = assemble_pair(monkeypatch, 'typed-text-result')
    assert 'instruction' not in value['nodes'][1]['imports'][0]
    assert communication_classification(parse_graph(value, tokenizer_fingerprint=FP))['communication'] is True
    with pytest.raises(ValueError, match='contradicts'):
        assemble_pair(monkeypatch, 'typed-text-result', declared='instruction')


def entries(spec):
    return [{'id': ident, 'communication': True, 'payload_sha256': digests} for ident, digests in spec]


def test_default_cap_allows_three_readers_per_identical_payload():
    assert DEFAULT_MAX_READERS_PER_PAYLOAD == 3
    rows = entries([(f'r{i}', ['same']) for i in range(10)] + [('own', ['distinct'])])
    report = {}
    quota = communication_minimum_quotas(rows, ['native-communication'] * len(rows), 7, report=report)
    assert len(quota) == 4 and quota['own'] == 7 and set(quota.values()) == {7}
    assert sum(ident.startswith('r') for ident in quota) == 3
    assert report == {'max_readers_per_payload': 3, 'capped_rows': 7, 'unknown_payload_rows': 0, 'distinct_payloads': 2}
    # Deterministic and independent of source order.
    reordered = list(reversed(rows))
    assert communication_minimum_quotas(reordered, ['native-communication'] * len(rows), 7) == quota


def test_cap_counts_every_payload_of_a_row_and_keeps_unidentified_rows():
    rows = entries([('a', ['p']), ('b', ['p']), ('c', ['p', 'q']), ('d', ['p', 'q'])]) + [
        {'id': 'old', 'communication': True}]
    kept, report = cap_readers_per_payload(rows, {row['id'] for row in rows}, 2)
    assert 'old' in kept and report['unknown_payload_rows'] == 1
    # p is full after two readers; a row keeps its quota only while one of its payloads has room.
    assert sum(ident in kept for ident in 'abcd') >= 2
    assert report['capped_rows'] == 4 - sum(ident in kept for ident in 'abcd')
    kept, report = cap_readers_per_payload(rows, {row['id'] for row in rows}, None)
    assert len(kept) == 5 and report['capped_rows'] == 0
    with pytest.raises(ValueError, match='positive integer'):
        cap_readers_per_payload(rows, {'a'}, 0)


def test_instruction_only_rows_are_never_in_the_quota_family():
    rows = [{'id': 'inline', 'communication': False}, {'id': 'value', 'communication': True, 'payload_sha256': ['x']}]
    with pytest.raises(ValueError, match='non-communication'):
        communication_minimum_quotas(rows, ['native-communication', 'native-communication'], 7)
    assert communication_minimum_quotas(rows, ['natlang', 'native-communication'], 7) == {'value': 7}
