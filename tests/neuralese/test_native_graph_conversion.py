from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.data.native_graph_conversion import (
    locate_attested_role_span, text_row_graph, witnessed_pair_graph,
)


class AsciiTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(value) for value in ids)

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        result = {'input_ids': list(text.encode())}
        if return_offsets_mapping:
            result['offset_mapping'] = [(index, index+1) for index in range(len(text))]
        return result


class RoleAsciiTokenizer(AsciiTokenizer):
    markers = ('<|produce|>', '<|/produce|>', '<|consume|>', '<|/consume|>')
    marker_ids = (1001, 1002, 1003, 1004)

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets, position = [], [], 0
        while position < len(text):
            found = next(((marker, token) for marker, token in zip(self.markers, self.marker_ids)
                          if text.startswith(marker, position)), None)
            if found:
                marker, token = found
                ids.append(token); offsets.append((position, position + len(marker)))
                position += len(marker)
            else:
                ids.append(text[position].encode()[0]); offsets.append((position, position + 1))
                position += 1
        result = {'input_ids': ids}
        if return_offsets_mapping: result['offset_mapping'] = offsets
        return result

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.markers[self.marker_ids.index(value)] if value in self.marker_ids
                       else bytes([value]).decode() for value in ids)


def row(identifier, text, *, split='train', groups=None, boundary=None):
    return {'id': identifier, 'text': text, 'token_ids': list(text.encode()),
            'tokenizer_sha256': 'a' * 64, 'split': split,
            'source_groups': groups or ['group:' + identifier],
            'supervised_suffix_start': len(text) // 2 if boundary is None else boundary,
            'neuralese_context_attestations': []}


def test_text_row_conversion_preserves_full_tokens_and_exact_target_boundary():
    source = row('ordinary', 'complete prompt and target')
    graph = text_row_graph(source, source_tokenizer_fingerprint='a' * 64,
        tokenizer=AsciiTokenizer(), expanded_tokenizer_fingerprint='b' * 64, context_weight=.05)
    example = parse_graph(graph, tokenizer_fingerprint='b' * 64)
    ids = source['token_ids']
    boundary = source['supervised_suffix_start']
    assert list(example.plan.invocations[0].token_ids) == ids
    assert list(example.target_weights) == [.0] + [.05] * (boundary-1) + [1.] * (len(ids)-boundary)
    assert graph['conversion']['kind'] == 'static-transformation'


def test_marker_looking_text_stays_complete_ordinary_text_without_attested_role():
    source = row('marker-example', 'Prompt says <|neuralese_produce|> literally; target is <|/neuralese_produce|>.')
    graph = text_row_graph(source, source_tokenizer_fingerprint='a' * 64,
        tokenizer=AsciiTokenizer(), expanded_tokenizer_fingerprint='b' * 64)
    example = parse_graph(graph, tokenizer_fingerprint='b' * 64)
    node = example.plan.invocations[0]
    assert list(node.token_ids) == source['token_ids']
    assert not node.producing and not node.imports
    assert graph['conversion']['role_interpretation'] == 'ordinary-text'


def test_witnessed_pair_encodes_roles_import_and_model_neutral_payload_alignment():
    import hashlib
    payload = 'verified payload'
    source = row('writer', 'prompt:' + payload + ':answer', boundary=7)
    target = row('reader', 'context:' + payload + ':outcome', boundary=10)
    span_start = source['text'].index(payload)
    read_start = target['text'].index(payload)
    graph = witnessed_pair_graph(source, target, source_tokenizer=AsciiTokenizer(),
        source_tokenizer_fingerprint='a' * 64, expanded_tokenizer_fingerprint='b' * 64,
        role_ids=(1001, 1002, 1003, 1004),
        writer_span={'start': span_start, 'stop': span_start + len(payload)},
        reader_span={'start': read_start, 'stop': read_start + len(payload)},
        payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        source_span_id='span:source-1', edge_id='edge:exact-1', chain_id='chain:1')
    example = parse_graph(graph, tokenizer_fingerprint='b' * 64)
    producer, consumer = example.plan.invocations
    assert producer.producing == ((span_start + 1, span_start + 1 + len(payload)),)
    assert consumer.imports[0].at == read_start + 1
    assert consumer.imports[0].source == 'producer'
    assert graph['source_pairing']['payloads'][0]['text'] == payload
    assert graph['source_pairing']['payloads'][0]['text_sha256'] == hashlib.sha256(payload.encode()).hexdigest()
    assert example.target_weights[example.plan.offsets[1] + read_start + 1] == 0.
    assert example.target_weights[-1] == 1.


def test_witnessed_pair_rejects_mismatched_payload_and_cross_split():
    import hashlib
    from pytest import raises

    writer = row('writer', 'prefix:value:suffix')
    reader = row('reader', 'context:value:answer')
    common = dict(source_tokenizer=AsciiTokenizer(), source_tokenizer_fingerprint='a' * 64,
        expanded_tokenizer_fingerprint='b' * 64, role_ids=(1001, 1002, 1003, 1004),
        writer_span={'start': 7, 'stop': 12}, reader_span={'start': 8, 'stop': 13},
        payload_sha256=hashlib.sha256(b'wrong').hexdigest(), source_span_id='span',
        edge_id='edge', chain_id='chain')
    with raises(ValueError, match='hash differs'):
        witnessed_pair_graph(writer, reader, **common)
    held = row('reader', 'context:value:answer', split='test')
    with raises(ValueError, match='share one explicit split'):
        witnessed_pair_graph(writer, held, **common)


def test_attested_span_uses_exact_payload_hash_and_removes_only_witnessed_wrapper():
    import hashlib
    from pytest import raises

    payload = 'verified payload'
    text = 'prompt <|neuralese|>BODY<|/neuralese|> then <|neuralese|>' + payload + '<|/neuralese|>'
    source = row('writer', text)
    span = locate_attested_role_span(source, tokenizer=AsciiTokenizer(),
        payload_sha256=hashlib.sha256(payload.encode()).hexdigest())
    assert span['text'] == payload
    assert text[span['container_start']:span['container_stop']] == '<|neuralese|>' + payload + '<|/neuralese|>'
    with raises(ValueError, match='expected one'):
        locate_attested_role_span(row('ambiguous', text + ' <|neuralese|>' + payload + '<|/neuralese|>'),
            tokenizer=AsciiTokenizer(), payload_sha256=hashlib.sha256(payload.encode()).hexdigest())


def test_e2e_pair_occurrence_binds_structural_write_and_exact_read_not_text_search():
    import importlib.util
    from pathlib import Path
    import pytest

    script = Path(__file__).parents[2] / 'scripts/neuralese_data/build_native_semantic_pairs_e2e.py'
    spec = importlib.util.spec_from_file_location('e2e_pair_builder', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = 'same body appears more than once'
    writer = {'target': {'tool_calls': [{'function': {'arguments':
        '{"result":{"$write":{"name":"r","source":"same body appears more than once"}}}'}}]}}
    reader = {'messages': [{'role': 'tool', 'content': [
        {'type': 'read', 'name': 'r', 'source': payload},
        {'type': 'text', 'text': payload}]}]}
    occurrence = module.source_occurrences(writer, reader, 'r', payload,
        module.sha256_bytes(payload.encode()))
    assert occurrence['writer_json_pointer'].endswith('/$write')
    assert occurrence['reader_json_pointer'] == '/messages/0/content/0'
    duplicate_reader = {'messages': [{'type': 'read', 'name': 'r', 'source': payload},
                                     {'type': 'read', 'name': 'r', 'source': payload}]}
    with pytest.raises(ValueError, match='ambiguous'):
        module.source_occurrences(writer, duplicate_reader, 'r', payload,
            module.sha256_bytes(payload.encode()))


def test_prebounded_typed_pair_spans_reuse_markers_without_duplicate_wrapping():
    import hashlib
    tok = RoleAsciiTokenizer()
    payload = 'typed value'
    writer_text = 'prompt' + tok.markers[0] + payload + tok.markers[1]
    reader_text = 'read:' + tok.markers[2] + payload + tok.markers[3] + ':answer'
    writer, reader = row('writer', writer_text, boundary=6), row('reader', reader_text, boundary=18)
    writer['token_ids'] = tok(writer_text)['input_ids']
    reader['token_ids'] = tok(reader_text)['input_ids']
    # Corrected text rows must carry their expanded tokenizer fingerprint.
    writer['tokenizer_sha256'] = reader['tokenizer_sha256'] = 'a' * 64
    writer_span = {'container_start': 6, 'start': 7, 'stop': 7 + len(payload),
                   'container_stop': 8 + len(payload), 'role_wrapped': True}
    reader_span = {'container_start': 5, 'start': 6, 'stop': 6 + len(payload),
                   'container_stop': 7 + len(payload), 'role_wrapped': True}
    graph = witnessed_pair_graph(writer, reader, source_tokenizer=tok,
        source_tokenizer_fingerprint='a' * 64, expanded_tokenizer_fingerprint='a' * 64,
        role_ids=tok.marker_ids, writer_span=writer_span, reader_span=reader_span,
        payload_sha256=hashlib.sha256(payload.encode()).hexdigest(), source_span_id='write:1',
        edge_id='edge:1', chain_id='chain:1')
    example = parse_graph(graph, tokenizer_fingerprint='a' * 64)
    producer, consumer = example.plan.invocations
    assert producer.producing == ((7, 7 + len(payload)),)
    assert len(producer.token_ids) == len(tok(writer_text)['input_ids'])
    assert consumer.imports[0].at == 6


def test_contextual_body_uses_its_actual_standalone_tokenization():
    import hashlib

    class SegmentedTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return ''.join({8: 'a', 9: 'b', 20: 'prefix', 21: 'suffix', 77: 'ab',
                            1001: '<|produce|>', 1002: '<|/produce|>',
                            1003: '<|consume|>', 1004: '<|/consume|>'}[value]
                           for value in ids)

    tok = SegmentedTokenizer()
    body = 'ab'
    writer = {'id': 'writer', 'text': 'prefix<|produce|>ab<|/produce|>suffix', 'token_ids': [20, 1001, 77, 1002, 21],
              'tokenizer_sha256': 'a' * 64, 'split': 'train', 'source_groups': ['g'],
              'supervised_suffix_start': 1}
    reader = {'id': 'reader', 'text': 'prefix<|consume|>ab<|/consume|>suffix', 'token_ids': [20, 1003, 77, 1004, 21],
              'tokenizer_sha256': 'a' * 64, 'split': 'train', 'source_groups': ['g'],
              'supervised_suffix_start': 4}
    standalone = {'container_start': 1, 'container_stop': 4, 'body_text': body,
                  'payload_token_ids': [8, 9], 'tokenize_payload_alone': True}
    graph = witnessed_pair_graph(writer, reader, source_tokenizer=tok,
        source_tokenizer_fingerprint='a' * 64, expanded_tokenizer_fingerprint='b' * 64,
        role_ids=(1001, 1002, 1003, 1004), writer_span=standalone,
        reader_span={**standalone}, payload_sha256=hashlib.sha256(body.encode()).hexdigest(),
        source_span_id='write:segmented', edge_id='edge:segmented', chain_id='chain:segmented')
    parsed = parse_graph(graph, tokenizer_fingerprint='b' * 64)
    producer, consumer = parsed.plan.invocations
    assert producer.token_ids == (20, 1001, 8, 9, 1002, 21)
    assert producer.producing == ((2, 4),)
    assert consumer.token_ids == (20, 1003, 8, 9, 1004, 21)
    assert consumer.imports[0].at == 2
    assert parsed.target_weights[2:4] == (1., 1.)
    assert parsed.target_weights[parsed.plan.offsets[1] + 2:parsed.plan.offsets[1] + 4] == (0., 0.)
