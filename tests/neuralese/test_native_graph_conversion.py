from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.data.native_graph_conversion import (
    locate_attested_role_span, text_row_graph, whole_trajectory_text_row_graph, witnessed_pair_graph,
)


def test_pair_source_corpus_repair_rebinds_all_witnesses_without_changing_model_graphs():
    import copy
    import pytest
    from types import SimpleNamespace
    from test_native_state_sft import authenticated_pair
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import rebind_pair_source_corpus
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from natlang_neuralese.data.native_pair_schema import parse_pair
    row=authenticated_pair('actual-source-edge');row['conversion']={'source_corpus':'wrong-rest-default'}
    for view in row['models'].values():
        view['graph']['source_pairing']['communication_sha256']=row['communication_sha256']
    original=copy.deepcopy(row)
    repaired=rebind_pair_source_corpus(row,'neuralese-e2e-cloud-big2-protected-20261015-v1')
    assert row==original and repaired['id']==original['id']
    assert repaired['communication']['source_binding']['corpus_id']==repaired['conversion']['source_corpus']
    assert repaired['communication_sha256']==canonical_json_sha256_hex(repaired['communication'])
    assert repaired['communication_sha256']!=original['communication_sha256']
    for name,view in repaired['models'].items():
        assert view['graph']['nodes']==original['models'][name]['graph']['nodes']
        assert view['export']==original['models'][name]['export']
        assert view['witness']['communication_sha256']==repaired['communication_sha256']
        assert view['graph']['source_pairing']['communication_sha256']==repaired['communication_sha256']
    participants={name:SimpleNamespace(config={'tokenizer_fingerprint':view['graph']['tokenizer_fingerprint']})
                  for name,view in row['models'].items()}
    assert set(parse_pair(repaired,participants,required_split='train'))==set(participants)
    bad=copy.deepcopy(row);next(iter(bad['models'].values()))['witness']['communication_sha256']='changed'
    with pytest.raises(ValueError,match='original model witness changed'):
        rebind_pair_source_corpus(bad,'correct')


def test_e2e_conversion_requires_explicit_source_corpus_before_opening_inputs(monkeypatch,capsys):
    import pytest
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import main
    args=['builder']
    for flag in ['records','pieces','edges','out','lfm-tokenizer','mellum-tokenizer','records-sha256','edges-sha256']:
        args.extend(['--'+flag,'not-opened'])
    monkeypatch.setattr('sys.argv',args)
    with pytest.raises(SystemExit) as error:main()
    assert error.value.code==2
    assert '--source-corpus-id' in capsys.readouterr().err


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


def test_pair_spans_preserve_actual_segmented_chat_tokens_instead_of_retokenizing():
    import hashlib
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import exact_marked_spans
    class MergedTokenizer(RoleAsciiTokenizer):
        def __call__(self, text, **kwargs):
            encoded = super().__call__(text, **kwargs)
            if text.startswith('AB'):
                encoded['input_ids'][:2] = [2001]
                if 'offset_mapping' in encoded:
                    encoded['offset_mapping'][:2] = [(0,2)]
            return encoded
        def decode(self, ids, **kwargs):
            return ''.join('AB' if value == 2001 else super(MergedTokenizer,self).decode([value], **kwargs)
                           for value in ids)
        def convert_tokens_to_ids(self, marker):
            return self.marker_ids[self.markers.index(marker)]
    text = 'AB<|consume|>x<|/consume|>'
    actual = RoleAsciiTokenizer()(text, return_offsets_mapping=True)
    source = {'text':text, 'token_ids':actual['input_ids'], 'token_offsets':actual['offset_mapping']}
    assert MergedTokenizer()(text)['input_ids'] != source['token_ids']
    spans = exact_marked_spans(source, MergedTokenizer(), 'x', open_marker='<|consume|>',
        close_marker='<|/consume|>', expected_sha256=hashlib.sha256(b'x').hexdigest(), expected_count=1)
    assert spans[0]['container_start'] == 2 and spans[0]['container_stop'] == 5
    assert spans[0]['payload_token_ids'] == [ord('x')]
    source['token_offsets'][2] = (0,1)
    import pytest
    with pytest.raises(ValueError, match='role markers do not bind'):
        exact_marked_spans(source, MergedTokenizer(), 'x', open_marker='<|consume|>',
            close_marker='<|/consume|>', expected_sha256=hashlib.sha256(b'x').hexdigest(), expected_count=1)


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
    assert graph['conversion']['source_record_ids'] == ['ordinary']


class ChatRoleAsciiTokenizer(AsciiTokenizer):
    special = ('<|im_start|>', '<|im_sep|>', '<|im_end|>', '<tool_response>', '</tool_response>',
               '<think>', '</think>', 'system', 'user', 'assistant', 'tool')
    token_ids = {value: 2000 + index for index, value in enumerate(special)}
    unk_token_id = -1

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets, position = [], [], 0
        while position < len(text):
            match = next((value for value in self.special if text.startswith(value, position)), None)
            if match is None:
                ids.append(ord(text[position])); offsets.append((position, position + 1)); position += 1
            else:
                ids.append(self.token_ids[match]); offsets.append((position, position + len(match)))
                position += len(match)
        result = {'input_ids': ids}
        if return_offsets_mapping:
            result['offset_mapping'] = offsets
        return result

    def decode(self, ids, skip_special_tokens=False):
        reverse = {value: key for key, value in self.token_ids.items()}
        return ''.join(reverse.get(value, chr(value)) for value in ids)

    def convert_tokens_to_ids(self, value):
        return self.token_ids.get(value, self.unk_token_id)


def test_whole_trajectory_conversion_reuses_exact_role_weight_policy():
    import pytest
    from natlang_neuralese.data.native_graph import parse_graph
    from natlang_neuralese.train.text_supervision import ROLE_CODES, chat_roles, segmented_position_weights
    import torch

    tokenizer = ChatRoleAsciiTokenizer()
    text = ('<|im_start|>user\n<tool_response>tool output</tool_response><|im_end|>'
            '<|im_start|>assistant<|im_sep|><think>reasoning</think>answer<|im_end|>')
    ids = tokenizer(text)['input_ids']
    source = {'id': 'whole-trajectory:example', 'text': text, 'token_ids': ids,
        'tokenizer_sha256': 'a' * 64, 'split': 'train', 'source_groups': ['group:trajectory'],
        'supervision': 'whole-trajectory', 'source_record_ids': ['record:1', 'record:2']}
    graph = whole_trajectory_text_row_graph(source, source_tokenizer_fingerprint='a' * 64,
        tokenizer=tokenizer, expanded_tokenizer_fingerprint='b' * 64)
    example = parse_graph(graph, tokenizer_fingerprint='b' * 64)
    role_ids = {tokenizer.convert_tokens_to_ids(name): name for name in ('system', 'user', 'assistant', 'tool')}
    prefixes = tuple(tuple(tokenizer(prefix)['input_ids']) for prefix in ('\n<tool_response>', '\n\n<tool_response>'))
    labels = chat_roles(ids, start_id=tokenizer.convert_tokens_to_ids('<|im_start|>'), role_ids=role_ids,
        think_open=tokenizer.convert_tokens_to_ids('<think>'), think_close=tokenizer.convert_tokens_to_ids('</think>'),
        tool_response_prefixes=prefixes)
    assistant = {ROLE_CODES.index('assistant_reasoning'), ROLE_CODES.index('assistant_reply')}
    spans = []
    pos = 0
    while pos < len(labels):
        if labels[pos] not in assistant:
            pos += 1
            continue
        begin = pos
        while pos < len(labels) and labels[pos] in assistant:
            pos += 1
        spans.append((begin, pos))
    expected = segmented_position_weights(torch.ones((1, len(ids))), spans,
        roles=torch.tensor([list(labels)]), context_weight=.05, feedback_weight=.25)[0].tolist()
    expected[0] = 0.
    assert list(example.plan.invocations[0].token_ids) == ids
    assert list(example.target_weights) == pytest.approx(expected)
    assert graph['conversion']['supervision_policy']['name'] == 'whole-trajectory-context-feedback-failed-turns-v2'
    tool_positions = [i for i, role in enumerate(labels) if role == ROLE_CODES.index('tool')]
    assistant_positions = [i for i, role in enumerate(labels) if role == ROLE_CODES.index('assistant_reply')]
    assert tool_positions and assistant_positions
    assert all(expected[i] < expected[assistant_positions[-1]] for i in tool_positions)
    assert expected[0] == 0.
    source.pop('supervision')
    with pytest.raises(ValueError, match='declared supervision'):
        whole_trajectory_text_row_graph(source, source_tokenizer_fingerprint='a' * 64,
            tokenizer=tokenizer, expanded_tokenizer_fingerprint='b' * 64)


def test_whole_trajectory_masks_failed_assistant_action_and_downweights_runtime_opening():
    import pytest
    from natlang_neuralese.data.native_graph import parse_graph
    from natlang_neuralese.train.text_supervision import ROLE_CODES, chat_roles, segmented_position_weights
    import torch

    tokenizer = ChatRoleAsciiTokenizer()
    text = ('<|im_start|>assistant<|im_sep|>runtime input<|im_end|>'
            '<|im_start|>assistant<|im_sep|>rejected call<|im_end|>'
            '<|im_start|>user\n<tool_response>rejected action: bad argument</tool_response><|im_end|>'
            '<|im_start|>assistant<|im_sep|>corrected answer<|im_end|>')
    ids = tokenizer(text)['input_ids']
    source = {'id': 'whole-trajectory:failure', 'text': text, 'token_ids': ids,
        'tokenizer_sha256': 'a' * 64, 'split': 'train', 'source_groups': ['group:failure'],
        'supervision': 'whole-trajectory', 'source_record_ids': ['record:failure'],
        'runtime_opening_assistant_turns': 1}
    graph = whole_trajectory_text_row_graph(source, source_tokenizer_fingerprint='a' * 64,
        tokenizer=tokenizer, expanded_tokenizer_fingerprint='b' * 64)
    example = parse_graph(graph, tokenizer_fingerprint='b' * 64)
    weights = list(example.target_weights)
    labels = chat_roles(ids, start_id=tokenizer.convert_tokens_to_ids('<|im_start|>'),
        role_ids={tokenizer.convert_tokens_to_ids(name): name for name in ('system', 'user', 'assistant', 'tool')},
        think_open=tokenizer.convert_tokens_to_ids('<think>'),
        think_close=tokenizer.convert_tokens_to_ids('</think>'),
        tool_response_prefixes=tuple(tuple(tokenizer(prefix)['input_ids'])
                                     for prefix in ('\n<tool_response>', '\n\n<tool_response>')))
    assistant = {ROLE_CODES.index('assistant_reasoning'), ROLE_CODES.index('assistant_reply')}
    assistant_runs = []
    pos = 0
    while pos < len(labels):
        if labels[pos] not in assistant:
            pos += 1
            continue
        begin = pos
        while pos < len(labels) and labels[pos] in assistant:
            pos += 1
        assistant_runs.append((begin, pos))
    assert len(assistant_runs) == 3
    opening, failed, successful = assistant_runs
    context_labels = list(labels)
    context_labels[opening[0]:opening[1]] = [ROLE_CODES.index('other')] * (opening[1] - opening[0])
    expected = segmented_position_weights(torch.ones((1, len(ids))), [successful],
        roles=torch.tensor([context_labels]), context_weight=.05, feedback_weight=.25,
        exclude=torch.tensor([[0 if not (failed[0] <= i < failed[1]) else 1 for i in range(len(ids))]],
                             dtype=torch.bool))[0].tolist()
    expected[0] = 0.
    assert weights == pytest.approx(expected)
    assert all(weights[i] < weights[successful[0]] for i in range(*opening))
    assert all(weights[i] == 0 for i in range(*failed))
    context_positions = [i for i, role in enumerate(labels)
                         if role == ROLE_CODES.index('other') and i != 0]
    assert context_positions and all(weights[successful[0]] > weights[i] for i in context_positions)
    tool_positions = [i for i, role in enumerate(labels) if role == ROLE_CODES.index('tool')]
    assert tool_positions and all(weights[i] < weights[successful[0]] for i in tool_positions)


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
    # The consumer receives only imported state between role markers; the
    # plaintext body remains source provenance, never an own-token bypass.
    assert len(consumer.token_ids) == len(target['token_ids']) - len(payload) + 2
    assert consumer.token_ids[consumer.imports[0].at] == 1004
    assert graph['source_pairing']['consumed_occurrences'][0]['payload_sha256'] == hashlib.sha256(payload.encode()).hexdigest()


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
    assert occurrence['reader_json_pointers'] == ['/messages/0/content/0']
    duplicate_reader = {'messages': [{'type': 'read', 'name': 'r', 'source': payload},
                                     {'type': 'read', 'name': 'r', 'source': payload}]}
    repeated = module.source_occurrences(writer, duplicate_reader, 'r', payload,
        module.sha256_bytes(payload.encode()))
    assert repeated['reader_json_pointers'] == ['/messages/0', '/messages/1']
    assert repeated['transform'] == 'static-shared-producer-multi-read'


def test_witnessed_pair_imports_every_exact_static_read_occurrence():
    import hashlib
    tok = RoleAsciiTokenizer()
    payload = 'typed value'
    writer_text = 'prompt' + tok.markers[0] + payload + tok.markers[1]
    reader_text = ('read:' + tok.markers[2] + payload + tok.markers[3] +
                   ',then:' + tok.markers[2] + payload + tok.markers[3] + ':answer')
    writer = row('writer', writer_text, boundary=6)
    reader = row('reader', reader_text, boundary=1)
    writer['token_ids'] = tok(writer_text)['input_ids']
    reader['token_ids'] = tok(reader_text)['input_ids']
    reader['supervised_suffix_start'] = len(reader['token_ids']) - 6
    writer['tokenizer_sha256'] = reader['tokenizer_sha256'] = 'a' * 64
    writer_open = writer['token_ids'].index(tok.marker_ids[0])
    reader_opens = [i for i, token in enumerate(reader['token_ids']) if token == tok.marker_ids[2]]
    reader_closes = [i for i, token in enumerate(reader['token_ids']) if token == tok.marker_ids[3]]
    writer_span = {'container_start': writer_open,
        'container_stop': writer['token_ids'].index(tok.marker_ids[1]) + 1,
        'body_text': payload, 'payload_token_ids': list(payload.encode()), 'tokenize_payload_alone': True}
    reader_spans = [{'container_start': start, 'container_stop': close + 1,
        'body_text': payload, 'payload_token_ids': list(payload.encode()), 'tokenize_payload_alone': True}
        for start, close in zip(reader_opens, reader_closes)]
    graph = witnessed_pair_graph(writer, reader, source_tokenizer=tok,
        source_tokenizer_fingerprint='a' * 64, expanded_tokenizer_fingerprint='a' * 64,
        role_ids=tok.marker_ids, writer_span=writer_span, reader_span=reader_spans,
        payload_sha256=hashlib.sha256(payload.encode()).hexdigest(), source_span_id='write:1',
        edge_id='edge:multi', chain_id='chain:multi')
    example = parse_graph(graph, tokenizer_fingerprint='a' * 64)
    producer, consumer = example.plan.invocations
    assert len(consumer.imports) == 2
    assert [edge.at for edge in consumer.imports] == [reader_opens[0] + 1,
        reader_opens[1] + 1 - len(payload)]
    assert all(edge.source == 'producer' for edge in consumer.imports)
    for edge in consumer.imports:
        assert example.target_weights[example.plan.offsets[1] + edge.at] == 0.
        assert consumer.token_ids[edge.at] == tok.marker_ids[3]
    expected_reader = reader_text.replace(tok.markers[2] + payload + tok.markers[3],
        tok.markers[2] + tok.markers[3])
    assert tok.decode(consumer.token_ids) == expected_reader


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
    assert consumer.token_ids == (20, 1003, 1004, 21)
    assert consumer.imports[0].at == 2
    assert parsed.target_weights[2:4] == (1., 1.)
    assert parsed.target_weights[parsed.plan.offsets[1] + 2] == 0.


def test_empty_authenticated_payload_keeps_structural_native_import_without_fake_tokens():
    import hashlib
    tok = RoleAsciiTokenizer()
    writer_text = 'q' + tok.markers[0] + tok.markers[1] + 'z'
    reader_text = 'read' + tok.markers[2] + tok.markers[3] + ':answer'
    writer, reader = row('empty-writer', writer_text, boundary=1), row('empty-reader', reader_text,
        boundary=len(tok(reader_text)['input_ids']) - len(':answer'))
    writer['token_ids'] = tok(writer_text)['input_ids']
    reader['token_ids'] = tok(reader_text)['input_ids']
    writer['tokenizer_sha256'] = reader['tokenizer_sha256'] = 'a' * 64
    writer_start = writer['token_ids'].index(tok.marker_ids[0])
    reader_start = reader['token_ids'].index(tok.marker_ids[2])
    empty_span = {'container_start': writer_start, 'container_stop': writer_start + 2,
                  'body_text': '', 'payload_token_ids': [], 'tokenize_payload_alone': True}
    read_span = {'container_start': reader_start, 'container_stop': reader_start + 2,
                 'body_text': '', 'payload_token_ids': [], 'tokenize_payload_alone': True}
    graph = witnessed_pair_graph(writer, reader, source_tokenizer=tok,
        source_tokenizer_fingerprint='a' * 64, expanded_tokenizer_fingerprint='a' * 64,
        role_ids=tok.marker_ids, writer_span=empty_span, reader_span=read_span,
        payload_sha256=hashlib.sha256(b'').hexdigest(), source_span_id='empty-write',
        edge_id='empty-edge', chain_id='empty-chain')
    parsed = parse_graph(graph, tokenizer_fingerprint='a' * 64)
    producer, consumer = parsed.plan.invocations
    assert producer.producing == ((writer_start + 1, writer_start + 1),)
    assert len(consumer.imports) == 1 and consumer.imports[0].start == consumer.imports[0].stop
    assert consumer.token_ids[consumer.imports[0].at] == tok.marker_ids[3]
    assert parsed.target_weights[parsed.plan.offsets[1] + consumer.imports[0].at] == 0.
    assert tok.decode(consumer.token_ids) == 'read' + tok.markers[2] + tok.markers[3] + ':answer'


def test_pair_resume_expansion_processes_uncovered_edges_in_a_completed_chain():
    import hashlib
    import importlib.util
    import json
    from pathlib import Path

    script = Path(__file__).parents[2] / 'scripts/neuralese_data/build_native_semantic_pairs_e2e.py'
    spec = importlib.util.spec_from_file_location('e2e_pair_builder_resume', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    edges = [{'trajectory': 'same-chain', 'id': i, 'split': 'train'} for i in range(3)]
    def identity(edge):
        return hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    # A prior selected-edge pass covered one edge. Expanding to all source
    # edges must revisit the same chain for both uncovered edge identities.
    pending = module._remaining_edges_by_chain({'same-chain': edges}, {identity(edges[0])})
    assert pending == {'same-chain': edges[1:]}
    assert {identity(edge) for group in pending.values() for edge in group} == {
        identity(edges[1]), identity(edges[2])}
    # Rebuild selection must hash each edge (including the final item) and
    # preserve both source splits when completing the full edge set.
    mixed = [{'trajectory': 'chain-a', 'id': 0, 'split': 'train'},
             {'trajectory': 'chain-a', 'id': 1, 'split': 'test'},
             {'trajectory': 'chain-b', 'id': 2, 'split': 'train'}]
    all_ids = module._selected_edge_ids({'chain-a': mixed[:2], 'chain-b': mixed[2:]})
    train_ids = module._selected_edge_ids({'chain-a': mixed[:2], 'chain-b': mixed[2:]}, 'train')
    assert all_ids == {identity(edge) for edge in mixed}
    assert train_ids == {identity(mixed[0]), identity(mixed[2])}
    previous = {'inputs': {'edges': {'sha256': 'b' * 64}, 'rebuild_all_pairs': {
        'mode': 'all-admitted-edges', 'edges_sha256': 'b' * 64}},
        'requested_edge_ids': [identity(mixed[0])]}
    expanded = {'edges': {'sha256': 'b' * 64}, 'rebuild_all_pairs': {
        'mode': 'all-admitted-edges', 'edges_sha256': 'b' * 64,
        'selected_edge_ids': {'sha256': 'c' * 64}}}
    assert module._pair_resume_base(previous['inputs']) == module._pair_resume_base(expanded)
    assert module._previous_requested_edges(previous, {identity(edge): edge for edge in mixed}) == {
        identity(mixed[0])}
    assert set(previous['requested_edge_ids']) < all_ids
    assert module._remaining_edges_by_chain({'chain-a': mixed[:2], 'chain-b': mixed[2:]},
        set(previous['requested_edge_ids'])) == {'chain-a': [mixed[1]], 'chain-b': [mixed[2]]}


def test_pair_worker_commit_removes_only_reported_edges_from_chain_frontier():
    import hashlib
    import importlib.util
    import json
    from pathlib import Path

    script = Path(__file__).parents[2] / 'scripts/neuralese_data/build_native_semantic_pairs_e2e.py'
    spec = importlib.util.spec_from_file_location('e2e_pair_builder_frontier', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    edges = [{'trajectory': 'same-chain', 'id': i, 'split': 'train' if i < 2 else 'test'}
             for i in range(4)]
    identity = lambda edge: hashlib.sha256(json.dumps(
        edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    frontier = {'same-chain': list(edges)}
    module._remove_completed_edges(frontier, 'same-chain', {identity(edges[1]), identity(edges[3])})
    assert frontier == {'same-chain': [edges[0], edges[2]]}
    module._remove_completed_edges(frontier, 'same-chain', {identity(edges[0]), identity(edges[2])})
    assert frontier == {}


def test_e2e_pair_candidates_include_only_exact_typed_block_identity():
    import importlib.util
    from pathlib import Path
    import pytest
    script=Path(__file__).parents[2]/'scripts/neuralese_data/build_native_semantic_pairs_e2e.py'
    spec=importlib.util.spec_from_file_location('typed_pair_occurrences',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    body='one exact writer body';block='nz1_'+'a'*52;name='soft-state:'+block
    writer={'target':{'$write':{'name':name,'source':body}}}
    reader={'messages':[{'role':'user','content':[{'type':'neuralese','id':block}]},
        {'role':'assistant','tool_calls':[{'function':{'arguments':[{'type':'neuralese','id':block}]}}]}]}
    occurrence=module.source_occurrences(writer,reader,name,body,module.sha256_bytes(body.encode()))
    assert occurrence['reader_json_pointers']==['/messages/0/content/0','/messages/1/tool_calls/0/function/arguments/0']
    reader['messages'][0]['content'][0]['id']='nz1_'+'b'*52
    reader['messages'][1]['tool_calls'][0]['function']['arguments'][0]['id']='nz1_'+'b'*52
    with pytest.raises(ValueError,match='reads=0'):
        module.source_occurrences(writer,reader,name,body,module.sha256_bytes(body.encode()))
