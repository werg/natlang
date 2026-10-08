import json
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_recurrence_audit():
    spec = importlib.util.spec_from_file_location("audit_neuralese_recurrence",
        ROOT / "scripts/audit_neuralese_recurrence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reports_both_dependency_directions(tmp_path):
    def row(name, reads=(), writes=()):
        return {'id': name, 'split': 'train', 'source_groups': ['one-world'],
                'messages': [{'type': 'read', 'name': value} for value in reads],
                'target': [{'$write': {'name': value, 'source': value}} for value in writes]}
    rows = [row('a', writes=['a']), row('b', writes=['b']),
            row('c', reads=['a', 'b']), row('d', reads=['a']), row('e', reads=['a'])]
    source, out = tmp_path/'rows.jsonl', tmp_path/'audit.json'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                    str(source), '--out', str(out)], check=True, capture_output=True)
    audit = json.loads(out.read_text())
    assert audit['schema'] == 'natlang.recurrence-audit/4'
    assert audit['max_producers_per_consumer'] == 2
    assert audit['max_consumers_per_producer'] == 3
    assert audit['linked_edges'] == 4
    assert audit['structurally_closed']
    assert 'max_branching' not in audit


def test_authenticated_context_read_is_external_root_not_synthetic_edge(tmp_path):
    import hashlib
    body = 'observed runtime input'
    body_sha = hashlib.sha256(body.encode()).hexdigest()
    hashes = {name: hashlib.sha256(name.encode()).hexdigest()
              for name in ('row', 'trace', 'transport', 'raw', 'rendered')}
    receipt = {
        'schema': 'natlang.provider-expanded-read-context/2',
        'origin': 'same-run-producer', 'writer_target_selected': False,
        'writer_source_class': 'modern-typed-text-result',
        'invocation_id': 'reader/1', 'parent_invocation_id': 'parent/1',
        'source_row_sha256': hashes['row'], 'trace_sha256': hashes['trace'],
        'transport_provenance_sha256': hashes['transport'], 'raw_request_sha256': hashes['raw'],
        'rendered_request_sha256': hashes['rendered'],
        'block': {'id': 'block-1', 'type': 'Neuralese<string>', 'body': body,
                  'body_sha256': body_sha},
        'producer_write': {'kind': 'block_write', 'producer': 'text-marker-emulation',
                           'source_kind': 'typed-text-result', 'block': 'block-1',
                           'result_type': 'Neuralese<string>', 'text_body_sha256': body_sha,
                           'call_id': 'writer/1', 'node': 'writer/1#4'},
        'block_read': {'kind': 'block_read', 'block': 'block-1', 'call_id': 'reader/1',
                       'turn': 'reader/1#turn1', 'node': 'reader/1#2',
                       'inputs': [{'node': 'writer/1#4', 'block': 'block-1'}]},
        'model_turn': {'kind': 'model_turn', 'call_id': 'reader/1', 'node': 'reader/1#turn1',
                       'inputs': [{'node': 'reader/1#2', 'block': 'block-1'}]},
    }
    context = {
        'schema': 'natlang.external-context-input/1', 'origin': 'same-run-producer',
        'block_id': 'block-1', 'type': 'Neuralese<string>', 'body_sha256': body_sha,
        'invocation_id': 'reader/1', 'parent_invocation_id': 'parent/1',
        'source_row_sha256': hashes['row'], 'trace_sha256': hashes['trace'],
        'transport_provenance_sha256': hashes['transport'], 'raw_request_sha256': hashes['raw'],
        'rendered_request_sha256': hashes['rendered'], 'producer_call_id': 'writer/1',
        'producer_write_node': 'writer/1#4', 'read_node': 'reader/1#2',
        'model_turn_node': 'reader/1#turn1', 'writer_target_selected': False,
        'writer_source_class': 'modern-typed-text-result',
        'learner_representation': 'typed-read-from-authenticated-runtime-writer-event-context-only',
    }
    row = {'id': 'reader', 'split': 'train', 'source_groups': ['one-world'],
           'source_ref': {'provider_expanded_read_contexts': [receipt]},
           'neuralese_conversion': {'external_context_inputs': [context]},
           'messages': [{'type': 'read', 'name': 'soft-state:block-1'}], 'target': []}
    source, out = tmp_path/'rows.jsonl', tmp_path/'audit.json'
    source.write_text(json.dumps(row) + '\n')
    subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                    str(source), '--out', str(out)], check=True, capture_output=True)
    audit = json.loads(out.read_text())
    assert audit['authenticated_external_context_roots'] == [
        {'reader': 'reader', 'name': 'soft-state:block-1'}]
    assert audit['failures']['missing_producers'] == []
    assert audit['structurally_closed']

    for source_kind, writer_class, witness in [
        ('return_result', 'legacy-text-marker-standin-return-result', {
            'kind': 'raw-return-result-value-equals-expanded-body', 'source': 'return_result',
            'host_result_call_id': 'writer/1', 'host_result_type': 'Neuralese<string>',
            'host_result_value_sha256': '6' * 64, 'raw_response_sha256': '7' * 64}),
        ('eval-finish', 'legacy-text-marker-standin-eval-finish', {
            'kind': 'completed-eval-finish-host-reference', 'source': 'eval-finish',
            'host_result_call_id': 'writer/1', 'host_result_type': 'Neuralese<string>',
            'host_result_value_sha256': '8' * 64}),
    ]:
        legacy_result = json.loads(json.dumps(row))
        legacy_receipt = legacy_result['source_ref']['provider_expanded_read_contexts'][0]
        legacy_receipt.update({'writer_source_class': writer_class, 'writer_witness': witness})
        legacy_receipt['producer_write'].update({'source': source_kind, 'marker_context': 'return-result'})
        legacy_result['neuralese_conversion']['external_context_inputs'][0].update(
            {'writer_source_class': writer_class, 'writer_witness': witness})
        source.write_text(json.dumps(legacy_result) + '\n')
        subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                        str(source), '--out', str(out)], check=True, capture_output=True)
        result_audit = json.loads(out.read_text())
        assert result_audit['authenticated_external_context_roots'] == [
            {'reader': 'reader', 'name': 'soft-state:block-1'}]
        bad = json.loads(json.dumps(legacy_result))
        bad['source_ref']['provider_expanded_read_contexts'][0]['writer_witness']['host_result_call_id'] = 'other'
        source.write_text(json.dumps(bad) + '\n')
        subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                        str(source), '--out', str(out)], check=True, capture_output=True)
        bad_audit = json.loads(out.read_text())
        assert bad_audit['authenticated_external_context_roots'] == []
        assert bad_audit['failures']['missing_producers']

    legacy = json.loads(json.dumps(row))
    legacy_receipt = legacy['source_ref']['provider_expanded_read_contexts'][0]
    legacy_receipt['writer_source_class'] = 'legacy-text-marker-standin-eval-code'
    legacy_receipt['producer_write'].pop('producer')
    legacy_receipt['producer_write'].pop('source_kind')
    legacy_receipt['producer_write'].update({'emulation_version': 'text-marker-standin/2',
                                              'marker_context': 'eval-code', 'learned_vectors': False})
    legacy['neuralese_conversion']['external_context_inputs'][0]['writer_source_class'] = \
        'legacy-text-marker-standin-eval-code'
    source.write_text(json.dumps(legacy) + '\n')
    subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                    str(source), '--out', str(out)], check=True, capture_output=True)
    audit = json.loads(out.read_text())
    assert audit['authenticated_external_context_roots'] == [
        {'reader': 'reader', 'name': 'soft-state:block-1'}]
    assert audit['failures']['missing_producers'] == []
    assert audit['structurally_closed']

    writer = {'id': 'selected-writer', 'split': 'train', 'source_groups': ['one-world'],
              'messages': [],
              'target': [{'$write': {'name': 'soft-state:block-1', 'source': body}}]}
    source.write_text(json.dumps(row) + '\n' + json.dumps(writer) + '\n')
    subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                    str(source), '--out', str(out)], check=True, capture_output=True)
    audit = json.loads(out.read_text())
    assert audit['authenticated_external_context_roots'] == []
    assert audit['failures']['missing_producers'] == []
    assert audit['structurally_closed']
    assert audit['linked_edges'] == 1

    context['body_sha256'] = '0' * 64
    source.write_text(json.dumps(row) + '\n')
    subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                    str(source), '--out', str(out)], check=True, capture_output=True)
    audit = json.loads(out.read_text())
    assert audit['authenticated_external_context_roots'] == []
    assert audit['failures']['missing_producers'] == [
        {'reader': 'reader', 'name': 'soft-state:block-1'}]
    assert not audit['structurally_closed']
    context['body_sha256'] = body_sha
    receipt['producer_write'].pop('call_id')
    context.pop('producer_call_id')
    source.write_text(json.dumps(row) + '\n')
    subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                    str(source), '--out', str(out)], check=True, capture_output=True)
    audit = json.loads(out.read_text())
    assert audit['authenticated_external_context_roots'] == []
    assert audit['failures']['missing_producers'] == [
        {'reader': 'reader', 'name': 'soft-state:block-1'}]


def test_old_modern_context_receipts_and_mixed_source_classes_are_bound_per_block():
    audit = _load_recurrence_audit()

    def make(block_id, body, invocation, writer, reader, *, legacy=False,
             legacy_finish=False, include_class=True):
        digest = hashlib.sha256(body.encode()).hexdigest()
        hashes = {key: hashlib.sha256((key + block_id).encode()).hexdigest()
                  for key in ('source', 'trace', 'transport', 'raw', 'rendered')}
        producer = {'kind': 'block_write', 'call_id': writer, 'node': writer + '#4',
                    'block': block_id, 'truncated': False, 'result_type': 'Neuralese<string>',
                    'text_body_sha256': digest}
        source_class = ('legacy-text-marker-standin-eval-finish' if legacy_finish else
                        'legacy-text-marker-standin-eval-code' if legacy else None)
        witness = None
        if legacy:
            producer.update({'emulation_version': 'text-marker-standin/2',
                             'marker_context': 'eval-code', 'learned_vectors': False})
        elif legacy_finish:
            source_class = 'legacy-text-marker-standin-eval-finish'
            witness = {'kind': 'completed-eval-finish-host-reference', 'source': 'eval-finish',
                       'host_result_call_id': writer, 'host_result_type': 'Neuralese<string>',
                       'host_result_value_sha256': hashlib.sha256((block_id + ' value').encode()).hexdigest()}
            producer.update({'producer': 'text-marker-emulation', 'source_kind': 'typed-text-result',
                             'source': 'eval-finish', 'marker_context': 'return-result'})
        else:
            producer.update({'producer': 'text-marker-emulation', 'source_kind': 'typed-text-result'})
            if include_class:
                source_class = 'modern-typed-text-result'
        read_node, turn_node = reader + '#2', reader + '#turn1'
        receipt = {'schema': 'natlang.provider-expanded-read-context/2',
            'origin': 'same-run-producer', 'writer_target_selected': False,
            'invocation_id': reader, 'parent_invocation_id': 'root/1',
            'source_row_sha256': hashes['source'], 'trace_sha256': hashes['trace'],
            'transport_provenance_sha256': hashes['transport'], 'raw_request_sha256': hashes['raw'],
            'rendered_request_sha256': hashes['rendered'],
            'block': {'id': block_id, 'type': 'Neuralese<string>', 'body': body,
                      'body_sha256': digest}, 'producer_write': producer,
            'block_read': {'kind': 'block_read', 'call_id': reader, 'node': read_node,
                'turn': turn_node, 'block': block_id,
                'inputs': [{'node': writer + '#4', 'block': block_id}]},
            'model_turn': {'kind': 'model_turn', 'call_id': reader, 'node': turn_node,
                'inputs': [{'node': read_node, 'block': block_id}]}}
        if source_class is not None:
            receipt['writer_source_class'] = source_class
        if witness is not None:
            receipt['writer_witness'] = witness
        metadata = {'schema': 'natlang.external-context-input/1', 'origin': 'same-run-producer',
            'block_id': block_id, 'type': 'Neuralese<string>', 'body_sha256': digest,
            'invocation_id': reader, 'parent_invocation_id': 'root/1',
            'source_row_sha256': hashes['source'], 'trace_sha256': hashes['trace'],
            'transport_provenance_sha256': hashes['transport'], 'raw_request_sha256': hashes['raw'],
            'rendered_request_sha256': hashes['rendered'], 'producer_call_id': writer,
            'producer_write_node': writer + '#4', 'read_node': read_node, 'model_turn_node': turn_node,
            'writer_target_selected': False,
            'learner_representation': 'typed-read-from-authenticated-runtime-writer-event-context-only'}
        if source_class is not None:
            metadata['writer_source_class'] = source_class
        if witness is not None:
            metadata['writer_witness'] = witness
        return receipt, metadata

    legacy = make('block-legacy', 'legacy body', 'reader/legacy', 'writer/legacy', 'reader/legacy',
                  legacy_finish=True)
    modern_old_v2 = make('block-modern-v2', 'modern body', 'reader/modern', 'writer/modern',
                         'reader/modern', include_class=False)
    modern = make('block-modern', 'modern newer body', 'reader/new-modern', 'writer/new-modern',
                  'reader/new-modern', include_class=True)
    # Put the legacy receipt last too: metadata validation must bind class and
    # witness to each matched receipt, not whichever loop item was visited last.
    row = {'id': 'mixed', 'source_ref': {'provider_expanded_read_contexts':
               [modern_old_v2[0], modern[0], legacy[0]]},
           'neuralese_conversion': {'external_context_inputs': [modern_old_v2[1], modern[1], legacy[1]]}}
    assert audit.authenticated_external_context_names(row) == {
        'soft-state:block-legacy', 'soft-state:block-modern-v2', 'soft-state:block-modern'}

    forged = json.loads(json.dumps(row))
    forged['source_ref']['provider_expanded_read_contexts'][1]['producer_write']['source_kind'] = 'untyped'
    assert audit.authenticated_external_context_names(forged) == {
        'soft-state:block-legacy', 'soft-state:block-modern-v2'}
    forged = json.loads(json.dumps(row))
    forged['source_ref']['provider_expanded_read_contexts'][2]['writer_witness']['host_result_call_id'] = 'other'
    assert audit.authenticated_external_context_names(forged) == {
        'soft-state:block-modern-v2', 'soft-state:block-modern'}


def test_selected_direct_typed_return_links_only_to_exact_authenticated_reader_event(tmp_path):
    audit = _load_recurrence_audit()
    body = 'A directly authored semantic note.'
    body_sha = hashlib.sha256(body.encode()).hexdigest()
    source_sha = hashlib.sha256(b'writer-source').hexdigest()
    block = 'nz1_' + 'a' * 52
    event_name = audit.direct_typed_result_event_name(block, 'run', 'writer-call', 'writer-call#9')
    args = {'status': 'success', 'value': body}
    typed = {'schema': 'natlang.typed-result-write/1', 'trajectory_id': 'run',
        'source_row_sha256': source_sha, 'invocation_id': 'writer-call', 'writer_call_id': 'writer-call',
        'writer_node': 'writer-call#9', 'block_id': block, 'source_kind': 'typed-text-result',
        'source': 'return_result', 'result_type': 'Neuralese<string>', 'result_path': ['return'],
        'model_turn_node': 'writer-call#3',
        'action_seq': 17, 'body_sha256': body_sha, 'body_source': body,
        'body_source_basis': 'exact-raw-model-result-string'}
    write = {'name': event_name, 'block_id': block, 'type': 'Neuralese<string>', 'source': body}
    writer = {'id': 'writer-row', 'split': 'train', 'source_groups': ['world'],
        'source_ref': {'trajectory_id': 'run', 'invocation_id': 'writer-call', 'source_row_sha256': source_sha},
        'decision': {'assistant': {'calls': [{'source_tool': 'return_result', 'arguments': args,
            'outcome': {'name': 'return_result', 'arguments': args, 'trace_seq': 17,
                        'typed_result_writes': [typed]}}]}},
        'outcome': {'execution_graph': [
            {'kind': 'model_turn', 'call_id': 'writer-call', 'node': 'writer-call#3', 'seq': 3},
            {'kind': 'block_write', 'call_id': 'writer-call', 'node': 'writer-call#9', 'seq': 9,
             'block': block, 'source_kind': 'typed-text-result', 'source': 'return_result',
             'marker_context': 'return-result', 'result_type': 'Neuralese<string>',
             'text_body_sha256': body_sha, 'truncated': False,
             'inputs': [{'node': 'writer-call#3', 'port': 'result-source'}]}]},
        'target': {'tool_calls': [{'id': 'return-id', 'function': {'name': 'return_result',
            'arguments': json.dumps({'status': 'success', 'value': {'$write': write}})}}]},
        'neuralese_conversion': {'selected_runtime_result_writes': [{
            'schema': 'natlang.selected-runtime-result-write/1',
            'role': 'selected-direct-typed-text-semantic-writer', 'trajectory_id': 'run',
            'source_row_sha256': source_sha, 'invocation_id': 'writer-call', 'writer_call_id': 'writer-call',
                'writer_node': 'writer-call#9', 'block_id': block, 'result_type': 'Neuralese<string>',
                'body_sha256': body_sha, 'result_path': ['return'], 'action_seq': 17,
                'typed_result_receipt_sha256': hashlib.sha256(audit.stable_json(typed).encode()).hexdigest(),
            'action_target_call_id': 'return-id',
            'action_arguments_sha256': hashlib.sha256(audit.stable_json(args).encode()).hexdigest(),
                'target_write_name': event_name,
            'target_write_sha256': hashlib.sha256(audit.stable_json({'$write': write}).encode()).hexdigest(),
            'body_source_basis': 'exact-raw-model-result-string'}]}}

    hashes = {key: hashlib.sha256(key.encode()).hexdigest()
              for key in ('reader-source', 'trace', 'transport', 'raw', 'rendered', 'source-request', 'source-response')}
    reader_call, read_node, turn_node = 'reader-call', 'reader-call#4', 'reader-call#turn2'
    provider_receipt = {'schema': 'natlang.provider-expanded-read-context/2',
        'origin': 'same-run-producer', 'writer_target_selected': False,
        'writer_source_class': 'modern-typed-text-result', 'invocation_id': reader_call,
        'parent_invocation_id': 'root-call', 'source_row_sha256': hashes['reader-source'],
        'trace_sha256': hashes['trace'], 'transport_provenance_sha256': hashes['transport'],
        'raw_request_sha256': hashes['raw'], 'rendered_request_sha256': hashes['rendered'],
        'source_request_sha256': hashes['source-request'], 'source_response_sha256': hashes['source-response'],
        'block': {'id': block, 'type': 'Neuralese<string>', 'body': body, 'body_sha256': body_sha},
        'producer_write': {'kind': 'block_write', 'producer': 'text-marker-emulation',
            'source_kind': 'typed-text-result', 'source': 'return_result', 'block': block, 'result_type': 'Neuralese<string>',
            'text_body_sha256': body_sha, 'call_id': 'writer-call', 'node': 'writer-call#9',
            'truncated': False},
        'block_read': {'kind': 'block_read', 'block': block, 'call_id': reader_call,
            'turn': turn_node, 'node': read_node,
            'seq': 1,
            'inputs': [{'node': 'writer-call#9', 'block': block}]},
        'model_turn': {'kind': 'model_turn', 'call_id': reader_call, 'node': turn_node, 'seq': 2,
            'inputs': [{'node': read_node, 'block': block}]}}
    context = {'schema': 'natlang.external-context-input/1', 'origin': 'same-run-producer',
        'writer_target_selected': False, 'block_id': block, 'type': 'Neuralese<string>',
        'target_write_name': event_name,
        'body_sha256': body_sha, 'invocation_id': reader_call, 'parent_invocation_id': 'root-call',
        'source_row_sha256': hashes['reader-source'], 'trace_sha256': hashes['trace'],
        'transport_provenance_sha256': hashes['transport'], 'raw_request_sha256': hashes['raw'],
        'rendered_request_sha256': hashes['rendered'], 'producer_call_id': 'writer-call',
        'producer_write_node': 'writer-call#9', 'read_node': read_node, 'model_turn_node': turn_node,
        'writer_source_class': 'modern-typed-text-result',
        'learner_representation': 'typed-read-from-authenticated-runtime-writer-event-context-only'}
    reader = {'id': 'reader-row', 'split': 'train', 'source_groups': ['world'],
        'source_ref': {'trajectory_id': 'run', 'invocation_id': reader_call,
            'source_row_sha256': hashes['reader-source'], 'provider_expanded_read_contexts': [provider_receipt]},
        'neuralese_conversion': {'external_context_inputs': [context]},
        'messages': [{'type': 'read', 'name': event_name}], 'target': []}
    assert audit.validate_selected_direct_return_receipts(writer)[0]
    assert audit.semantic_writer_matches_reader(
        writer['neuralese_conversion']['selected_runtime_result_writes'][0], reader)
    assert audit.authenticated_external_context_names(reader) == {event_name}

    def run(writer_row, reader_row):
        source, output = tmp_path/'candidate.jsonl', tmp_path/'audit.json'
        source.write_text(json.dumps(writer_row) + '\n' + json.dumps(reader_row) + '\n')
        subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                        str(source), '--out', str(output)], check=True, capture_output=True)
        return json.loads(output.read_text())

    good = run(writer, reader)
    assert good['linked_edges'] == 1
    assert good['selected_direct_typed_return_writers'] == 1
    assert good['structurally_closed'], 'direct semantic return and exact reader event form one structural edge'
    assert not good['authenticated_external_context_roots'], 'the selected producer takes precedence over its context-only receipt'
    # Numeric sequence IDs belong to their own invocation traces; writer 17 is
    # valid with reader-local read/turn sequence 1/2 because the node inputs bind them.
    assert not audit.semantic_writer_matches_reader(
        {**writer['neuralese_conversion']['selected_runtime_result_writes'][0], 'writer_node': 'writer-call#8'}, reader)

    # Native materialized rows do not retain the complete execution_graph. A
    # downstream provider receipt can still authenticate the selected write
    # when it carries the exact event and read/turn graph links for the same
    # source row. This preserves the context as a read, not a second writer.
    no_graph = json.loads(json.dumps(writer))
    no_graph.pop('outcome')
    bound_reader = json.loads(json.dumps(reader))
    bound_reader['source_ref']['source_row_sha256'] = source_sha
    bound_receipt = bound_reader['source_ref']['provider_expanded_read_contexts'][0]
    bound_receipt['source_row_sha256'] = source_sha
    bound_receipt['writer_source_class'] = 'legacy-text-marker-standin-return-result'
    bound_receipt['writer_witness'] = {'kind': 'raw-return-result-value-equals-expanded-body',
        'source': 'return_result', 'host_result_call_id': 'writer-call', 'host_result_type': 'Neuralese<string>',
        'host_result_value_sha256': hashlib.sha256(b'host-ref').hexdigest(),
        'raw_response_sha256': hashlib.sha256(b'raw-response').hexdigest()}
    producer = bound_receipt['producer_write']
    producer.update({'marker_context': 'return-result', 'seq': 9,
        'inputs': [{'node': 'writer-call#3', 'port': 'result-source'}]})
    bound_reader['neuralese_conversion']['external_context_inputs'][0]['source_row_sha256'] = source_sha
    bound_reader['neuralese_conversion']['external_context_inputs'][0]['writer_source_class'] = \
        'legacy-text-marker-standin-return-result'
    bound_reader['neuralese_conversion']['external_context_inputs'][0]['writer_witness'] = \
        bound_receipt['writer_witness']
    valid_receipts, receipt_failures = audit.validate_selected_direct_return_receipts(no_graph, [bound_receipt])
    assert valid_receipts, receipt_failures
    no_graph_audit = run(no_graph, bound_reader)
    assert no_graph_audit['linked_edges'] == 1 and no_graph_audit['structurally_closed']

    bad_writer = json.loads(json.dumps(writer))
    bad_writer['neuralese_conversion']['selected_runtime_result_writes'][0]['writer_node'] = 'writer-call#8'
    bad = run(bad_writer, reader)
    assert not bad['structurally_closed']
    assert (bad['failures']['selected_return_event_binding_mismatches']
            or bad['failures']['invalid_selected_runtime_result_writes'])
    bad_reader = json.loads(json.dumps(reader))
    bad_reader['source_ref']['provider_expanded_read_contexts'][0]['block']['body'] = 'different body'
    bad = run(writer, bad_reader)
    assert not bad['structurally_closed']
    assert bad['failures']['missing_producers'] or bad['failures']['selected_return_event_binding_mismatches']
    bad_reader = json.loads(json.dumps(reader))
    bad_reader['source_ref']['provider_expanded_read_contexts'][0]['block_read']['inputs'][0]['node'] = 'writer-call#8'
    bad = run(writer, bad_reader)
    assert not bad['structurally_closed']
    assert bad['failures']['selected_return_event_binding_mismatches']
    bad_writer = json.loads(json.dumps(writer))
    bad_writer['neuralese_conversion']['selected_runtime_result_writes'][0]['action_seq'] = 18
    bad = run(bad_writer, reader)
    assert not bad['structurally_closed']
    assert bad['failures']['invalid_selected_runtime_result_writes'], 'wrong local event sequence is rejected'
    bad_writer = json.loads(json.dumps(writer))
    bad_writer['outcome']['execution_graph'][1]['seq'] = 18
    bad = run(bad_writer, reader)
    assert not bad['structurally_closed']
    assert bad['failures']['invalid_selected_runtime_result_writes'], 'a write after return completion is rejected'
    bad_writer = json.loads(json.dumps(writer))
    bad_writer['outcome']['execution_graph'][1]['text_body_sha256'] = '0' * 64
    bad = run(bad_writer, reader)
    assert not bad['structurally_closed']
    assert bad['failures']['invalid_selected_runtime_result_writes'], 'graph body digest mismatch is rejected'
    bad_writer = json.loads(json.dumps(writer))
    bad_writer['source_ref']['trajectory_id'] = 'other-run'
    bad = run(bad_writer, reader)
    assert not bad['structurally_closed']
    assert bad['failures']['invalid_selected_runtime_result_writes'], 'source trajectory mismatch is rejected'
