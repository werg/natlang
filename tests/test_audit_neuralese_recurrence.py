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
    assert audit['schema'] == 'natlang.recurrence-audit/3'
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

    def make(block_id, body, invocation, writer, reader, *, legacy=False, include_class=True):
        digest = hashlib.sha256(body.encode()).hexdigest()
        hashes = {key: hashlib.sha256((key + block_id).encode()).hexdigest()
                  for key in ('source', 'trace', 'transport', 'raw', 'rendered')}
        producer = {'kind': 'block_write', 'call_id': writer, 'node': writer + '#4',
                    'block': block_id, 'truncated': False, 'result_type': 'Neuralese<string>',
                    'text_body_sha256': digest}
        source_class = 'legacy-text-marker-standin-eval-code' if legacy else None
        if legacy:
            producer.update({'emulation_version': 'text-marker-standin/2',
                             'marker_context': 'eval-code', 'learned_vectors': False})
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
        return receipt, metadata

    legacy = make('block-legacy', 'legacy body', 'reader/legacy', 'writer/legacy', 'reader/legacy', legacy=True)
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
