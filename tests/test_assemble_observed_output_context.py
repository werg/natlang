import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import assemble_neuralese_recurrence as assembler
import audit_observed_neuralese_outputs as observed_audit


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def base_row(ident, invocation, split, group):
    return {'id': ident, 'split': split, 'source_ids': ['source-world'],
        'source_groups': [group], 'messages': [], 'target': {'text': ident},
        'task': {'program_ir': {'source_groups': [group], 'license': 'project-generated',
            'gold_sources': ['constructed-world-oracle'],
            'curriculum': {'family': 'decision_skill_catalog'}}},
        'neuralese_conversion': {'version': 'natlang.neuralese-conversion/13'},
        'training_admission': {'approved': True},
        'outcome': {'accepted': True, 'oracle': {'level': 'exact', 'accepted': True}},
        'trace_admission': {'admitted': True},
        'provenance': {'trace_sha256': sha('trace')},
        'source_ref': {'trajectory_id': 'trajectory-1', 'source_row_sha256': sha('source-row'),
            'invocation_id': invocation}}


def make_inputs(tmp_path):
    body, block_id = 'observed answer', 'nz1_observed'
    body_hash, trajectory, source_hash = sha(body), 'trajectory-1', sha('source-row')
    writer_id, child_id, reader_id = 'eval/1', 'child/1', 'reader/1'
    writer_node, reader_node = writer_id + '#17', reader_id + '#2'
    handle = {'$neuralese': {'type': 'Neuralese<string>', 'id': block_id}}
    handle_sha = observed_audit.host_value_digest(handle)
    request_hash, response_hash = sha('request'), sha('response')
    transport_hash, raw_hash, rendered_hash = sha('transport'), sha('raw'), sha('rendered')
    arguments = {'code': 'const answer = await nl<string>`solve`(input); return answer;', 'finish': True}

    writer = base_row('writer-target', writer_id, 'train', 'group-a')
    writer['target'] = {'tool_calls': [{'name': 'eval', 'arguments': arguments}]}
    writer['source_ref']['host_result_capture'] = {'capture': {
        'call_id': writer_id, 'result_type': 'Neuralese<string>', 'terminal_action_seq': 18,
        'origin': 'observed-host-result; not a model-generated writer target',
        'value': handle, 'value_sha256': handle_sha}}
    writer['decision'] = {'assistant': {'calls': [{'source_tool': 'eval', 'call_id': 'tool-call-1',
        'arguments': arguments, 'outcome': {'status': 'completed', 'trace_seq': 18,
            'arguments': arguments, 'typed_result_writes': [{
                'schema': 'natlang.typed-result-write/1', 'trajectory_id': trajectory,
                'source_row_sha256': source_hash, 'invocation_id': writer_id,
                'writer_call_id': writer_id, 'writer_node': writer_node, 'block_id': block_id,
                'source_kind': 'typed-text-result', 'source': 'eval-finish',
                'result_type': 'Neuralese<string>', 'result_path': ['return'],
                'model_turn_node': writer_id + '#turn2', 'action_seq': 18,
                'body_sha256': body_hash, 'request_sha256': request_hash,
                'raw_response_sha256': response_hash,
                'body_source_basis': 'authenticated-final-host-output-reference'}]}}]}}

    child = base_row('child-target', child_id, 'train', 'group-a')
    child['source_ref']['host_result_capture'] = {'capture': {
        'call_id': child_id, 'result_type': 'string', 'terminal_action_seq': 4,
        'value': 'child text', 'value_sha256': sha('child text')}}

    reader = base_row('reader-target', reader_id, 'train', 'group-a')
    name = 'soft-state:' + block_id
    reader['messages'] = [{'role': 'user', 'content': [
        {'type': 'soft', 'name': name}, {'type': 'read', 'name': name}]}]
    witness = {'kind': 'completed-eval-finish-host-reference', 'source': 'eval-finish',
        'host_result_call_id': writer_id, 'host_result_type': 'Neuralese<string>',
        'host_result_value_sha256': handle_sha}
    reader['source_ref']['parent_invocation_id'] = 'parent/1'
    reader['source_ref']['provider_expanded_read_contexts'] = [{
        'schema': 'natlang.provider-expanded-read-context/2', 'origin': 'same-run-producer',
        'writer_target_selected': False, 'writer_source_class': 'legacy-text-marker-standin-eval-finish',
        'writer_witness': witness, 'invocation_id': reader_id, 'parent_invocation_id': 'parent/1',
        'source_row_sha256': source_hash, 'trace_sha256': sha('trace'),
        'transport_provenance_sha256': transport_hash, 'raw_request_sha256': raw_hash,
        'rendered_request_sha256': rendered_hash,
        'block': {'id': block_id, 'type': 'Neuralese<string>', 'body': body,
                  'body_sha256': body_hash, 'learned_vectors': False},
        'producer_write': {'kind': 'block_write', 'producer': 'text-marker-emulation',
            'source_kind': 'typed-text-result', 'source': 'eval-finish',
            'marker_context': 'return-result', 'call_id': writer_id, 'node': writer_node,
            'block': block_id, 'result_type': 'Neuralese<string>', 'text_body_sha256': body_hash,
            'inputs': [{'node': writer_id + '#turn2', 'port': 'result-source'}]},
        'block_read': {'kind': 'block_read', 'call_id': reader_id, 'node': reader_node,
            'turn': reader_id + '#turn1', 'block': block_id,
            'inputs': [{'node': writer_node, 'block': block_id}]},
        'model_turn': {'kind': 'model_turn', 'call_id': reader_id, 'node': reader_id + '#turn1',
            'inputs': [{'node': reader_node, 'block': block_id}]}}]
    reader['neuralese_conversion']['external_context_inputs'] = [{
        'schema': 'natlang.external-context-input/1', 'origin': 'same-run-producer',
        'block_id': block_id, 'type': 'Neuralese<string>', 'body_sha256': body_hash,
        'invocation_id': reader_id, 'parent_invocation_id': 'parent/1',
        'transport_provenance_sha256': transport_hash, 'raw_request_sha256': raw_hash,
        'rendered_request_sha256': rendered_hash, 'source_request_sha256': sha('source-request'),
        'source_response_sha256': sha('source-response'), 'source_row_sha256': source_hash,
        'trace_sha256': sha('trace'), 'read_node': reader_node, 'model_turn_node': reader_id + '#turn1',
        'producer_call_id': writer_id, 'producer_write_node': writer_node,
        'target_write_name': name, 'writer_source_class': 'legacy-text-marker-standin-eval-finish',
        'writer_witness': witness, 'writer_target_selected': False,
        'learner_representation': 'typed-read-from-authenticated-runtime-writer-event-context-only'}]

    heldout = base_row('existing-heldout', 'heldout/1', 'test', 'heldout-group')
    heldout['source_ref']['trajectory_id'] = 'heldout-trajectory'
    rows = [writer, child, reader, heldout]
    records = tmp_path / 'records.jsonl'
    pieces = tmp_path / 'pieces.jsonl'
    records.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    pieces.write_text(json.dumps({'name': name, 'kind': 'Neuralese<string>', 'text': body}) + '\n')

    graph = [
        {'kind': 'model_turn', 'call_id': writer_id, 'node': writer_id + '#turn2', 'seq': 16},
        {'kind': 'block_write', 'call_id': writer_id, 'node': writer_node, 'block': block_id,
         'source': 'eval-finish', 'result_type': 'Neuralese<string>', 'text_body_sha256': body_hash,
         'truncated': False, 'seq': 17,
         'inputs': [{'node': writer_id + '#turn2', 'port': 'result-source'}]},
        {'kind': 'block_read', 'call_id': reader_id, 'node': reader_node, 'turn': reader_id + '#turn1',
         'block': block_id, 'inputs': [{'node': writer_node, 'block': block_id}]},
        {'kind': 'model_turn', 'call_id': reader_id, 'node': reader_id + '#turn1', 'seq': 20,
         'inputs': [{'node': reader_node, 'block': block_id}]},
        {'kind': 'invocation', 'phase': 'start', 'call_id': child_id, 'node': 'call:' + child_id,
         'inputs': [{'node': 'call:' + writer_id, 'port': 'caller'}]},
    ]
    trace = {'outcome': {'invocation_ledger': [
        {'invocation_id': writer_id, 'parent_invocation_id': 'parent/1',
         'host_result': {'call_id': writer_id, 'result_type': 'Neuralese<string>',
            'terminal_action_seq': 18, 'value': handle, 'value_sha256': handle_sha}},
        {'invocation_id': child_id, 'parent_invocation_id': writer_id,
         'completion_status': 'done', 'host_result': {'result_type': 'string', 'value': 'child text',
            'value_sha256': sha('child text')}},
        {'invocation_id': reader_id, 'parent_invocation_id': 'parent/1'}],
        'action_ledger': [{'call_id': writer_id, 'seq': 18, 'name': 'eval',
                           'outcome': 'completed', 'arguments': arguments}],
        'execution_graph': graph}}
    trace_path = tmp_path / 'trace.json'
    trace_path.write_text(json.dumps(trace))
    report = observed_audit.audit(rows, [(trace_path, trace)])
    report['inputs'] = {'native_rows': {'path': str(records),
        'sha256': hashlib.sha256(records.read_bytes()).hexdigest()},
        'execution_traces': [{'path': str(trace_path),
            'sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest()}]}
    audit_path = tmp_path / 'observed-output-audit.json'
    audit_path.write_text(json.dumps(report))
    return records, pieces, audit_path, rows, name


@pytest.mark.parametrize('tamper', [None, 'missing-closure', 'missing-value', 'split-mismatch', 'source-group-mismatch'])
def test_assembler_can_consume_trace_authenticated_eval_context_without_output_loss(
        tmp_path, tamper):
    records, pieces, audit_path, rows, name = make_inputs(tmp_path)
    report = json.loads(audit_path.read_text())
    if tamper == 'missing-closure':
        report['events'][0]['closure']['reader_contexts'] = []
    elif tamper == 'missing-value':
        report['events'][0]['body'] = None
    elif tamper == 'split-mismatch':
        rows[2]['split'] = 'test'
        records.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        report['inputs']['native_rows']['sha256'] = hashlib.sha256(records.read_bytes()).hexdigest()
    elif tamper == 'source-group-mismatch':
        rows[2]['source_groups'] = ['different-group']
        rows[2]['task']['program_ir']['source_groups'] = ['different-group']
        records.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        report['inputs']['native_rows']['sha256'] = hashlib.sha256(records.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(report))
    review = {'schema': assembler.SOURCE_REVIEW_SCHEMA,
        'inputs': {'records': [{'path': str(records.resolve()),
            'sha256': hashlib.sha256(records.read_bytes()).hexdigest()}],
            'pieces': [{'path': str(pieces.resolve()),
            'sha256': hashlib.sha256(pieces.read_bytes()).hexdigest()}]},
        'allow': {'source_groups': [], 'target_ids': [row['id'] for row in rows]},
        'hold': {'source_groups': [], 'target_ids': []}}
    review_path = tmp_path / 'source-review.json'
    review_path.write_text(json.dumps(review))
    out = tmp_path / 'assembled'
    assembler.main(['--records', str(records), '--pieces', str(pieces),
        '--source-review', str(review_path), '--observed-output-audit', str(audit_path), '--out', str(out)])

    assembled = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
    ids = {row['id'] for row in assembled}
    if tamper is None:
        assert ids == {'writer-target', 'child-target', 'reader-target', 'existing-heldout'}
        recurrence = json.loads((out / 'recurrence-audit.json').read_text())
        assert recurrence['writer_records'] == 0
        assert recurrence['observed_output_context_root_count'] == 1
        assert recurrence['linked_edges'] == 0
        root = recurrence['observed_output_context_roots'][0]
        assert root['name'] == name and root['type'] == 'Neuralese<string>'
        assert root['trainable_producer_loss'] is False
        assert root['child_gradient_transport'] == 'not-claimed-through-opaque-eval-boundary'
        assert root['children'][0]['source_rows'][0]['id'] == 'child-target'
        train_rows = [json.loads(line) for line in (out / 'train.jsonl').read_text().splitlines()]
        reader = next(row for row in train_rows if row['id'] == 'reader-target')
        soft = next(part for message in reader['messages'] for part in message.get('content', [])
                    if part.get('type') == 'soft')
        piece = json.loads((out / 'pieces.jsonl').read_text())
        assert soft['name'] == piece['name'] and piece['text'] == 'observed answer'
    else:
        assert 'reader-target' not in ids
        held = [json.loads(line) for line in (out / 'held-targets.jsonl').read_text().splitlines()]
        assert any(item['id'] == 'reader-target' and item['reason'] == 'producer closure'
                   for item in held)
        assert 'existing-heldout' in ids
