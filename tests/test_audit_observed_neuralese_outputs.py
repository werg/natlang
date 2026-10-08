import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('audit_observed_outputs',
    ROOT / 'scripts/audit_observed_neuralese_outputs.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    body = 'observed answer'
    body_hash = sha(body)
    row_hash = sha('source-row')
    trajectory = 'trajectory-1'
    writer_id, child_id, reader_id = 'eval/1', 'child/1', 'reader/1'
    block_id, writer_node = 'nz1_block', writer_id + '#17'
    handle = {'$neuralese': {'type': 'Neuralese<string>', 'id': block_id}}
    handle_sha = AUDIT.host_value_digest(handle)
    arguments = {'code': 'const answer = await nl<string>`solve`(input); return answer;', 'finish': True}
    writer = {
        'id': 'writer-target', 'split': 'test', 'source_ids': ['source-a'],
        'source_groups': ['group-a'], 'training_admission': {'approved': False},
        'provenance': {'trace_sha256': sha('trace')},
        'source_ref': {'trajectory_id': trajectory, 'source_row_sha256': row_hash,
            'invocation_id': writer_id, 'host_result_capture': {'capture': {
                'call_id': writer_id, 'result_type': 'Neuralese<string>',
                'terminal_action_seq': 18,
                'origin': 'observed-host-result; not a model-generated writer target',
                'value': handle, 'value_sha256': handle_sha,
            }}},
        'decision': {'assistant': {'calls': [{'name': 'eval', 'tool_call_id': 'tool-call-1',
            'outcome': {'status': 'completed', 'trace_seq': 18, 'arguments': arguments,
                'typed_result_writes': [{'schema': 'natlang.typed-result-write/1',
                    'trajectory_id': trajectory, 'source_row_sha256': row_hash,
                    'invocation_id': writer_id, 'writer_call_id': writer_id,
                    'writer_node': writer_node, 'block_id': block_id,
                    'source_kind': 'typed-text-result', 'source': 'eval-finish',
                    'result_type': 'Neuralese<string>', 'result_path': ['return'],
                    'model_turn_node': writer_id + '#turn2', 'action_seq': 18,
                    'body_sha256': body_hash, 'request_sha256': sha('request'),
                    'raw_response_sha256': sha('response'),
                    'body_source_basis': 'authenticated-final-host-output-reference'}]}}]}}}
    reader = {
        'id': 'reader-target', 'split': 'test', 'source_ids': ['source-a'],
        'source_groups': ['group-a'], 'training_admission': {'approved': False},
        'provenance': {'trace_sha256': sha('trace')},
        'source_ref': {'trajectory_id': trajectory, 'source_row_sha256': row_hash,
            'invocation_id': reader_id, 'provider_expanded_read_contexts': [{
                'schema': 'natlang.provider-expanded-read-context/2',
                'origin': 'same-run-producer', 'writer_target_selected': False,
                'source_row_sha256': row_hash, 'invocation_id': reader_id,
                'block': {'id': block_id, 'type': 'Neuralese<string>', 'body': body,
                          'body_sha256': body_hash},
                'producer_write': {'kind': 'block_write', 'call_id': writer_id,
                    'node': writer_node, 'block': block_id,
                    'result_type': 'Neuralese<string>', 'text_body_sha256': body_hash},
                'block_read': {'call_id': reader_id, 'node': reader_id + '#2',
                    'turn': reader_id + '#turn1',
                    'inputs': [{'node': writer_node, 'block': block_id}]},
                'model_turn': {'call_id': reader_id, 'node': reader_id + '#turn1',
                    'inputs': [{'node': reader_id + '#2', 'block': block_id}]},
            }]}}
    child = {'id': 'child-target', 'source_ref': {'invocation_id': child_id}}
    trace = {'outcome': {
        'invocation_ledger': [
            {'invocation_id': writer_id, 'parent_invocation_id': 'parent/1',
             'host_result': {'call_id': writer_id, 'result_type': 'Neuralese<string>',
                'terminal_action_seq': 18, 'value': handle, 'value_sha256': handle_sha}},
            {'invocation_id': child_id, 'parent_invocation_id': writer_id,
             'completion_status': 'done', 'host_result': {'result_type': 'Neuralese<string>',
                'value': {'$neuralese': {'type': 'Neuralese<string>', 'id': block_id}},
                'value_sha256': sha('child-output')}}],
        'action_ledger': [{'call_id': writer_id, 'seq': 18, 'name': 'eval',
                           'outcome': 'completed', 'arguments': arguments}],
        'execution_graph': [
            {'kind': 'model_turn', 'call_id': writer_id, 'node': writer_id + '#turn2', 'seq': 16},
            {'kind': 'block_write', 'call_id': writer_id, 'node': writer_node,
             'block': block_id, 'source': 'eval-finish', 'result_type': 'Neuralese<string>',
             'text_body_sha256': body_hash, 'truncated': False, 'seq': 17,
             'inputs': [{'node': writer_id + '#turn2', 'port': 'result-source'}]},
            {'kind': 'block_read', 'call_id': reader_id, 'node': reader_id + '#2',
             'turn': reader_id + '#turn1', 'block': block_id,
             'inputs': [{'node': writer_node, 'block': block_id, 'port': 'block'}]},
            {'kind': 'model_turn', 'call_id': reader_id, 'node': reader_id + '#turn1', 'seq': 20,
             'inputs': [{'node': reader_id + '#2', 'block': block_id, 'port': 'read'}]},
            {'kind': 'invocation', 'phase': 'start', 'call_id': child_id,
             'node': 'call:' + child_id,
             'inputs': [{'node': 'call:' + writer_id, 'port': 'caller'}]},
        ]}}
    return [writer, child, reader], [('trace.json', trace)], child_id, block_id


def test_eval_output_is_observed_event_with_child_and_reader_links_not_writer_target():
    rows, traces, child_id, block_id = fixture()
    report = AUDIT.audit(rows, traces)
    assert report['observed_output_events'] == 1
    event = report['events'][0]
    assert event['closure']['status'] == 'trace-and-reader-closed'
    assert event['body'] == 'observed answer'
    assert event['body_sha256'] == sha('observed answer')
    assert event['result_type'] == 'Neuralese<string>'
    assert event['split'] == 'test' and event['source_groups'] == ['group-a']
    assert event['assistant_target_id'] == 'writer-target'
    assert event['output_is_assistant_target'] is False
    assert event['trainable_writer'] is False
    assert event['producer_class'] == 'opaque-eval-tool-result'
    assert event['assistant_target_contains_output_body'] is False
    child = event['child_invocations'][0]
    assert child['invocation_id'] == child_id
    assert child['source_rows'] == [{'id': 'child-target', 'split': None,
                                      'source_ids': [], 'source_groups': [],
                                      'same_split_as_parent_target': False,
                                      'shared_source_groups': []}]
    assert child['value_is_output_block'] is True
    assert child['output_dependency_on_eval_result'] == 'not-asserted-by-recorded-graph'
    assert report['trainable_output_targets'] == 0


def test_output_event_stays_held_when_reader_binding_is_wrong():
    rows, traces, _, _ = fixture()
    rows[-1]['source_ref']['provider_expanded_read_contexts'][0]['block_read']['inputs'][0]['node'] = 'other#write'
    report = AUDIT.audit(rows, traces)
    event = report['events'][0]
    assert event['closure']['status'] == 'trace-closed-reader-context-not-found'
    assert event['body'] is None
    assert event['trainable_writer'] is False


def test_forged_reader_node_is_not_closed_by_embedded_receipt_alone():
    rows, traces, _, _ = fixture()
    rows[-1]['source_ref']['provider_expanded_read_contexts'][0]['model_turn']['node'] = 'forged#turn'
    report = AUDIT.audit(rows, traces)
    assert report['events'][0]['closure']['status'] == 'trace-closed-reader-context-not-found'


def test_reader_split_mismatch_holds_observed_context_root():
    rows, traces, _, _ = fixture()
    rows[-1]['split'] = 'train'
    report = AUDIT.audit(rows, traces)
    assert report['events'][0]['closure']['status'] == 'trace-closed-reader-context-not-found'


def test_missing_observed_value_holds_reader_closure():
    rows, traces, _, _ = fixture()
    rows[-1]['source_ref']['provider_expanded_read_contexts'][0]['block']['body'] = None
    report = AUDIT.audit(rows, traces)
    assert report['events'][0]['closure']['status'] == 'trace-closed-reader-context-not-found'
