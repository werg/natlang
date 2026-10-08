#!/usr/bin/env python3
"""Audit eval-produced Neuralese values as observed, non-trainable graph events.

An eval-finish value is an output of the host execution boundary. It is not an
assistant target. Nested NL invocations are recorded as child invocation nodes
when the trace proves the caller edge, but their outputs are not silently
promoted to the eval result or treated as differentiable through the host.
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def host_block_id(row):
    capture = row.get('source_ref', {}).get('host_result_capture', {})
    capture = capture.get('capture', {}) if isinstance(capture, dict) else {}
    value = capture.get('value') if isinstance(capture, dict) else None
    neuralese = value.get('$neuralese') if isinstance(value, dict) else None
    return neuralese.get('id') if isinstance(neuralese, dict) else None


def eval_finish_receipts(row):
    calls = row.get('decision', {}).get('assistant', {}).get('calls', [])
    for action in calls if isinstance(calls, list) else []:
        if not isinstance(action, dict):
            continue
        outcome = action.get('outcome')
        if not isinstance(outcome, dict) or (action.get('source_tool') or action.get('name')) != 'eval':
            continue
        for receipt in outcome.get('typed_result_writes', []):
            if isinstance(receipt, dict) and receipt.get('source') == 'eval-finish':
                yield action, outcome, receipt


def reader_context_matches(event, reader):
    """Require provider-expanded graph evidence for the exact observed event."""
    source = reader.get('source_ref', {})
    refs = source.get('provider_expanded_read_contexts', []) if isinstance(source, dict) else []
    for ref in refs if isinstance(refs, list) else []:
        if not isinstance(ref, dict) or ref.get('schema') != 'natlang.provider-expanded-read-context/2':
            continue
        block = ref.get('block')
        producer, read, turn = ref.get('producer_write'), ref.get('block_read'), ref.get('model_turn')
        if not all(isinstance(x, dict) for x in (block, producer, read, turn)):
            continue
        body = block.get('body')
        if (ref.get('writer_target_selected') is not False
                or ref.get('source_row_sha256') != event['source_row_sha256']
                or block.get('id') != event['block_id']
                or block.get('type') != event['result_type']
                or not isinstance(body, str)
                or sha256_bytes(body.encode()) != event['body_sha256']
                or block.get('body_sha256') != event['body_sha256']
                or producer.get('kind') != 'block_write'
                or producer.get('call_id') != event['invocation_id']
                or producer.get('node') != event['writer_node']
                or producer.get('block') != event['block_id']
                or producer.get('result_type') != event['result_type']
                or producer.get('text_body_sha256') != event['body_sha256']
                or read.get('call_id') != ref.get('invocation_id')
                or not any(isinstance(edge, dict) and edge.get('node') == event['writer_node']
                           and edge.get('block') == event['block_id'] for edge in read.get('inputs', []))
                or turn.get('call_id') != ref.get('invocation_id')
                or turn.get('node') != read.get('turn')
                or not any(isinstance(edge, dict) and edge.get('node') == read.get('node')
                           and edge.get('block') == event['block_id'] for edge in turn.get('inputs', []))):
            continue
        return {'reader_invocation_id': ref.get('invocation_id'),
                'reader_record_id': reader.get('id'), 'body': body,
                'body_sha256': event['body_sha256'], 'result_type': event['result_type'],
                'read_node': read.get('node'), 'model_turn_node': turn.get('node'),
                'scope': 'authenticated-provider-read-of-observed-event'}
    return None


def audit(rows, trace_documents):
    rows = list(rows)
    rows_by_invocation = {}
    for row in rows:
        source = row.get('source_ref', {})
        if isinstance(source, dict) and source.get('invocation_id'):
            rows_by_invocation.setdefault(source['invocation_id'], []).append(row)
    traces_by_invocation = {}
    for path, doc in trace_documents:
        outcome = doc.get('outcome', {})
        ledger = outcome.get('invocation_ledger', []) if isinstance(outcome, dict) else []
        graph = outcome.get('execution_graph', []) if isinstance(outcome, dict) else []
        try:
            trace_file_sha256 = sha256_bytes(Path(path).read_bytes())
        except (OSError, TypeError):
            trace_file_sha256 = None
        for item in ledger if isinstance(ledger, list) else []:
            if isinstance(item, dict) and isinstance(item.get('invocation_id'), str):
                traces_by_invocation.setdefault(item['invocation_id'], []).append(
                    {'path': str(path), 'file_sha256': trace_file_sha256,
                     'document': doc, 'ledger': ledger, 'graph': graph})

    events, failures = [], []
    for row in rows:
        source = row.get('source_ref', {})
        if not isinstance(source, dict):
            continue
        for action, outcome, receipt in eval_finish_receipts(row):
            inv = source.get('invocation_id')
            capture = source.get('host_result_capture', {})
            capture = capture.get('capture', {}) if isinstance(capture, dict) else {}
            block_id = receipt.get('block_id')
            event = {
                'schema': 'natlang.observed-neuralese-output-event/1',
                'event_id': hashlib.sha256(stable_json([source.get('trajectory_id'), inv,
                    receipt.get('writer_node'), block_id]).encode()).hexdigest(),
                'source_record_id': row.get('id'),
                'source_row_sha256': source.get('source_row_sha256'),
                'trace_sha256': row.get('provenance', {}).get('trace_sha256'),
                'trajectory_id': source.get('trajectory_id'),
                'invocation_id': inv,
                'writer_call_id': receipt.get('writer_call_id'),
                'writer_node': receipt.get('writer_node'),
                'block_id': block_id,
                'result_type': receipt.get('result_type'),
                'body_sha256': receipt.get('body_sha256'),
                'host_result_value_sha256': capture.get('value_sha256'),
                'action_seq': receipt.get('action_seq'),
                'action_target_call_id': action.get('tool_call_id') or action.get('call_id'),
                'action_arguments_sha256': sha256_bytes(stable_json(outcome.get('arguments', {})).encode())
                    if isinstance(outcome.get('arguments'), dict) else None,
                'action_code_sha256': sha256_bytes(outcome.get('arguments', {}).get('code', '').encode())
                    if isinstance(outcome.get('arguments'), dict)
                    and isinstance(outcome.get('arguments', {}).get('code'), str) else None,
                'split': row.get('split'),
                'source_ids': row.get('source_ids', []),
                'source_groups': row.get('source_groups', []),
                'training_admission': row.get('training_admission'),
                'assistant_target_id': row.get('id'),
                'assistant_target_is_eval_action': True,
                'assistant_target_contains_output_body': 'unassessed-until-reader-body-is-available',
                'output_is_assistant_target': False,
                'trainable_writer': False,
                'producer_class': 'opaque-eval-tool-result',
                'gradient_transport': 'no-through-model-gradient-claim-across-eval-boundary',
                'closure': {'status': 'unverified'},
                'child_invocations': [],
            }
            reasons = []
            if (outcome.get('status') != 'completed' or receipt.get('result_type') != 'Neuralese<string>'
                    or receipt.get('source_kind') != 'typed-text-result'
                    or receipt.get('trajectory_id') != source.get('trajectory_id')
                    or receipt.get('source_row_sha256') != source.get('source_row_sha256')
                    or receipt.get('invocation_id') != inv or receipt.get('writer_call_id') != inv
                    or receipt.get('block_id') != host_block_id(row)
                    or receipt.get('action_seq') != outcome.get('trace_seq')
                    or capture.get('call_id') != inv or capture.get('result_type') != receipt.get('result_type')
                    or capture.get('terminal_action_seq') != receipt.get('action_seq')
                    or capture.get('origin') != 'observed-host-result; not a model-generated writer target'):
                reasons.append('native-action-receipt-host-capture-binding-failed')
            trace_matches = traces_by_invocation.get(inv, [])
            if len(trace_matches) != 1:
                reasons.append('execution-trace-invocation-not-unique')
            else:
                trace = trace_matches[0]
                graph = trace['graph']
                raw_actions = trace['document'].get('outcome', {}).get('action_ledger', [])
                action_args = outcome.get('arguments')
                trace_actions = [item for item in raw_actions if isinstance(item, dict)
                    and item.get('call_id') == inv and item.get('seq') == receipt.get('action_seq')
                    and item.get('name') == 'eval' and item.get('outcome') == 'completed'
                    and item.get('arguments') == action_args]
                if len(trace_actions) != 1:
                    reasons.append('native-eval-action-does-not-match-trace-action')
                event_nodes = [node for node in graph if isinstance(node, dict)
                    and node.get('kind') == 'block_write' and node.get('call_id') == inv
                    and node.get('node') == receipt.get('writer_node')
                    and node.get('block') == block_id and node.get('source') == 'eval-finish'
                    and node.get('result_type') == receipt.get('result_type')
                    and node.get('text_body_sha256') == receipt.get('body_sha256')
                    and node.get('truncated') is False]
                turn = [node for node in graph if isinstance(node, dict)
                        and node.get('kind') == 'model_turn' and node.get('call_id') == inv
                        and node.get('node') == receipt.get('model_turn_node')]
                ledger_item = next((x for x in trace['ledger'] if isinstance(x, dict)
                                    and x.get('invocation_id') == inv), {})
                trace_capture = ledger_item.get('host_result', {})
                trace_capture_matches = (isinstance(trace_capture, dict)
                    and trace_capture.get('call_id') == inv
                    and trace_capture.get('result_type') == receipt.get('result_type')
                    and trace_capture.get('terminal_action_seq') == receipt.get('action_seq')
                    and trace_capture.get('value') == capture.get('value')
                    and trace_capture.get('value_sha256') == capture.get('value_sha256'))
                if not trace_capture_matches:
                    reasons.append('host-result-capture-does-not-match-invocation-trace')
                if len(event_nodes) != 1 or len(turn) != 1 or not any(
                        isinstance(edge, dict) and edge.get('node') == turn[0].get('node')
                        and edge.get('port') == 'result-source' for edge in event_nodes[0].get('inputs', [])
                        ) or not (isinstance(event_nodes[0].get('seq'), int)
                                  and isinstance(turn[0].get('seq'), int)
                                  and turn[0]['seq'] < event_nodes[0]['seq'] < receipt['action_seq']
                        ) if len(event_nodes) == 1 and len(turn) == 1 else True:
                    reasons.append('trace-result-event-graph-binding-failed')
                children = [x for x in trace['ledger'] if isinstance(x, dict)
                            and x.get('parent_invocation_id') == inv]
                graph_children = {node.get('call_id')
                    for node in graph if isinstance(node, dict) and node.get('kind') == 'invocation'
                    and node.get('phase') == 'start' and isinstance(node.get('inputs'), list)
                    and any(isinstance(edge, dict) and edge.get('node') == 'call:' + inv
                            and edge.get('port') == 'caller' for edge in node['inputs'])}
                for child in children:
                    child_id = child.get('invocation_id')
                    if child_id not in graph_children:
                        continue
                    child_host = child.get('host_result', {})
                    child_value = child_host.get('value') if isinstance(child_host, dict) else None
                    event['child_invocations'].append({
                        'invocation_id': child_id,
                        'invocation_start_link': 'caller-edge-in-authenticated-execution-graph',
                        'source_rows': [{'id': x.get('id'), 'split': x.get('split'),
                                         'source_ids': x.get('source_ids', []),
                                         'source_groups': x.get('source_groups', []),
                                         'same_split_as_parent_target': x.get('split') == row.get('split'),
                                         'shared_source_groups': sorted(set(x.get('source_groups', []))
                                                                        & set(row.get('source_groups', [])))}
                                        for x in rows_by_invocation.get(child_id, [])],
                        'result_type': child_host.get('result_type') if isinstance(child_host, dict) else None,
                        'value_sha256': child_host.get('value_sha256') if isinstance(child_host, dict) else None,
                        'completion_status': child.get('completion_status'),
                        'output_dependency_on_eval_result': 'not-asserted-by-recorded-graph',
                        'value_is_output_block': (isinstance(child_value, dict)
                            and isinstance(child_value.get('$neuralese'), dict)
                            and child_value['$neuralese'].get('id') == block_id),
                    })
            if not reasons:
                event['closure']['status'] = 'trace-event-authenticated'
                event['closure']['trace_path'] = trace_matches[0]['path']
                event['closure']['trace_file_sha256'] = trace_matches[0]['file_sha256']
                matched_readers = []
                for reader in rows:
                    match = reader_context_matches(event, reader)
                    if match:
                        matched_readers.append(match)
                event['closure']['reader_contexts'] = matched_readers
                event['closure']['status'] = ('trace-and-reader-closed' if matched_readers
                                              else 'trace-closed-reader-context-not-found')
                event['body'] = matched_readers[0]['body'] if matched_readers else None
                if matched_readers:
                    action_code = outcome.get('arguments', {}).get('code', '')
                    event['assistant_target_contains_output_body'] = (
                        event['body'] in action_code if isinstance(action_code, str) else None)
                    event['assistant_target_body_overlap_basis'] = 'exact-substring-check-in-supervised-eval-source'
            else:
                event['closure']['status'] = 'held'
            event['validation_failures'] = reasons
            events.append(event)
            if reasons:
                failures.append({'event_id': event['event_id'], 'reasons': reasons})
    return {
        'schema': 'natlang.observed-neuralese-output-audit/1',
        'input_records': len(rows),
        'observed_output_events': len(events),
        'trace_authenticated_events': sum(e['closure']['status'] in
            ('trace-event-authenticated', 'trace-and-reader-closed', 'trace-closed-reader-context-not-found')
            for e in events),
        'trace_and_reader_closed_events': sum(e['closure']['status'] == 'trace-and-reader-closed' for e in events),
        'events_with_child_invocations': sum(bool(e['child_invocations']) for e in events),
        'child_invocation_nodes': sum(len(e['child_invocations']) for e in events),
        'child_invocations_with_selected_rows': sum(bool(x['source_rows']) for e in events
                                                    for x in e['child_invocations']),
        'child_outputs_equal_event_block': sum(any(x['value_is_output_block'] for x in e['child_invocations'])
                                               for e in events),
        'trainable_output_targets': sum(e['output_is_assistant_target'] for e in events),
        'gradient_transport': 'opaque-eval-results-are-context; child invocation edges are provenance only unless a separate qualified differentiable runtime path exists',
        'failures': failures,
        'events': events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('native_rows', type=Path)
    parser.add_argument('--trace-glob', action='append', default=[],
                        help='glob(s) for source result JSONL files containing outcome.execution_graph')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.native_rows.read_text().splitlines() if line.strip()]
    trace_docs = []
    for pattern in args.trace_glob:
        for filename in sorted(glob.glob(pattern, recursive=True)):
            path = Path(filename)
            payload = path.read_text()
            if not payload.strip():
                continue
            doc = json.loads(payload)
            trace_docs.append((path, doc))
    result = audit(rows, trace_docs)
    result['inputs'] = {
        'native_rows': {'path': str(args.native_rows),
                        'sha256': sha256_bytes(args.native_rows.read_bytes())},
        'execution_traces': [{'path': str(path), 'sha256': sha256_bytes(path.read_bytes())}
                             for path, _ in trace_docs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'events'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
