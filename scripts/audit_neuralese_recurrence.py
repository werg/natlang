#!/usr/bin/env python3
"""Audit producer closure and whole-graph split isolation without loading contexts.

Equal text is not evidence of invocation identity. Ambiguous producers are held.
This is a structural gate; correctness and causal-use evaluation remain separate.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def names(value, kind):
    out = set()
    if isinstance(value, dict):
        if kind == 'read' and value.get('type') == 'read': out.add(value['name'])
        if kind == 'write' and isinstance(value.get('$write'), dict): out.add(value['$write']['name'])
        for k,v in value.items():
            if k == 'arguments' and isinstance(v, str):
                try: out |= names(json.loads(v), kind)
                except ValueError: pass
            else: out |= names(v, kind)
    elif isinstance(value, list):
        for v in value: out |= names(v, kind)
    return out


def semantic_writer_matches_reader(writer, reader):
    """Bind a selected direct return to the exact producer event read downstream.

    Sequence numbers are local to each invocation trace. Cross-invocation ordering
    is therefore proven by the reader graph's producer node input, not by comparing
    the two numeric sequence fields.
    """
    source = reader.get('source_ref', {})
    refs = source.get('provider_expanded_read_contexts', []) if isinstance(source, dict) else []
    for ref in refs if isinstance(refs, list) else []:
        if not isinstance(ref, dict) or ref.get('schema') != 'natlang.provider-expanded-read-context/2':
            continue
        block = ref.get('block')
        producer, read, turn = ref.get('producer_write'), ref.get('block_read'), ref.get('model_turn')
        if not all(isinstance(x, dict) for x in (block, producer, read, turn)):
            continue
        expected_block = writer.get('block_id')
        if (block.get('id') != expected_block or not isinstance(block.get('body'), str)
                or hashlib.sha256(block['body'].encode()).hexdigest() != block.get('body_sha256')
                or producer.get('kind') != 'block_write'
                or producer.get('block') != expected_block or producer.get('call_id') != writer.get('writer_call_id')
                or producer.get('node') != writer.get('writer_node')
                or producer.get('result_type') != writer.get('result_type')
                or producer.get('text_body_sha256') != writer.get('body_sha256')
                or block.get('type') != writer.get('result_type')
                or block.get('body_sha256') != writer.get('body_sha256')
                or read.get('block') != expected_block or read.get('call_id') != ref.get('invocation_id')
                or not any(isinstance(edge, dict) and edge.get('node') == writer.get('writer_node')
                           and edge.get('block') == expected_block for edge in read.get('inputs', []))
                or turn.get('call_id') != ref.get('invocation_id') or turn.get('node') != read.get('turn')
                or not any(isinstance(edge, dict) and edge.get('node') == read.get('node')
                           and edge.get('block') == expected_block for edge in turn.get('inputs', []))):
            continue
        if ref.get('writer_target_selected') is not False:
            continue
        if (writer.get('trajectory_id') != source.get('trajectory_id')
                or not isinstance(writer.get('source_row_sha256'), str)
                or not isinstance(writer.get('invocation_id'), str)):
            continue
        event_name = direct_typed_result_event_name(expected_block, writer.get('trajectory_id'), writer.get('writer_call_id'),
                                                    writer.get('writer_node'))
        if event_name not in authenticated_external_context_names(reader):
            continue
        return True
    return False


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def direct_typed_result_event_name(block_id, trajectory_id, call_id, node):
    event = hashlib.sha256(json.dumps([trajectory_id, call_id, node], ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()[:12]
    return f'soft-state:{block_id}@{event}'


def validate_selected_direct_return_receipts(row, provider_contexts=()):
    import re
    conversion = row.get('neuralese_conversion')
    receipts = conversion.get('selected_runtime_result_writes', []) if isinstance(conversion, dict) else []
    target = row.get('target')
    calls = target.get('tool_calls', []) if isinstance(target, dict) else []
    source_ref = row.get('source_ref', {})
    raw_decision = row.get('decision', {})
    raw_calls = (raw_decision.get('assistant', {}).get('calls', [])
                 if isinstance(raw_decision, dict) else [])
    valid, failures = {}, []
    for receipt in receipts if isinstance(receipts, list) else []:
        reason = None
        if not isinstance(receipt, dict) or receipt.get('schema') != 'natlang.selected-runtime-result-write/1' or \
                receipt.get('role') != 'selected-direct-typed-text-semantic-writer':
            reason = 'receipt-schema-or-role-invalid'
        else:
            block, name = receipt.get('block_id'), receipt.get('target_write_name')
            required = (receipt.get('body_sha256'), receipt.get('source_row_sha256'),
                        receipt.get('action_arguments_sha256'), receipt.get('target_write_sha256'),
                        receipt.get('typed_result_receipt_sha256'))
            if (not isinstance(block, str) or re.fullmatch(r'nz1_[a-z2-7]{20,}', block) is None
                    or name != direct_typed_result_event_name(block, receipt.get('trajectory_id'),
                                                              receipt.get('writer_call_id'), receipt.get('writer_node'))
                    or receipt.get('trajectory_id') != source_ref.get('trajectory_id')
                    or receipt.get('source_row_sha256') != source_ref.get('source_row_sha256')
                    or not isinstance(receipt.get('trajectory_id'), str) or not receipt['trajectory_id']
                    or not isinstance(receipt.get('source_row_sha256'), str)
                    or re.fullmatch(r'[0-9a-f]{64}', receipt['source_row_sha256']) is None
                    or receipt.get('invocation_id') != source_ref.get('invocation_id')
                    or receipt.get('writer_call_id') != source_ref.get('invocation_id')
                    or not isinstance(receipt.get('writer_node'), str) or not receipt['writer_node']
                    or receipt.get('result_type') != 'Neuralese<string>'
                    or receipt.get('body_source_basis') != 'exact-raw-model-result-string'
                    or any(not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None for value in required)
                    or receipt.get('result_path') != ['return']
                    or not isinstance(receipt.get('action_seq'), int) or receipt['action_seq'] < 0
                    or not isinstance(receipt.get('action_target_call_id'), str)):
                reason = 'event-binding-shape-invalid'
            else:
                body_sha = receipt['body_sha256']
                target_matches = []
                for call in calls if isinstance(calls, list) else []:
                    if not isinstance(call, dict) or call.get('id') != receipt['action_target_call_id']:
                        continue
                    fn = call.get('function')
                    if not isinstance(fn, dict) or fn.get('name') != 'return_result':
                        continue
                    try:
                        args = json.loads(fn.get('arguments', ''))
                    except (TypeError, json.JSONDecodeError):
                        continue
                    write = args.get('value', {}).get('$write') if isinstance(args, dict) and isinstance(args.get('value'), dict) else None
                    if (isinstance(write, dict) and args.get('status') == 'success'
                            and write.get('name') == name and write.get('block_id') == block
                            and write.get('type') == 'Neuralese<string>' and isinstance(write.get('source'), str)
                            and hashlib.sha256(write['source'].encode()).hexdigest() == body_sha
                            and hashlib.sha256(stable_json({'$write': write}).encode()).hexdigest() == receipt['target_write_sha256']):
                        target_matches.append(write)
                source_matches = []
                matched_typed_receipts = []
                for action in raw_calls if isinstance(raw_calls, list) else []:
                    if not isinstance(action, dict) or action.get('source_tool') != 'return_result':
                        continue
                    outcome = action.get('outcome')
                    if not isinstance(outcome, dict) or outcome.get('name') != 'return_result':
                        continue
                    args = outcome.get('arguments')
                    if (not isinstance(args, dict) or args.get('status') != 'success'
                            or args.get('value') is None or not isinstance(args.get('value'), str)
                            or outcome.get('trace_seq') != receipt.get('action_seq')
                            or hashlib.sha256(stable_json(args).encode()).hexdigest() != receipt['action_arguments_sha256']
                            or hashlib.sha256(args['value'].encode()).hexdigest() != body_sha):
                        continue
                    typed = outcome.get('typed_result_writes', [])
                    typed_matches = [item for item in typed if isinstance(item, dict) and item.get('block_id') == block
                           and item.get('writer_node') == receipt.get('writer_node')
                           and item.get('result_type') == 'Neuralese<string>'
                           and item.get('source_kind') == 'typed-text-result'
                           and item.get('source') == 'return_result'
                           and item.get('body_sha256') == body_sha
                           and item.get('body_source') == args['value']
                           and item.get('body_source_basis') == 'exact-raw-model-result-string'
                           and item.get('result_path') == ['return']
                           and item.get('action_seq') == receipt.get('action_seq')] if isinstance(typed, list) else []
                    if (len(typed_matches) == 1 and
                            hashlib.sha256(stable_json(typed_matches[0]).encode()).hexdigest() ==
                            receipt['typed_result_receipt_sha256']):
                        source_matches.append(action)
                        matched_typed_receipts.append(typed_matches[0])
                graph = (row.get('outcome') or {}).get('execution_graph', [])
                graph_writers = [event for event in graph if isinstance(event, dict)
                    and event.get('kind') == 'block_write' and event.get('call_id') == receipt.get('writer_call_id')
                    and event.get('node') == receipt.get('writer_node') and event.get('block') == block
                    and event.get('source_kind') == 'typed-text-result' and event.get('source') == 'return_result'
                    and event.get('marker_context') == 'return-result'
                    and event.get('result_type') == 'Neuralese<string>'
                    and event.get('text_body_sha256') == body_sha and event.get('truncated') is False]
                graph_turns = [event for event in graph if isinstance(event, dict)
                    and event.get('kind') == 'model_turn' and event.get('call_id') == receipt.get('writer_call_id')
                    and event.get('node') == (matched_typed_receipts[0].get('model_turn_node')
                                              if matched_typed_receipts else None)]
                event_ordered = (len(graph_writers) == 1 and len(graph_turns) == 1
                    and isinstance(graph_writers[0].get('seq'), int)
                    and isinstance(graph_turns[0].get('seq'), int)
                    and graph_turns[0]['seq'] < graph_writers[0]['seq'] < receipt['action_seq']
                    and any(isinstance(edge, dict) and edge.get('node') == graph_turns[0].get('node')
                            and edge.get('port') == 'result-source'
                            for edge in graph_writers[0].get('inputs', [])))
                if not event_ordered and not graph:
                    # Materialized decision rows omit the raw execution graph.
                    # An authenticated same-run provider read receipt can carry
                    # the exact producer event and both causal graph edges; join
                    # it only to this selected target's full source-row and
                    # event identity. It remains context evidence, not a new
                    # writer target.
                    matching_context_events = {}
                    for context in provider_contexts:
                        if not isinstance(context, dict) or context.get('schema') != 'natlang.provider-expanded-read-context/2':
                            continue
                        block_context = context.get('block')
                        producer = context.get('producer_write')
                        read, turn = context.get('block_read'), context.get('model_turn')
                        witness = context.get('writer_witness')
                        if not all(isinstance(value, dict) for value in (block_context, producer, read, turn)):
                            continue
                        body = block_context.get('body')
                        if (context.get('origin') != 'same-run-producer' or context.get('writer_target_selected') is not False
                                or context.get('source_row_sha256') != receipt.get('source_row_sha256')
                                or context.get('invocation_id') == receipt.get('writer_call_id')
                                or block_context.get('id') != block or block_context.get('type') != 'Neuralese<string>'
                                or not isinstance(body, str) or hashlib.sha256(body.encode()).hexdigest() != body_sha
                                or block_context.get('body_sha256') != body_sha
                                or producer.get('kind') != 'block_write' or producer.get('block') != block
                                or producer.get('call_id') != receipt.get('writer_call_id')
                                or producer.get('node') != receipt.get('writer_node')
                                or producer.get('producer') != 'text-marker-emulation'
                                or producer.get('source_kind') != 'typed-text-result'
                                or producer.get('source') != 'return_result'
                                or producer.get('marker_context') != 'return-result'
                                or producer.get('result_type') != 'Neuralese<string>'
                                or producer.get('text_body_sha256') != body_sha or producer.get('truncated') is not False
                                or not any(isinstance(edge, dict) and edge.get('node') == matched_typed_receipts[0].get('model_turn_node')
                                           and edge.get('port') == 'result-source' for edge in producer.get('inputs', []))
                                or not isinstance(read.get('node'), str) or not read.get('node')
                                or read.get('kind') != 'block_read' or read.get('block') != block
                                or read.get('call_id') != context.get('invocation_id')
                                or not any(isinstance(edge, dict) and edge.get('node') == receipt.get('writer_node')
                                           and edge.get('block') == block for edge in read.get('inputs', []))
                                or turn.get('kind') != 'model_turn' or turn.get('call_id') != context.get('invocation_id')
                                or turn.get('node') != read.get('turn')
                                or not any(isinstance(edge, dict) and edge.get('node') == read.get('node')
                                           and edge.get('block') == block for edge in turn.get('inputs', []))
                                or context.get('writer_source_class') != 'legacy-text-marker-standin-return-result'
                                or not isinstance(witness, dict)
                                or witness.get('kind') != 'raw-return-result-value-equals-expanded-body'
                                or witness.get('source') != 'return_result'
                                or witness.get('host_result_call_id') != producer.get('call_id')
                                or witness.get('host_result_type') != producer.get('result_type')
                                or any(not isinstance(context.get(key), str) or re.fullmatch(r'[0-9a-f]{64}', context[key]) is None
                                       for key in ('trace_sha256', 'transport_provenance_sha256', 'raw_request_sha256',
                                                   'rendered_request_sha256', 'source_request_sha256', 'source_response_sha256'))
                                or any(not isinstance(witness.get(key), str) or re.fullmatch(r'[0-9a-f]{64}', witness[key]) is None
                                       for key in ('host_result_value_sha256', 'raw_response_sha256'))):
                            continue
                        event_name = direct_typed_result_event_name(block, receipt.get('trajectory_id'),
                                                                   receipt.get('writer_call_id'), receipt.get('writer_node'))
                        if event_name == name:
                            event_key = (producer.get('call_id'), producer.get('node'), producer.get('block'),
                                         producer.get('text_body_sha256'), matched_typed_receipts[0].get('model_turn_node'))
                            matching_context_events[event_key] = (producer, matched_typed_receipts[0])
                    event_ordered = len(matching_context_events) == 1
                if len(target_matches) != 1 or len(source_matches) != 1 or not event_ordered:
                    reason = 'raw-action-or-target-write-not-uniquely-bound'
                else:
                    valid[name] = receipt
        if reason:
            failures.append({'record': row.get('id'),
                             'block_id': receipt.get('block_id') if isinstance(receipt, dict) else None,
                             'reason': reason})
    return valid, failures


def authenticated_external_context_names(row):
    """Return same-run typed reads that are evidence-backed context roots.

    These are observed runtime inputs, not selected model writer examples, so
    they must not create synthetic recurrence edges. Require agreement between
    converter metadata and the full provider receipt carried by the source row.
    """
    import re
    refs = row.get('source_ref', {}).get('provider_expanded_read_contexts', [])
    receipts = {}
    for ref in refs if isinstance(refs, list) else []:
        if not isinstance(ref, dict) or ref.get('schema') != 'natlang.provider-expanded-read-context/2':
            continue
        block = ref.get('block')
        producer = ref.get('producer_write')
        read = ref.get('block_read')
        turn = ref.get('model_turn')
        writer_source_class = ref.get('writer_source_class')
        producer_source_valid = (
            (writer_source_class in (None, 'modern-typed-text-result')
             and producer.get('producer') == 'text-marker-emulation'
             and producer.get('source_kind') == 'typed-text-result') if isinstance(producer, dict) else False)
        if isinstance(producer, dict) and writer_source_class == 'legacy-text-marker-standin-eval-code':
            producer_source_valid = (producer.get('emulation_version') == 'text-marker-standin/2'
                                     and producer.get('marker_context') == 'eval-code'
                                     and producer.get('learned_vectors') is False)
        witness = ref.get('writer_witness')
        if isinstance(producer, dict) and writer_source_class == 'legacy-text-marker-standin-return-result':
            producer_source_valid = (producer.get('producer') == 'text-marker-emulation'
                                     and producer.get('source_kind') == 'typed-text-result'
                                     and producer.get('source') == 'return_result'
                                     and producer.get('marker_context') == 'return-result'
                                     and isinstance(witness, dict)
                                     and witness.get('kind') == 'raw-return-result-value-equals-expanded-body'
                                     and witness.get('source') == 'return_result'
                                     and witness.get('host_result_call_id') == producer.get('call_id')
                                     and witness.get('host_result_type') == block.get('type')
                                     and all(isinstance(witness.get(key), str) and len(witness[key]) == 64
                                             and all(char in '0123456789abcdef' for char in witness[key])
                                             for key in ('host_result_value_sha256', 'raw_response_sha256')))
        if isinstance(producer, dict) and writer_source_class == 'legacy-text-marker-standin-eval-finish':
            producer_source_valid = (producer.get('producer') == 'text-marker-emulation'
                                     and producer.get('source_kind') == 'typed-text-result'
                                     and producer.get('source') == 'eval-finish'
                                     and producer.get('marker_context') == 'return-result'
                                     and isinstance(witness, dict)
                                     and witness.get('kind') == 'completed-eval-finish-host-reference'
                                     and witness.get('source') == 'eval-finish'
                                     and witness.get('host_result_call_id') == producer.get('call_id')
                                     and witness.get('host_result_type') == block.get('type')
                                     and isinstance(witness.get('host_result_value_sha256'), str)
                                     and len(witness['host_result_value_sha256']) == 64
                                     and all(char in '0123456789abcdef' for char in witness['host_result_value_sha256']))
        if (ref.get('origin') != 'same-run-producer' or ref.get('writer_target_selected') is not False
                or not isinstance(block, dict) or not isinstance(block.get('id'), str)
                or not isinstance(block.get('body'), str) or not block.get('type')
                or not isinstance(producer, dict) or producer.get('kind') != 'block_write'
                or not producer_source_valid
                or producer.get('block') != block.get('id')
                or producer.get('result_type') != block.get('type')
                or not isinstance(producer.get('call_id'), str) or not producer.get('call_id')
                or not isinstance(producer.get('node'), str) or not producer.get('node')
                or not isinstance(read, dict) or read.get('kind') != 'block_read'
                or read.get('block') != block.get('id') or read.get('call_id') != ref.get('invocation_id')
                or not isinstance(read.get('node'), str) or not read.get('node')
                or not isinstance(turn, dict) or turn.get('kind') != 'model_turn'
                or turn.get('call_id') != ref.get('invocation_id')
                or not isinstance(turn.get('node'), str) or not turn.get('node')
                or read.get('turn') != turn.get('node')
                or not isinstance(ref.get('invocation_id'), str) or not ref.get('invocation_id')
                or ('parent_invocation_id' in ref and ref.get('parent_invocation_id') is not None
                    and not isinstance(ref.get('parent_invocation_id'), str))):
            continue
        body_sha = hashlib.sha256(block['body'].encode()).hexdigest()
        required_hashes = (body_sha, block.get('body_sha256'), ref.get('source_row_sha256'),
                           ref.get('trace_sha256'), ref.get('transport_provenance_sha256'),
                           ref.get('raw_request_sha256'), ref.get('rendered_request_sha256'),
                           producer.get('text_body_sha256'))
        if (any(not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None
                for value in required_hashes)
                or body_sha != block.get('body_sha256')
                or producer.get('text_body_sha256') != body_sha
                or not any(isinstance(edge, dict) and edge.get('node') == producer.get('node')
                           and edge.get('block') == block.get('id') for edge in read.get('inputs', []))
                or not any(isinstance(edge, dict) and edge.get('node') == read.get('node')
                           and edge.get('block') == block.get('id') for edge in turn.get('inputs', []))):
            continue
        receipts[block['id']] = ref

    valid = set()
    conversion = row.get('neuralese_conversion')
    metadata = conversion.get('external_context_inputs', []) if isinstance(conversion, dict) else []
    for item in metadata if isinstance(metadata, list) else []:
        if not isinstance(item, dict) or item.get('schema') != 'natlang.external-context-input/1':
            continue
        if item.get('origin') != 'same-run-producer' or item.get('writer_target_selected') is not False:
            continue
        block_id = item.get('block_id')
        ref = receipts.get(block_id)
        if ref is None:
            continue
        block = ref['block']
        producer = ref.get('producer_write') or {}
        read = ref.get('block_read') or {}
        turn = ref.get('model_turn') or {}
        pairs = [(item.get('type'), block.get('type')),
                 (item.get('body_sha256'), block.get('body_sha256')),
                 (item.get('invocation_id'), ref.get('invocation_id')),
                 (item.get('parent_invocation_id'), ref.get('parent_invocation_id')),
                 (item.get('source_row_sha256'), ref.get('source_row_sha256')),
                 (item.get('trace_sha256'), ref.get('trace_sha256')),
                 (item.get('transport_provenance_sha256'), ref.get('transport_provenance_sha256')),
                 (item.get('raw_request_sha256'), ref.get('raw_request_sha256')),
                 (item.get('rendered_request_sha256'), ref.get('rendered_request_sha256')),
                 (item.get('producer_call_id'), producer.get('call_id')),
                 (item.get('writer_source_class'), ref.get('writer_source_class')),
                 (item.get('writer_witness'), ref.get('writer_witness')),
                 (item.get('producer_write_node'), producer.get('node')),
                 (item.get('read_node'), read.get('node')),
                 (item.get('model_turn_node'), turn.get('node'))]
        if any(left != right for left, right in pairs):
            continue
        if item.get('learner_representation') != 'typed-read-from-authenticated-runtime-writer-event-context-only':
            continue
        if (not isinstance(block_id, str) or not block_id
                or not isinstance(item.get('producer_write_node'), str) or not item.get('producer_write_node')
                or not isinstance(item.get('producer_call_id'), str) or not item.get('producer_call_id')
                or not isinstance(item.get('read_node'), str) or not item.get('read_node')
                or not isinstance(item.get('model_turn_node'), str) or not item.get('model_turn_node')
                or any(not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None
                       for value in (item.get('body_sha256'), item.get('source_row_sha256'),
                                     item.get('trace_sha256'), item.get('transport_provenance_sha256'),
                                     item.get('raw_request_sha256'), item.get('rendered_request_sha256')))):
            continue
        producer = ref.get('producer_write') or {}
        base_name = 'soft-state:' + block_id
        event_name = (direct_typed_result_event_name(block_id, row.get('source_ref', {}).get('trajectory_id'),
                                                     producer.get('call_id'), producer.get('node'))
                      if producer.get('source') == 'return_result'
                      and producer.get('source_kind') == 'typed-text-result'
                      and producer.get('result_type') == 'Neuralese<string>'
                      and isinstance(producer.get('call_id'), str)
                      and isinstance(producer.get('node'), str) else base_name)
        declared_name = item.get('target_write_name', base_name)
        if declared_name not in (base_name, event_name):
            continue
        valid.add(declared_name if declared_name == event_name else base_name)
    return valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    provider_contexts_by_source = collections.defaultdict(list)
    for path in a.inputs:
        with path.open() as f:
            for line in f:
                if not line.strip(): continue
                candidate = json.loads(line)
                source = candidate.get('source_ref', {})
                if not isinstance(source, dict): continue
                key = (source.get('trajectory_id'), source.get('source_row_sha256'))
                contexts = source.get('provider_expanded_read_contexts', [])
                if isinstance(contexts, list): provider_contexts_by_source[key].extend(contexts)
    rows, producers = {}, collections.defaultdict(list)
    semantic_producers = collections.defaultdict(list)
    invalid_semantic_receipts = []
    for path in a.inputs:
        with path.open() as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                if row['id'] in rows: raise ValueError('duplicate record id: ' + row['id'])
                target = names(row.get('target'), 'write')
                reads = names(row.get('messages'), 'read') - target
                external_candidates = reads & authenticated_external_context_names(row)
                source = row.get('source_ref', {})
                context_key = (source.get('trajectory_id'), source.get('source_row_sha256')) if isinstance(source, dict) else (None, None)
                semantic_writes, semantic_failures = validate_selected_direct_return_receipts(
                    row, provider_contexts_by_source[context_key])
                invalid_semantic_receipts.extend(semantic_failures)
                rows[row['id']] = {'reads': reads, 'external_candidates': external_candidates,
                    'external_context_roots': [],
                    'semantic_writes': semantic_writes, 'raw': row,
                    'writes': target, 'split': row.get('split', 'unspecified'),
                    'source_groups': row.get('source_groups', []),
                    'run': row.get('source_ref', {}).get('trajectory_id')}
                for name in target:
                    producers[name].append(row['id'])
                    if name in semantic_writes:
                        semantic_producers[name].append((row['id'], semantic_writes[name]))
    # A selected producer always takes precedence over context-root metadata:
    # the reader then keeps its real cohort edge and its usual split checks.
    for row in rows.values():
        external = {name for name in row['external_candidates'] if not producers.get(name)}
        row['reads'] -= external
        row['external_context_roots'] = sorted(external)
    missing, ambiguous, cross_split, cycles, event_binding_failures = [], [], [], [], []
    adjacency = collections.defaultdict(set)
    for ident,row in rows.items():
        for name in row['reads']:
            ps = producers.get(name, [])
            if not ps: missing.append({'reader': ident, 'name': name}); continue
            if len(ps) != 1: ambiguous.append({'reader': ident, 'name': name, 'producers': ps}); continue
            selected_events = semantic_producers.get(name, [])
            if selected_events:
                bound = [(producer_id, event) for producer_id, event in selected_events
                         if semantic_writer_matches_reader(event, row['raw'])
                         and name in row['external_candidates']]
                if len(bound) != 1 or bound[0][0] != ps[0]:
                    event_binding_failures.append({'reader': ident, 'producer': ps[0], 'name': name,
                        'reason': 'selected-return-event-does-not-match-authenticated-reader-event'})
                    continue
            adjacency[ident].add(ps[0])
            if row['split'] != rows[ps[0]]['split']:
                cross_split.append({'reader': ident, 'producer': ps[0], 'name': name})
    depths = {}
    def depth(node, visiting):
        if node in visiting:
            cycles.append(node); return 0
        if node not in depths:
            depths[node] = max([1 + depth(child, visiting | {node}) for child in adjacency[node]] or [0])
        return depths[node]
    for ident in rows: depth(ident, set())
    group_splits = collections.defaultdict(set)
    for row in rows.values():
        for group in row['source_groups']: group_splits[group].add(row['split'])
        if row['run']: group_splits['run:' + str(row['run'])].add(row['split'])
    mixed = {k: sorted(v) for k,v in group_splits.items() if len(v) > 1}
    consumers = collections.defaultdict(set)
    for reader, sources in adjacency.items():
        for producer in sources: consumers[producer].add(reader)
    external_roots = [{'reader': ident, 'name': name} for ident,row in rows.items()
                      for name in row['external_context_roots']]
    summary = {'schema': 'natlang.recurrence-audit/4', 'records': len(rows), 'writer_records': sum(bool(r['writes']) for r in rows.values()),
        'reader_records': sum(bool(r['reads']) for r in rows.values()), 'linked_edges': sum(map(len, adjacency.values())),
        'selected_direct_typed_return_writers': sum(len(r['semantic_writes']) for r in rows.values()),
        'authenticated_external_context_roots': external_roots,
        'depth_histogram': dict(sorted(collections.Counter(depths.values()).items())),
        'max_producers_per_consumer': max(map(len, adjacency.values()), default=0),
        'max_consumers_per_producer': max(map(len, consumers.values()), default=0),
        'degree_scope': 'distinct decision records; context rereads count as consumers, not new runtime invocations',
        'failures': {'missing_producers': missing, 'ambiguous_producers': ambiguous, 'cross_split_edges': cross_split,
                     'cycles': sorted(set(cycles)), 'mixed_split_source_groups': mixed,
                     'invalid_selected_runtime_result_writes': invalid_semantic_receipts,
                     'selected_return_event_binding_mismatches': event_binding_failures},
        'structurally_closed': not any((missing, ambiguous, cross_split, cycles, mixed,
                                        invalid_semantic_receipts, event_binding_failures)),
        'admission': 'structural-audit-only; still requires source/quality/causal-use checks'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: ({n:len(v) for n,v in x.items()} if k == 'failures' else x) for k,x in summary.items()}, indent=2))

if __name__ == '__main__': main()
