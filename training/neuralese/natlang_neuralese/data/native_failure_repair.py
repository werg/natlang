"""Admission of verified whole-graph failure/repair preference pairs.

The caller supplies graph captures already reconstructed and verified by the
native episode/replay capture path. This module only binds their identity and
builds one disjoint-union graph, preserving all internal causal edges.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex
from .native_graph_schema import SCHEMA, parse_graph

KIND = 'native-failure-repair-dpo'
PAIR_SCHEMA = 'natlang.native-failure-repair-pair/1'
OBJECTIVE_SCHEMA = 'natlang.native-failure-repair-dpo-objective/1'


def _validate_pair(record, tokenizer_fingerprint):
    if record.get('schema') != PAIR_SCHEMA or record.get('split') != 'train':
        raise ValueError('failure-repair admission accepts training pairs only')
    if record.get('protected') is not False or record.get('held_source_groups'):
        raise ValueError('protected or held failure evidence cannot enter repair training')
    binding = record.get('original_task') or {}
    required = ('task_sha256', 'world_sha256', 'oracle_sha256', 'prompt_sha256')
    if any(not isinstance(binding.get(key), str) or not binding[key] for key in required):
        raise ValueError('original unhinted task, world, oracle, source, and prompt bindings are required')
    if (not isinstance(binding.get('source_groups'), list) or not binding['source_groups']
            or any(not isinstance(group,str) or not group for group in binding['source_groups'])):
        raise ValueError('original task source groups are required')
    protection=record.get('source_protection')
    if (not isinstance(protection,dict) or protection.get('schema')!='natlang.failure-repair-source-protection/1'
            or protection.get('split')!='train' or protection.get('protected') is not False
            or protection.get('source_groups')!=binding['source_groups']
            or not isinstance(protection.get('held_source_groups'),list)
            or set(protection['held_source_groups']) & set(binding['source_groups'])
            or record.get('source_protection_sha256')!=canonical_json_sha256_hex(protection)):
        raise ValueError('source protection receipt does not establish an unheld training-only task')
    if record.get('hint_text') is not None or record.get('repair_hint') is not None:
        raise ValueError('repair hints are evidence only and may not be copied into training records')
    alternatives = record.get('alternatives')
    if not isinstance(alternatives, dict) or set(alternatives) != {'failed', 'repaired'}:
        raise ValueError('both failed and repaired whole-graph captures are required')
    reference_identities=set()
    for label, alt in alternatives.items():
        graph = alt.get('graph')
        if not isinstance(graph, dict) or graph.get('schema') != SCHEMA:
            raise ValueError(f'{label} alternative must retain its complete native graph')
        if graph.get('split') != 'train' or graph.get('source_groups') != binding['source_groups']:
            raise ValueError('alternative graph split/source groups differ from original training task')
        if graph.get('tokenizer_fingerprint') != tokenizer_fingerprint:
            raise ValueError('alternative graph tokenizer differs from the student tokenizer')
        if graph.get('external_states'):
            raise ValueError('failure repair requires the captured producer graph; detached external state fallback is forbidden')
        if alt.get('task_sha256') != binding['task_sha256'] or alt.get('world_sha256') != binding['world_sha256']:
            raise ValueError('failed and repaired executions must use the exact original task and world')
        if alt.get('oracle_sha256') != binding['oracle_sha256'] or alt.get('source_groups') != binding['source_groups']:
            raise ValueError('failed and repaired executions changed the oracle or source groups')
        if alt.get('prompt_sha256') != binding['prompt_sha256'] or alt.get('prompt_view') != 'original-unhinted':
            raise ValueError('student execution must be bound to the explicit original unhinted prompt view')
        if alt.get('hint_in_student_prompt') is not False:
            raise ValueError('hint leakage status must be explicitly false')
        if not alt.get('execution_capture_sha256') or not alt.get('graph_replay_sha256'):
            raise ValueError('exact execution and native graph replay receipts are required')
        if alt.get('complete_graph') is not True or alt.get('occurrence_bindings_verified') is not True:
            raise ValueError('whole graph and occurrence bindings must be verified')
        actor = alt.get('actor') or {}
        if not all(isinstance(actor.get(k), str) and actor[k] for k in ('checkpoint_sha256','runtime_sha256','actor_id')):
            raise ValueError('each student execution needs exact checkpoint/runtime/actor provenance')
        reference = alt.get('reference_score') or {}
        if not all(isinstance(reference.get(k), str) and reference[k] for k in
                   ('checkpoint_sha256','tokenizer_fingerprint','runtime_sha256','graph_action_sha256','receipt_sha256')):
            raise ValueError('fixed reference score receipt lacks checkpoint/tokenizer/runtime/action identity')
        if reference.get('tokenizer_fingerprint') != tokenizer_fingerprint:
            raise ValueError('reference tokenizer differs from the graph tokenizer')
        if reference.get('normalization') != 'token-mean-v1':
            raise ValueError('reference policy scores require explicit token-mean normalization')
        reference_identities.add(tuple(reference[k] for k in
            ('checkpoint_sha256','tokenizer_fingerprint','runtime_sha256')))
        actions = alt.get('actions')
        if not isinstance(actions, list) or not actions:
            raise ValueError('every captured sampled/forced action must be explicitly scored')
        by_name = {node['name']: node for node in graph.get('nodes') or []}
        for action in actions:
            node = by_name.get(action.get('node'))
            pos, token = action.get('position'), action.get('token_id')
            if (node is None or type(pos) is not int or not 0 < pos < len(node['token_ids'])
                    or type(token) is not int or node['token_ids'][pos] != token
                    or action.get('kind') not in ('sampled','forced')
                    or type(action.get('reference_logp')) not in (int,float)
                    or action['reference_logp'] > 1e-6):
                raise ValueError('action capture or fixed reference log probability is malformed')
        capture_receipt=alt.get('execution_receipt')
        actor=alt.get('actor') or {}
        if (not isinstance(capture_receipt,dict)
                or capture_receipt.get('schema')!='natlang.native-failure-execution-capture/1'
                or capture_receipt.get('task_sha256')!=binding['task_sha256']
                or capture_receipt.get('world_sha256')!=binding['world_sha256']
                or capture_receipt.get('prompt_sha256')!=binding['prompt_sha256']
                or capture_receipt.get('actions')!=actions
                or capture_receipt.get('actor')!=actor
                or canonical_json_sha256_hex(capture_receipt)!=alt.get('execution_capture_sha256')):
            raise ValueError('full sampled/forced action capture receipt does not match this exact unhinted task')
        replay_receipt=alt.get('graph_replay_receipt')
        if (not isinstance(replay_receipt,dict)
                or replay_receipt.get('schema')!='natlang.native-graph-replay-receipt/1'
                or replay_receipt.get('graph_sha256')!=canonical_json_sha256_hex(graph)
                or replay_receipt.get('occurrence_bindings_verified') is not True
                or replay_receipt.get('execution_capture_sha256')!=alt['execution_capture_sha256']
                or canonical_json_sha256_hex(replay_receipt)!=alt.get('graph_replay_sha256')):
            raise ValueError('native graph replay receipt does not bind the complete captured occurrences')
        if reference['graph_action_sha256'] != canonical_json_sha256_hex(actions):
            raise ValueError('reference scores are not bound to the exact action occurrence inventory')
        receipt=reference.get('receipt')
        if (not isinstance(receipt,dict) or receipt.get('schema')!='natlang.native-fixed-reference-scores/1'
                or receipt.get('actions')!=actions
                or any(receipt.get(k)!=reference.get(k) for k in
                    ('checkpoint_sha256','tokenizer_fingerprint','runtime_sha256','graph_action_sha256','normalization'))
                or reference.get('receipt_sha256')!=canonical_json_sha256_hex(receipt)):
            raise ValueError('fixed reference score receipt content or digest does not match')
        parsed = parse_graph(graph, tokenizer_fingerprint=tokenizer_fingerprint)
        if not parsed.plan.invocations:
            raise ValueError('empty captured native graph')
    if len(reference_identities)!=1:
        raise ValueError('failed and repaired scores must use the same fixed reference policy')
    return binding, alternatives


def _merge_graphs(record, tokenizer_fingerprint):
    binding, alternatives = _validate_pair(record, tokenizer_fingerprint)
    merged_nodes, merged_external, suffixes, counts = [], {}, {}, {}
    graph_hashes = {}
    for label in ('failed','repaired'):
        alt = alternatives[label]; graph = copy.deepcopy(alt['graph'])
        prefix = label + '/'
        name_map = {node['name']: prefix + node['name'] for node in graph['nodes']}
        ext_map = {name: prefix + name for name in (graph.get('external_states') or {})}
        for name, value in (graph.get('external_states') or {}).items():
            merged_external[ext_map[name]] = value
        action_map = {(action['node'], action['position']): action for action in alt['actions']}
        actions = []
        for node in graph['nodes']:
            node['name'] = name_map[node['name']]
            for edge in node.get('imports', []):
                edge['source'] = name_map.get(edge['source'], ext_map.get(edge['source'], edge['source']))
            for index, token in enumerate(node['token_ids']):
                node['target_weights'][index] = 0.
            actions.extend(action for (old_name, _), action in action_map.items() if old_name == node['name'][len(prefix):])
            merged_nodes.append(node)
        actions = sorted(actions, key=lambda a: (a['node'], a['position']))
        if len(actions) != len(alt['actions']):
            raise ValueError('an action occurrence is missing from its full graph')
        suffixes[label] = [{'node': name_map[a['node']], 'position': a['position'],
                            'token_id': a['token_id'], 'reference_logp': float(a['reference_logp']),
                            'kind': a['kind']} for a in actions]
        counts[label] = len(actions)
        graph_hashes[label] = canonical_json_sha256_hex(graph)
    identity = 'failure-repair-dpo:' + canonical_json_sha256_hex({
        'record_sha256': canonical_json_sha256_hex(record), 'graph_hashes': graph_hashes})
    row = {'schema': SCHEMA, 'id': identity, 'split': 'train',
           'source_groups': binding['source_groups'], 'tokenizer_fingerprint': tokenizer_fingerprint,
           'nodes': merged_nodes, 'external_states': merged_external,
           'required_objective': KIND,
           'conversion': {'kind': KIND, 'offline': True, 'native_on_policy': False,
               'original_task_sha256': binding['task_sha256'], 'original_prompt_sha256': binding['prompt_sha256'],
               'graph_hashes': graph_hashes, 'complete_alternatives': ['failed','repaired'],
               'ordinary_ce_targets': 0}}
    parsed = parse_graph(row, tokenizer_fingerprint=tokenizer_fingerprint)
    from ..rl.native_replay import graph_identity
    objective_row = {'graph_sha256': canonical_json_sha256_hex(graph_identity(parsed)),
        'split':'train', 'source_groups': binding['source_groups'], 'suffixes':suffixes,
        'token_counts':counts, 'source_record_sha256':canonical_json_sha256_hex(record),
        'reference': {label: {k: alternatives[label]['reference_score'][k] for k in
             ('checkpoint_sha256','tokenizer_fingerprint','runtime_sha256','graph_action_sha256','receipt_sha256','normalization','receipt')}
             for label in ('failed','repaired')},
        'task_binding': {k:binding[k] for k in ('task_sha256','world_sha256','oracle_sha256','prompt_sha256','source_groups')}}
    return row, objective_row


def admit_jsonl(input_path, output_dir, *, tokenizer_fingerprint):
    """Convert verified pair receipts into a content-addressed graph/objective source."""
    from ..common.jsonio import open_jsonl_text, write_json_durable, write_canonical_jsonl
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError('failure-repair admission output directory must be empty and immutable')
    graph_path = output_dir / 'repair-graphs.jsonl'; objective_path = output_dir / 'repair-objective.json'
    graph_rows, objective_rows = [], {}
    with open_jsonl_text(input_path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip(): continue
            try: record = json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f'invalid pair receipt at line {line_number}') from exc
            graph, objective = _merge_graphs(record, tokenizer_fingerprint)
            if graph['id'] in objective_rows: raise ValueError('duplicate repair pair identity')
            graph_rows.append(graph); objective_rows[graph['id']] = objective
    if not graph_rows: raise ValueError('no verified failure-repair pairs to admit')
    write_canonical_jsonl(graph_path,graph_rows)
    objective = {'schema':OBJECTIVE_SCHEMA,'kind':KIND,'beta':0.1,'normalization':'token-mean-v1',
                 'rows':objective_rows}
    write_json_durable(objective_path, objective)
    manifest_path = output_dir / 'repair-sources.json'
    write_json_durable(manifest_path, {'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs','path':graph_path.name,'sha256':sha256_file_hex(graph_path)}]})
    return {'graphs':str(graph_path),'objective':str(objective_path),'sources':str(manifest_path),
            'pairs':len(graph_rows)}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Admit verified whole-graph failure-repair DPO pairs')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--tokenizer-fingerprint', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(admit_jsonl(args.input, args.output_dir,
        tokenizer_fingerprint=args.tokenizer_fingerprint), sort_keys=True))


if __name__ == '__main__': raise SystemExit(main())
