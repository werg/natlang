"""Build verified whole-graph failure/repair preference pairs from run captures.

The executable path reads actual task-rollout artifacts, recomputes outcomes,
replays the native captures, scores the fixed reference policy, and emits
content-bound graph/objective/recipe inputs. The merge helper preserves the
internal causal edges of both alternatives.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex
from .native_graph_schema import SCHEMA, parse_graph

KIND = 'native-failure-repair-dpo'
PAIR_SCHEMA = 'natlang.native-failure-repair-pair/1'
OBJECTIVE_SCHEMA = 'natlang.native-failure-repair-dpo-objective/1'


def _validate_pair(record, tokenizer_fingerprint, *, protected_groups):
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
    if set(binding['source_groups']) & set(protected_groups):
        raise ValueError('task source groups intersect the loaded protected-source manifests')
    oracle=record.get('oracle_evidence') or {}
    if set(oracle)!={'failed','repaired'}:
        raise ValueError('actual failed and repaired oracle evaluation receipts are required')
    failed=oracle['failed'];repaired=oracle['repaired']
    if (failed.get('outcome') not in {'invalid_action','repetition','budget_exhaustion','model_incomplete','semantic_mismatch'}
            or repaired.get('outcome')!='task_success'
            or any(value.get('task_sha256')!=binding['task_sha256']
                   or value.get('world_sha256')!=binding['world_sha256']
                   or value.get('oracle_sha256')!=binding['oracle_sha256']
                   or value.get('source_groups')!=binding['source_groups']
                   or not value.get('packet_sha256') or not value.get('result_sha256')
                   or not isinstance(value.get('evaluation_receipt'),dict)
                   or value.get('evaluation_receipt_sha256')!=canonical_json_sha256_hex(value['evaluation_receipt'])
                   or value['evaluation_receipt'].get('outcome')!=value.get('outcome')
                   or value['evaluation_receipt'].get('task_sha256')!=value.get('task_sha256')
                   or value['evaluation_receipt'].get('world_sha256')!=value.get('world_sha256')
                   or value['evaluation_receipt'].get('oracle_sha256')!=value.get('oracle_sha256')
                   or value['evaluation_receipt'].get('source_groups')!=value.get('source_groups')
                   or value['evaluation_receipt'].get('packet_sha256')!=value.get('packet_sha256')
                   or value['evaluation_receipt'].get('result_sha256')!=value.get('result_sha256')
                   or value['evaluation_receipt'].get('collection_binding_sha256')!=value.get('collection_binding_sha256')
                   or not value['evaluation_receipt'].get('evaluator_sha256')
                   for value in (failed,repaired))):
        raise ValueError('oracle receipts do not prove a task-caused failure and exact repaired success on the same task')
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
                    or not math.isfinite(action['reference_logp']) or action['reference_logp'] > 1e-6):
                raise ValueError('action capture or fixed reference log probability is malformed')
        capture_receipt=alt.get('execution_receipt')
        actor=alt.get('actor') or {}
        capture_actions=[{key:action.get(key) for key in ('node','position','token_id','kind')}
            for action in actions]
        if (not isinstance(capture_receipt,dict)
                or capture_receipt.get('schema')!='natlang.native-failure-execution-capture/1'
                or capture_receipt.get('task_sha256')!=binding['task_sha256']
                or capture_receipt.get('world_sha256')!=binding['world_sha256']
                or capture_receipt.get('prompt_sha256')!=binding['prompt_sha256']
                or capture_receipt.get('actions')!=capture_actions
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


def _merge_graphs(record, tokenizer_fingerprint, *, protected_groups=frozenset()):
    binding, alternatives = _validate_pair(record, tokenizer_fingerprint,protected_groups=protected_groups)
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


def _write_admitted_rows(records, output_dir, *, tokenizer_fingerprint, protected_groups,
                         stage_inputs=None):
    """Write pairs assembled by the verified run/replay path, never hand-authored records."""
    from ..common.jsonio import write_json_durable, write_canonical_jsonl
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError('failure-repair admission output directory must be empty and immutable')
    graph_path = output_dir / 'repair-graphs.jsonl'; objective_path = output_dir / 'repair-objective.json'
    graph_rows, objective_rows = [], {}
    for record in records:
        graph, objective = _merge_graphs(record, tokenizer_fingerprint,protected_groups=protected_groups)
        if graph['id'] in objective_rows: raise ValueError('duplicate repair pair identity')
        graph_rows.append(graph); objective_rows[graph['id']] = objective
    if not graph_rows: raise ValueError('no verified failure-repair pairs to admit')
    write_canonical_jsonl(graph_path,graph_rows)
    objective = {'schema':OBJECTIVE_SCHEMA,'kind':KIND,'beta':float(stage_inputs['beta']) if stage_inputs else 0.1,'normalization':'token-mean-v1',
                 'rows':objective_rows}
    write_json_durable(objective_path, objective)
    manifest_path = output_dir / 'repair-sources.json'
    write_json_durable(manifest_path, {'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs','path':graph_path.name,'sha256':sha256_file_hex(graph_path)}]})
    result={'graphs':str(graph_path),'objective':str(objective_path),'sources':str(manifest_path),
            'pairs':len(graph_rows)}
    if stage_inputs is not None:
        from ..common.hashing import sha256_file_hex_cached
        stage_id=stage_inputs['stage_id'];bindings={}
        def bind(role,path):
            path=Path(path).resolve(strict=True);key='failure_repair_'+role
            bindings[key]={'path':str(path),'sha256':sha256_file_hex_cached(path)}
            return key
        binding_names={
            'native_graphs':bind('graphs',manifest_path),
            'objective_spec':bind('objective',objective_path),
            'held_native_graphs':bind('held',stage_inputs['held_native_graphs']),
            'continue_from':bind('parent_checkpoint',stage_inputs['continue_from'])}
        stage={'id':stage_id,'kind':'native_offline_preference',
            'requires':list(stage_inputs.get('requires') or []),'inputs':binding_names,
            'parameters':{'preference_beta':stage_inputs['beta'],'preference_margin':0}}
        addition={'schema':'natlang.native-failure-repair-recipe-addition/1',
            'training_admission':'pending-normal-recipe-source-and-custody-validation',
            'input_bindings':bindings,'stage':stage,
            'continuation':{'objective_continuation':True,'weights_optimizer_rng_exposure':'preserved',
                'append_policy':'exact prior graph/objective row prefix; new pairs receive their own exposure visits'},
            'objective':{'kind':'native-failure-repair-dpo','beta':stage_inputs['beta'],
                'reference_normalization':'token-mean-v1','protected_failures_admitted':False}}
        addition_path=output_dir/'recipe-stage-addition.json';write_json_durable(addition_path,addition)
        result['recipe_stage_addition']=str(addition_path)
    return result


def _run_member(run_dir, task_id, sample):
    """Recover one task result from the real collector run and recompute its oracle class."""
    import copy
    from ..common.hashing import sha256_file_hex_cached
    from ..eval.task_packet import classify_case,read_jsonl
    from ..rl.task_rollouts import reward_of,rollout_turns,_budget_terminated_prefix
    run_dir=Path(run_dir).resolve(strict=True)
    preparation=json.loads((run_dir/'preparation.json').read_text())
    packet_path=run_dir/'packet.jsonl';packet=read_jsonl(packet_path)
    packet_sha=sha256_file_hex_cached(packet_path)
    if preparation.get('schema')!='natlang.task-rollout-preparation/1' or preparation.get('packet_sha256')!=packet_sha:
        raise ValueError('task rollout run lost its exact prepared packet binding')
    queue=read_jsonl(run_dir/'queue.jsonl')
    journal_paths=[run_dir/'journal.jsonl',*sorted((run_dir/'workers').glob('journal-*.jsonl'))]
    journals=[read_jsonl(path) for path in journal_paths if path.is_file()]
    finishes={row['key']:row for journal in journals for row in journal if row.get('event')=='finish'}
    candidates=[]
    for entry in queue:
        index=entry.get('index')
        if type(index)is int and 0<=index<len(packet) and packet[index].get('id')==task_id and entry.get('sample')==sample:
            candidates.append((entry,packet[index]))
    if len(candidates)!=1:raise ValueError('collector run must contain exactly one requested task/sample')
    entry,task=candidates[0];result_path=Path(entry['output']).resolve()
    results=read_jsonl(result_path) if result_path.is_file() else []
    finish=finishes.get(entry['key']);partial_evidence=None
    if len(results)==1:
        result=results[0]
    elif not results:
        prefix=_budget_terminated_prefix(entry,task,finish,packet_source_path=packet_path,
            packet_source_sha256=packet_sha)
        if prefix is None:
            reason=(finish or {}).get('resource_limit_reason') or (finish or {}).get('status') or 'missing terminal receipt'
            raise ValueError(f'zero-result failed task is not a complete task_rollouts model-budget prefix ({reason})')
        partial_evidence=prefix['evidence']
        result={'version':'natlang.teacher_trajectory.native/1','task':{'program_ir':task},
            'outcome':{'status':'failed','accepted':False,'detail':partial_evidence['failure_reason'],
                'execution_graph':prefix['execution_graph']},'trajectory':prefix['trajectory']}
    else:raise ValueError('requested task run has multiple result rows')
    if (result.get('task',{}).get('program_ir')!=task or finish is None
            or result.get('outcome',{}).get('status') not in ('done','failed','quiesced')):
        raise ValueError('collector result, terminal journal and exact task packet disagree')
    teacher_forced=[]
    for turn_ordinal,turn in enumerate(result.get('trajectory') or []):
        capture=((turn.get('model_response') or {}).get('rl_sampling') or {})
        if capture.get('schema')!='natlang.native-teacher-forced-capture/1':continue
        source_receipt=capture.get('teacher_source_receipt')
        verified_source=_verify_teacher_turn_source(source_receipt)
        expected_prompt=canonical_json_sha256_hex({'messages':turn.get('context'),'tools':turn.get('tools_offered') or []})
        if capture.get('replay_prompt_sha256')!=expected_prompt:
            raise ValueError('Qwen forced target did not execute against the actual original unhinted student request')
        teacher_forced.append({'source_receipt':source_receipt,'source_row_id':source_receipt['row_id'],
            'source_program_ir_sha256':verified_source['program_ir_sha256'],'replay_prompt_sha256':expected_prompt,
            'turn_ordinal':turn_ordinal})
    binding_path,expected_identity,runtime_sha,collector_runtime_sha=_collector_actor_binding(run_dir,preparation)
    for step in result.get('trajectory') or []:
        capture=((step.get('model_response') or {}).get('rl_sampling') or {})
        policy=capture.get('policy')
        if policy is not None and policy!=expected_identity:
            raise ValueError('captured native policy identity differs from exact collection binding')
    outcome=classify_case(result,finish);reward=reward_of(outcome,finish)
    if reward not in (0.,1.):raise ValueError('infrastructure, unresolved, or unscored task outcome cannot be DPO evidence')
    rollout={'schema':'natlang.rollout/1','task_id':task_id,'split':task.get('split'),
        'group':entry['group'],'sample':sample,'policy':{'checkpoint_sha256':preparation.get('policy_checkpoint_sha256')},
        'turns':rollout_turns(result),
        'execution_graph':copy.deepcopy((result.get('outcome') or {}).get('execution_graph') or []),
        'outcome':outcome,'reward':{'quality':reward}}
    captured_policies=[(turn.get('sampling') or {}).get('policy') for turn in rollout['turns']]
    if not captured_policies or any(not isinstance(value,dict) or
            value.get('checkpoint_sha256')!=preparation.get('policy_checkpoint_sha256')
            for value in captured_policies):
        raise ValueError('actual captured actor identity differs from the exact run preparation checkpoint')
    result_sha=(canonical_json_sha256_hex({'schema':'natlang.task-rollout-budget-prefix-result/1',
        'partial_evidence':partial_evidence,'finish':finish}) if partial_evidence is not None else
        sha256_file_hex_cached(result_path))
    evidence={'outcome':outcome,'task_sha256':canonical_json_sha256_hex(task),
        'world_sha256':canonical_json_sha256_hex(task.get('world',task.get('environment',task))),
        'oracle_sha256':canonical_json_sha256_hex(task.get('oracle',task.get('semantics',task))),
        'source_groups':sorted(set(task.get('source_groups') or [])),'packet_sha256':packet_sha,
        'result_sha256':result_sha,**({'result_path':str(result_path)} if partial_evidence is None else
            {'partial_evidence':partial_evidence}),
        'finish_sha256':canonical_json_sha256_hex(finish),
        'collection_binding_sha256':sha256_file_hex_cached(binding_path),
        'actor_runtime_sha256':runtime_sha,'collector_runtime_inventory_sha256':collector_runtime_sha}
    if teacher_forced:evidence['teacher_forced_sources']=teacher_forced
    evaluator=Path(__file__).resolve().parents[1]/'eval/task_packet.py'
    evidence['evaluation_receipt']={'schema':'natlang.task-oracle-evaluation/1',
        **{key:evidence[key] for key in ('outcome','task_sha256','world_sha256','oracle_sha256',
            'source_groups','packet_sha256','result_sha256','finish_sha256')},
        'task_id':task_id,'evaluator_sha256':sha256_file_hex_cached(evaluator),
        'collection_binding_sha256':evidence['collection_binding_sha256']}
    evidence['evaluation_receipt_sha256']=canonical_json_sha256_hex(evidence['evaluation_receipt'])
    return task,rollout,evidence


def _collector_actor_binding(run_dir,preparation):
    """Read the exact native actor/runtime receipt sealed by this collector round."""
    from ..common.hashing import sha256_file_hex_cached
    path=Path(run_dir)/'collection-binding.json';value=json.loads(path.read_text())
    expected=value.get('expected_native_actor') or {};identity=expected.get('native_identity') or {}
    runtime=expected.get('python_runtime') or {};digest=runtime.get('sha256')
    collector_runtime=value.get('runtime')
    if (value.get('schema')!='natlang.rl-collection-binding/1'
            or value.get('policy_id')!=preparation.get('policy_checkpoint_sha256')
            or identity.get('checkpoint_sha256')!=preparation.get('policy_checkpoint_sha256')
            or runtime.get('schema')!='natlang.python-runtime/1'
            or type(runtime.get('files')) is not int or runtime['files']<1
            or not isinstance(digest,str) or len(digest)!=64
            or any(char not in '0123456789abcdef' for char in digest)
            or not isinstance(value.get('collector_sha256'),str) or len(value['collector_sha256'])!=64
            or any(char not in '0123456789abcdef' for char in value['collector_sha256'])
            or not isinstance(collector_runtime,dict) or not collector_runtime
            or any(not isinstance(name,str) or not name.endswith('.js') or name.startswith('/') or '..' in Path(name).parts
                or not isinstance(digest,str) or len(digest)!=64 or any(char not in '0123456789abcdef' for char in digest)
                for name,digest in collector_runtime.items())):
        raise ValueError('collector binding lacks its exact native actor/runtime receipt')
    return path,identity,digest,canonical_json_sha256_hex(collector_runtime)


def _replay_rollout(rollout, *, identity, roles, fetch_block):
    from ..rl.episode_graph import whole_episode_turns
    from ..rl.native_replay import compile_replay,capture_invocation_actors
    turns=whole_episode_turns(rollout)
    schemas={turn['sampling'].get('schema') for turn in turns}
    structural=bool(schemas & {'natlang.native-structural-capture/1','natlang.native-structural-capture/2',
        'natlang.native-teacher-forced-capture/1'})
    actors=capture_invocation_actors(turns,identity)
    replay=compile_replay(turns,identity=identity,roles=roles,fetch_block=fetch_block,
        invocation_actors=actors,structural=structural)
    return replay,actors


def _graph_from_replay(rollout,replay,*,tokenizer_fingerprint,source_groups):
    from ..data.native_graph_schema import SCHEMA,parse_graph
    from ..model.native_graph_plan import invocation_dict
    from ..rl.native_replay import graph_identity
    from ..serve.store import native_state_id
    nodes=[];offsets={}
    for node,offset in zip(replay.plan.invocations,replay.plan.offsets):
        value=invocation_dict(node);value['token_ids']=list(node.token_ids)
        value['target_weights']=[0.]*len(node.token_ids);nodes.append(value);offsets[node.name]=offset
    external={name:{'id':native_state_id(block),'token_count':len(block.token_ids)}
        for name,block in replay.external_blocks.items()}
    row={'schema':SCHEMA,'id':'failure-alt:'+canonical_json_sha256_hex(rollout),
        'split':'train','source_groups':list(source_groups),'tokenizer_fingerprint':tokenizer_fingerprint,
        'nodes':nodes,'external_states':external}
    parsed=parse_graph(row,tokenizer_fingerprint=tokenizer_fingerprint)
    actions=[]
    for item in replay.input_actions:
        name=item['turn'];node=next(value for value in replay.plan.invocations if value.name==name)
        position=item['query']-offsets[name]+1
        if not 0<position<len(node.token_ids) or node.token_ids[position]!=item['token_id']:
            raise ValueError('native replay action occurrence is outside its exact full graph node')
        actions.append({'node':name,'position':position,'token_id':item['token_id'],'kind':item['kind']})
    return row,parsed,actions


def _score_reference(model,example,actions,*,reference_identity,runtime_sha256,chunk_tokens=128):
    import torch
    from ..rl.native_replay import action_logps,graph_identity
    if example.external_states:
        raise ValueError('failure DPO scoring requires complete internal producer graphs; external detached state is unsupported')
    model_actions=[{'query':example.plan.offsets[next(i for i,node in enumerate(example.plan.invocations)
        if node.name==item['node'])]+item['position']-1,'token_id':item['token_id'],'temperature':1.,
        'allowed_token_ids':None,'banned_token_ids':None,'behavior_logp':0.} for item in actions]
    with torch.no_grad():
        result=model(example.plan,retain_states=False)
        logps,_=action_logps(model,result['h_final'][0],model_actions,chunk_tokens=chunk_tokens)
    if len(actions)!=len(logps) or not torch.isfinite(logps).all():
        raise ValueError('reference model produced incomplete or nonfinite exact action scores')
    scores=[float(value) for value in logps.detach().cpu()]
    bound_actions=[{**item,'reference_logp':value} for item,value in zip(actions,scores)]
    action_sha=canonical_json_sha256_hex(bound_actions)
    receipt={'schema':'natlang.native-fixed-reference-scores/1',**reference_identity,
        'runtime_sha256':runtime_sha256,'graph_sha256':canonical_json_sha256_hex(graph_identity(example)),
        'graph_action_sha256':action_sha,'normalization':'token-mean-v1','actions':bound_actions}
    return bound_actions,receipt


def _verify_teacher_turn_source(receipt):
    """Authenticate a Qwen forced-turn source against its retained tranche and actor proof files."""
    from ..common.hashing import sha256_file_hex_cached
    from ..eval.task_packet import read_jsonl
    if (not isinstance(receipt,dict) or receipt.get('schema')!='natlang.teacher-turn-source/1'
            or receipt.get('source_view') not in ('unhinted','privileged-hint')
            or type(receipt.get('ordinal')) is not int or receipt['ordinal']<0):
        raise ValueError('forced repair capture lacks an explicit teacher turn source view and ordinal')
    path_keys=('source_path','source_verification_path','transfer_receipt_path')
    hash_keys=('source_file_sha256','source_verification_sha256','transfer_receipt_sha256')
    actor=receipt.get('actor') or {};actor_path_keys=('actor_source_path','execution_identity_path')
    actor_hash_keys=('actor_source_sha256','execution_identity_sha256')
    if (any(not isinstance(receipt.get(key),str) or not receipt[key] for key in path_keys+actor_path_keys)
            or any(not isinstance(receipt.get(key),str) or len(receipt[key])!=64 for key in hash_keys+actor_hash_keys)
            or actor.get('model')!='nvidia/Qwen3.6-35B-A3B-NVFP4'
            or actor.get('revision')!='1355db6a052410cfd62085d94b58866fd0f2c3c5'):
        raise ValueError('teacher source receipt is missing selected Qwen source, actor, or proof bindings')
    paths={key:Path(receipt[key]).resolve(strict=True) for key in (*path_keys,*actor_path_keys)}
    expected={**dict(zip(path_keys,hash_keys)),
        **dict(zip(actor_path_keys,actor_hash_keys))}
    for path_key,hash_key in expected.items():
        if sha256_file_hex_cached(paths[path_key])!=receipt[hash_key]:
            raise ValueError(f'teacher source proof bytes changed: {path_key}')
    source_proof=json.loads(paths['source_verification_path'].read_text())
    transfer=json.loads(paths['transfer_receipt_path'].read_text())
    actor_doc=json.loads(paths['actor_source_path'].read_text())
    execution=json.loads(paths['execution_identity_path'].read_text())
    if (source_proof.get('schema')!='natlang.teacher-live-tranche-canonical-source-verification/1'
            or source_proof.get('status')!='passed' or source_proof.get('training_admission') is not False
            or source_proof.get('source',{}).get('path')!=str(paths['source_path'])
            or source_proof.get('source',{}).get('sha256')!=receipt['source_file_sha256']
            or source_proof.get('actor',{}).get('sha256')!=receipt['actor']['actor_source_sha256']
            or source_proof.get('execution_identity',{}).get('sha256')!=receipt['actor']['execution_identity_sha256']
            or transfer.get('schema')!='natlang.verified-teacher-trajectory-transfer/1'):
        raise ValueError('Qwen source verification and transfer manifests do not bind this forced-turn receipt')
    transfer_files=transfer.get('files') or []
    expected_files={str(paths['actor_source_path']):receipt['actor']['actor_source_sha256'],
        str(paths['execution_identity_path']):receipt['actor']['execution_identity_sha256']}
    if any(not any(item.get('path')==path and item.get('sha256')==digest for item in transfer_files)
            for path,digest in expected_files.items()):
        raise ValueError('Qwen actor proof paths are not present in the retained transfer receipt')
    if (actor_doc.get('schema')!='natlang.selected-teacher-actor-source/1'
            or actor_doc.get('model')!=receipt['actor']['model'] or actor_doc.get('revision')!=receipt['actor']['revision']
            or execution.get('schema')!='natlang.self-hosted-teacher-execution-identity/1'
            or execution.get('model_id')!='qwen3.6-35b-a3b'
            or execution.get('actor',{}).get('sha256')!=receipt['actor']['actor_source_sha256']
            or transfer.get('teacher',{}).get('model')!=actor_doc.get('model')
            or transfer.get('teacher',{}).get('revision')!=actor_doc.get('revision')):
        raise ValueError('Qwen forced turn was not produced by the selected self-hosted teacher actor')
    rows=read_jsonl(str(paths['source_path']))
    candidates=[row for row in rows if row.get('id')==receipt.get('row_id')]
    if len(candidates)!=1:raise ValueError('Qwen source row is missing or ambiguous in the verified tranche')
    row=candidates[0]
    if (canonical_json_sha256_hex(row.get('task'))!=receipt.get('task_sha256')
            or canonical_json_sha256_hex((row.get('task') or {}).get('program_ir'))!=receipt.get('program_ir_sha256')
            or (row.get('provenance') or {}).get('model')!='qwen3.6-35b-a3b'
            or (row.get('outcome') or {}).get('accepted') is not True):
        raise ValueError('Qwen source row task, actor, or verified outcome differs from its turn receipt')
    turns=[turn for turn in row.get('trajectory') or [] if turn.get('invocation_id')==receipt.get('invocation_id')]
    ordinal=receipt['ordinal']
    if ordinal>=len(turns):raise ValueError('Qwen teacher source invocation/turn ordinal is outside its trajectory')
    turn=turns[ordinal]
    prompt={'messages':turn.get('context'),'tools':turn.get('tools_offered') or []}
    if (canonical_json_sha256_hex(turn)!=receipt.get('turn_sha256')
            or canonical_json_sha256_hex(prompt)!=receipt.get('prompt_sha256')):
        raise ValueError('Qwen teacher source turn or its teacher-visible prompt changed')
    return {'row':row,'turn':turn,'source_view':receipt['source_view'],
        'program_ir_sha256':receipt['program_ir_sha256']}


def admit_run_pair(*,failed_run_dir,repaired_run_dir,task_id,failed_sample,repaired_sample,reference_checkpoint,
                   base,held_native_graphs,failed_block_bundle=None,repaired_block_bundle=None,output_dir,continue_from,stage_id,
                   requires=(),device='cpu',beta=.1,_context=None):
    """Build DPO data from actual result files, replay captures and current held-source manifests."""
    import torch
    from ..common.hashing import sha256_file_hex_cached
    from ..model.native_model import load_native_checkpoint,native_identity
    from ..data.text_corpus import tokenizer_fingerprint
    from ..data.native_sources import NativeGraphSource
    from ..model.shared_state import RoleTokens
    from ..rl.graph_replay import NativeBundleStore
    from ..common.hashing import canonical_json_sha256_hex
    task_a,failed,failed_evidence=_run_member(failed_run_dir,task_id,failed_sample)
    task_b,repaired,repaired_evidence=_run_member(repaired_run_dir,task_id,repaired_sample)
    if canonical_json_sha256_hex(task_a)!=canonical_json_sha256_hex(task_b):
        raise ValueError('failed and repaired task packets differ in source, world, or oracle')
    for source in repaired_evidence.get('teacher_forced_sources',[]):
        if source.get('source_program_ir_sha256')!=canonical_json_sha256_hex(task_a):
            raise ValueError('Qwen correction row is not bound to this exact original task packet')
    if failed_evidence['outcome'] not in {'invalid_action','repetition','budget_exhaustion','model_incomplete','semantic_mismatch'}:
        raise ValueError('original case did not have an exact task-caused failure outcome')
    if repaired_evidence['outcome']!='task_success':
        raise ValueError('corrected execution does not pass the exact task packet oracle')
    shared_context=_context is not None
    if _context is None:
        model,tokenizer,roles,config,_=load_native_checkpoint(reference_checkpoint,base=base,device=device,allow_weights_only=True)
        model.eval();fingerprint=tokenizer_fingerprint(tokenizer)
        held=NativeGraphSource(held_native_graphs,fingerprint,required_split='test',role_ids=roles.ids)
        from scripts.source_review import pending_unpinned_source_groups,pending_family_hold_reason
        protected=set(held.groups)|pending_unpinned_source_groups()
        reference_id=native_identity(model,config,checkpoint_sha256=sha256_file_hex_cached(reference_checkpoint))
        _context={'model':model,'tokenizer':tokenizer,'roles':roles,'config':config,'fingerprint':fingerprint,
            'held':held,'protected':protected,'reference_id':reference_id}
    else:
        model=_context['model'];tokenizer=_context['tokenizer'];roles=_context['roles'];config=_context['config']
        fingerprint=_context['fingerprint'];held=_context['held'];protected=_context['protected'];reference_id=_context['reference_id']
    if config['tokenizer_fingerprint']!=fingerprint:
        raise ValueError('loaded reference checkpoint/tokenizer fingerprint mismatch')
    from scripts.source_review import pending_family_hold_reason
    task_groups=set(task_a.get('source_groups') or [])
    if not task_groups or task_groups & protected or pending_family_hold_reason(task_a):
        raise ValueError('source task intersects actual held or source-review protection')
    if failed['split']!='train' or repaired['split']!='train':raise ValueError('only original train tasks can enter repair DPO')
    captured_policies=[turn['sampling']['policy'] for rollout in (failed,repaired) for turn in rollout['turns']]
    if not captured_policies or any(value['tokenizer_fingerprint']!=fingerprint for value in captured_policies):
        raise ValueError('both complete captures must be explicitly replayable in the reference tokenizer')
    architectures={value.get('architecture') for value in captured_policies}
    if len(architectures)!=1 or None in architectures:
        raise ValueError('failed and repaired captures must use the same declared native model architecture')
    # The capture runtime actor can differ from the fixed DPO reference policy;
    # replay is bound to the exact actual invocation identity in each capture.
    bundles={
        'failed':NativeBundleStore(failed_block_bundle,policy_id=failed['turns'][0]['sampling']['policy']['checkpoint_sha256']) if failed_block_bundle else None,
        'repaired':NativeBundleStore(repaired_block_bundle,policy_id=repaired['turns'][0]['sampling']['policy']['checkpoint_sha256']) if repaired_block_bundle else None,
    }
    replays=[]
    for label,rollout in (('failed',failed),('repaired',repaired)):
        identity=rollout['turns'][0]['sampling']['policy']
        bundle=bundles[label]
        def fetch_block(block_id,bundle=bundle,role=label):
            if bundle is None:raise ValueError(f'{role} native graph requires its exact actor block bundle')
            return bundle.state(block_id)
        replay,actors=_replay_rollout(rollout,identity=identity,roles=roles,fetch_block=fetch_block)
        if replay.external_blocks:
            raise ValueError('failure repair capture is incomplete: every imported producer must be in the captured graph')
        graph,example,actions=_graph_from_replay(rollout,replay,tokenizer_fingerprint=fingerprint,source_groups=task_groups)
        replays.append((graph,example,actions,actors))
    runtime_files=['rl/native_replay.py','rl/episode_graph.py','model/native_graph_plan.py','model/shared_state.py',
        'serve/native_protocol.py','data/native_failure_repair.py','train/native_preference.py']
    package=Path(__file__).resolve().parents[1]
    runtime_sha=canonical_json_sha256_hex({name:sha256_file_hex_cached(package/name) for name in runtime_files})
    alternatives={}
    for label,rollout,evidence,(graph,example,actions,actors) in zip(
            ('failed','repaired'),(failed,repaired),(failed_evidence,repaired_evidence),replays):
        bound_actions,score_receipt=_score_reference(model,example,actions,reference_identity={
            'checkpoint_sha256':sha256_file_hex_cached(reference_checkpoint),
            'tokenizer_fingerprint':fingerprint},runtime_sha256=runtime_sha,chunk_tokens=128)
        score_receipt_sha=canonical_json_sha256_hex(score_receipt)
        actor_identity=rollout['turns'][0]['sampling']['policy']
        actor={'checkpoint_sha256':actor_identity['checkpoint_sha256'],
            'runtime_sha256':evidence['actor_runtime_sha256'],
            'actor_id':canonical_json_sha256_hex(actor_identity)}
        capture_actions=[{key:action[key] for key in ('node','position','token_id','kind')}
            for action in actions]
        prompt_sha=canonical_json_sha256_hex({'messages':rollout['turns'][0]['messages'],
            'tools':rollout['turns'][0].get('tools') or []})
        capture_receipt={'schema':'natlang.native-failure-execution-capture/1',
            'task_sha256':evidence['task_sha256'],'world_sha256':evidence['world_sha256'],
            'prompt_sha256':prompt_sha,'actions':capture_actions,'actor':actor,
            'result_sha256':evidence['result_sha256'],'execution_graph_sha256':canonical_json_sha256_hex(rollout['execution_graph'])}
        capture_sha=canonical_json_sha256_hex(capture_receipt)
        replay_receipt={'schema':'natlang.native-graph-replay-receipt/1',
            'graph_sha256':canonical_json_sha256_hex(graph),'execution_capture_sha256':capture_sha,
            'occurrence_bindings_verified':True}
        alternatives[label]={'graph':graph,'actions':bound_actions,'split':'train','source_groups':sorted(task_groups),
            'task_sha256':evidence['task_sha256'],'world_sha256':evidence['world_sha256'],
            'oracle_sha256':evidence['oracle_sha256'],'prompt_sha256':prompt_sha,
            'prompt_view':'original-unhinted','hint_in_student_prompt':False,
            'execution_capture_sha256':capture_sha,'execution_receipt':capture_receipt,
            'graph_replay_sha256':canonical_json_sha256_hex(replay_receipt),
            'graph_replay_receipt':replay_receipt,'complete_graph':True,
            'occurrence_bindings_verified':True,'actor':actor,
            'reference_score':{**{key:score_receipt[key] for key in
                ('checkpoint_sha256','tokenizer_fingerprint','runtime_sha256','graph_action_sha256','normalization')},
                'receipt':score_receipt,'receipt_sha256':score_receipt_sha},
            'capture_actor_identity':actor_identity,'capture_invocation_actors':actors}
    prompt_sha=alternatives['failed']['prompt_sha256']
    if alternatives['repaired']['prompt_sha256']!=prompt_sha:
        raise ValueError('corrected execution prompt differs from the original unhinted task view')
    if (failed['turns'][0]['messages']!=repaired['turns'][0]['messages'] or
            (failed['turns'][0].get('tools') or [])!=(repaired['turns'][0].get('tools') or [])):
        raise ValueError('same-task repaired capture does not use the exact original unhinted first-turn prompt')
    record={'schema':PAIR_SCHEMA,'split':'train','protected':False,
        'original_task':{'task_sha256':failed_evidence['task_sha256'],'world_sha256':failed_evidence['world_sha256'],
            'oracle_sha256':failed_evidence['oracle_sha256'],'source_groups':sorted(task_groups),'prompt_sha256':prompt_sha},
        'oracle_evidence':{'failed':failed_evidence,'repaired':repaired_evidence},'alternatives':alternatives}
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    result=_write_admitted_rows([record],output,tokenizer_fingerprint=fingerprint,protected_groups=protected,
        stage_inputs=None if shared_context else {'held_native_graphs':held_native_graphs,'continue_from':continue_from,
                      'stage_id':stage_id,'requires':requires,'beta':beta})
    return {**result,'reference_checkpoint_sha256':reference_id['checkpoint_sha256'],
        'reference_tokenizer_fingerprint':fingerprint,'task_id':task_id,'outcomes':{
            'failed':failed_evidence['outcome'],'repaired':repaired_evidence['outcome']}}


def admit_run_pairs(*,pairs,reference_checkpoint,base,held_native_graphs,output_dir,continue_from,stage_id,
                    prior_admission_dir=None,requires=(),device='cpu',beta=.1):
    """Admit a batch with one fixed-reference model/source load and append-only prior graph bindings."""
    import tempfile
    from ..common.hashing import sha256_file_hex_cached
    from ..common.jsonio import write_json_durable,write_canonical_jsonl
    from ..model.native_model import load_native_checkpoint,native_identity
    from ..data.text_corpus import tokenizer_fingerprint
    from ..data.native_sources import NativeGraphSource
    from ..eval.task_packet import read_jsonl
    from scripts.source_review import pending_unpinned_source_groups
    if not pairs:raise ValueError('failure-repair DPO batch must contain at least one task pair')
    model,tokenizer,roles,config,_=load_native_checkpoint(reference_checkpoint,base=base,device=device,allow_weights_only=True)
    model.eval();fingerprint=tokenizer_fingerprint(tokenizer)
    if config['tokenizer_fingerprint']!=fingerprint:raise ValueError('loaded reference checkpoint/tokenizer fingerprint mismatch')
    held=NativeGraphSource(held_native_graphs,fingerprint,required_split='test',role_ids=roles.ids)
    context={'model':model,'tokenizer':tokenizer,'roles':roles,'config':config,'fingerprint':fingerprint,
        'held':held,'protected':set(held.groups)|pending_unpinned_source_groups(),
        'reference_id':native_identity(model,config,checkpoint_sha256=sha256_file_hex_cached(reference_checkpoint))}
    graph_rows=[];objective_rows={}
    if prior_admission_dir is not None:
        prior=Path(prior_admission_dir).resolve(strict=True)
        old_graphs=read_jsonl(prior/'repair-graphs.jsonl');old_objective=json.loads((prior/'repair-objective.json').read_text())
        old_ids=[row.get('id') for row in old_graphs]
        if (old_objective.get('schema')!=OBJECTIVE_SCHEMA or old_objective.get('kind')!=KIND
                or len(old_ids)!=len(set(old_ids)) or set(old_ids)!=set(old_objective.get('rows',{}))):
            raise ValueError('prior repair admission graph/objective bindings disagree')
        graph_rows.extend(old_graphs);objective_rows.update(old_objective['rows'])
    with tempfile.TemporaryDirectory(prefix='failure-repair-batch-') as temp:
        for index,pair in enumerate(pairs):
            options=dict(pair)
            result=admit_run_pair(**options,reference_checkpoint=reference_checkpoint,base=base,
                held_native_graphs=held_native_graphs,output_dir=Path(temp)/str(index),
                continue_from=continue_from,stage_id=stage_id,requires=requires,device=device,beta=beta,_context=context)
            new_graphs=read_jsonl(result['graphs']);new_objective=json.loads(Path(result['objective']).read_text())
            for graph in new_graphs:
                if graph['id'] in objective_rows:raise ValueError('repair batch duplicates an existing pair identity')
                graph_rows.append(graph);objective_rows[graph['id']]=new_objective['rows'][graph['id']]
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('failure-repair batch output directory must be empty and immutable')
    graph_path=output/'repair-graphs.jsonl';objective_path=output/'repair-objective.json';sources_path=output/'repair-sources.json'
    write_canonical_jsonl(graph_path,graph_rows)
    write_json_durable(objective_path,{'schema':OBJECTIVE_SCHEMA,'kind':KIND,'beta':float(beta),
        'normalization':'token-mean-v1','rows':objective_rows})
    write_json_durable(sources_path,{'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs','path':graph_path.name,'sha256':sha256_file_hex(graph_path)}]})
    bindings={}
    def bind(role,path):
        path=Path(path).resolve(strict=True);name='failure_repair_'+role
        bindings[name]={'path':str(path),'sha256':sha256_file_hex_cached(path)};return name
    inputs={'native_graphs':bind('graphs',sources_path),'objective_spec':bind('objective',objective_path),
        'held_native_graphs':bind('held',held_native_graphs),'continue_from':bind('parent_checkpoint',continue_from)}
    stage={'id':stage_id,'kind':'native_offline_preference','requires':list(requires),'inputs':inputs,
        'parameters':{'preference_beta':beta,'preference_margin':0}}
    addition={'schema':'natlang.native-failure-repair-recipe-addition/1',
        'training_admission':'pending-normal-recipe-source-and-custody-validation','input_bindings':bindings,'stage':stage,
        'continuation':{'objective_continuation':True,'weights_optimizer_rng_exposure':'preserved',
            'append_policy':'exact prior graph/objective row prefix; new pairs receive their own exposure visits'},
        'objective':{'kind':KIND,'beta':beta,'reference_normalization':'token-mean-v1','protected_failures_admitted':False},
        'pairs':len(graph_rows),'new_pairs':len(pairs),'prior_pair_count':len(graph_rows)-len(pairs)}
    addition_path=output/'recipe-stage-addition.json';write_json_durable(addition_path,addition)
    return {'graphs':str(graph_path),'objective':str(objective_path),'sources':str(sources_path),
        'recipe_stage_addition':str(addition_path),'pairs':len(graph_rows),'new_pairs':len(pairs),
        'reference_checkpoint_sha256':context['reference_id']['checkpoint_sha256'],
        'reference_tokenizer_fingerprint':fingerprint}


def materialize_qwen_repair_run(*,failed_run_dir,task_id,failed_sample,repaired_sample,capture_path,
                                output_dir,endpoint):
    """Turn an actual executeProgram Qwen replay capture into the normal collector-run contract.

    The actor binding and task packet are inherited byte-for-byte from the failed collection. The exact endpoint is
    checked against both that binding and its live native identity before its retained state artifacts are used.
    """
    import urllib.request
    from ..common.hashing import sha256_file_hex_cached,canonical_json_sha256_hex
    from ..common.jsonio import write_json_durable,write_canonical_jsonl
    from ..eval.task_packet import read_jsonl
    from ..rl.task_rollouts import rollout_turns
    from ..rl.graph_replay import NativeBundleStore,save_bundle,verified_bundle_bytes
    from ..rl.actor_client import NativeActorClient
    failed_dir=Path(failed_run_dir).resolve(strict=True);capture=json.loads(Path(capture_path).read_text())
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Qwen repair materialization output must be an empty immutable directory')
    if capture.get('schema')!='natlang.qwen-teacher-repair-execution/1':
        raise ValueError('Qwen replay did not produce the exact execution-capture schema')
    preparation=json.loads((failed_dir/'preparation.json').read_text())
    binding=json.loads((failed_dir/'collection-binding.json').read_text())
    packet_path=failed_dir/'packet.jsonl';packet=read_jsonl(packet_path);packet_sha=sha256_file_hex_cached(packet_path)
    if preparation.get('schema')!='natlang.task-rollout-preparation/1' or preparation.get('packet_sha256')!=packet_sha:
        raise ValueError('failed run lost its exact task packet preparation binding')
    queue=read_jsonl(failed_dir/'queue.jsonl')
    candidates=[(index,row) for index,row in enumerate(packet) if row.get('id')==task_id]
    if len(candidates)!=1:raise ValueError('failed task packet id is missing or ambiguous')
    packet_index,task=candidates[0]
    record=(capture.get('row') or {}).get('task',{}).get('program_ir')
    if record!=task.get('program_ir') or task.get('split')!='train':
        raise ValueError('teacher replay did not execute the exact original train task packet')
    selected=[entry for entry in queue if entry.get('index')==packet_index and entry.get('sample')==failed_sample]
    if len(selected)!=1:raise ValueError('source run queue does not uniquely bind the failed task/sample')
    if (binding.get('schema')!='natlang.rl-collection-binding/1' or
            preparation.get('policy_checkpoint_sha256')!=binding.get('policy_id')):
        raise ValueError('failed run collection and checkpoint bindings disagree')
    with urllib.request.urlopen(endpoint.rstrip('/')+'/v1/neuralese/info',timeout=30) as response:
        info=json.loads(response.read())
    live=(info.get('rl_policy_binding') or {}).get('native_identity')
    old_identity=(binding.get('expected_native_actor') or {}).get('native_identity')
    live_runtime=(info.get('rl_policy_binding') or {}).get('python_runtime')
    target_actor=capture.get('target_actor') or {}
    if (target_actor.get('endpoint','').rstrip('/')!=endpoint.rstrip('/') or
            target_actor.get('binding_sha256')!=canonical_json_sha256_hex(info.get('rl_policy_binding')) or
            target_actor.get('native_identity')!=live or target_actor.get('python_runtime')!=live_runtime):
        raise ValueError('repair endpoint actor/runtime changed between forced capture and artifact materialization')
    if (not isinstance(old_identity,dict) or not isinstance(live,dict) or
            live.get('architecture')!=old_identity.get('architecture') or
            live.get('tokenizer_fingerprint')!=old_identity.get('tokenizer_fingerprint') or
            not isinstance(live.get('checkpoint_sha256'),str) or len(live['checkpoint_sha256'])!=64 or
            not isinstance(live.get('embedding_weight_sha256'),str) or len(live['embedding_weight_sha256'])!=64):
        raise ValueError('live repair actor must have an exact identity in the failed model architecture and tokenizer family')
    if not isinstance(live_runtime,dict) or live_runtime.get('schema')!='natlang.python-runtime/1':
        raise ValueError('live repair endpoint has no authenticated Python runtime inventory')
    run=capture.get('run');trajectory=capture.get('trajectory')
    if not isinstance(run,dict) or not isinstance(run.get('outcome'),dict) or not isinstance(trajectory,list) or not trajectory:
        raise ValueError('Qwen replay artifact lacks complete actual trajectory, outcome, or execution graph')
    source_rows=[turn for turn in trajectory if ((turn.get('model_response') or {}).get('rl_sampling') or {}).get('schema')==
        'natlang.native-teacher-forced-capture/1']
    if not source_rows:raise ValueError('Qwen replay produced no actual native forced-turn capture')
    for turn in source_rows:
        capture_receipt=turn['model_response']['rl_sampling']
        if capture_receipt.get('replay_prompt_sha256')!=canonical_json_sha256_hex(
                {'messages':turn.get('context'),'tools':turn.get('tools_offered') or []}):
            raise ValueError('Qwen replay target was not forced under the exact unhinted execution prompt')
    result={'version':'natlang.teacher_trajectory.native/1','id':f'qwen-repair:{capture.get("run_id")}',
        'task':{'program_ir':task['program_ir']},'provenance':{'model':str(live.get('architecture','native-student')),
            'checkpoint_sha256':live['checkpoint_sha256'],'native_identity':live},
        'execution_identity':{'schema':'natlang.qwen-teacher-repair-execution/1','run_id':capture.get('run_id')},
        'outcome':run['outcome'],'trajectory':trajectory,'capture_limits':[]}
    result_path=output/'results'/'repaired.jsonl';result_path.parent.mkdir(parents=True)
    write_canonical_jsonl(result_path,[result])
    queue_entry=dict(selected[0]);queue_entry['sample']=repaired_sample
    queue_entry['key']=f'qwen-repair:{task_id}:{repaired_sample}:{capture.get("run_id")}'
    queue_entry['output']=str(result_path.resolve())
    write_canonical_jsonl(output/'queue.jsonl',[queue_entry])
    preparation=dict(preparation);preparation['policy_checkpoint_sha256']=live['checkpoint_sha256']
    preparation['capture_mode']='offline-teacher-forced-repair'
    write_json_durable(output/'preparation.json',preparation)
    (output/'packet.jsonl').write_bytes(packet_path.read_bytes())
    repository=Path(__file__).resolve().parents[4];ts_root=repository/'ts-host'
    runtime_inventory={str(path.relative_to(ts_root)):sha256_file_hex_cached(path)
        for path in sorted((ts_root/'dist').rglob('*.js'))}
    if not runtime_inventory:raise ValueError('exact TypeScript collector runtime inventory is empty')
    replay_binding={'schema':'natlang.rl-collection-binding/1','cases_sha256':binding.get('cases_sha256'),
        'policy_id':live['checkpoint_sha256'],'rows':1,'offset':packet_index,'group':queue_entry.get('group'),
        'root_mode':'teacher-forced-offline-repair','seed_base':None,'temperature':0.,'guided':False,
        'case_seconds':None,'max_model_requests':None,'max_turns':None,'endpoints':[endpoint],
        'neuralese_server':True,'runtime':runtime_inventory,
        'collector_sha256':sha256_file_hex_cached(repository/'scripts/run_bonsai_queue.py'),
        'expected_native_actor':{'native_identity':live,'python_runtime':live_runtime},
        'capture_executor':{'schema':'natlang.qwen-teacher-replay-executor/1',
            'source_collection_binding_sha256':sha256_file_hex_cached(failed_dir/'collection-binding.json'),
            'source_task_sha256':canonical_json_sha256_hex(task),
            'sampling_semantics':'exact-forced-sequence; behavior-log-probabilities-absent'}}
    write_json_durable(output/'collection-binding.json',replay_binding)
    finish={'event':'finish','key':queue_entry['key'],'status':'complete','exit_code':0,
        'resource_limit_reason':None,'time':0,'output_accounting':{'complete':True,'expected_jobs':1,
            'completed_jobs':1,'missing_jobs':0}}
    workers=output/'workers';workers.mkdir()
    write_canonical_jsonl(workers/'journal-0.jsonl',[finish])
    repaired_rollout={'schema':'natlang.rollout/1','task_id':task_id,'split':task['split'],
        'group':queue_entry.get('group',task_id),'sample':repaired_sample,
        'policy':{'checkpoint_sha256':preparation['policy_checkpoint_sha256']},
        'turns':rollout_turns(result),'execution_graph':run['outcome'].get('execution_graph') or [],
        'outcome':'task_success' if run['outcome'].get('accepted') is True else 'unresolved',
        'reward':{'quality':1. if run['outcome'].get('accepted') is True else None}}
    failed_task,failed_rollout,_=_run_member(failed_dir,task_id,failed_sample)
    if canonical_json_sha256_hex(failed_task)!=canonical_json_sha256_hex(task):
        raise ValueError('source failed rollout changed while preparing the repaired result')
    bundle_dir=output/'blocks';bundle_dir.mkdir()
    old_manifest=failed_dir/'blocks'/'manifest.json'
    repaired_policy_id=live['checkpoint_sha256'];server=NativeActorClient(endpoint)
    def fetch(block_id):
        kind='native-states' if block_id.startswith('nzs1_') else 'native-tensor-artifacts' if block_id.startswith('nnt1_') else 'blocks'
        return server.get_bytes('/v1/neuralese/'+kind+'/'+block_id)
    manifest=save_bundle([repaired_rollout],bundle_dir,policy_id=repaired_policy_id,fetch=fetch,
        manifest_fields={'collection_binding':replay_binding})
    return {'run_dir':str(output),'result':str(result_path),'block_bundle':str(bundle_dir/'manifest.json'),
        'failed_block_bundle':str(old_manifest) if old_manifest.is_file() else None,
        'task_id':task_id,'sample':repaired_sample,'outcome':repaired_rollout['outcome'],
        'native_blocks':len(manifest['blocks']),'collection_binding_sha256':sha256_file_hex_cached(output/'collection-binding.json')}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Build failure-repair DPO from actual collector results and native captures')
    parser.add_argument('--pairs-json',type=Path,
        help='JSON array of exact run-pair objects; each names failed_run_dir, repaired_run_dir, task_id and samples; optional teacher_replay runs authenticated Qwen capture and materializes the repaired run')
    parser.add_argument('--prior-admission-dir',type=Path)
    parser.add_argument('--failed-run-dir', type=Path)
    parser.add_argument('--repaired-run-dir', type=Path)
    parser.add_argument('--task-id')
    parser.add_argument('--failed-sample',type=int,default=0)
    parser.add_argument('--repaired-sample',type=int,default=1)
    parser.add_argument('--reference-checkpoint',required=True,type=Path)
    parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--held-native-graphs',required=True,type=Path)
    parser.add_argument('--failed-block-bundle',type=Path,
        help='exact native artifact bundle for the failed capture actor')
    parser.add_argument('--repaired-block-bundle',type=Path,
        help='exact native artifact bundle for the repaired capture actor')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--continue-from',required=True,type=Path)
    parser.add_argument('--stage-id',required=True)
    parser.add_argument('--requires',action='append',default=[])
    parser.add_argument('--beta',type=float,default=.1)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args(argv)
    if args.pairs_json:
        jobs=json.loads(args.pairs_json.read_text())
        if not isinstance(jobs,list) or any(not isinstance(job,dict) for job in jobs):
            parser.error('--pairs-json must contain a JSON array of pair objects')
        if any('teacher_replay' in job for job in jobs):
            import os,subprocess,tempfile
            from ..common.jsonio import write_json_durable
            import shutil
            node=os.environ.get('NATLANG_NODE') or shutil.which('node')
            if not node:parser.error('Node is required for native teacher replay; set NATLANG_NODE or add node to PATH')
            cli=Path(__file__).resolve().parents[4]/'ts-host'/'dist'/'teacher'/'native-forced-replay-cli.js'
            if not cli.is_file():parser.error(f'Qwen replay CLI is missing; build ts-host first: {cli}')
            with tempfile.TemporaryDirectory(prefix='qwen-repair-admission-') as temp_root:
                temp=Path(temp_root)
                for index,job in enumerate(jobs):
                    replay=job.pop('teacher_replay',None)
                    if replay is None:continue
                    if not isinstance(replay,dict):parser.error('teacher_replay must be a JSON object per pair')
                    capture_path=temp/f'capture-{index}.json';repaired_dir=temp/f'repaired-{index}'
                    spec={**replay,'schema':'natlang.qwen-teacher-repair-job/1',
                        'failed_run_dir':job['failed_run_dir'],'task_id':job['task_id'],
                        'failed_sample':job['failed_sample'],'output_capture_path':str(capture_path)}
                    spec_path=temp/f'replay-{index}.json';write_json_durable(spec_path,spec)
                    subprocess.run([node,str(cli),'--spec',str(spec_path)],check=True)
                    materialized=materialize_qwen_repair_run(failed_run_dir=job['failed_run_dir'],task_id=job['task_id'],
                        failed_sample=job['failed_sample'],repaired_sample=job['repaired_sample'],capture_path=capture_path,
                        output_dir=repaired_dir,endpoint=spec['target_endpoint'])
                    job['repaired_run_dir']=materialized['run_dir'];job['repaired_block_bundle']=materialized['block_bundle']
                    if materialized['failed_block_bundle'] is not None:
                        job['failed_block_bundle']=materialized['failed_block_bundle']
        result=admit_run_pairs(pairs=jobs,reference_checkpoint=args.reference_checkpoint,base=args.base,
            held_native_graphs=args.held_native_graphs,output_dir=args.output_dir,continue_from=args.continue_from,
            stage_id=args.stage_id,prior_admission_dir=args.prior_admission_dir,requires=args.requires,
            device=args.device,beta=args.beta)
    else:
        if not all((args.failed_run_dir,args.repaired_run_dir,args.task_id)):
            parser.error('single-pair mode requires --failed-run-dir, --repaired-run-dir and --task-id')
        result=admit_run_pair(failed_run_dir=args.failed_run_dir,repaired_run_dir=args.repaired_run_dir,
            task_id=args.task_id,failed_sample=args.failed_sample,repaired_sample=args.repaired_sample,
            reference_checkpoint=args.reference_checkpoint,base=args.base,held_native_graphs=args.held_native_graphs,
            failed_block_bundle=args.failed_block_bundle,repaired_block_bundle=args.repaired_block_bundle,
            output_dir=args.output_dir,continue_from=args.continue_from,
            stage_id=args.stage_id,requires=args.requires,device=args.device,beta=args.beta)
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__': raise SystemExit(main())
