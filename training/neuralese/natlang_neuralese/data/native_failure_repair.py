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
    from ..rl.task_rollouts import reward_of,rollout_turns
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
    entry,task=candidates[0];result_path=Path(entry['output']).resolve(strict=True)
    results=read_jsonl(result_path)
    if len(results)!=1:raise ValueError('requested task run must retain exactly one complete result row')
    result=results[0];finish=finishes.get(entry['key'])
    if (result.get('task',{}).get('program_ir')!=task or finish is None
            or result.get('outcome',{}).get('status') not in ('done','failed','quiesced')):
        raise ValueError('collector result, terminal journal and exact task packet disagree')
    outcome=classify_case(result,finish);reward=reward_of(outcome,finish)
    if reward not in (0.,1.):raise ValueError('infrastructure, unresolved, or unscored task outcome cannot be DPO evidence')
    runtime_candidates=[preparation.get('policy_runtime_sha256'),preparation.get('runtime_sha256'),
        entry.get('policy_runtime_sha256'),entry.get('runtime_sha256'),result.get('policy_runtime_sha256'),
        (result.get('policy') or {}).get('runtime_sha256')]
    runtime_sha=next((value for value in runtime_candidates if isinstance(value,str) and len(value)==64
        and all(char in '0123456789abcdef' for char in value)),None)
    if runtime_sha is None:
        raise ValueError('collector evidence lacks the exact policy runtime digest; actor runtime cannot be inferred')
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
    evidence={'outcome':outcome,'task_sha256':canonical_json_sha256_hex(task),
        'world_sha256':canonical_json_sha256_hex(task.get('world',task.get('environment',task))),
        'oracle_sha256':canonical_json_sha256_hex(task.get('oracle',task.get('semantics',task))),
        'source_groups':task.get('source_groups') or [],'packet_sha256':packet_sha,
        'result_sha256':sha256_file_hex_cached(result_path),'result_path':str(result_path),
        'finish_sha256':canonical_json_sha256_hex(finish),'actor_runtime_sha256':runtime_sha}
    evaluator=Path(__file__).resolve().parents[1]/'eval/task_packet.py'
    evidence['evaluation_receipt']={'schema':'natlang.task-oracle-evaluation/1',
        **{key:evidence[key] for key in ('outcome','task_sha256','world_sha256','oracle_sha256',
            'source_groups','packet_sha256','result_sha256','finish_sha256')},
        'task_id':task_id,'evaluator_sha256':sha256_file_hex_cached(evaluator)}
    evidence['evaluation_receipt_sha256']=canonical_json_sha256_hex(evidence['evaluation_receipt'])
    return task,rollout,evidence


def _replay_rollout(rollout, *, identity, roles, fetch_block):
    from ..rl.episode_graph import whole_episode_turns
    from ..rl.native_replay import compile_replay,capture_invocation_actors
    turns=whole_episode_turns(rollout)
    schemas={turn['sampling'].get('schema') for turn in turns}
    structural=bool(schemas & {'natlang.native-structural-capture/1','natlang.native-structural-capture/2'})
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
    with torch.no_grad():result=model(example.plan,retain_states=False)
    model_actions=[{'query':example.plan.offsets[next(i for i,node in enumerate(example.plan.invocations)
        if node.name==item['node'])]+item['position']-1,'token_id':item['token_id'],'temperature':1.,
        'allowed_token_ids':None,'banned_token_ids':None,'behavior_logp':0.} for item in actions]
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


def admit_run_pair(*,failed_run_dir,repaired_run_dir,task_id,failed_sample,repaired_sample,reference_checkpoint,
                   base,held_native_graphs,block_bundle,output_dir,continue_from,stage_id,
                   requires=(),device='cpu',beta=.1):
    """Build DPO data from actual result files, replay captures and current held-source manifests."""
    import torch
    from ..common.hashing import sha256_file_hex_cached
    from ..model.native_model import load_native_checkpoint,native_identity
    from ..data.text_corpus import tokenizer_fingerprint
    from ..data.native_sources import NativeGraphSource
    from ..model.shared_state import RoleTokens
    from ..rl.graph_replay import NativeBundleStore
    from ..rl.native_replay import capture_invocation_actors
    from ..common.hashing import canonical_json_sha256_hex
    task_a,failed,failed_evidence=_run_member(failed_run_dir,task_id,failed_sample)
    task_b,repaired,repaired_evidence=_run_member(repaired_run_dir,task_id,repaired_sample)
    if canonical_json_sha256_hex(task_a)!=canonical_json_sha256_hex(task_b):
        raise ValueError('failed and repaired task packets differ in source, world, or oracle')
    if failed_evidence['outcome'] not in {'invalid_action','repetition','budget_exhaustion','model_incomplete','semantic_mismatch'}:
        raise ValueError('original case did not have an exact task-caused failure outcome')
    if repaired_evidence['outcome']!='task_success':
        raise ValueError('corrected execution does not pass the exact task packet oracle')
    model,tokenizer,roles,config,_=load_native_checkpoint(reference_checkpoint,base=base,device=device,allow_weights_only=True)
    model.eval();fingerprint=tokenizer_fingerprint(tokenizer)
    if config['tokenizer_fingerprint']!=fingerprint:
        raise ValueError('loaded reference checkpoint/tokenizer fingerprint mismatch')
    held=NativeGraphSource(held_native_graphs,fingerprint,required_split='test',role_ids=roles.ids)
    from scripts.source_review import pending_unpinned_source_groups,pending_family_hold_reason
    protected=set(held.groups)|pending_unpinned_source_groups()
    task_groups=set(task_a.get('source_groups') or [])
    if not task_groups or task_groups & protected or pending_family_hold_reason(task_a):
        raise ValueError('source task intersects actual held or source-review protection')
    reference_id=native_identity(model,config,checkpoint_sha256=sha256_file_hex_cached(reference_checkpoint))
    if failed['split']!='train' or repaired['split']!='train':raise ValueError('only original train tasks can enter repair DPO')
    captured_policies=[turn['sampling']['policy'] for rollout in (failed,repaired) for turn in rollout['turns']]
    if not captured_policies or any(value['tokenizer_fingerprint']!=fingerprint for value in captured_policies):
        raise ValueError('both complete captures must be explicitly replayable in the reference tokenizer')
    # The capture runtime actor can differ from the fixed DPO reference policy;
    # replay is bound to the exact actual invocation identity in each capture.
    capture_base=captured_policies[0]
    bundle=NativeBundleStore(block_bundle,policy_id=capture_base['checkpoint_sha256'])
    if any(value.get('checkpoint_sha256')!=capture_base['checkpoint_sha256'] for value in captured_policies):
        raise ValueError('one complete actor bundle is required per graph-capture policy; mixed actor bundles need explicit per-turn custody')
    replays=[]
    for rollout in (failed,repaired):
        identity=rollout['turns'][0]['sampling']['policy']
        replay,actors=_replay_rollout(rollout,identity=identity,roles=roles,fetch_block=bundle.state)
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
        score_receipt['receipt_sha256']=canonical_json_sha256_hex(score_receipt)
        actor_identity=rollout['turns'][0]['sampling']['policy']
        capture_actions=[{key:action[key] for key in ('node','position','token_id','kind')}
            for action in actions]
        prompt_sha=canonical_json_sha256_hex(rollout['turns'][0]['messages'])
        capture_receipt={'schema':'natlang.native-failure-execution-capture/1',
            'task_sha256':evidence['task_sha256'],'world_sha256':evidence['world_sha256'],
            'prompt_sha256':prompt_sha,'actions':capture_actions,'actor':actor_identity,
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
            'occurrence_bindings_verified':True,'actor':{'checkpoint_sha256':actor_identity['checkpoint_sha256'],
                'runtime_sha256':evidence['actor_runtime_sha256'],'actor_id':canonical_json_sha256_hex(actor_identity)},
            'reference_score':{**{key:score_receipt[key] for key in
                ('checkpoint_sha256','tokenizer_fingerprint','runtime_sha256','graph_action_sha256','normalization')},
                'receipt':score_receipt,'receipt_sha256':score_receipt['receipt_sha256']},
            'capture_actor_identity':actor_identity,'capture_invocation_actors':actors}
    prompt_sha=alternatives['failed']['prompt_sha256']
    if alternatives['repaired']['prompt_sha256']!=prompt_sha:
        raise ValueError('corrected execution prompt differs from the original unhinted task view')
    if failed['turns'][0]['messages']!=repaired['turns'][0]['messages']:
        raise ValueError('same-task repaired capture does not use the exact original unhinted first-turn prompt')
    record={'schema':PAIR_SCHEMA,'split':'train','protected':False,
        'original_task':{'task_sha256':failed_evidence['task_sha256'],'world_sha256':failed_evidence['world_sha256'],
            'oracle_sha256':failed_evidence['oracle_sha256'],'source_groups':sorted(task_groups),'prompt_sha256':prompt_sha},
        'oracle_evidence':{'failed':failed_evidence,'repaired':repaired_evidence},'alternatives':alternatives}
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    result=_write_admitted_rows([record],output,tokenizer_fingerprint=fingerprint,protected_groups=protected,
        stage_inputs={'held_native_graphs':held_native_graphs,'continue_from':continue_from,
                      'stage_id':stage_id,'requires':requires,'beta':beta})
    return {**result,'reference_checkpoint_sha256':reference_id['checkpoint_sha256'],
        'reference_tokenizer_fingerprint':fingerprint,'task_id':task_id,'outcomes':{
            'failed':failed_evidence['outcome'],'repaired':repaired_evidence['outcome']}}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Build failure-repair DPO from actual collector results and native captures')
    parser.add_argument('--failed-run-dir', required=True, type=Path)
    parser.add_argument('--repaired-run-dir', required=True, type=Path)
    parser.add_argument('--task-id',required=True)
    parser.add_argument('--failed-sample',type=int,default=0)
    parser.add_argument('--repaired-sample',type=int,default=1)
    parser.add_argument('--reference-checkpoint',required=True,type=Path)
    parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--held-native-graphs',required=True,type=Path)
    parser.add_argument('--block-bundle',required=True,type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--continue-from',required=True,type=Path)
    parser.add_argument('--stage-id',required=True)
    parser.add_argument('--requires',action='append',default=[])
    parser.add_argument('--beta',type=float,default=.1)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args(argv)
    print(json.dumps(admit_run_pair(failed_run_dir=args.failed_run_dir,repaired_run_dir=args.repaired_run_dir,
        task_id=args.task_id,failed_sample=args.failed_sample,repaired_sample=args.repaired_sample,
        reference_checkpoint=args.reference_checkpoint,
        base=args.base,held_native_graphs=args.held_native_graphs,block_bundle=args.block_bundle,
        output_dir=args.output_dir,continue_from=args.continue_from,stage_id=args.stage_id,
        requires=args.requires,device=args.device,beta=args.beta),sort_keys=True))


if __name__ == '__main__': raise SystemExit(main())
