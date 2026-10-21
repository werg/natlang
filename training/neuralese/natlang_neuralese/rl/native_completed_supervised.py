"""Account a sealed completed quantum before adapting the future recipe.

Policy remains with the supervisor; this module verifies durable boundaries.
It creates neither optimizer-owner evidence nor earned visits for new sources.
"""
from __future__ import annotations
import copy
import contextlib
import fcntl
import json
import math
import signal
from pathlib import Path

from ..common.jsonio import write_json_durable,write_bytes_durable
from ..train.native_readiness import _bound
from ..train.native_resident import binding,code_inventory

SCHEMA='natlang.native-completed-supervised-adaptation/1'
SCHEMA_RAW='natlang.native-completed-raw-adaptation/1'
CONSUMED='natlang.native-completed-supervised-consumption/1'
CONSUMED_RAW='natlang.native-completed-raw-consumption/1'


def _read(reference):return json.loads(_bound(reference).read_text())


def _artifact_binding(reference):
    """Check path/SHA and optional byte count before comparing artifact identity."""
    if (not isinstance(reference,dict) or set(reference) not in (
            {'path','sha256'},{'path','sha256','bytes'})
            or not isinstance(reference.get('path'),str)
            or not isinstance(reference.get('sha256'),str)):
        raise ValueError('completed artifact reference needs exact path/SHA and optional bytes')
    path=_bound(reference)
    if 'bytes'in reference and (type(reference['bytes'])is not int
            or reference['bytes']<0 or reference['bytes']!=path.stat().st_size):
        raise ValueError('completed artifact reference byte count changed')
    return {'path':reference['path'],'sha256':reference['sha256']}


def _controls(config_path):
    path=Path(config_path).resolve();config=json.loads(path.read_text());root=path.parent
    if Path(config['out']).resolve()!=root/'production':
        raise ValueError('completed quantum lost its actual controller namespace')
    return config,{'config':path,'cursor':root/'cycle.json','state':root/'production/state.json'}


def _no_owner(controller):
    from .native_raw_handoff import observe,same_process
    current=observe(controller['pid'])
    if same_process(controller,current)and current['state']not in('Z','X'):
        raise ValueError('completed adaptation requires its exact controller to exit')


def _no_active_operation(root):
    root=Path(root).resolve()
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():continue
        try:
            raw=(process/'cmdline').read_bytes()
            if not raw.endswith(b'\0'):continue
            args=[part.decode()for part in raw[:-1].split(b'\0')]
            critical=any(part in args for part in('natlang_neuralese.train.native_state_sft',
                'natlang_neuralese.train.native_preference','natlang_neuralese.rl.native_stage')) or any(
                    part.endswith(('/worker-entry.py','/serve_direct_checkpoint.py'))for part in args)
            if critical and any(part.startswith(str(root)+'/')for part in args):
                raise ValueError('completed adaptation still has an actual operation owner')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue


def _operation(config,cursor,request_ref,result_ref):
    from .native_bridge import read_request,verify_result_inventory
    from .native_raw_rounds import _supervised_stage_inventory,cycle_work_directory
    from ..train.recipe import resolve_recipe_data
    request,result=verify_result_inventory(_bound(request_ref),_bound(result_ref))
    root=Path(config['out']).parent
    stages=_supervised_stage_inventory(resolve_recipe_data(config['production_recipe']),config['production_stage'])
    index=cursor.get('next_stage');cycle=cursor.get('cycle')
    if (cursor.get('phase')!='sft' or cursor.get('pending_rl_round')is not None
            or type(index)is not int or not 0<=index<len(stages) or type(cycle)is not int
            or request.get('native_actor_mode')!='raw-state' or request['kind']!='round'
            or result.get('status')!='completed'
            or Path(request['out']).resolve()!=cycle_work_directory(root,cursor)/'supervised'/f'{index:03d}'):
        raise ValueError('completed supervised result differs from the current cycle/stage')
    continuation=_read(request['config']);stage=stages[index]
    if (continuation.get('mandatory_sft_only')is not True
            or continuation.get('production_stage')!=stage['id']
            or continuation.get('trainer_parameters',{}).get('stage_id')!=stage['id']
            or binding(continuation['production_recipe'])!=binding(config['production_recipe'])
            or request['parent_checkpoint']['sha256']!=cursor['current_checkpoint']['sha256']):
        raise ValueError('completed supervised stage changed its sealed recipe/parent')
    return request,result,continuation,stage


def _raw_operation(config,cursor,request_ref,result_ref,operation_state=None):
    """Authenticate a completed raw round without interpreting it as CE work."""
    from .native_bridge import verify_result_inventory
    from .native_raw_rounds import cycle_work_directory
    request,result=verify_result_inventory(_bound(request_ref),_bound(result_ref))
    operation=_read(operation_state or result['state'])
    prior=_read(request['state']) if request.get('state') else None
    invocation=_read(request['config'])
    root=Path(config['out']).parent;cycle=cursor.get('cycle')
    if (request.get('kind')!='round' or request.get('native_actor_mode')!='raw-state'
            or result.get('status')!='completed' or not result.get('checkpoint')
            or cursor.get('phase')!='rl' or type(cycle)is not int
            or type(cursor.get('pending_rl_round'))is not int
            or Path(request['out']).resolve()!=Path(config['out']).resolve()
            or Path(request_ref['path']).resolve().parent!=cycle_work_directory(root,cursor)/'rl-bridge'
            or invocation.get('mandatory_sft_only')is True or invocation.get('semantic_adapter')
            or invocation.get('coefficient_policy') or request.get('semantic_adapter')
            or binding(invocation['production_recipe'])!=binding(config['production_recipe'])
            or invocation.get('raw_model_id')!=config.get('raw_model_id')
            or any(request['parent_checkpoint'].get(k)!=cursor.get('current_checkpoint',{}).get(k)
                   for k in('path','sha256'))):
        raise ValueError('completed raw result differs from its exact owned cycle/request')
    if (not isinstance(prior,dict) or prior.get('phase')!='collect'
            or operation.get('phase')!='collect'
            or operation.get('round')!=cursor['pending_rl_round']
            or operation.get('round')!=prior.get('round',-1)+1
            or operation.get('trained_rounds')!=prior.get('trained_rounds',-1)+1
            or operation.get('pending_native_checkpoint')is not None
            or operation.get('production_complete')is not False
            or binding(prior['checkpoint'])['sha256']!=request['parent_checkpoint']['sha256']
            or binding(operation['checkpoint'])['sha256']!=result['checkpoint']['sha256']
            or result['checkpoint']['sha256']==request['parent_checkpoint']['sha256']
            or (operation_state is not None and operation_state['sha256']!=result['state']['sha256'])):
        raise ValueError('completed raw result lost its actual published update boundary')
    return request,result,invocation,operation


def _raw_receipt(saved,checkpoint,parent):
    """Use the actual policy receipt; no mandatory-stage completion is inferred."""
    from ..train.native_continuation import collect_sft_progress,progress_obligations
    from ..train.native_operation_report import compact_obligations
    from ..common.hashing import canonical_json_sha256_hex
    receipt_path=Path(checkpoint['path']).parent/'native-training-receipt.json'
    receipt=json.loads(receipt_path.read_text());training=saved['training'];options=training.get('options')or{}
    updates=(receipt.get('continuation_boundary')or{}).get('new_optimizer_updates')
    expected=compact_obligations(progress_obligations(collect_sft_progress(training,borrow_frames=True)),
        {**checkpoint,'bytes':Path(checkpoint['path']).stat().st_size})
    if (receipt.get('schema')!='natlang.native-training-readiness/1'
            or receipt.get('policy_operation_completed')is not True or receipt.get('completed')is not False
            or receipt.get('checkpoint',{}).get('sha256')!=checkpoint['sha256']
            or receipt.get('continuation',{}).get('checkpoint_sha256')!=parent['sha256']
            or (options.get('continuation')or{}).get('checkpoint_sha256')!=parent['sha256']
            or receipt.get('mandatory_sft')!=expected
            or type(updates)is not int or updates<=0
            or receipt.get('resident_adoption')
            or not isinstance(options.get('stage_id'),str) or receipt.get('stage_id')!=options['stage_id']):
        raise ValueError('completed raw checkpoint lost its policy-only receipt/full carried progress')
    # Bind the existing admission to the actual sampled parent and saved rows.
    admission=receipt_path.parent/'rl-data-receipt.json';data=json.loads(admission.read_text())
    objective=receipt_path.parent/'rl-native-objective.json';graphs=receipt_path.parent/'rl-native-graphs.jsonl'
    if (data.get('schema')!='natlang.native-rl-admission/1'
            or data.get('policy',{}).get('checkpoint_sha256')!=parent['sha256']
            or data.get('native_graphs_sha256')!=binding(graphs)['sha256']
            or data.get('objective_sha256')!=binding(objective)['sha256']
            or not data.get('tasks_selected')or any(row.get('split')!='train'for row in data['tasks_selected'])
            or (training.get('source_receipts')or [{}])[0].get('sha256')!=data['native_graphs_sha256']):
        raise ValueError('completed raw checkpoint lost its sampled-policy admission/source binding')
    return receipt_path,receipt,updates,{'admission':binding(admission),'objective':binding(objective),
        'graphs':binding(graphs),'mandatory_sft_sha256':canonical_json_sha256_hex(expected)}



def _requested_refresh_signal(records):
    """Authenticate the existing StopSignal producer's durable record shape."""
    if not isinstance(records,list) or not records:
        return False
    names={item.name for item in signal.Signals}
    for record in records:
        if (not isinstance(record,dict) or set(record)!={'signal','unix'}
                or not isinstance(record['signal'],str) or record['signal']not in names
                or type(record['unix'])not in(int,float)):
            return False
        try:
            if not math.isfinite(record['unix']):return False
        except OverflowError:
            return False
    return any(record['signal']=='SIGUSR2'for record in records)


def _qualified_refresh_updates(config,stage,receipt,continuation=None):
    boundary=receipt.get('continuation_boundary')or{}
    updates=boundary.get('new_optimizer_updates')
    quantum=config.get('supervised_stage_quantum_updates',{}).get(stage['id'],config.get('supervised_quantum_updates'))
    if type(quantum)is not int or quantum<=0 or type(updates)is not int or not 0<updates<=quantum:
        raise ValueError('completed supervised result differs from its declared finite update quantum')
    if updates<quantum and (boundary.get('kind')!='semantic-refresh'
            or boundary.get('reached')is not True
            or not _requested_refresh_signal(boundary.get('requested_signals'))
            or type((continuation or{}).get('trainer_parameters',{}).get('refresh_after_updates'))is not int
            or (continuation or{}).get('trainer_parameters',{}).get('refresh_after_updates')!=quantum):
        raise ValueError('completed finite update quantum requires its actual qualified requested early refresh')
    return updates


@contextlib.contextmanager
def _completed_leases(config_path,owned_controller_fd=None,*,production=True):
    from .native_runtime_handoff import _runtime_leases
    config,paths=_controls(config_path)
    canonical=paths['config'].parent.parent/'controller.lock'
    locks=(canonical,paths['state'].parent/'controller.lock')if production else(canonical,)
    with _runtime_leases(locks,canonical,owned_controller_fd):
        yield config,paths


def _verified_completed_pair(request_ref,result_ref,verified_checkpoint=None):
    from .native_bridge import verify_result
    if verified_checkpoint is None:
        return verify_result(_bound(request_ref),_bound(result_ref),return_raw_checkpoint=True)
    return verify_result(_bound(request_ref),_bound(result_ref),return_raw_checkpoint=True,
                         verified_raw_checkpoint=verified_checkpoint)


def register_publication(config_path,publication,out,*,controller,verified_checkpoint,owned_controller_fd=None):
    """Authenticate a real already-applied ordinary publication without replaying effects.

    The supervisor's bounded publisher receipt is evidence, never a substitute
    for the original request/result, before snapshots, or exact normal effects.
    """
    from . import native_raw_rounds as raw
    with _completed_leases(config_path,owned_controller_fd)as(current,paths):
        _no_owner(controller);_no_active_operation(paths['config'].parent)
        published=_read(publication)
        if (published.get('schema')!='natlang.completed-worker-normal-publication-commit/1'
                or published.get('passed')is not True
                or published.get('normal_shared_publisher_reused')is not True
                or published.get('whole_recipe_completed')is not False
                or any(published.get(name)!=0 for name in('full_decodes','new_optimizer_updates','new_exposure'))):
            raise ValueError('completed publication lost its honest finite-request authority')
        intent=_read(published['intent'])
        if any(intent[key]!=published[key]for key in intent):
            raise ValueError('completed publication changed its durable intent')
        directory=Path(publication['path']).parent
        snapshots={}
        for key,path in paths.items():
            if (published['before'][key]['path']!=str(path)
                    or binding(path)!=published['after'][key]
                    or binding(path)['sha256']!=published['candidates'][key]['sha256']):
                raise ValueError('completed publication has a later or unknown successor')
            snapshot=directory/(key+'-before.json')
            if binding(snapshot)['sha256']!=published['before'][key]['sha256']:
                raise ValueError('completed publication lost its original control snapshot')
            _bound(published['candidates'][key]);snapshots[key]=binding(snapshot)
        config=_read(snapshots['config']);cursor=_read(snapshots['cursor']);state=_read(snapshots['state'])
        request,result,continuation,stage=_operation(config,cursor,published['request'],published['result'])
        result,pair=_verified_completed_pair(published['request'],published['result'],verified_checkpoint)
        receipt_path=Path(pair[0]['path']).parent/'native-training-receipt.json'
        receipt=json.loads(receipt_path.read_text());updates=_qualified_refresh_updates(config,stage,receipt,continuation)
        if (pair[1]['training'].get('global_step')!=published['global_step']
                or published['actual_request_updates']!=updates
                or receipt.get('stage_id')!=stage['id']
                or receipt.get('checkpoint',{}).get('sha256')!=pair[0]['sha256']
                or receipt.get('continuation',{}).get('checkpoint_sha256')!=request['parent_checkpoint']['sha256']
                or receipt.get('continuation_boundary',{}).get('checkpoint_sha256')!=pair[0]['sha256']):
            raise ValueError('completed publication lost its actual saved update boundary')
        operation=_read(result['state']);effects={}
        originals=(raw._publish_checkpoint,raw.write_json_durable,raw.write_bytes_durable)
        def capture(path,data):
            path=Path(path)
            if path not in paths.values():
                # Existing resident snapshots/recipe aliases must already be
                # exact; no new writes are allowed during reconciliation.
                if not path.is_file()or path.read_bytes()!=data:
                    raise ValueError('completed reconciliation would create a new publication effect')
                return
            if path in effects and effects[path]!=data:
                raise ValueError('completed publisher produced conflicting effects')
            effects[path]=data
        def capture_json(path,value):capture(path,json.dumps(value,indent=2,sort_keys=True).encode())
        def capture_checkpoint(cfg,selected):
            if cfg['out']!=config['out']:raise ValueError('completed publisher changed its production namespace')
            capture_json(paths['state'],raw._published_checkpoint_state(state,selected))
        try:
            raw._publish_checkpoint=capture_checkpoint
            raw.write_json_durable=capture_json;raw.write_bytes_durable=capture
            raw._publish_supervised_result(continuation,Path(request['out']),config,cursor,operation)
        finally:
            raw._publish_checkpoint,raw.write_json_durable,raw.write_bytes_durable=originals
        cursor.update(next_stage=cursor['next_stage']+1,current_checkpoint=binding(pair[0]['path']))
        capture_json(paths['cursor'],cursor)
        if paths['config']not in effects:capture_json(paths['config'],config)
        for key,path in paths.items():
            if effects.get(path)!=_bound(published['candidates'][key]).read_bytes():
                raise ValueError('completed publication differs from the existing shared publisher')
        value={'schema':CONSUMED,'controller':controller,'request':published['request'],'result':published['result'],
            'before':published['before'],'after':published['after'],'snapshots':snapshots,
            'candidates':published['candidates'],'checkpoint':result['checkpoint'],'stage_id':stage['id'],
            'global_step':published['global_step'],'new_optimizer_updates':updates,
            'training_receipt':binding(receipt_path),'publication':publication,
            'cursor_advances_from_registration':0,'whole_recipe_completed':False}
        out=Path(out)
        if out.exists():
            if json.loads(out.read_text())!=value:raise ValueError('completed registration changed its authority')
        else:write_json_durable(out,value)
        return value,pair

def consume(config_path,request_ref,result_ref,out,*,controller,verified_checkpoint=None,owned_controller_fd=None):
    """Use the existing publication body, advancing this completed stage once.

    Return the same process's verified full checkpoint for subsequent preparation.
    The default bridge verifier still performs all artifact and resume checks.
    """
    from .native_bridge import verify_result
    from .native_raw_rounds import _publish_supervised_result,_current_checkpoint,_published_checkpoint_state
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    config,paths=_controls(config_path);_no_owner(controller);_no_active_operation(paths['config'].parent)
    with _completed_leases(config_path,owned_controller_fd,production=False):
        if out.exists():
            receipt=json.loads(out.read_text())
            if (receipt.get('schema')!=CONSUMED or receipt['request']!=request_ref
                    or receipt['result']!=result_ref or receipt['controller']!=controller):
                raise ValueError('completed consumption retry changed its authority')
            cursor=json.loads(paths['cursor'].read_text());state=json.loads(paths['state'].read_text())
            if (binding(paths['cursor'])!=receipt['after']['cursor']
                    or binding(paths['state'])!=receipt['after']['state']):
                raise ValueError('completed consumption has a later successor; do not rewind')
            from ..train.checkpoint_policy import CheckpointPolicy
            if verified_checkpoint is not None:
                _,pair=_verified_completed_pair(request_ref,result_ref,verified_checkpoint)
                return receipt,pair
            return receipt,(receipt['checkpoint'],CheckpointPolicy.load(_bound(receipt['checkpoint'])))
        intent=out.with_name(out.stem+'-intent.json')
        recovery=json.loads(intent.read_text())if intent.exists()else None
        if recovery:
            if recovery['request']!=request_ref or recovery['result']!=result_ref or recovery['controller']!=controller:
                raise ValueError('completed consumption intent changed its exact request')
            before=recovery['before'];cursor=_read(recovery['snapshots']['cursor']);state=_read(recovery['snapshots']['state'])
            if binding(paths['config'])!=before['config']:
                raise ValueError('completed consumption config changed during recovery')
            for key in('state','cursor'):
                if binding(paths[key])['sha256']not in(before[key]['sha256'],recovery['candidates'][key]['sha256']):
                    raise ValueError('completed consumption found an unknown control successor')
        else:
            before={key:binding(path)for key,path in paths.items()}
            cursor=json.loads(paths['cursor'].read_text());state=json.loads(paths['state'].read_text())
        request,result,continuation,stage=_operation(config,cursor,request_ref,result_ref)
        if binding(state['checkpoint'])['sha256']!=request['parent_checkpoint']['sha256']:
            raise ValueError('completed consumption current production parent changed')
        verified,pair=_verified_completed_pair(request_ref,result_ref,verified_checkpoint)
        operation_state=_read(verified['state']);saved=pair[1];training=saved['training']
        options=training.get('options')or{}
        receipt_path=Path(verified['checkpoint']['path']).parent/'native-training-receipt.json'
        training_receipt=json.loads(receipt_path.read_text());boundary=training_receipt.get('continuation_boundary')or{}
        updates=_qualified_refresh_updates(config,stage,training_receipt,continuation)
        if (options.get('stage_id')!=stage['id'] or training_receipt.get('stage_id')!=stage['id']
                or training_receipt.get('checkpoint',{}).get('sha256')!=pair[0]['sha256']
                or training_receipt.get('continuation',{}).get('checkpoint_sha256')!=request['parent_checkpoint']['sha256']
                or type(updates)is not int or updates<=0
                or boundary.get('checkpoint_sha256')!=pair[0]['sha256']
                or training_receipt.get('resident_adoption')):
            raise ValueError('completed consumption lost its actual saved stage/update lineage')
        _no_owner(controller)
        value={'schema':CONSUMED,'controller':controller,'request':request_ref,'result':result_ref,
            'before':before,'checkpoint':pair[0],'stage_id':stage['id'],'new_optimizer_updates':updates,
            'global_step':training.get('global_step'),'training_receipt':binding(receipt_path),'whole_recipe_completed':False}
        checkpoint=binding(pair[0]['path'])
        next_state=_published_checkpoint_state(state,checkpoint)
        next_cursor={**cursor,'next_stage':cursor['next_stage']+1,'current_checkpoint':checkpoint}
        next_cursor.pop('readiness',None)
        directory=out.parent/(out.stem+'-controls');directory.mkdir(exist_ok=True)
        snapshots={};candidates={}
        for key,current in(('state',state),('cursor',cursor)):
            path=directory/('original-'+key+'.json')
            if not path.exists():write_bytes_durable(path,_bound(before[key]).read_bytes())
            snapshots[key]=binding(path)
        for key,current in(('state',next_state),('cursor',next_cursor)):
            path=directory/('candidate-'+key+'.json');write_json_durable(path,current);candidates[key]=binding(path)
        value.update(snapshots=snapshots,candidates=candidates)
        if recovery:
            if recovery!=value:raise ValueError('completed consumption recovery changed its candidates')
        else:
            if any(binding(path)!=before[key]for key,path in paths.items()):
                raise ValueError('completed controls changed before publication')
            write_json_durable(intent,value)
        _publish_supervised_result(continuation,Path(request['out']),config,cursor,operation_state)
        if binding(paths['state'])['sha256']!=candidates['state']['sha256']:
            raise ValueError('completed publication differs from the existing shared state transition')
        write_bytes_durable(paths['cursor'],_bound(candidates['cursor']).read_bytes())
        value.update(intent=binding(intent),after={key:binding(path)for key,path in paths.items()})
        write_json_durable(out,value)
        return value,pair


def consume_raw(config_path,request_ref,result_ref,out,*,controller):
    """Close the real completed raw cursor, keeping every supervised stage pending."""
    from .native_bridge import verify_result
    from .native_raw_rounds import completed_raw_cursor,cycle_work_directory
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    config,paths=_controls(config_path);_no_owner(controller);_no_active_operation(paths['config'].parent)
    with(paths['config'].parent.parent/'controller.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists():
            receipt=json.loads(out.read_text())
            if (receipt.get('schema')!=CONSUMED_RAW or receipt['request']!=request_ref
                    or receipt['result']!=result_ref or receipt['controller']!=controller
                    or any(binding(path)!=receipt['after'][key]for key,path in paths.items())):
                raise ValueError('completed raw consumption retry changed its authority/current controls')
            from ..train.checkpoint_policy import CheckpointPolicy
            return receipt,(receipt['checkpoint'],CheckpointPolicy.load(_bound(receipt['checkpoint'])))
        intent=out.with_name(out.stem+'-intent.json');recovery=_read(binding(intent))if intent.exists()else None
        if recovery:
            if recovery['request']!=request_ref or recovery['result']!=result_ref or recovery['controller']!=controller:
                raise ValueError('completed raw intent changed its exact operation')
            before=recovery['before'];cursor=_read(recovery['snapshots']['cursor'])
            if (binding(paths['config'])!=before['config']or binding(paths['state'])!=before['state']
                    or binding(paths['cursor'])['sha256']not in(before['cursor']['sha256'],recovery['candidate_cursor']['sha256'])):
                raise ValueError('completed raw recovery found an unknown control successor')
            operation_ref=recovery['operation_state']
        else:
            before={key:binding(path)for key,path in paths.items()};cursor=json.loads(paths['cursor'].read_text());operation_ref=None
        request,result,_,operation=_raw_operation(config,cursor,request_ref,result_ref,operation_ref)
        if before['state']['sha256']!=result['state']['sha256']:
            raise ValueError('completed raw production state differs from its actual result')
        verified,pair=verify_result(_bound(request_ref),_bound(result_ref),return_raw_checkpoint=True)
        receipt_path,training_receipt,updates,evidence=_raw_receipt(pair[1],pair[0],request['parent_checkpoint'])
        next_cursor=completed_raw_cursor(config,cursor,operation)
        if (next_cursor.get('next_stage')!=0 or next_cursor.get('phase')!='sft'
                or any((cycle_work_directory(paths['config'].parent,cursor)/'supervised').rglob('*.request.json'))):
            raise ValueError('completed raw consumption lost its unstarted supervised-stage boundary')
        directory=out.parent/(out.stem+'-controls');directory.mkdir(exist_ok=True)
        snapshots={}
        for key in('state','cursor'):
            target=directory/('original-'+key+'.json')
            if not target.exists():write_bytes_durable(target,_bound(before[key]).read_bytes())
            snapshots[key]=binding(target)
        candidate=directory/'candidate-cursor.json';write_json_durable(candidate,next_cursor)
        value={'schema':CONSUMED_RAW,'controller':controller,'request':request_ref,'result':result_ref,
            'before':before,'snapshots':snapshots,'operation_state':snapshots['state'],
            'candidate_cursor':binding(candidate),'checkpoint':pair[0],'stage_id':None,
            'operation_stage_id':training_receipt['stage_id'],'new_optimizer_updates':updates,
            'global_step':pair[1]['training']['global_step'],'training_receipt':binding(receipt_path),
            'raw_admission':evidence,'supervised_stage_credits_added':0,'whole_recipe_completed':False}
        if recovery:
            if recovery!=value:raise ValueError('completed raw recovery changed its exact candidate')
        else:
            if any(binding(path)!=before[key]for key,path in paths.items()):
                raise ValueError('completed raw controls changed before durable intent')
            write_json_durable(intent,value)
        _no_owner(controller);write_bytes_durable(paths['cursor'],_bound(value['candidate_cursor']).read_bytes())
        value.update(intent=binding(intent),after={key:binding(path)for key,path in paths.items()})
        write_json_durable(out,value);return value,pair


def _stable(plan):
    from .native_raw_handoff import _stable_handoff_plan
    return _stable_handoff_plan(plan)


def _future_config(config,recipe_reference,recipe,checkpoint,lineage,*,declaration=None,plan_reference=None):
    from .native_supervised import continuation_config,bind_recipe_lineage
    from .native_raw_rounds import _supervised_quantum
    foundation=next(stage for stage in recipe['stages']if stage['id']==config['production_stage'])
    future=copy.deepcopy(config);future['production_recipe']=str(_bound(recipe_reference))
    configured,_=continuation_config(future,foundation,checkpoint=checkpoint['path'],adapter=None,out=config['out'])
    for key in ('trainer_parameters','seed_command','seed_native_graphs','held_native_graphs',
                'foundation_native_graphs','continued_native_graphs','continued_held_native_graphs',
                'pending_native_sources','withdrawn_native_graphs','objective_spec'):
        if key in configured:future[key]=configured[key]
        else:future.pop(key,None)
    future['seed_trainer_parameters']=copy.deepcopy(future['trainer_parameters'])
    _supervised_quantum(future,future['seed_trainer_parameters'])
    directory=str(_bound(recipe_reference).parent/'resident-data')
    if directory:
        future['resident_data_directory']=directory
        future['seed_trainer_parameters']['resident_data_directory']=directory
    else:future.pop('resident_data_directory',None)
    bind_recipe_lineage(future,lineage)
    if declaration is not None:
        from .native_rounds import configure_provider
        from .native_raw_rounds import _scheduling_controls
        # This is a closed completed-request boundary, not an in-flight
        # collection. Bind the new actor command/environment/inventory together.
        from .production import actor_runtime_configuration
        protected_fields=('cases','runtime','task_source','source_admission')
        protected={key:copy.deepcopy(future[key])for key in protected_fields if key in future}
        future.update(actor_runtime_configuration(declaration['actor_runtime']))
        configure_provider(future,declaration,{},config['bootstrap_binding'],Path(config['out']).parent/'config.json',write=False)
        for key in protected_fields:
            if key in protected:future[key]=protected[key]
            else:future.pop(key,None)
        _scheduling_controls(future,declaration,_bound(plan_reference),{},pending=True)
        future.pop('runtime_handoff',None)
    return future


def _future_global(reference,old,role,recipe,model_id,package):
    target=_read(reference)
    old_rows={row['model']:row for row in old['models']}
    rows={row['model']:row for row in target['models']}
    if (len(rows)!=len(target['models'])or set(rows)!=set(old_rows)
            or target.get('owned_models')!=old.get('owned_models')):
        raise ValueError('completed runtime transition changed the owned student cohort')
    for name,row in rows.items():
        previous=old_rows[name]
        if row.get('handoff_requirement',{}).get('model_id')!=previous.get('handoff_requirement',{}).get('model_id'):
            raise ValueError('completed runtime transition substituted a student/base identity')
    selected=rows[role]
    if (_artifact_binding(selected['production_recipe'])!=_artifact_binding(recipe)
            or selected.get('handoff_requirement',{}).get('model_id')!=model_id
            or selected.get('actor_runtime')!=str(Path(package).resolve().parents[2])
            or selected.get('production_stage')!=old_rows[role].get('production_stage')
            or 'completed_supervised_adaptation'in selected):
        raise ValueError('completed runtime transition lost its exact future owned invocation')
    return target,selected


def _closed_withdrawals(training,frames,old_plan):
    from ..train.recipe import resolve_stage_inputs
    histories={}
    for frame in(training,*frames.values()):
        for history in(frame.get('options')or{}).get('withdrawal_history')or[]:
            histories.setdefault(history['proof_sha256'],[]).append(history)
    all_bindings=[frame.get('source_row_bindings')or{}for frame in(training,*frames.values())]
    for versions in histories.values():
        for history in versions:
            if any(row['id']in bindings for row in history['rows']for bindings in all_bindings):
                raise ValueError('completed adaptation reintroduced a closed withdrawal in saved frames')
    verified=[]
    for stage in old_plan['recipe']['stages']:
        reference=old_plan['stage_inputs'].get(stage['id'],{}).get('withdrawn_native_graphs')
        if not reference:continue
        proof=_read(reference);versions=histories.get(reference['sha256'],[])
        triples=lambda rows:{(row['id'],row['binding'],row['source_sha256'])for row in rows}
        if (proof.get('schema')!='natlang.native-graph-withdrawal/1' or not versions
                or any(triples(item['rows'])!=triples(proof['rows'])for item in versions)):
            raise ValueError('completed adaptation lost its exact closed prior withdrawal proof')
        verified.append({'proof':reference,'rows':len(proof['rows'])})
    return verified


def prepare(config_path,plan_path,role,recipe_reference,consumption,verified_checkpoint,out,*,package,old_package=None,future_global_plan=None,owned_controller_fd=None):
    with _completed_leases(config_path,owned_controller_fd):
        return _prepare(config_path,plan_path,role,recipe_reference,consumption,verified_checkpoint,out,
                        package=package,old_package=old_package,future_global_plan=future_global_plan)


def _prepare(config_path,plan_path,role,recipe_reference,consumption,verified_checkpoint,out,*,package,old_package=None,future_global_plan=None):
    """Validate the future recipe using this invocation's sole verified saved object."""
    from .native_raw_handoff import _recipe_plan
    from ..train.recipe import lineage_handoff,native_recipe_binding,native_recipe_provenance
    from ..train.native_continuation import collect_sft_progress
    from .native_supervised import continuation_config,bind_recipe_lineage
    from .native_raw_rounds import _supervised_quantum
    _artifact_binding(recipe_reference)
    config,paths=_controls(config_path);consumed=_read(consumption);reference,saved=verified_checkpoint
    if (consumed.get('schema')not in(CONSUMED,CONSUMED_RAW) or reference!=consumed['checkpoint']
            or binding(reference['path'])['sha256']!=reference['sha256']
            or any(binding(path)!=consumed['after'][key]for key,path in paths.items())):
        raise ValueError('future adaptation lost its exact consumed checkpoint/control pair')
    _no_owner(consumed['controller']);_no_active_operation(paths['config'].parent)
    training=saved['training'];options=training.get('options')or{};lineage=config['recipe_lineage_sha256']
    if (native_recipe_provenance(options.get('recipe_binding'))!=native_recipe_binding(Path(config['production_recipe']),lineage)
            or training.get('global_step')!=consumed['global_step']
            or saved['model_config']['architecture']+':'+saved['model_config']['base_inventory_sha256']!=config['raw_model_id']):
        raise ValueError('future adaptation saved recipe/model lineage changed')
    package=Path(package).resolve();code=code_inventory(package)
    old_package=Path(old_package).resolve()if old_package is not None else package
    old_code=code_inventory(old_package)
    old_plan=_recipe_plan(Path(config['production_recipe']),'cuda');old_plan.update(code=old_code,recipe_sha256=lineage)
    proposed=_recipe_plan(_bound(recipe_reference),'cuda');proposed['code']=code
    frames=collect_sft_progress(training,borrow_frames=True)
    started=[stage['id']for stage in old_plan['recipe']['stages']if stage['id']in frames]
    recorded=_read(consumed['training_receipt']).get('mandatory_sft',{}).get('stages',{})
    if set(started)!=(set(recorded)&{stage['id']for stage in old_plan['recipe']['stages']}) or(consumed['schema']==CONSUMED and consumed['stage_id']not in started):
        raise ValueError('future adaptation lost its actual saved stage inventory')
    closed_withdrawals=_closed_withdrawals(training,frames,old_plan)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    transitioned=lineage_handoff(old_plan,proposed,old_package,package,out.parent/'plan.json',
        out=out.parent,dry_run=True,started_stage_ids=started)
    if transitioned['recipe_sha256']!=lineage:raise ValueError('future adaptation reset its recipe lineage')
    # This seam changes future data/operational policy, never numerical code.
    if transitioned['code']!=code:raise ValueError('future adaptation changed its numerical package')
    declaration=None
    if future_global_plan is not None:
        _,declaration=_future_global(future_global_plan,json.loads(Path(plan_path).read_text()),role,
                                    recipe_reference,config['raw_model_id'],package)
    elif old_code!=code:
        # A changed package can be inspected/qualified before activation, but
        # configuration must have an explicit future invocation contract.
        declaration=None
    future=_future_config(config,recipe_reference,proposed['recipe'],reference,lineage,
                          declaration=declaration,plan_reference=future_global_plan)
    for name,value in(('old-plan.json',old_plan),('future-plan.json',proposed),('lineage-handoff.json',transitioned),('future-config.json',future)):
        write_json_durable(out.parent/name,value)
    old_config=out.parent/'original-config.json';write_bytes_durable(old_config,Path(config_path).read_bytes())
    old_global=out.parent/'original-global-plan.json';write_bytes_durable(old_global,Path(plan_path).read_bytes())
    value={'schema':SCHEMA_RAW if consumed['schema']==CONSUMED_RAW else SCHEMA,'role':role,'consumption':consumption,'checkpoint':reference,
        'global_step':consumed['global_step'],'package':str(package),'code':code,'started_stage_ids':started,
        'old_package':str(old_package),'old_code':old_code,'fresh_runtime_qualification_required':old_code!=code,
        'future_global_plan':future_global_plan,
        'closed_prior_withdrawals':closed_withdrawals,'old_plan':binding(out.parent/'old-plan.json'),'future_plan':binding(out.parent/'future-plan.json'),
        'lineage_handoff':binding(out.parent/'lineage-handoff.json'),'future_config':binding(out.parent/'future-config.json'),
        'old_config':binding(old_config),'old_global_plan':binding(old_global),'recipe':recipe_reference,
        'controls':consumed['after'],'lineage':lineage,'new_sources_earned_visits':0,'whole_recipe_completed':False}
    write_json_durable(out,value);verify(binding(out));return binding(out)


def _verify_controller_recipe(declaration, config, cursor, lineage):
    """Authenticate an already-accounted append against its frozen controller."""
    actual = binding(config['production_recipe'])
    if _artifact_binding(declaration['production_recipe']) == actual:
        return
    reference = cursor.get('resident_current')
    if reference is None:
        raise ValueError('completed adaptation changed recipe without accounted resident history')
    from .native_raw_rounds import _release_requirement
    from ..train.native_resident import current
    # This also authenticates an earlier completed/runtime adaptation before
    # replaying the committed append contracts from that declared recipe.
    requirement = _release_requirement(declaration, {'resident_current': reference})
    state = current(_bound(reference).parent)
    if not state:
        raise ValueError('completed adaptation lost its accounted resident history')
    payload = state['payload']; recipe = payload['recipe']
    expected = {'path': recipe['path'],
                'sha256': recipe.get('file_sha256', recipe['sha256'])}
    if (requirement.get('recipe_lineage_sha256') != lineage
            or requirement.get('recipe_sha256') != recipe['sha256']
            or payload['plan']['recipe_sha256'] != lineage
            or _artifact_binding(expected) != actual):
        raise ValueError('completed adaptation config differs from its accounted resident recipe')


from ..train.recipe import same_replayed_recipe_plan as _same_replayed_recipe_plan


def verify(reference):
    """Replay policy verification over immutable controls; no historical tensor decode."""
    from ..train.recipe import lineage_handoff
    value=_read(reference)
    if value.get('schema')not in(SCHEMA,SCHEMA_RAW) or value.get('new_sources_earned_visits')!=0 or value.get('whole_recipe_completed')is not False:
        raise ValueError('completed adaptation cannot grant future exposure/completion')
    consumed=_read(value['consumption']);config=_read(value['old_config'])
    if (consumed.get('schema')not in(CONSUMED,CONSUMED_RAW) or consumed['checkpoint']!=value['checkpoint']
            or (value['schema']==SCHEMA_RAW)!=(consumed['schema']==CONSUMED_RAW)
            or consumed['global_step']!=value['global_step']or consumed['after']!=value['controls']
            or value['controls']['config']['sha256']!=value['old_config']['sha256']
            or config.get('recipe_lineage_sha256')!=value['lineage']):
        raise ValueError('completed adaptation changed its earned controls')
    _bound(consumed['request']);_bound(consumed['result']);_bound(value['checkpoint'])
    training_receipt=_read(consumed['training_receipt'])
    old_cursor=_read(consumed['snapshots']['cursor'])
    if consumed['schema']==CONSUMED_RAW:
        request,result,_,operation=_raw_operation(config,old_cursor,consumed['request'],consumed['result'],consumed['operation_state'])
        stage={'id':None}
        evidence=consumed.get('raw_admission')or{}
        if set(evidence)!={'admission','objective','graphs','mandatory_sft_sha256'}:
            raise ValueError('completed raw proof lost its exact admission evidence')
        admission=_read(evidence['admission']);_bound(evidence['objective']);_bound(evidence['graphs'])
        from ..common.hashing import canonical_json_sha256_hex
        if (admission.get('schema')!='natlang.native-rl-admission/1'
                or admission.get('policy',{}).get('checkpoint_sha256')!=request['parent_checkpoint']['sha256']
                or admission.get('objective_sha256')!=evidence['objective']['sha256']
                or admission.get('native_graphs_sha256')!=evidence['graphs']['sha256']
                or Path(evidence['admission']['path']).parent!=Path(consumed['training_receipt']['path']).parent
                or Path(evidence['objective']['path'])!=Path(evidence['admission']['path']).parent/'rl-native-objective.json'
                or Path(evidence['graphs']['path'])!=Path(evidence['admission']['path']).parent/'rl-native-graphs.jsonl'
                or canonical_json_sha256_hex(training_receipt.get('mandatory_sft'))!=evidence['mandatory_sft_sha256']):
            raise ValueError('completed raw proof changed its actual policy admission/source')
        if (consumed.get('supervised_stage_credits_added')!=0 or consumed.get('stage_id')is not None
                or training_receipt.get('policy_operation_completed')is not True
                or training_receipt.get('completed')is not False
                or training_receipt.get('stage_id')!=consumed.get('operation_stage_id')):
            raise ValueError('completed raw proof granted supervised credit or lost its real policy stage')
        from .native_raw_rounds import completed_raw_cursor
        if completed_raw_cursor(config,old_cursor,operation)!=_read(consumed['candidate_cursor']):
            raise ValueError('completed raw proof changed the shared raw-to-supervised transition')
        if consumed['after']['state']!=consumed['before']['state'] or consumed['after']['config']!=consumed['before']['config']:
            raise ValueError('completed raw consumption rewrote its production result/config')
    else:
        operation_config=_read(consumed['snapshots']['config'])if consumed.get('publication')else config
        request,result,_,stage=_operation(operation_config,old_cursor,consumed['request'],consumed['result'])
    if (stage['id']!=consumed['stage_id'] or result['checkpoint']!=value['checkpoint']
            or training_receipt.get('checkpoint',{}).get('sha256')!=value['checkpoint']['sha256']
            or training_receipt.get('continuation',{}).get('checkpoint_sha256')!=request['parent_checkpoint']['sha256']
            or training_receipt.get('continuation_boundary',{}).get('new_optimizer_updates')!=consumed['new_optimizer_updates']):
        raise ValueError('completed adaptation changed its exact consumed result/receipt')
    package=Path(value['package']);old_package=Path(value.get('old_package',value['package']));old_code=value.get('old_code',value['code']);old=_read(value['old_plan']);future=_read(value['future_plan'])
    from .native_raw_handoff import _recipe_plan
    expected_old=_recipe_plan(Path(config['production_recipe']),'cuda');expected_old.update(code=old_code,recipe_sha256=value['lineage'])
    expected_future=_recipe_plan(_bound(value['recipe']),'cuda');expected_future['code']=value['code']
    if not _same_replayed_recipe_plan(old,expected_old) or not _same_replayed_recipe_plan(future,expected_future):raise ValueError('completed adaptation changed its exact old/future recipe plans')
    declared=[ref for roles in old['stage_inputs'].values()for key,ref in roles.items()if key=='withdrawn_native_graphs']
    if [item['proof']for item in value['closed_prior_withdrawals']]!=declared:
        raise ValueError('completed adaptation changed its prior withdrawal declarations')
    for item in value['closed_prior_withdrawals']:
        if len(_read(item['proof'])['rows'])!=item['rows']:raise ValueError('completed adaptation prior withdrawal count changed')
    recorded=set(training_receipt.get('mandatory_sft',{}).get('stages',{}))&{stage['id']for stage in old['recipe']['stages']}
    if set(value['started_stage_ids'])!=recorded:raise ValueError('completed adaptation changed actual started stages')
    if (code_inventory(package)!=value['code']or code_inventory(old_package)!=old_code
            or old['code']!=old_code or future['code']!=value['code']
            or value.get('fresh_runtime_qualification_required',False)!=(old_code!=value['code'])):
        raise ValueError('completed adaptation package changed')
    reproduced=lineage_handoff(old,future,old_package,package,Path(reference['path']).parent/'recheck.json',
        out=Path(reference['path']).parent,dry_run=True,started_stage_ids=value['started_stage_ids'])
    if _stable(reproduced)!=_stable(_read(value['lineage_handoff'])):
        raise ValueError('completed adaptation future policy differs from shared lineage verifier')
    future_config=_read(value['future_config'])
    original_global=_read(value['old_global_plan'])
    declaration=None
    if value.get('future_global_plan')is not None:
        _,declaration=_future_global(value['future_global_plan'],original_global,value['role'],
                                    value['recipe'],config['raw_model_id'],package)
    expected_config=_future_config(config,value['recipe'],future['recipe'],value['checkpoint'],value['lineage'],
                                   declaration=declaration,plan_reference=value.get('future_global_plan'))
    if future_config!=expected_config:raise ValueError('completed adaptation changed its declared future invocation')
    if (binding(future_config['production_recipe'])!=_artifact_binding(value['recipe'])
            or future_config['raw_model_id']!=config['raw_model_id']
            or future_config['bootstrap_binding']!=config['bootstrap_binding']
            or (declaration is None and future_config.get('runtime_handoff')!=config.get('runtime_handoff'))
            or (declaration is not None and 'runtime_handoff'in future_config)):
        raise ValueError('completed adaptation changed its original runtime/student authority')
    original_global=_read(value['old_global_plan'])
    declaration=next((item for item in original_global['models']if item['model']==value['role']),None)
    authority=config.get('runtime_binding',config.get('bootstrap_binding',{}))
    if (declaration is None
            or authority.get('plan_sha256')!=value['old_global_plan']['sha256']):
        raise ValueError('completed adaptation old plan differs from its actual controller authority')
    operation_config=_read(consumed['snapshots']['config'])if consumed.get('publication')else config
    _verify_controller_recipe(declaration,operation_config,old_cursor,value['lineage'])
    return value


def verify_declaration(reference,declaration):
    """Authenticate only the proof's declared owned-row recipe transition."""
    proof=verify(reference)
    old_global=_read(proof['old_global_plan'])
    matches=[row for row in old_global['models']if row['model']==proof['role']]
    if len(matches)!=1:
        raise ValueError('completed adaptation lost its exact owned declaration')
    base=matches[0]
    if proof.get('future_global_plan')is not None:
        _,base=_future_global(proof['future_global_plan'],old_global,proof['role'],proof['recipe'],
                              _read(proof['old_config'])['raw_model_id'],proof['package'])
    expected={**base,'production_recipe':proof['recipe'],
        'completed_supervised_adaptation':reference}
    actual={**declaration,'production_recipe':_artifact_binding(declaration['production_recipe'])}
    expected['production_recipe']=_artifact_binding(expected['production_recipe'])
    if actual!=expected:
        raise ValueError('completed adaptation declaration changed outside its owned recipe transition')
    return proof


def dependencies(reference):
    """Exact verifier/recipe/loader inputs; bound JSON provenance is not executable.

    Normal recipe collectors define source, index, CE and admission dependencies.
    Opaque objectives/receipts keep all original bytes without recursively fetching
    historic source paths or mutable resident pointers recorded inside them.
    """
    from ..train.recipe import recipe_file_dependencies, _binding_files
    from .native_raw_handoff import _recipe_plan
    proof=verify(reference);found={};loaded_controls=set()
    def add(ref):
        path=_bound(ref)
        if 'bytes'in ref and (type(ref['bytes'])is not int or ref['bytes']<0
                or path.stat().st_size!=ref['bytes']):
            raise ValueError('completed dependency byte binding changed')
        actual={**binding(path),'bytes':path.stat().st_size}
        if str(path)in found and found[str(path)]!=actual:
            raise ValueError('completed dependency changed during enumeration')
        found[str(path)]=actual
        return path
    def read(ref):return json.loads(add(ref).read_text())
    def controls(ref):
        path=add(ref)
        marker=(str(path),ref['sha256'])
        if marker in loaded_controls:return
        loaded_controls.add(marker)
        value=json.loads(path.read_text());schema=value.get('schema')
        if schema=='natlang.source-host-publishing-control-closure/1':
            add(value['base_loader']);controls(value['base_manifest'])
            add(value['role_authorities'])
            for item in value['controls'].values():add(item)
        elif schema=='natlang.external-owned-controller-closure/1':
            for item in value['modules']:add(item['source'])
        elif schema=='natlang.owned-whole-recipe-controller/1':
            selected_controls(value)
        else:raise ValueError('completed dependency control manifest schema differs')
    def selected_controls(value):
        # These are the exact outside-world loaders selected by the current
        # controller/publisher contract, not arbitrary provenance references.
        for key in ('controller','binder','control_loader','publisher_loader',
                    'role_mechanism_authorities','entry_wrapper'):
            if value.get(key):add(value[key])
        for key in ('control_manifest','publisher_manifest','controller_controls'):
            if value.get(key):controls(value[key])
        for item in value.get('controls',{}).values():add(item)
        for item in value.get('frozen_refs',[]):add(item)
        for item in value.get('operational_inputs',[]):add(item)
        if value.get('native_checkpoint_publication') is not None:
            from ..model.native_checkpoint_control import validate_publication_selection
            selection=validate_publication_selection(value['native_checkpoint_publication'])
            add(selection['producer']);add(selection['dependency_producer'])
    def global_plan(ref):
        value=read(ref)
        selected_controls(value)
        for declaration in value['models']:selected_controls(declaration)
        return value
    def recipe(path):
        # _recipe_plan uses resolve_stage_inputs, whose typed role collectors
        # authenticate CE masks/classification/withdrawal/pending/source DAGs.
        for item in recipe_file_dependencies(path):add(item)
        plan=_recipe_plan(Path(path),'cuda')
        for inputs in plan['stage_inputs'].values():
            for value in inputs.values():
                for item in _binding_files(value):add(item)
    add(reference);add(binding(__file__))
    for path_name,code_name in (('package','code'),('old_package','old_code')):
        package=Path(proof.get(path_name,proof['package']))
        inventory=proof.get(code_name,proof['code'])
        if code_inventory(package)!=inventory:raise ValueError('completed dependency package changed')
        for relative,digest in inventory.items():add({'path':str(package/relative),'sha256':digest})
    consumed=read(proof['consumption'])
    old_config=read(proof['old_config'])
    for key in ('old_plan','future_plan','lineage_handoff','checkpoint'):add(proof[key])
    future_config=read(proof['future_config']);selected_controls(future_config)
    add(proof['old_global_plan'])
    # An explicit later plan supersedes old publisher source ownership. When
    # absent, the original plan is also the selected future declaration.
    global_plan(proof.get('future_global_plan') or proof['old_global_plan'])
    recipe(old_config['production_recipe']);recipe(add(proof['recipe']))
    for item in proof['closed_prior_withdrawals']:add(item['proof'])
    for key in ('training_receipt','operation_state','candidate_cursor'):
        if consumed.get(key):add(consumed[key])
    # Snapshot/candidate bindings are immutable. before/after/controls name the
    # rolling slots only and must never be read after a newer owner publishes.
    for key in ('snapshots','candidates'):
        for item in consumed.get(key,{}).values():add(item)
    if consumed.get('publication'):
        publication=read(consumed['publication'])
        if publication.get('intent'):add(publication['intent'])
        for key in ('snapshots','candidates'):
            for item in publication.get(key,{}).values():add(item)
    request=read(consumed['request']);add(consumed['result'])
    add(request['config'])
    if consumed['schema']==CONSUMED_RAW and request.get('state'):add(request['state'])
    for key in ('admission','objective','graphs'):
        if consumed.get('raw_admission'):add(consumed['raw_admission'][key])
    return found


def _install_resident_plan(proof,config):
    from .native_raw_rounds import _restore_recipe_alias,_resident_release_requirement
    from ..train import native_resident as resident
    from ..train.recipe import recipe_identity_sha256
    plan=_read(proof['lineage_handoff']);future=_read(proof['future_plan'])
    if (plan['recipe_sha256']!=proof['lineage'] or plan['code']!=proof['code']
            or plan['recipe']!=future['recipe'] or plan['stage_inputs']!=future['stage_inputs']):
        raise ValueError('completed adaptation resident plan lost its preserved lineage')
    directory=Path(config['resident_data_directory'])
    if (directory!=_bound(proof['recipe']).parent/'resident-data'
            or config['seed_trainer_parameters'].get('resident_data_directory')!=str(directory)):
        raise ValueError('completed adaptation resident namespace changed')
    target=directory.parent/'recipe-plan.json'
    stage=config['production_stage']
    with resident.locked(directory):
        state=resident.current(directory)
        if not target.exists() and any((directory/name).exists()for name in ('initial.json','current.json')):
            raise ValueError('completed adaptation resident plan missing after adoption')
        active_plan=plan
        if state:
            initial=json.loads((directory/'initial.json').read_text());payload=state['payload']
            active_plan=payload['plan'];old_adoptions=plan.get('resident_adoptions',[])
            adoptions=active_plan.get('resident_adoptions',[]);suffix=adoptions[len(old_adoptions):]
            if (initial['plan_sha256']!=resident.digest(plan) or initial['stage_id']!=stage
                    or payload['stage_id']!=stage or active_plan['code']!=plan['code']
                    or active_plan.get('device')!=plan.get('device')
                    or active_plan.get('handoffs',[])!=plan.get('handoffs',[])
                    or adoptions[:len(old_adoptions)]!=old_adoptions or not suffix
                    or suffix[0]['predecessor']!=resident.digest(plan)
                    or [item['recipe']for item in suffix]!=[item['recipe']for item in state['history']]
                    or any(item['stage_id']!=stage for item in suffix)
                    or payload['prior_recipe']!=initial['recipe']
                    or state['history'][-1]['generation']!=state['generation']):
                raise ValueError('completed adaptation resident history lost its installed-plan anchor')
            requirement={'recipe_sha256':recipe_identity_sha256(_bound(proof['recipe'])),
                'recipe_lineage_sha256':proof['lineage'],'stage_ids':[stage]}
            _resident_release_requirement(requirement,{'resident_current':binding(directory/'current.json')})
            if json.loads(target.read_text())!=active_plan:
                raise ValueError('completed adaptation resident projection differs from its committed plan')
            recipe=payload['recipe']
            if binding(config['production_recipe'])['sha256']!=recipe.get('file_sha256',recipe['sha256']):
                raise ValueError('completed adaptation resident configuration lost its current recipe')
        inputs=active_plan['stage_inputs'][stage]
        for key,source in (('seed_native_graphs','native_graphs'),('held_native_graphs','held_native_graphs')):
            active=config[key];path=_bound(active)if isinstance(active,dict)else Path(active)
            if binding(path)['sha256']!=inputs[source]['sha256']:
                raise ValueError('completed adaptation resident active source changed')
        if state:return binding(target)
        reference={'path':str(target),'sha256':proof['lineage_handoff']['sha256']}
        _restore_recipe_alias(reference,proof['lineage_handoff'])
        return reference


def install_resident_plan(reference,config):
    """Authenticate cached setup without a checkpoint load or authority bypass."""
    if config.get('completed_supervised_adaptation')!=reference:
        raise ValueError('completed adaptation resident configuration belongs to another proof')
    return _install_resident_plan(verify(reference),config)


def configure(plan_path,declaration,bootstrap,config_path,*,owned_controller_fd=None):
    with _completed_leases(config_path,owned_controller_fd):
        return _configure(plan_path,declaration,bootstrap,config_path)


def _configure(plan_path,declaration,bootstrap,config_path):
    """Apply only the proof-bound future invocation at the consumed boundary."""
    from .native_supervised import bind_recipe_lineage
    from .native_bootstrap_metadata import save
    reference=declaration['completed_supervised_adaptation'];proof=verify_declaration(reference,declaration)
    old_global=_read(proof['old_global_plan']);actual=json.loads(Path(plan_path).read_text())
    expected=copy.deepcopy(_read(proof['future_global_plan'])if proof.get('future_global_plan')else old_global);role=proof['role']
    row=next(item for item in expected['models']if item['model']==role)
    # verify_declaration already authenticated this exact recipe identity and
    # any declared byte count; preserve the actual plan's reference shape.
    row.update(production_recipe=declaration['production_recipe'],completed_supervised_adaptation=reference)
    if actual!=expected or next(item for item in actual['models']if item['model']==role)!=declaration:
        raise ValueError('completed adaptation global plan changed outside its owned role')
    path=Path(config_path);prior=json.loads(path.read_text());old=_read(proof['old_config']);future=_read(proof['future_config'])
    if bootstrap!=old['bootstrap_binding']['handoff']or path.resolve()!=Path(proof['controls']['config']['path']).resolve():
        raise ValueError('completed adaptation belongs to another controller/bootstrap')
    if proof.get('fresh_runtime_qualification_required')and proof.get('future_global_plan')is None:
        raise ValueError('completed runtime activation requires its explicit future provider declaration')
    future['completed_supervised_adaptation']=reference
    future['runtime_binding']={'plan_sha256':binding(plan_path)['sha256'],'handoff':bootstrap}
    for key in('supervised_quantum_control','supervised_stage_quantum_control'):
        if key in future:
            declared=future[key].get('plan',{})
            old_authority=(declared.get('sha256')==proof['old_global_plan']['sha256']
                           and declared==(old.get(key)or{}).get('plan'))
            new_authority=(not old_authority and proof.get('future_global_plan')is not None
                           and _artifact_binding(declared)==_artifact_binding(proof['future_global_plan']))
            if not(old_authority or new_authority):
                raise ValueError('completed adaptation changed its declared scheduling authority')
            future[key]={**future[key],'plan':binding(plan_path)}
    if prior==future:
        _install_resident_plan(proof,future)
        save(path.parent/'bootstrap-verification.json',plan_path,declaration,bootstrap,future)
        return path
    from ..train.native_resident import current
    from .native_raw_rounds import _resident_configuration
    directory=Path(future['resident_data_directory']);adopted=current(directory)
    if adopted:
        cursor=json.loads(Path(proof['controls']['cursor']['path']).read_text())
        snapshot=cursor.get('resident_current')
        if not snapshot or current(_bound(snapshot).parent)!=adopted:
            raise ValueError('completed adaptation resident config lacks its exact accounted cursor snapshot')
        expected=_resident_configuration(copy.deepcopy(future),adopted,directory,proof['checkpoint'])
        if prior!=expected:
            raise ValueError('completed adaptation resident configuration changed outside its adopted sources')
        _install_resident_plan(proof,prior)
        save(path.parent/'bootstrap-verification.json',plan_path,declaration,bootstrap,prior)
        return path
    if binding(path)!=proof['controls']['config']:
        raise ValueError('completed adaptation current config has an unknown successor')
    for key in('cursor','state'):
        if binding(proof['controls'][key]['path'])!=proof['controls'][key]:
            raise ValueError('completed adaptation cursor/state changed before application')
    _no_owner(_read(proof['consumption'])['controller']);_no_active_operation(path.parent)
    _install_resident_plan(proof,future)
    write_json_durable(path,future)
    save(path.parent/'bootstrap-verification.json',plan_path,declaration,bootstrap,future)
    return path
