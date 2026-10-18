"""Account a sealed completed quantum before adapting the future recipe.

Policy remains with the supervisor; this module verifies durable boundaries.
It creates neither optimizer-owner evidence nor earned visits for new sources.
"""
from __future__ import annotations
import copy
import fcntl
import json
from pathlib import Path

from ..common.jsonio import write_json_durable,write_bytes_durable
from ..train.native_readiness import _bound
from ..train.native_resident import binding,code_inventory

SCHEMA='natlang.native-completed-supervised-adaptation/1'
CONSUMED='natlang.native-completed-supervised-consumption/1'


def _read(reference):return json.loads(_bound(reference).read_text())


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
    from .native_raw_rounds import _supervised_stage_inventory
    from ..train.recipe import resolve_recipe_data
    request,result=verify_result_inventory(_bound(request_ref),_bound(result_ref))
    root=Path(config['out']).parent
    stages=_supervised_stage_inventory(resolve_recipe_data(config['production_recipe']),config['production_stage'])
    index=cursor.get('next_stage');cycle=cursor.get('cycle')
    if (cursor.get('phase')!='sft' or cursor.get('pending_rl_round')is not None
            or type(index)is not int or not 0<=index<len(stages) or type(cycle)is not int
            or request.get('native_actor_mode')!='raw-state' or request['kind']!='round'
            or result.get('status')!='completed'
            or Path(request['out']).resolve()!=root/'cycles'/f'{cycle:06d}'/'supervised'/f'{index:03d}'):
        raise ValueError('completed supervised result differs from the current cycle/stage')
    continuation=_read(request['config']);stage=stages[index]
    if (continuation.get('mandatory_sft_only')is not True
            or continuation.get('production_stage')!=stage['id']
            or continuation.get('trainer_parameters',{}).get('stage_id')!=stage['id']
            or binding(continuation['production_recipe'])!=binding(config['production_recipe'])
            or request['parent_checkpoint']['sha256']!=cursor['current_checkpoint']['sha256']):
        raise ValueError('completed supervised stage changed its sealed recipe/parent')
    return request,result,continuation,stage


def consume(config_path,request_ref,result_ref,out,*,controller):
    """Use the existing publication body, advancing this completed stage once.

    Return the same process's verified full checkpoint for subsequent preparation.
    The default bridge verifier still performs all artifact and resume checks.
    """
    from .native_bridge import verify_result
    from .native_raw_rounds import _publish_supervised_result,_current_checkpoint,_published_checkpoint_state
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    config,paths=_controls(config_path);_no_owner(controller);_no_active_operation(paths['config'].parent)
    with(paths['config'].parent.parent/'controller.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
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
        verified,pair=verify_result(_bound(request_ref),_bound(result_ref),return_raw_checkpoint=True)
        operation_state=_read(verified['state']);saved=pair[1];training=saved['training']
        options=training.get('options')or{}
        receipt_path=Path(verified['checkpoint']['path']).parent/'native-training-receipt.json'
        training_receipt=json.loads(receipt_path.read_text());boundary=training_receipt.get('continuation_boundary')or{}
        updates=boundary.get('new_optimizer_updates')
        quantum=config.get('supervised_stage_quantum_updates',{}).get(stage['id'],config.get('supervised_quantum_updates'))
        if type(quantum)is not int or quantum<=0 or updates!=quantum:
            raise ValueError('completed supervised result differs from its declared finite update quantum')
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


def _stable(plan):
    from .native_raw_handoff import _stable_handoff_plan
    return _stable_handoff_plan(plan)


def _future_config(config,recipe_reference,recipe,checkpoint,lineage):
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
    directory=foundation['parameters'].get('resident_data_directory')
    if directory:future['resident_data_directory']=directory
    else:future.pop('resident_data_directory',None)
    bind_recipe_lineage(future,lineage)
    return future


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


def prepare(config_path,plan_path,role,recipe_reference,consumption,verified_checkpoint,out,*,package):
    """Validate the future recipe using this invocation's sole verified saved object."""
    from .native_raw_handoff import _recipe_plan
    from ..train.recipe import lineage_handoff,native_recipe_binding
    from ..train.native_continuation import collect_sft_progress
    from .native_supervised import continuation_config,bind_recipe_lineage
    from .native_raw_rounds import _supervised_quantum
    config,paths=_controls(config_path);consumed=_read(consumption);reference,saved=verified_checkpoint
    if (consumed.get('schema')!=CONSUMED or reference!=consumed['checkpoint']
            or binding(reference['path'])['sha256']!=reference['sha256']
            or any(binding(path)!=consumed['after'][key]for key,path in paths.items())):
        raise ValueError('future adaptation lost its exact consumed checkpoint/control pair')
    _no_owner(consumed['controller']);_no_active_operation(paths['config'].parent)
    training=saved['training'];options=training.get('options')or{};lineage=config['recipe_lineage_sha256']
    if (options.get('recipe_binding')!=native_recipe_binding(Path(config['production_recipe']),lineage)
            or training.get('global_step')!=consumed['global_step']
            or saved['model_config']['architecture']+':'+saved['model_config']['base_inventory_sha256']!=config['raw_model_id']):
        raise ValueError('future adaptation saved recipe/model lineage changed')
    package=Path(package).resolve();code=code_inventory(package)
    old_plan=_recipe_plan(Path(config['production_recipe']),'cuda');old_plan.update(code=code,recipe_sha256=lineage)
    proposed=_recipe_plan(_bound(recipe_reference),'cuda');proposed['code']=code
    frames=collect_sft_progress(training,borrow_frames=True)
    started=[stage['id']for stage in old_plan['recipe']['stages']if stage['id']in frames]
    recorded=_read(consumed['training_receipt']).get('mandatory_sft',{}).get('stages',{})
    if set(started)!=(set(recorded)&{stage['id']for stage in old_plan['recipe']['stages']}) or consumed['stage_id']not in started:
        raise ValueError('future adaptation lost its actual saved stage inventory')
    closed_withdrawals=_closed_withdrawals(training,frames,old_plan)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    transitioned=lineage_handoff(old_plan,proposed,package,package,out.parent/'plan.json',
        out=out.parent,dry_run=True,started_stage_ids=started)
    if transitioned['recipe_sha256']!=lineage:raise ValueError('future adaptation reset its recipe lineage')
    # This seam changes future data/operational policy, never numerical code.
    if transitioned['code']!=code:raise ValueError('future adaptation changed its numerical package')
    future=_future_config(config,recipe_reference,proposed['recipe'],reference,lineage)
    for name,value in(('old-plan.json',old_plan),('future-plan.json',proposed),('lineage-handoff.json',transitioned),('future-config.json',future)):
        write_json_durable(out.parent/name,value)
    old_config=out.parent/'original-config.json';write_bytes_durable(old_config,Path(config_path).read_bytes())
    old_global=out.parent/'original-global-plan.json';write_bytes_durable(old_global,Path(plan_path).read_bytes())
    value={'schema':SCHEMA,'role':role,'consumption':consumption,'checkpoint':reference,
        'global_step':consumed['global_step'],'package':str(package),'code':code,'started_stage_ids':started,
        'closed_prior_withdrawals':closed_withdrawals,'old_plan':binding(out.parent/'old-plan.json'),'future_plan':binding(out.parent/'future-plan.json'),
        'lineage_handoff':binding(out.parent/'lineage-handoff.json'),'future_config':binding(out.parent/'future-config.json'),
        'old_config':binding(old_config),'old_global_plan':binding(old_global),'recipe':recipe_reference,
        'controls':consumed['after'],'lineage':lineage,'new_sources_earned_visits':0,'whole_recipe_completed':False}
    write_json_durable(out,value);verify(binding(out));return binding(out)


def verify(reference):
    """Replay policy verification over immutable controls; no historical tensor decode."""
    from ..train.recipe import lineage_handoff
    value=_read(reference)
    if value.get('schema')!=SCHEMA or value.get('new_sources_earned_visits')!=0 or value.get('whole_recipe_completed')is not False:
        raise ValueError('completed adaptation cannot grant future exposure/completion')
    consumed=_read(value['consumption']);config=_read(value['old_config'])
    if (consumed.get('schema')!=CONSUMED or consumed['checkpoint']!=value['checkpoint']
            or consumed['global_step']!=value['global_step']or consumed['after']!=value['controls']
            or value['controls']['config']['sha256']!=value['old_config']['sha256']
            or config.get('recipe_lineage_sha256')!=value['lineage']):
        raise ValueError('completed adaptation changed its earned controls')
    _bound(consumed['request']);_bound(consumed['result']);_bound(value['checkpoint'])
    training_receipt=_read(consumed['training_receipt'])
    old_cursor=_read(consumed['snapshots']['cursor'])
    request,result,_,stage=_operation(config,old_cursor,consumed['request'],consumed['result'])
    if (stage['id']!=consumed['stage_id'] or result['checkpoint']!=value['checkpoint']
            or training_receipt.get('checkpoint',{}).get('sha256')!=value['checkpoint']['sha256']
            or training_receipt.get('continuation',{}).get('checkpoint_sha256')!=request['parent_checkpoint']['sha256']
            or training_receipt.get('continuation_boundary',{}).get('new_optimizer_updates')!=consumed['new_optimizer_updates']):
        raise ValueError('completed adaptation changed its exact consumed result/receipt')
    package=Path(value['package']);old=_read(value['old_plan']);future=_read(value['future_plan'])
    from .native_raw_handoff import _recipe_plan
    expected_old=_recipe_plan(Path(config['production_recipe']),'cuda');expected_old.update(code=value['code'],recipe_sha256=value['lineage'])
    expected_future=_recipe_plan(_bound(value['recipe']),'cuda');expected_future['code']=value['code']
    if old!=expected_old or future!=expected_future:raise ValueError('completed adaptation changed its exact old/future recipe plans')
    declared=[ref for roles in old['stage_inputs'].values()for key,ref in roles.items()if key=='withdrawn_native_graphs']
    if [item['proof']for item in value['closed_prior_withdrawals']]!=declared:
        raise ValueError('completed adaptation changed its prior withdrawal declarations')
    for item in value['closed_prior_withdrawals']:
        if len(_read(item['proof'])['rows'])!=item['rows']:raise ValueError('completed adaptation prior withdrawal count changed')
    recorded=set(training_receipt.get('mandatory_sft',{}).get('stages',{}))&{stage['id']for stage in old['recipe']['stages']}
    if set(value['started_stage_ids'])!=recorded:raise ValueError('completed adaptation changed actual started stages')
    if code_inventory(package)!=value['code']or old['code']!=value['code']or future['code']!=value['code']:
        raise ValueError('completed adaptation package changed')
    reproduced=lineage_handoff(old,future,package,package,Path(reference['path']).parent/'recheck.json',
        out=Path(reference['path']).parent,dry_run=True,started_stage_ids=value['started_stage_ids'])
    if _stable(reproduced)!=_stable(_read(value['lineage_handoff'])):
        raise ValueError('completed adaptation future policy differs from shared lineage verifier')
    future_config=_read(value['future_config'])
    expected_config=_future_config(config,value['recipe'],future['recipe'],value['checkpoint'],value['lineage'])
    if future_config!=expected_config:raise ValueError('completed adaptation changed its declared future invocation')
    if (binding(future_config['production_recipe'])!=value['recipe']
            or future_config['raw_model_id']!=config['raw_model_id']
            or future_config['bootstrap_binding']!=config['bootstrap_binding']
            or future_config.get('runtime_handoff')!=config.get('runtime_handoff')):
        raise ValueError('completed adaptation changed its original runtime/student authority')
    original_global=_read(value['old_global_plan'])
    declaration=next((item for item in original_global['models']if item['model']==value['role']),None)
    authority=config.get('runtime_binding',config.get('bootstrap_binding',{}))
    if (declaration is None or declaration['production_recipe']!=binding(config['production_recipe'])
            or authority.get('plan_sha256')!=value['old_global_plan']['sha256']):
        raise ValueError('completed adaptation old plan differs from its actual controller authority')
    return value


def configure(plan_path,declaration,bootstrap,config_path):
    """Apply only the proof-bound future invocation at the consumed boundary."""
    from .native_supervised import bind_recipe_lineage
    from .native_bootstrap_metadata import save
    reference=declaration['completed_supervised_adaptation'];proof=verify(reference)
    old_global=_read(proof['old_global_plan']);actual=json.loads(Path(plan_path).read_text())
    expected=copy.deepcopy(old_global);role=proof['role']
    row=next(item for item in expected['models']if item['model']==role)
    row.update(production_recipe=proof['recipe'],completed_supervised_adaptation=reference)
    if actual!=expected or next(item for item in actual['models']if item['model']==role)!=declaration:
        raise ValueError('completed adaptation global plan changed outside its owned role')
    path=Path(config_path);prior=json.loads(path.read_text());old=_read(proof['old_config']);future=_read(proof['future_config'])
    if bootstrap!=old['bootstrap_binding']['handoff']or path.resolve()!=Path(proof['controls']['config']['path']).resolve():
        raise ValueError('completed adaptation belongs to another controller/bootstrap')
    future['completed_supervised_adaptation']=reference
    future['runtime_binding']={'plan_sha256':binding(plan_path)['sha256'],'handoff':bootstrap}
    for key in('supervised_quantum_control','supervised_stage_quantum_control'):
        if key in future:
            if (future[key].get('plan',{}).get('sha256')!=proof['old_global_plan']['sha256']
                    or Path(future[key]['plan']['path']).resolve()!=Path(plan_path).resolve()):
                raise ValueError('completed adaptation changed its declared scheduling authority')
            future[key]={**future[key],'plan':binding(plan_path)}
    if prior==future:
        save(path.parent/'bootstrap-verification.json',plan_path,declaration,bootstrap,future)
        return path
    if binding(path)!=proof['controls']['config']:
        raise ValueError('completed adaptation current config has an unknown successor')
    for key in('cursor','state'):
        if binding(proof['controls'][key]['path'])!=proof['controls'][key]:
            raise ValueError('completed adaptation cursor/state changed before application')
    _no_owner(_read(proof['consumption'])['controller']);_no_active_operation(path.parent)
    write_json_durable(path,future)
    save(path.parent/'bootstrap-verification.json',plan_path,declaration,bootstrap,future)
    return path
