"""Exact later-owner authority; bootstrap, checkpoints and work cursors stay put."""
from __future__ import annotations
import contextlib
import fcntl
import json
from pathlib import Path
from ..common.jsonio import write_bytes_durable,write_json_durable
from .native_raw_handoff import binding,verify_handoff,verify_handoff_metadata,verify_drained_recipe_handoff
from ..train.native_readiness import _bound

SCHEMA='natlang.native-raw-runtime-handoff/1'


def _read(reference):return json.loads(_bound(reference).read_text())


def _snap(reference,directory,name):
    raw=_bound(reference).read_bytes()
    if binding(reference['path'])!=reference:raise ValueError('runtime handoff control changed during snapshot')
    path=directory/name
    if path.exists() and path.read_bytes()!=raw:raise ValueError('runtime handoff immutable snapshot differs')
    write_bytes_durable(path,raw)
    return binding(path)


def _paths(config_path,config):
    root=Path(config_path).resolve().parent
    if Path(config['out']).resolve().parent!=root:
        raise ValueError('runtime handoff requires the existing student controller namespace')
    return root,root/'cycle.json',Path(config['out'])/'state.json'


def _transition(reference,handoff,record,saved):
    """A policy owner carries SFT credits; it must never impersonate its old owner."""
    declared=_read(reference)
    source_handoff=declared['handoff']
    if source_handoff==handoff:
        return verify_drained_recipe_handoff(reference,handoff=handoff,verified_handoff=(record,saved))
    source_record,source_saved=verify_handoff(source_handoff)
    transition=verify_drained_recipe_handoff(reference,handoff=source_handoff,
        verified_handoff=(source_record,source_saved))
    from .native_raw_handoff import _drained_plan_authority
    _,_,stage=_drained_plan_authority(source_record,source_saved)
    old=source_saved['training'];current=saved['training']
    old_frame=(old.get('sft_progress') or {}).get(stage)
    carried=(current.get('sft_progress') or {}).get(stage)
    from ..train.native_continuation import _TRANSIENT
    def equal(left,right):
        if type(left)!=type(right):return False
        if isinstance(left,dict):return left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
        if isinstance(left,(list,tuple)):return len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
        import torch
        if isinstance(left,torch.Tensor):return left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
        return left==right
    if (not isinstance(old_frame,dict) or not equal(old_frame,carried)
            or any(key not in old_frame or not equal(value,old_frame[key])
                for key,value in old.items() if key not in _TRANSIENT)
            or any(source_saved['model_config'].get(key)!=saved['model_config'].get(key)
                for key in ('architecture','base_inventory_sha256'))
            or int(current.get('global_step',current.get('step',-1)))<int(old.get('global_step',old.get('step',0)))):
        raise ValueError('runtime handoff current optimizer lost its exact historical source-plan frame')
    return transition


def _bare_authority(authority,record,saved,config):
    """Authenticate an already executed policy, never manufacture a resident plan."""
    from ..train.native_resident import code_inventory
    from ..train.recipe import resolve_recipe_data,resolve_stage_inputs,native_recipe_binding
    from ..common.hashing import canonical_json_sha256_hex as digest
    if record.get('resident'):
        raise ValueError('bare runtime authority cannot replace a genuine resident plan')
    training=saved['training'];options=training.get('options') or {}
    if options.get('resident_data_directory') or options.get('resident_adoption'):
        raise ValueError('bare runtime authority omitted its actual resident owner evidence')
    actual=options.get('recipe_binding')
    if isinstance(actual,str):actual=json.loads(actual)
    lineage=config.get('recipe_lineage_sha256')
    expected=native_recipe_binding(_bound(authority['recipe']),lineage)
    if actual!=expected:
        raise ValueError('bare runtime authority must bind the actual saved current recipe and lineage')
    package=Path(authority['package']);code=code_inventory(package)
    if code!=authority['code'] or digest(saved['model_config'])!=authority['model_config_sha256']:
        raise ValueError('bare runtime authority changed its executed package or model/profile identity')
    stage_id=config['production_stage'];recipe=resolve_recipe_data(expected['path'])
    stages=[item for item in recipe['stages'] if item['id']==stage_id]
    if len(stages)!=1:raise ValueError('bare runtime authority lost its production stage')
    stage=stages[0];frame=training if options.get('stage_id')==stage_id else (training.get('sft_progress') or {}).get(stage_id)
    if not isinstance(frame,dict):raise ValueError('bare runtime authority lost its actual carried production stage')
    frame_options=frame.get('options') or {};frame_recipe=frame_options.get('recipe_binding')
    if isinstance(frame_recipe,str):frame_recipe=json.loads(frame_recipe)
    if not frame_recipe or frame_options.get('stage_id')!=stage_id:
        raise ValueError('bare runtime authority has no authenticated production stage recipe')
    frame_path=_bound({'path':frame_recipe['path'],'sha256':frame_recipe.get('file_sha256',frame_recipe['sha256'])})
    if native_recipe_binding(frame_path,lineage)!=frame_recipe:
        raise ValueError('bare runtime authority changed its carried recipe lineage')
    if [item for item in resolve_recipe_data(frame_path)['stages'] if item['id']==stage_id]!=[stage]:
        raise ValueError('bare runtime authority cannot silently change a carried production stage')
    inputs=resolve_stage_inputs(recipe,stage,{})
    receipts=frame.get('source_receipts') or []
    if len(receipts)<2 or any(receipts[i].get('sha256')!=inputs[role]['sha256'] for i,role in enumerate(('native_graphs','held_native_graphs'))):
        raise ValueError('bare runtime authority production sources differ from saved credits')
    qualification=_read(authority['qualification']);remembered=options.get('memory_qualification') or {}
    if (remembered.get('passed') is not True or qualification.get('passed') is not True
            or any(remembered.get(key)!=qualification.get(key) for key in ('request','parent_checkpoint_sha256'))):
        raise ValueError('bare runtime authority requires the actual saved passed qualification')
    checks=qualification.get('checks',[qualification])
    if not checks or any(check.get('passed') is not True or check.get('runtime_code_sha256')!=digest(code)
            or check.get('identity',{}).get('model_config')!=saved['model_config'] for check in checks):
        raise ValueError('bare runtime authority qualification changed its runtime/model/profile')
    return {'recipe':authority['recipe'],'new_package':str(package),'new_code':code,
            'lineage':lineage,'inputs':inputs,'bare':True}


def _authority(value,handoff,record,saved,config):
    if value.get('bare_direct_owner') is not None:
        if value.get('recipe_handoff') is not None:
            raise ValueError('runtime handoff cannot mix bare and resident authority')
        return _bare_authority(value['bare_direct_owner'],record,saved,config)
    return _transition(value['recipe_handoff'],handoff,record,saved)


def prepare_runtime_handoff(bootstrap,handoff,recipe_handoff,config_path,out,*,operation=None,bare_direct_owner=None):
    """Snapshot an actual later drain under the existing coordinator/worker leases."""
    config_path=Path(config_path);out=Path(out)
    if out.exists():
        value=verify_runtime_handoff(binding(out),bootstrap=bootstrap)
        if (value['handoff']!=handoff or value['recipe_handoff']!=recipe_handoff
                or (value.get('bare_direct_owner') or {}).get('requested')!=bare_direct_owner
                or (value.get('operation') or {}).get('original')!=operation
                or value['sources']['config']['path']!=str(config_path.resolve())):
            raise ValueError('runtime handoff retry changed its requested authority')
        return binding(out)
    config=json.loads(config_path.read_text());root,cursor,state=_paths(config_path,config)
    locks=[root.parent/'controller.lock',Path(config['out'])/'controller.lock']
    if operation:
        request=_read(operation['request']);locks.append(Path(request['out'])/'controller.lock')
    with contextlib.ExitStack() as stack:
        for path in sorted(set(locks)):
            path.parent.mkdir(parents=True,exist_ok=True)
            lease=stack.enter_context(path.open('a'));fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources={'config':binding(config_path),'cursor':binding(cursor),'state':binding(state)}
        current=json.loads(state.read_text());published=binding(current['checkpoint'])
        record,saved=verify_handoff(handoff)
        verify_handoff_metadata(bootstrap)
        authority={'recipe_handoff':recipe_handoff}
        if bare_direct_owner is not None:
            if recipe_handoff is not None:raise ValueError('choose bare or resident runtime authority')
            from ..train.native_resident import code_inventory
            from ..common.hashing import canonical_json_sha256_hex as digest
            directory=out.parent/(out.stem+'-control');directory.mkdir(parents=True,exist_ok=True)
            authority['bare_direct_owner']={**bare_direct_owner,'requested':bare_direct_owner,
                'qualification':_snap(bare_direct_owner['qualification'],directory,'qualification.json'),
                'code':code_inventory(Path(bare_direct_owner['package'])),
                'model_config_sha256':digest(saved['model_config'])}
        transition=_authority(authority,handoff,record,saved,config)
        if operation:
            from .native_bridge import verify_result
            result=verify_result(_bound(operation['request']),_bound(operation['result']))
            if (request.get('native_actor_mode')!='raw-state'
                    or (record['checkpoint']['sha256']!=published['sha256'] and request['parent_checkpoint']['sha256']!=published['sha256'])
                    or result.get('checkpoint',{}).get('sha256')!=record['checkpoint']['sha256']):
                raise ValueError('runtime handoff partial lacks its actual current native operation')
        elif record['checkpoint']['sha256']!=published['sha256']:
            raise ValueError('runtime handoff parent differs from current published optimizer; verified partial required')
        directory=out.parent/(out.stem+'-control');directory.mkdir(parents=True,exist_ok=True)
        snapshots={key:_snap(ref,directory,key+'.json') for key,ref in sources.items()}
        operation_snapshot=None
        if operation:
            operation_state=_read(result['state'])
            selected=(operation_state.get('pending_native_checkpoint') or
                (operation_state.get('partial_checkpoint') if operation_state.get('phase') in {'learner_pending','requested_stop'} else None)
                or operation_state['checkpoint'])
            operation_snapshot={'original':operation,
                'selected_path':str(Path(selected).absolute()),'selected_resolved_path':str(Path(selected).resolve()),
                'request':_snap(operation['request'],directory,'operation-request.json'),
                'result':_snap(operation['result'],directory,'operation-result.json'),
                'state':_snap(result['state'],directory,'operation-state.json')}
        value={'schema':SCHEMA,'bootstrap':bootstrap,'handoff':handoff,'recipe_handoff':recipe_handoff,
            'bare_direct_owner':authority.get('bare_direct_owner'),
            'checkpoint':record['checkpoint'],'published_checkpoint':published,'sources':sources,
            'snapshots':snapshots,'operation':operation_snapshot,'model_id':config.get('raw_model_id'),
            'lineage':config.get('recipe_lineage_sha256'),'previous':config.get('runtime_handoff')}
        _verify(value,bootstrap,record,saved,transition)
        if any(binding(ref['path'])!=ref for ref in sources.values()):
            raise ValueError('runtime handoff current control changed before commit')
        write_json_durable(out,value)
    return binding(out)


def _verify(value,bootstrap,record,saved,transition):
    if value.get('schema')!=SCHEMA or value.get('bootstrap')!=bootstrap or value.get('checkpoint')!=record['checkpoint']:
        raise ValueError('runtime handoff changed original bootstrap or actual later parent')
    config=_read(value['snapshots']['config']);cursor=_read(value['snapshots']['cursor']);state=_read(value['snapshots']['state'])
    root,cursor_path,state_path=_paths(value['sources']['config']['path'],config)
    if (set(value['sources'])!={'config','cursor','state'} or set(value['snapshots'])!=set(value['sources'])
            or any(value['sources'][key]['sha256']!=value['snapshots'][key]['sha256'] for key in value['sources'])
            or value['sources']['cursor']['path']!=str(cursor_path)
            or value['sources']['state']['path']!=str(state_path)):
        raise ValueError('runtime handoff changed its exact original control snapshots')
    actual=saved['model_config']['architecture']+':'+saved['model_config']['base_inventory_sha256']
    plan=None if transition.get('bare') else _read(transition['new_plan'])
    if (config.get('bootstrap_binding',{}).get('handoff')!=bootstrap
            or config.get('raw_model_id')!=actual or value.get('model_id')!=actual
            or value.get('lineage')!=config.get('recipe_lineage_sha256')
            or value['lineage']!=(transition['lineage'] if transition.get('bare') else plan['recipe_sha256'])):
        raise ValueError('runtime handoff replaced its student/base or preserved recipe lineage')
    from .native_supervised import recipe_lineage
    if recipe_lineage(config)!=value['lineage']:
        raise ValueError('runtime handoff config has conflicting recipe lineage')
    if str(Path(state['checkpoint']).resolve())!=value['published_checkpoint']['path']:
        raise ValueError('runtime handoff changed its published current checkpoint')
    if value.get('previous')!=config.get('runtime_handoff'):
        raise ValueError('runtime handoff lost its previous authority')
    if value.get('previous'):
        previous=_read(value['previous'])
        if (previous.get('schema')!=SCHEMA or previous.get('bootstrap')!=bootstrap
                or previous.get('model_id')!=actual or previous.get('lineage')!=value['lineage']):
            raise ValueError('runtime handoff previous authority belongs to another lineage')
    if value.get('operation'):
        from .native_bridge import read_request,RESULT_SCHEMA
        request=read_request(_bound(value['operation']['request']))
        result=_read(value['operation']['result']);operation_state=_read(value['operation']['state'])
        original=value['operation']['original']
        selected=(operation_state.get('pending_native_checkpoint') or
            (operation_state.get('partial_checkpoint') if operation_state.get('phase') in {'learner_pending','requested_stop'} else None)
            or operation_state['checkpoint'])
        # The original full bridge verifier ran before snapshot publication.
        # Recheck its immutable request/result/state joins against the actual
        # drained full checkpoint without reading subsequently mutable job files.
        artifact={item['path']:item for item in result.get('artifacts',[])}
        if (request.get('native_actor_mode')!='raw-state'
                or original['request']['sha256']!=value['operation']['request']['sha256']
                or original['result']['sha256']!=value['operation']['result']['sha256']
                or result.get('schema')!=RESULT_SCHEMA or result.get('kind')!=request['kind']
                or result.get('request_id')!=request['request_id']
                or result.get('request_sha256')!=value['operation']['request']['sha256']
                or result.get('status') not in {'completed','stopped','failed'}
                or result.get('state',{}).get('sha256')!=value['operation']['state']['sha256']
                or artifact.get(result.get('state',{}).get('path'),{}).get('sha256')!=value['operation']['state']['sha256']
                or artifact.get(result.get('checkpoint',{}).get('path'),{}).get('sha256')!=record['checkpoint']['sha256']
                or (record['checkpoint']['sha256']!=value['published_checkpoint']['sha256']
                    and request['parent_checkpoint']['sha256']!=value['published_checkpoint']['sha256'])
                or result.get('checkpoint',{}).get('sha256')!=record['checkpoint']['sha256']
                or str(Path(selected).absolute())!=value['operation']['selected_path']
                or value['operation']['selected_resolved_path']!=record['checkpoint_pin']['original_slot']['path']
                or (saved['training'].get('options') or {}).get('continuation',{}).get('checkpoint_sha256')!=request['parent_checkpoint']['sha256']
                or not Path(request['out']).resolve().is_relative_to(Path(value['sources']['config']['path']).parent)):
            raise ValueError('runtime handoff partial operation differs from current student/parent')
    elif record['checkpoint']['sha256']!=value['published_checkpoint']['sha256']:
        raise ValueError('runtime handoff does not contain its current published checkpoint')
    return config,cursor,state


def verify_runtime_handoff(reference,*,bootstrap):
    value=_read(reference)
    verify_handoff_metadata(bootstrap)
    record,saved=verify_handoff(value['handoff'])
    transition=_authority(value,value['handoff'],record,saved,_read(value['snapshots']['config']))
    _verify(value,bootstrap,record,saved,transition)
    return value


def configure_runtime_handoff(plan_path,declaration,bootstrap,config_path):
    """Rebind invocation metadata; never republish a parent or rewrite a cursor."""
    reference=declaration['runtime_handoff'];proof=verify_runtime_handoff(reference,bootstrap=bootstrap)
    path=Path(config_path);prior=json.loads(path.read_text());old=_read(proof['snapshots']['config'])
    if str(path.resolve())!=proof['sources']['config']['path']:
        raise ValueError('runtime handoff belongs to another current controller namespace')
    if any(prior.get(key)!=old.get(key) for key in ('bootstrap_binding','raw_model_id','out','checkpoint')):
        raise ValueError('runtime handoff replaced its original bootstrap/controller parent')
    bare=proof.get('bare_direct_owner')
    transition=({'recipe':bare['recipe'],'new_code':bare['code']} if bare else _read(proof['recipe_handoff']))
    stage_id=declaration['production_stage']
    if bare:
        from ..train.recipe import resolve_recipe_data,resolve_stage_inputs
        recipe=resolve_recipe_data(_bound(bare['recipe']));stage=next(item for item in recipe['stages'] if item['id']==stage_id)
        inputs=resolve_stage_inputs(recipe,stage,{})
    else:
        plan=_read(transition['new_plan']);inputs=plan['stage_inputs'][stage_id]
    if (declaration['handoff_requirement']['model_id']!=proof['model_id']
            or stage_id!=old['production_stage']):
        raise ValueError('runtime handoff declaration replaced its actual student')
    already=prior.get('runtime_handoff')==reference
    if not already:
        if binding(path)!=proof['sources']['config']:
            raise ValueError('runtime handoff current config changed before application')
        for key in ('cursor','state'):
            if binding(proof['sources'][key]['path'])!=proof['sources'][key]:
                raise ValueError('runtime handoff current operation changed before application')
    config=json.loads(json.dumps(prior if already else old))
    if already and binding(config['production_recipe'])['sha256']!=transition['recipe']['sha256']:
        if bare:raise ValueError('bare runtime handoff current recipe changed without new authority')
        from ..train.native_resident import current
        from ..train.recipe import recipe_identity_sha256
        resident_reference=json.loads(Path(proof['sources']['cursor']['path']).read_text()).get('resident_current')
        adopted=current(_bound(resident_reference).parent) if resident_reference else None
        if (not adopted or binding(config['production_recipe'])['sha256']!=adopted['payload']['recipe']['file_sha256']
                or adopted['payload']['plan']['recipe_sha256']!=proof['lineage']
                or recipe_identity_sha256(_bound(transition['recipe'])) not in [adopted['payload']['prior_recipe']['sha256'],*[item['recipe']['sha256'] for item in adopted['history']]]):
            raise ValueError('runtime handoff later recipe lacks its actual resident adoption history')
    if not already:
        config.update(production_recipe=transition['recipe']['path'],seed_native_graphs=inputs['native_graphs']['path'],
            held_native_graphs=inputs['held_native_graphs']['path'])
    from ..train.recipe import resolve_recipe_data
    recipe=resolve_recipe_data(config['production_recipe']);stage=next(item for item in recipe['stages'] if item['id']==stage_id)
    from .native_supervised import continuation_config,bind_recipe_lineage
    continued,_=continuation_config(config,stage,checkpoint=proof['checkpoint']['path'],adapter=None,out=config['out'])
    # Preserve all controller/case/selection/parent fields; only the recipe-owned
    # supervised invocation contract and provider hooks may change here.
    for key in ('trainer_parameters','seed_native_graphs','held_native_graphs','foundation_native_graphs',
                'continued_native_graphs','continued_held_native_graphs','pending_native_sources','objective_spec'):
        if key in continued:config[key]=continued[key]
        else:config.pop(key,None)
    config['seed_trainer_parameters']={**config['trainer_parameters'],'stage_id':stage_id,'refresh_at_checkpoint':True}
    bind_recipe_lineage(config,proof['lineage'])
    from .native_raw_rounds import _scheduling_controls
    from .native_rounds import configure_provider
    record=_read(proof['handoff'])
    cursor=json.loads(Path(proof['sources']['cursor']['path']).read_text())
    pending=cursor.get('phase')=='sft' or bool(cursor.get('pending_rl_round'))
    _scheduling_controls(config,declaration,plan_path,record,pending=pending)
    config['runtime_handoff']=reference
    config['runtime_binding']={'plan_sha256':binding(plan_path)['sha256'],'handoff':bootstrap}
    original=config['bootstrap_binding']
    protected={key:config[key] for key in ('cases','runtime','task_source','source_admission','actor_command','before_collect') if key in config}
    configure_provider(config,declaration,{},original,path,write=False)
    if pending:
        for key in ('cases','runtime','task_source','source_admission','actor_command','before_collect'):
            if key in protected:config[key]=protected[key]
            else:config.pop(key,None)
    else:
        from .native_raw_rounds import _task_cases
        config['cases']=str(_task_cases(declaration))
        config['runtime']=str(Path(declaration['runtime']).resolve(strict=True))
    # The existing command hook must select the reviewed numerical package;
    # keeping an old env prefix would execute the wrong trainer despite new proof.
    command=config.get('seed_command') or []
    paths=[part.split('=',1)[1] for part in command if part.startswith('PYTHONPATH=')]
    from ..train.native_resident import code_inventory
    selected=None
    if paths:
        selected=next((Path(p)/'natlang_neuralese' for p in paths[-1].split(':') if (Path(p)/'natlang_neuralese').is_dir()),None)
    if selected is None or code_inventory(selected)!=transition['new_code']:
        raise ValueError('runtime handoff seed command must explicitly bind its reviewed Python package')
    write_json_durable(path,config)
    from .native_bootstrap_metadata import save
    save(path.parent/'bootstrap-verification.json',plan_path,declaration,bootstrap,config)
    return path


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',required=True,type=Path)
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--handoff',type=Path)
    parser.add_argument('--recipe-handoff',type=Path)
    parser.add_argument('--bare-recipe',type=Path)
    parser.add_argument('--package',type=Path)
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--operation-request',type=Path)
    parser.add_argument('--operation-result',type=Path)
    args=parser.parse_args(argv)
    bootstrap=binding(args.bootstrap)
    if args.verify:
        verify_runtime_handoff(binding(args.verify),bootstrap=bootstrap)
        reference=binding(args.verify)
    else:
        if not all((args.handoff,args.config,args.out)) or bool(args.recipe_handoff)==bool(args.bare_recipe):
            parser.error('prepare requires --handoff, --config, --out and exactly one of --recipe-handoff/--bare-recipe')
        if args.bare_recipe and not (args.package and args.qualification):
            parser.error('bare authority requires --package and actual --qualification')
        if bool(args.operation_request)!=bool(args.operation_result):
            parser.error('partial operation requires both exact request and result')
        operation=({'request':binding(args.operation_request),'result':binding(args.operation_result)}
                   if args.operation_request else None)
        bare=({'recipe':binding(args.bare_recipe),'package':str(args.package.resolve()),
               'qualification':binding(args.qualification)} if args.bare_recipe else None)
        reference=prepare_runtime_handoff(bootstrap,binding(args.handoff),binding(args.recipe_handoff) if args.recipe_handoff else None,
            args.config,args.out,operation=operation,bare_direct_owner=bare)
    print(json.dumps(reference,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
