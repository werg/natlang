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


def _owned_launch_chain(record, snapshots=None):
    """Follow only the byte-bound launch observed during genuine live capture."""
    captured=_read(record['intent']);observed=_read(captured['launch'])
    if observed.get('inprocess_dispatch'):
        from .native_owned_dispatch import verify
        return verify(captured,observed,snapshots)
    def read(name,reference):
        selected=(snapshots or {}).get(name,reference)
        if selected['sha256']!=reference['sha256']:
            raise ValueError('ordinary owner launch snapshot differs from captured authority')
        return _read(selected),reference
    original,original_ref=read('original',observed['original_launch'])
    spawn,spawn_ref=read('spawn',observed['spawn'])
    launch_intent,intent_ref=read('intent',original['intent'])
    continuation,continuation_ref=read('continuation',launch_intent['continuation'])
    argv,argv_ref=read('argv',launch_intent['argv_binding'])
    supervisor_ref=launch_intent['supervisor'];supervisor=(snapshots or {}).get('supervisor',supervisor_ref)
    if supervisor['sha256']!=supervisor_ref['sha256']:
        raise ValueError('ordinary owner supervisor snapshot differs from captured authority')
    _bound(supervisor)
    from .native_raw_handoff import same_process,process_argv
    if (not same_process(original['owner'],captured['owner'])
            or not same_process(spawn['identity'],captured['owner'])
            or original['intent']!=spawn['intent'] or original['argv']!=argv
            or ('script' in spawn and spawn['script']!=launch_intent.get('supervisor'))
            or supervisor_ref['path'] not in
                process_argv(captured['owner'])
            or not any(child['ppid']==captured['owner']['pid'] and
                process_argv(child)==argv
                for child in captured['children'])):
        raise ValueError('ordinary capacity lacks its exact captured supervisor and trainer argv')
    refs=dict(original=original_ref,spawn=spawn_ref,intent=intent_ref,
              continuation=continuation_ref,argv=argv_ref,supervisor=supervisor_ref)
    return continuation,argv,refs


def _ordinary_capacity_authority(authority,record,saved,options,code):
    from ..train.native_capacity_continuation import verify_capacity_certificate
    continuation,argv,launch_refs=_owned_launch_chain(record,authority.get('launch_control'))
    certificate_ref=authority['ordinary_capacity']
    original=authority.get('requested',authority)['ordinary_capacity']
    def argument(flag):
        if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv):
            raise ValueError('ordinary owner needs one explicit '+flag)
        return argv[argv.index(flag)+1]
    actual_recipe=options['recipe_binding']
    if isinstance(actual_recipe,str):actual_recipe=json.loads(actual_recipe)
    parent=(options.get('continuation') or {}).get('checkpoint_sha256')
    if (options.get('objective_spec') or (saved['training'].get('identity') or {}).get('objective')
            or options.get('memory_qualification') or not parent
            or options.get('resident_geometry_qualification')!=original['path']
            or argument('--resident-geometry-qualification')!=original['path']
            or continuation['certificate']!=original or original['sha256']!=certificate_ref['sha256']
            or continuation['parent']['sha256']!=parent or continuation['recipe']!=actual_recipe
            or json.loads(argument('--recipe-binding'))!=actual_recipe
            or argument('--stage-id')!=options.get('stage_id')):
        raise ValueError('ordinary capacity differs from its executed recipe/parent/certificate')
    launch_intent=_read((authority.get('launch_control') or launch_refs)['intent'])
    # The captured supervisor executes this exact configured import environment.
    import_path=(launch_intent.get('env') or {}).get('PYTHONPATH','').split(':')
    selected=next((Path(path)/'natlang_neuralese' for path in import_path
                   if path and (Path(path)/'natlang_neuralese').is_dir()),None)
    if selected is None or selected.resolve()!=Path(authority['package']).resolve():
        raise ValueError('ordinary capacity owner did not execute its declared package')
    certificate=_read(certificate_ref);identity=certificate['identity']
    expected={key:options.get(key) for key in
        ('batch_graphs','batch_tokens','checkpoint_layers','chunk_tokens','checkpoint_input_offload')}
    if (identity.get('model_config')!=saved['model_config'] or identity.get('objective') is not None
            or any(identity.get(key)!=value for key,value in expected.items())
            or bool(identity.get('share_causal_prefixes'))!=bool(options.get('share_causal_prefixes'))
            or identity.get('checkpoint_input_offload_directory')!=options.get('checkpoint_input_offload_directory')
            or any(identity.get('fp8',{}).get(key)!=options.get(option) for key,option in
                (('kind','fp8_training'),('recipe','fp8_recipe'),('amax_history_len','fp8_amax_history_len')))):
        raise ValueError('ordinary capacity changed its saved model/packing/backend identity')
    verify_capacity_certificate(certificate_ref,identity=identity,code=code,parent=parent)


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
    if bool(authority.get('qualification'))==bool(authority.get('ordinary_capacity')):
        raise ValueError('bare runtime authority needs exactly one executed capacity authority')
    if authority.get('ordinary_capacity'):
        _ordinary_capacity_authority(authority,record,saved,options,code)
    else:
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


def _loaded_handoff(reference,verified=None):
    """Reuse this invocation's exact decoded immutable pin, never a disk cache."""
    if verified is None:return verify_handoff(reference)
    record,saved=verified
    if record!=_read(reference) or binding(record['checkpoint']['path'])!=record['checkpoint']:
        raise ValueError('reused handoff decode differs from its exact bound immutable parent')
    return record,saved


def prepare_runtime_handoff(bootstrap,handoff,recipe_handoff,config_path,out,*,operation=None,bare_direct_owner=None,owned_operations=None,deferred_owner_bridge=None,verified_handoff=None):
    """Snapshot an actual later drain under the existing coordinator/worker leases."""
    config_path=Path(config_path);out=Path(out)
    if out.exists():
        value=verify_runtime_handoff(binding(out),bootstrap=bootstrap)
        if (value['handoff']!=handoff or value['recipe_handoff']!=recipe_handoff
                or (value.get('bare_direct_owner') or {}).get('requested')!=bare_direct_owner
                or (value.get('operation') or {}).get('original')!=operation
                or value.get('owned_operations')!=owned_operations
                or value.get('deferred_owner_bridge')!=deferred_owner_bridge
                or value['sources']['config']['path']!=str(config_path.resolve())):
            raise ValueError('runtime handoff retry changed its requested authority')
        return binding(out)
    config=json.loads(config_path.read_text());root,cursor,state=_paths(config_path,config)
    if deferred_owner_bridge is not None and (operation is not None or owned_operations is not None):
        raise ValueError('deferred-owner bridge cannot be combined with another runtime operation authority')
    locks=[root.parent/'controller.lock',Path(config['out'])/'controller.lock']
    if operation:
        request=_read(operation['request']);locks.append(Path(request['out'])/'controller.lock')
    with contextlib.ExitStack() as stack:
        for path in sorted(set(locks)):
            path.parent.mkdir(parents=True,exist_ok=True)
            lease=stack.enter_context(path.open('a'));fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources={'config':binding(config_path),'cursor':binding(cursor),'state':binding(state)}
        current=json.loads(state.read_text());published=binding(current['checkpoint'])
        record,saved=_loaded_handoff(handoff,verified_handoff)
        verify_handoff_metadata(bootstrap)
        authority={'recipe_handoff':recipe_handoff}
        if bare_direct_owner is not None:
            if recipe_handoff is not None:raise ValueError('choose bare or resident runtime authority')
            from ..train.native_resident import code_inventory
            from ..common.hashing import canonical_json_sha256_hex as digest
            directory=out.parent/(out.stem+'-control');directory.mkdir(parents=True,exist_ok=True)
            if bool(bare_direct_owner.get('qualification'))==bool(bare_direct_owner.get('ordinary_capacity')):
                raise ValueError('bare runtime authority needs exactly one executed capacity authority')
            selected='ordinary_capacity' if bare_direct_owner.get('ordinary_capacity') else 'qualification'
            authority['bare_direct_owner']={**bare_direct_owner,'requested':bare_direct_owner,
                selected:_snap(bare_direct_owner[selected],directory,selected+'.json'),
                'code':code_inventory(Path(bare_direct_owner['package'])),
                'model_config_sha256':digest(saved['model_config'])}
            if selected=='ordinary_capacity':
                _,_,refs=_owned_launch_chain(record)
                authority['bare_direct_owner']['launch_control']={name:_snap(ref,directory,'owner-'+name+'.json')
                    for name,ref in refs.items()}
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
        bridge=None
        if deferred_owner_bridge is not None:
            bridge=_verified_deferred_owner_bridge(deferred_owner_bridge,record['checkpoint'],sources)
        value={'schema':SCHEMA,'bootstrap':bootstrap,'handoff':handoff,'recipe_handoff':recipe_handoff,
            'bare_direct_owner':authority.get('bare_direct_owner'),
            'checkpoint':record['checkpoint'],'published_checkpoint':published,'sources':sources,
            'snapshots':snapshots,'operation':operation_snapshot,'owned_operations':owned_operations,
            'deferred_owner_bridge':bridge,'model_id':config.get('raw_model_id'),
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
    if value.get('owned_operations'):
        owned=_read(value['owned_operations']);intent=_read(owned['intent'])
        from ..common.hashing import canonical_json_sha256_hex as digest
        if (owned.get('schema')!='natlang.native-owned-operation-reconciliation/1'
                or intent.get('schema')!=owned.get('schema')
                or owned.get('request_sha256')!=digest(intent['request'])
                or owned.get('global_step')!=saved['training'].get('global_step',saved['training'].get('step'))
                or owned.get('acknowledged_finite_stages')!=intent.get('new_stage_ids')
                or bool(owned.get('retain_mandatory_schedule'))!=bool(intent.get('retain_mandatory_schedule'))
                or intent['request']['operations'][-1]['handoff']!=value['handoff']
                or owned.get('published_checkpoint')!=record['checkpoint']
                or owned.get('whole_recipe_completed')is not False
                or intent['request']['config']!=value['sources']['config']['path']
                or intent['request']['recipe']!=transition.get('recipe')
                or any(intent['candidates'][key]['sha256']!=value['snapshots'][key]['sha256']for key in('state','cursor'))
                or intent['before']['config']['sha256']!=value['snapshots']['config']['sha256']):
            raise ValueError('runtime handoff lost its exact owned-operation publication and stage acknowledgement')
    if value.get('deferred_owner_bridge'):
        _verified_deferred_owner_bridge(value['deferred_owner_bridge'],record['checkpoint'],value['sources'],
            cursor=cursor,state=state,handoff=value['handoff'])
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


def _verified_deferred_owner_bridge(reference,checkpoint,sources,*,cursor=None,state=None,handoff=None):
    """Verify a no-credit bridge as the boundary authority for a fresh SFT schedule."""
    from ..common.hashing import canonical_json_sha256_hex as digest
    if (not isinstance(reference,dict) or not {'path','sha256'}<=set(reference)
            or set(reference)-{'path','sha256','bytes'}):
        raise ValueError('runtime handoff needs an exact deferred-owner bridge receipt')
    path=_bound(reference)
    if binding(path)!=reference:raise ValueError('deferred-owner bridge receipt changed')
    receipt=_read(reference)
    if (receipt.get('schema')!='natlang.native-raw-deferred-owner-bridge/1'
            or receipt.get('published_checkpoint')!=checkpoint
            or receipt.get('whole_recipe_completed') is not False
            or receipt.get('training_credit')!=0 or receipt.get('raw_completion_credit')!=0
            or receipt.get('readiness_granted') is not False or receipt.get('runtime_handoff_required') is not True):
        raise ValueError('deferred-owner bridge does not bind this exact no-credit checkpoint')
    intent_ref=receipt.get('intent');intent=_read(intent_ref)
    request=intent.get('request')
    if (intent.get('schema')!='natlang.native-raw-deferred-owner-bridge/1'
            or receipt.get('request_sha256')!=digest(request)
            or intent.get('whole_recipe_completed') is not False or intent.get('training_credit')!=0
            or intent.get('raw_completion_credit')!=0 or request.get('child_checkpoint')!=checkpoint
            or request.get('config')!=sources['config']
            or request.get('handoff') is None):
        raise ValueError('deferred-owner bridge intent is not the source of this checkpoint')
    if handoff is not None and request.get('handoff')!=handoff:
        raise ValueError('deferred-owner bridge names another completed handoff')
    config=json.loads(_bound(sources['config']).read_text())
    root,cursor_path,state_path=_paths(sources['config']['path'],config)
    paths={'state':state_path,'cursor':cursor_path}
    if (set(intent.get('before',{}))!=set(paths) or set(intent.get('snapshots',{}))!=set(paths)
            or set(intent.get('candidates',{}))!=set(paths)):
        raise ValueError('deferred-owner bridge omitted an exact controller snapshot or candidate')
    originals={}
    for name,path in paths.items():
        before=intent['before'][name];snapshot=intent['snapshots'][name];candidate=intent['candidates'][name]
        if (before.get('path')!=str(path) or snapshot.get('sha256')!=before.get('sha256')
                or binding(_bound(snapshot))!=snapshot):
            raise ValueError('deferred-owner bridge snapshot bindings differ from its intent')
        if not candidate.get('path') or binding(_bound(candidate))!=candidate:
            raise ValueError('deferred-owner bridge candidate binding changed')
        originals[name]=_read(snapshot)
    expected_state={**originals['state'],'checkpoint':checkpoint['path'],'phase':'collect',
        'actor_policy_pending_load':checkpoint['sha256']}
    expected_state.pop('readiness',None)
    expected_cursor={**originals['cursor'],'current_checkpoint':checkpoint}
    expected_cursor.pop('readiness',None);expected_cursor.pop('foundation_qualified',None)
    expected_cursor['deferred_owner_handoff_bridge']={'request_sha256':receipt['request_sha256'],
        'checkpoint':checkpoint,'whole_recipe_completed':False,'raw_completion_credit':0,
        'readiness_granted':False,'runtime_handoff_required':True}
    for name,expected in (('state',expected_state),('cursor',expected_cursor)):
        candidate=intent['candidates'][name]
        if (json.loads(_bound(candidate).read_text())!=expected
                or sources[name].get('sha256')!=candidate.get('sha256')):
            raise ValueError('deferred-owner bridge changed controller debt or is not the current successor')
    if cursor is not None and state is not None:
        marker=cursor.get('deferred_owner_handoff_bridge') or {}
        if (marker.get('request_sha256')!=receipt['request_sha256']
                or marker.get('checkpoint')!=checkpoint or marker.get('whole_recipe_completed') is not False
                or marker.get('raw_completion_credit')!=0 or marker.get('readiness_granted') is not False
                or marker.get('runtime_handoff_required') is not True
                or cursor.get('phase')!='sft' or cursor.get('next_stage')!=0
                or 'pending_rl_round' in cursor
                or state.get('phase')!='collect' or state.get('checkpoint')!=checkpoint['path']
                or state.get('pending_native_checkpoint') is not None
                or state.get('pending_native_round_dir') is not None
                or state.get('pending_semantic_source') is not None):
            raise ValueError('deferred-owner bridge cursor is not its exact SFT-first boundary')
    return reference


def _owned_cycle_boundary(proof,cursor,state,recipe,production_stage):
    """An authenticated end-of-cycle acknowledgement has no sealed next packet."""
    if not proof.get('owned_operations'):
        bridge=proof.get('deferred_owner_bridge')
        if not bridge:return False
        try:
            from .native_raw_rounds import _supervised_stage_inventory
            if not _supervised_stage_inventory(recipe,production_stage):return False
            _verified_deferred_owner_bridge(bridge,proof['checkpoint'],proof['sources'],
                cursor=cursor,state=state,handoff=proof['handoff'])
        except (ValueError,KeyError,TypeError,OSError):return False
        return True
    from .native_raw_rounds import _supervised_stage_inventory
    ids=[stage['id']for stage in _supervised_stage_inventory(recipe,production_stage)]
    owned=_read(proof['owned_operations']);ack=cursor.get('owned_operation_reconciliation')or{}
    retained=owned.get('retain_mandatory_schedule')is True
    if retained:
        intent=_read(owned['intent'])
        actual=[]
        for item in intent['operations']:
            if not actual or actual[-1]!=item['stage_id']:actual.append(item['stage_id'])
        schedule_valid=(intent['request'].get('retain_mandatory_schedule')is True
            and intent.get('mandatory_stage_ids')==ids
            and ack.get('retain_mandatory_schedule')is True
            and cursor.get('next_stage')==0
            and owned.get('acknowledged_finite_stages')==actual
            and bool(actual) and set(actual)<=set(ids))
    else:
        schedule_valid=(cursor.get('next_stage')==len(ids) and owned.get('acknowledged_finite_stages')==ids)
    return (cursor.get('phase')=='sft' and 'pending_rl_round'not in cursor
            and schedule_valid
            and ack.get('receipt')==proof['owned_operations']['path']
            and ack.get('request_sha256')==owned.get('request_sha256')
            and ack.get('runtime_configuration_required')is True
            and state.get('phase')=='collect'
            and not state.get('pending_native_checkpoint') and not state.get('pending_semantic_source'))


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
    if bare:
        config.pop('resident_data_directory',None)
        for parameters in ('trainer_parameters','seed_trainer_parameters'):
            config[parameters].pop('resident_data_directory',None)
    bind_recipe_lineage(config,proof['lineage'])
    from .native_raw_rounds import _scheduling_controls
    from .native_rounds import configure_provider
    record=_read(proof['handoff'])
    cursor=json.loads(Path(proof['sources']['cursor']['path']).read_text())
    state=json.loads(Path(proof['sources']['state']['path']).read_text())
    pending=(cursor.get('phase')=='sft' or 'pending_rl_round'in cursor)
    if _owned_cycle_boundary(proof,cursor,state,recipe,stage_id):pending=False
    _scheduling_controls(config,declaration,plan_path,record,pending=pending)
    config['runtime_handoff']=reference
    config['runtime_binding']={'plan_sha256':binding(plan_path)['sha256'],'handoff':bootstrap}
    original=config['bootstrap_binding']
    protected={key:config[key] for key in ('cases','runtime','task_source','source_admission','actor_command','before_collect','after_actor_ready') if key in config}
    configure_provider(config,declaration,{},original,path,write=False)
    if pending:
        for key in ('cases','runtime','task_source','source_admission','actor_command','before_collect','after_actor_ready'):
            if key in protected:config[key]=protected[key]
            else:config.pop(key,None)
    else:
        from .native_raw_rounds import _task_cases
        config['cases']=str(_task_cases(declaration))
        config['runtime']=str(Path(declaration['runtime']).resolve(strict=True))
    # The existing command hook must select the reviewed numerical package;
    # keeping an old env prefix would execute the wrong trainer despite new proof.
    command=config.get('seed_command') or []
    if bare and any(part=='--resident-data-directory' or part.startswith('--resident-data-directory=') for part in command):
        raise ValueError('bare runtime handoff seed command retains a stale resident owner hook')
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
    parser.add_argument('--ordinary-capacity',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--operation-request',type=Path)
    parser.add_argument('--operation-result',type=Path)
    parser.add_argument('--owned-operations',type=Path,help='bound completed owned operation-chain JSON; no stage is marked complete')
    parser.add_argument('--deferred-owner-bridge',type=Path,
        help='bound no-credit bridge from the selected startup parent to its completed owner checkpoint')
    args=parser.parse_args(argv)
    bootstrap=binding(args.bootstrap)
    if args.verify:
        verify_runtime_handoff(binding(args.verify),bootstrap=bootstrap)
        reference=binding(args.verify)
    else:
        if not all((args.handoff,args.config,args.out)) or bool(args.recipe_handoff)==bool(args.bare_recipe):
            parser.error('prepare requires --handoff, --config, --out and exactly one of --recipe-handoff/--bare-recipe')
        if args.bare_recipe and (not args.package or bool(args.qualification)==bool(args.ordinary_capacity)):
            parser.error('bare authority requires --package and exactly one actual --qualification/--ordinary-capacity')
        if bool(args.operation_request)!=bool(args.operation_result):
            parser.error('partial operation requires both exact request and result')
        operation=({'request':binding(args.operation_request),'result':binding(args.operation_result)}
                   if args.operation_request else None)
        bare=({'recipe':binding(args.bare_recipe),'package':str(args.package.resolve()),
               **({'qualification':binding(args.qualification)} if args.qualification else
                  {'ordinary_capacity':binding(args.ordinary_capacity)})} if args.bare_recipe else None)
        owned=None;verified=None
        if args.owned_operations:
            if operation:parser.error('owned operations cannot substitute another bridge operation')
            manifest_path=_bound(binding(args.owned_operations));manifest=json.loads(manifest_path.read_text())
            if manifest.get('schema')!='natlang.native-owned-training-operations/1':
                parser.error('owned operations require an explicit completed operation-chain manifest')
            operations=manifest.get('operations')
            if (not isinstance(operations,list) or not operations
                    or any(not isinstance(item,dict)or not item.get('receipt')
                        or bool(item.get('handoff'))==bool(item.get('saved_update_recovery'))for item in operations)
                    or operations[-1].get('handoff')!=binding(args.handoff)):
                parser.error('owned operation chain must end at the actual requested full handoff')
            for item in operations:
                _bound(item.get('handoff')or item['saved_update_recovery']);_bound(item['receipt'])
            verified=verify_handoff(binding(args.handoff))
            from .native_raw_rounds import reconcile_owned_operations
            recipe_reference=bare['recipe'] if bare else _read(binding(args.recipe_handoff))['recipe']
            owned=reconcile_owned_operations(args.config,manifest['operations'],recipe_reference,
                args.out.with_name(args.out.stem+'-owned-reconciliation.json'),
                bare_direct_owner=bare,recipe_handoff=binding(args.recipe_handoff) if args.recipe_handoff else None,verified_handoff=verified)
        reference=prepare_runtime_handoff(bootstrap,binding(args.handoff),binding(args.recipe_handoff) if args.recipe_handoff else None,
            args.config,args.out,operation=operation,bare_direct_owner=bare,
            owned_operations=owned,deferred_owner_bridge=(binding(args.deferred_owner_bridge)
                if args.deferred_owner_bridge else None),verified_handoff=verified)
    print(json.dumps(reference,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
