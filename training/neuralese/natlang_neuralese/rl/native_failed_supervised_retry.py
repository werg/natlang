"""Authenticate and replay a no-credit retry after a pre-update SFT failure."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from ..common.jsonio import write_bytes_durable, write_json_durable
from ..train.native_readiness import _bound
from ..train.native_resident import binding

SCHEMA = 'natlang.native-failed-supervised-retry/1'
FAILURE = 'ValueError: memory qualification request differs from actual restored model/runtime/packing'


def _read(reference):
    return json.loads(_bound(reference).read_text())


def _digest(value):
    from ..common.hashing import canonical_json_sha256_hex
    return canonical_json_sha256_hex(value)


def _artifact_binding(reference):
    if isinstance(reference,str):reference=binding(reference)
    if not isinstance(reference,dict) or not {'path','sha256'}<=set(reference):
        raise ValueError('retry artifact lacks exact path/SHA binding')
    path=_bound(reference)
    if 'bytes'in reference and (type(reference['bytes'])is not int or path.stat().st_size!=reference['bytes']):
        raise ValueError('retry artifact byte count changed')
    return {'path':reference['path'],'sha256':reference['sha256']}


def _failure_log_source(reference):
    """Return the exact captured worker log bytes and their original remote identity."""
    if isinstance(reference,dict) and {'source','snapshot'}<=set(reference):
        source=reference['source'];snapshot=reference['snapshot']
        if (not isinstance(source,dict) or not {'path','sha256','bytes'}<=set(source)
                or type(source['bytes']) is not int or not isinstance(snapshot,dict)):
            raise ValueError('captured failure log lacks original source and local snapshot bindings')
        path=_bound(snapshot)
        if (path.stat().st_size!=source['bytes'] or snapshot.get('bytes',source['bytes'])!=source['bytes']
                or snapshot['sha256']!=source['sha256']):
            raise ValueError('captured worker log snapshot differs from its original path/SHA/bytes')
        return source,path
    source=reference
    path=_bound(source)
    if 'bytes' not in source or type(source['bytes']) is not int or path.stat().st_size!=source['bytes']:
        raise ValueError('worker failure log requires its exact byte count')
    return source,path


def _verify_failure(request_ref, result_ref, failure_log_ref, *, config, cursor, state, verified_checkpoint):
    from .native_bridge import verify_result_inventory
    from .native_raw_rounds import _supervised_stage_inventory, cycle_work_directory
    from ..train.recipe import resolve_recipe_data
    request, result = verify_result_inventory(_bound(request_ref), _bound(result_ref))
    failure_log_source,failure_log_path=_failure_log_source(failure_log_ref)
    failure_log=failure_log_path.read_text(errors='replace')
    error=result.get('error') or ''
    if (result.get('status') != 'failed' or not error.startswith('CalledProcessError: Command ')
            or 'returned non-zero exit status 1.' not in error
            or Path(failure_log_source['path']).name!='worker.log'
            or Path(failure_log_source['path']).parent.name!=request.get('request_id')
            or 'ValueError: '+FAILURE not in failure_log
            or "raise ValueError('"+FAILURE+"')" not in failure_log
            or 'run_prepared' not in failure_log or '_run_one_prepared' not in failure_log
            or any(result.get(key) is not None for key in ('checkpoint','checkpoint_publication','deployment'))
            or any(Path(item['path']).name in ('checkpoint.pt','native-training-receipt.json',
                   'training-progress.json','progress.json')
                   for item in result.get('artifacts',[]))
            or request.get('kind') != 'round' or request.get('native_actor_mode') != 'raw-state'
            or request.get('parent_checkpoint', {}).get('sha256') != cursor['current_checkpoint']['sha256']
            or request.get('parent_checkpoint', {}).get('path') != cursor['current_checkpoint']['path']):
        raise ValueError('retry authority is not the exact failed pre-update capacity request operation')
    index = cursor.get('next_stage')
    stages = _supervised_stage_inventory(resolve_recipe_data(config['production_recipe']),config['production_stage'])
    if (cursor.get('phase') != 'sft' or cursor.get('pending_rl_round') is not None
            or type(index) is not int or not 0 <= index < len(stages)
            or Path(request['out']).resolve() != cycle_work_directory(Path(config['out']).parent,cursor)/'supervised'/f'{index:03d}'):
        raise ValueError('failed request is not the current same-stage RAW supervised operation')
    invocation = _read(request['config']); parameters = invocation.get('trainer_parameters') or {}
    if (invocation.get('mandatory_sft_only') is not True
            or invocation.get('production_stage') != stages[index]['id']
            or parameters.get('stage_id') != stages[index]['id']):
        raise ValueError('failed request changed mandatory stage identity')
    selected, saved = verified_checkpoint
    if (selected['path'] != cursor['current_checkpoint']['path']
            or selected['sha256'] != cursor['current_checkpoint']['sha256']
            or str(Path(state['checkpoint']).resolve()) != str(Path(selected['path']).resolve())
            or saved.get('model_config') is None):
        raise ValueError('failed operation parent is not the verified current full checkpoint')
    model=saved['model_config']
    model_id=str(model.get('architecture',''))+':'+str(model.get('base_inventory_sha256',''))
    if config.get('raw_model_id')!=model_id:
        raise ValueError('failed retry checkpoint belongs to another RAW model/base')
    memory_ref = parameters.get('memory_qualification')
    if not memory_ref: raise ValueError('failed request omitted its memory qualification binding')
    return request,result,invocation,stages[index],memory_ref,saved


def _future(config, plan_path, role, recipe_reference, future_global_plan,
            package, verified_checkpoint):
    """Reuse shared completed-adaptation recipe/runtime verifiers."""
    from .native_completed_supervised import _future_config, _future_global
    from .native_raw_handoff import _recipe_plan
    from ..train.native_resident import code_inventory
    from ..train.recipe import lineage_handoff
    from ..train.native_continuation import collect_sft_progress
    selected,saved=verified_checkpoint; lineage=config['recipe_lineage_sha256']
    package=Path(package).resolve(); code=code_inventory(package)
    old_plan=_recipe_plan(Path(config['production_recipe']),'cuda')
    old_plan.update(code=code,recipe_sha256=lineage)
    future_plan=_recipe_plan(_bound(recipe_reference),'cuda');future_plan['code']=code
    frames=collect_sft_progress(saved['training'],borrow_frames=True)
    started=[stage['id'] for stage in old_plan['recipe']['stages'] if stage['id'] in frames]
    lineage_proof=lineage_handoff(old_plan,future_plan,package,package,
        Path(recipe_reference['path']).parent/'failed-stage-lineage-handoff.json',
        out=Path(recipe_reference['path']).parent, dry_run=True, started_stage_ids=started)
    if lineage_proof.get('recipe_sha256')!=lineage or lineage_proof.get('code')!=code:
        raise ValueError('retry future recipe resets saved lineage or changes numerical runtime')
    declaration=None
    if future_global_plan is not None:
        _,declaration=_future_global(future_global_plan,json.loads(Path(plan_path).read_text()),
            role,recipe_reference,config['raw_model_id'],package)
    future=_future_config(config,recipe_reference,future_plan['recipe'],selected,lineage,
        declaration=declaration,plan_reference=future_global_plan,preserve_runtime_handoff=True)
    return future,lineage_proof,future_plan,started


def _corrected_request(reference, *, config, verified_checkpoint, package, stage_id):
    from ..common.hashing import canonical_json_sha256_hex
    from ..train.native_resident import code_inventory
    request=_read(reference); selected,saved=verified_checkpoint
    if request.get('schema')=='natlang.native-preference-capacity-case/1':
        from ..common.hashing import sha256_file_hex_cached
        params=config.get('trainer_parameters') or {}
        source=config.get('seed_native_graphs')
        objective=config.get('objective_spec')
        if not isinstance(source,(str,dict)) or not isinstance(objective,(str,dict)):
            raise ValueError('corrected live preference descriptor lacks restored source/objective bindings')
        objective_path=_bound(objective) if isinstance(objective,dict) else Path(objective).resolve(strict=True)
        spec=json.loads(objective_path.read_text())
        expected_objective={'kind':spec.get('kind'),'sha256':sha256_file_hex_cached(objective_path),
            'beta':params.get('preference_beta',spec.get('beta')),
            'margin':params.get('preference_margin',spec.get('margin'))}
        case=request.get('case');reference_case=request.get('numerical_reference')
        def valid_pair(value):
            geometry=value.get('geometry') if isinstance(value,dict) else None
            return (isinstance(value,dict) and isinstance(value.get('id'),str) and value['id']
                and isinstance(value.get('graph_sha256'),str)
                and re.fullmatch(r'[0-9a-f]{64}',value['graph_sha256']) is not None
                and value.get('split')=='train' and isinstance(geometry,dict)
                and type(geometry.get('forward_tokens'))is int and geometry['forward_tokens']>0
                and type(geometry.get('kv_slots'))is int and geometry['kv_slots']>=0)
        if (request.get('schema')!='natlang.native-preference-capacity-case/1'
                or _artifact_binding(request.get('source'))!=_artifact_binding(source)
                or request.get('objective')!=expected_objective
                or not valid_pair(case) or not valid_pair(reference_case)):
            raise ValueError('corrected live preference descriptor changed its restored source/objective/cases')
        return request
    if (request.get('schema')!='natlang.native-memory-qualification-request/1'
            or request.get('runtime_code_sha256')!=canonical_json_sha256_hex(code_inventory(package))
            or request.get('identity',{}).get('model_config')!=saved.get('model_config')):
        raise ValueError('corrected capacity request is not bound to the actual future runtime/model')
    params=config.get('trainer_parameters') or {}
    identity=request['identity']
    for key in ('batch_graphs','batch_tokens','checkpoint_layers','chunk_tokens','ffn_chunk_tokens',
                'checkpoint_input_offload','share_causal_prefixes'):
        if key not in identity or identity[key]!=params.get(key,False if key in ('checkpoint_input_offload','share_causal_prefixes') else None):
            raise ValueError('corrected capacity request differs from the future trainer packing: '+key)
    if config.get('production_stage')!=stage_id:
        raise ValueError('corrected capacity request belongs to another production stage')
    return request


def _request_target(value):
    if isinstance(value,dict) and value.get('path'):
        return str(Path(value['path']).resolve())
    if isinstance(value,str):
        try: parsed=json.loads(value)
        except (ValueError,TypeError): parsed=value
        if isinstance(parsed,str) and Path(parsed).exists():return str(Path(parsed).resolve())
        if isinstance(parsed,dict) and parsed.get('path'):return str(Path(parsed['path']).resolve())
    return None


def prepare(config_path, plan_path, role, recipe_reference, failed_request,
            failed_result, failure_log, corrected_request, out, *, package, future_global_plan,
            controller, owner_census, verified_checkpoint, owned_controller_fd=None):
    """Prepare a durable no-credit retry; canonical files change only in configure()."""
    from .native_completed_supervised import _completed_leases, _no_owner, _no_active_operation
    from ..train.native_resident import owner_alive
    config_path=Path(config_path).resolve(strict=True);plan_path=Path(plan_path).resolve(strict=True)
    out=Path(out).resolve();config=json.loads(config_path.read_text());root=config_path.parent
    if Path(config['out']).resolve()!=root/'production':raise ValueError('retry lost exact RAW namespace')
    if not isinstance(controller,dict) or type(controller.get('pid')) is not int:raise ValueError('retry requires captured controller identity')
    if not isinstance(owner_census,list) or not owner_census or any(owner_alive(item) for item in owner_census):
        raise ValueError('retry requires exact captured operation owners to have exited')
    paths={'config':config_path,'cursor':root/'cycle.json','state':Path(config['out'])/'state.json'}
    if out.exists():
        receipt=_read(binding(out));intent=_read(receipt['intent']);prior=intent['request']
        expected={'config':binding(config_path),'plan':binding(plan_path),'recipe':recipe_reference,
            'future_global_plan':future_global_plan,'role':role,'package':str(Path(package).resolve()),
            'failed_request':failed_request,'failed_result':failed_result,'failure_log':failure_log,
            'corrected_request':corrected_request,
            'controller':controller,'owner_census':owner_census,
            'parent':verified_checkpoint[0],'training_credit':0}
        if (receipt.get('schema')!=SCHEMA or prior.get('schema')!=SCHEMA
                or any(prior.get(key)!=value for key,value in expected.items())
                or prior.get('stage_id')!=receipt.get('stage_id')
                or receipt.get('training_credit')!=0 or receipt.get('whole_recipe_completed') is not False
                or receipt.get('readiness_granted') is not False):
            raise ValueError('failed retry receipt belongs to another exact no-credit request')
        return binding(out)
    with _completed_leases(config_path,owned_controller_fd):
        _no_owner(controller);_no_active_operation(root)
        cursor=json.loads(paths['cursor'].read_text());state=json.loads(paths['state'].read_text())
        request,result,invocation,stage,old_memory,saved=_verify_failure(
            failed_request,failed_result,failure_log,config=config,cursor=cursor,state=state,
            verified_checkpoint=verified_checkpoint)
        old_targets={target for target in (_request_target(value) for value in
            (old_memory if isinstance(old_memory,list) else [old_memory])) if target}
        new_target=_request_target(corrected_request)
        if new_target is None or new_target in old_targets:
            raise ValueError('corrected capacity input must be a distinct versioned request/descriptor')
        future,lineage_proof,future_plan,started_stage_ids=_future(config,plan_path,role,recipe_reference,
            future_global_plan,package,verified_checkpoint)
        effective_plan=future_global_plan or binding(plan_path)
        plan_sha=_artifact_binding(effective_plan)['sha256']
        handoff=config.get('bootstrap_binding',{}).get('handoff')
        future['runtime_binding']={'plan_sha256':plan_sha,'handoff':handoff}
        from .native_raw_rounds import _supervised_stage_inventory
        old_stages=_supervised_stage_inventory(
            __import__('natlang_neuralese.train.recipe',fromlist=['resolve_recipe_data']).resolve_recipe_data(config['production_recipe']),
            config['production_stage'])
        new_stages=_supervised_stage_inventory(future_plan['recipe'],config['production_stage'])
        if (len(new_stages)<=cursor['next_stage'] or new_stages[cursor['next_stage']]['id']!=stage['id']
                or old_stages[cursor['next_stage']]['id']!=stage['id']):
            raise ValueError('future recipe changed the exact uncompleted supervised stage')
        corrected_binding=binding(_bound(corrected_request))
        future['trainer_parameters']['memory_qualification']=corrected_binding
        future['seed_trainer_parameters']['memory_qualification']=corrected_binding
        _corrected_request(corrected_request,config=future,verified_checkpoint=verified_checkpoint,
            package=package,stage_id=stage['id'])
        from .native_raw_rounds import cycle_work_directory
        generation=cursor.get('raw_retry_generation',0)
        while cycle_work_directory(root,{**cursor,'raw_retry_generation':generation}).exists():generation+=1
        future_cursor=copy.deepcopy(cursor);future_cursor['raw_retry_generation']=generation
        future_state=copy.deepcopy(state)
        if (future_cursor['current_checkpoint'].get('sha256')!=verified_checkpoint[0]['sha256']
                or future_cursor['current_checkpoint'].get('path')!=verified_checkpoint[0]['path']
                or str(Path(state['checkpoint']).resolve())!=str(Path(verified_checkpoint[0]['path']).resolve())):
            raise ValueError('failed retry parent changed during preparation')
        from ..common.jsonio import write_json_durable
        from ..train.native_resident import binding as bind
        directory=out.parent/(out.stem+'-control');intent_path=directory/'intent.json'
        source_log,local_log=_failure_log_source(failure_log)
        directory.mkdir(parents=True,exist_ok=True)
        log_snapshot_path=directory/'failure-worker.log'
        if log_snapshot_path.exists() and log_snapshot_path.read_bytes()!=local_log.read_bytes():
            raise ValueError('retry transaction worker-log snapshot differs from its exact source')
        if not log_snapshot_path.exists():write_bytes_durable(log_snapshot_path,local_log.read_bytes())
        log_snapshot=binding(log_snapshot_path)
        if log_snapshot['sha256']!=source_log['sha256']:
            raise ValueError('retry transaction worker-log snapshot lost its original source SHA')
        lineage_path=directory/'lineage-handoff.json';future_plan_path=directory/'future-plan.json'
        if lineage_path.exists():
            if json.loads(lineage_path.read_text())!=lineage_proof:
                raise ValueError('retry lineage handoff snapshot changed')
        else:write_json_durable(lineage_path,lineage_proof)
        if future_plan_path.exists():
            if json.loads(future_plan_path.read_text())!=future_plan:
                raise ValueError('retry future plan snapshot changed')
        else:write_json_durable(future_plan_path,future_plan)
        lineage_handoff_ref=binding(lineage_path);future_plan_ref=binding(future_plan_path)
        request_obj={'schema':SCHEMA,'config':bind(config_path),'plan':bind(plan_path),
            'recipe':recipe_reference,'future_global_plan':future_global_plan,'role':role,
            'package':str(Path(package).resolve()),'failed_request':failed_request,
            'failed_result':failed_result,'failure_log':failure_log,'corrected_request':corrected_request,
            'controller':controller,'owner_census':owner_census,'parent':verified_checkpoint[0],
            'stage_id':stage['id'],'next_stage':cursor['next_stage'],
            'lineage_handoff':lineage_handoff_ref,'future_plan':future_plan_ref,
            'started_stage_ids':started_stage_ids,
            'lineage_handoff_sha256':_digest(lineage_proof),
            'training_credit':0,'failure_log_snapshot':log_snapshot}
        if out.exists():
            existing=_read(bind(out));intent=_read(existing['intent'])
            if existing.get('schema')!=SCHEMA or intent.get('request')!=request_obj:
                raise ValueError('failed retry receipt exists for another exact request')
            return bind(out)
        if intent_path.exists():
            intent=json.loads(intent_path.read_text())
            if intent.get('request')!=request_obj:raise ValueError('failed retry interrupted intent changed')
            before=intent.get('before');snapshots=intent.get('snapshots');candidates=intent.get('candidates')
            if (not isinstance(before,dict) or not isinstance(snapshots,dict) or not isinstance(candidates,dict)
                    or intent.get('failure_log_source')!=source_log
                    or intent.get('failure_log_snapshot')!=log_snapshot
                    or binding(log_snapshot_path)!=log_snapshot):
                raise ValueError('failed retry interrupted intent lost its exact snapshots or failure log')
        else:
            before={name:bind(path) for name,path in paths.items()};directory.mkdir(parents=True,exist_ok=True)
            snapshots={};candidates={}
            for name,path in paths.items():
                snap=directory/f'original-{name}.json';write_bytes_durable(snap,path.read_bytes());snapshots[name]=bind(snap)
            retry_marker={'receipt_path':str(out),'request_sha256':_digest(request_obj),
                'stage_id':stage['id'],'checkpoint':verified_checkpoint[0],
                'training_credit':0,'readiness_granted':False}
            future['failed_supervised_retry']=retry_marker
            future_cursor['failed_supervised_retry']=retry_marker
            values={'config':future,'cursor':future_cursor,'state':future_state}
            for name,value in values.items():
                candidate=directory/f'candidate-{name}.json'
                if name=='state':write_bytes_durable(candidate,paths['state'].read_bytes())
                else:write_json_durable(candidate,value)
                candidates[name]=bind(candidate)
            if any(bind(path)!=before[name] for name,path in paths.items()):raise ValueError('retry predecessor changed before intent')
            intent={'schema':SCHEMA,'request':request_obj,'before':before,'snapshots':snapshots,
                'failure_log_source':source_log,'failure_log_snapshot':log_snapshot,
                'candidates':candidates,'whole_recipe_completed':False,'training_credit':0,
                'readiness_granted':False,'failed_stage_index':cursor['next_stage']}
            write_json_durable(intent_path,intent)
        receipt={'schema':SCHEMA,'intent':bind(intent_path),'request_sha256':_digest(request_obj),
            'stage_id':stage['id'],'checkpoint':verified_checkpoint[0],'training_credit':0,
            'whole_recipe_completed':False,'readiness_granted':False}
        write_json_durable(out,receipt)
        return bind(out)


def configure(reference, plan_path, declaration, bootstrap, config_path, *, owned_controller_fd=None):
    """Install the exact future config at the verified failed-stage boundary."""
    from .native_completed_supervised import _completed_leases, _no_owner, _no_active_operation
    from ..train.native_resident import owner_alive
    declared_retry=declaration.get('failed_supervised_retry')
    if isinstance(declared_retry,str):declared_retry=binding(declared_retry)
    if binding(reference['path'])!=reference:raise ValueError('retry receipt binding changed')
    receipt=_read(reference)
    if not isinstance(receipt.get('intent'),dict) or binding(receipt['intent']['path'])!=receipt['intent']:
        raise ValueError('retry intent binding changed')
    intent=_read(receipt['intent']);request=intent.get('request')
    if (receipt.get('schema')!=SCHEMA or request.get('schema')!=SCHEMA
            or receipt.get('request_sha256')!=_digest(request)
            or intent.get('schema')!=SCHEMA
            or intent.get('training_credit')!=0
            or intent.get('whole_recipe_completed') is not False
            or intent.get('readiness_granted') is not False
            or receipt.get('training_credit')!=0 or receipt.get('whole_recipe_completed') is not False
            or receipt.get('readiness_granted') is not False
            or declared_retry!=reference
            or _artifact_binding(declaration.get('production_recipe'))!=_artifact_binding(request.get('recipe'))
            or Path(config_path).resolve()!=Path(request['config']['path']).resolve()
            or binding(plan_path)['sha256']!=_artifact_binding(request.get('future_global_plan') or request['plan'])['sha256']):
        raise ValueError('retry declaration does not bind its exact no-credit intent')
    log_source=(request.get('failure_log') or {}).get('source') if isinstance(request.get('failure_log'),dict) else None
    if log_source is None:log_source=request.get('failure_log')
    log_snapshot=intent.get('failure_log_snapshot')
    if (not isinstance(log_source,dict) or not isinstance(log_snapshot,dict)
            or request.get('failure_log_snapshot')!=log_snapshot
            or intent.get('failure_log_source')!=log_source
            or binding(log_snapshot['path'])!=log_snapshot
            or log_snapshot['sha256']!=log_source.get('sha256')
            or ('bytes' in log_source and Path(log_snapshot['path']).stat().st_size!=log_source['bytes'])):
        raise ValueError('retry durable intent lost its exact worker failure-log evidence')
    captured_log=Path(log_snapshot['path']).read_text(errors='replace')
    if ('ValueError: '+FAILURE not in captured_log or 'run_prepared' not in captured_log
            or '_run_one_prepared' not in captured_log):
        raise ValueError('retry worker failure log no longer proves the same pre-update failure')
    lineage_reference=request.get('lineage_handoff');future_plan_reference=request.get('future_plan')
    if (not isinstance(lineage_reference,dict) or not isinstance(future_plan_reference,dict)
            or binding(lineage_reference['path'])!=lineage_reference
            or binding(future_plan_reference['path'])!=future_plan_reference):
        raise ValueError('retry intent lost its durable future recipe lineage bindings')
    lineage_value=_read(lineage_reference);future_plan_value=_read(future_plan_reference)
    if (lineage_value.get('recipe_sha256')!=_read(intent['snapshots']['config']).get('recipe_lineage_sha256')
            or lineage_value.get('code')!=future_plan_value.get('code')
            or lineage_value.get('recipe')!=future_plan_value.get('recipe')
            or lineage_value.get('stage_inputs')!=future_plan_value.get('stage_inputs')):
        raise ValueError('retry durable lineage proof differs from its exact old/future plan authority')
    _verify_lineage_replay(request,_read(intent['snapshots']['config']))
    if request.get('future_global_plan'):
        from .native_completed_supervised import _future_global
        original_config=_read(intent['snapshots']['config'])
        if (intent['before']['config']['path']!=request['config']['path']
                or intent['snapshots']['config']['sha256']!=request['config']['sha256']):
            raise ValueError('retry original config snapshot differs from exact prepared parent')
        _future_global(request['future_global_plan'],json.loads(_bound(request['plan']).read_text()),
            request['role'],request['recipe'],original_config['raw_model_id'],
            Path(request['package']))
    path=Path(config_path).resolve();root=path.parent
    paths={'config':path,'cursor':root/'cycle.json','state':Path(json.loads(path.read_text())['out'])/'state.json'}
    with _completed_leases(path,owned_controller_fd):
        _no_owner(request['controller']);_no_active_operation(root)
        if any(owner_alive(item) for item in request['owner_census']):raise ValueError('retry operation owner revived')
        active=json.loads(path.read_text());live_cursor=json.loads(paths['cursor'].read_text())
        expected_config=json.loads(_bound(intent['candidates']['config']).read_text())
        identity={'receipt':reference,'stage_id':receipt['stage_id'],'checkpoint':receipt['checkpoint'],
            'training_credit':0,'readiness_granted':False}
        history_complete=(identity in (live_cursor.get('failed_supervised_retry_history') or []))
        if history_complete:
            retry_marker=active.get('failed_supervised_retry') or {}
            if retry_marker.get('receipt_path')!=str(Path(reference['path']).resolve()):
                raise ValueError('completed retry history differs from the active config receipt')
            active.pop('failed_supervised_retry',None)
            write_json_durable(path,active)
            from .native_bootstrap_metadata import save
            save(root/'bootstrap-verification.json',plan_path,declaration,bootstrap,active)
            _install_retry_resident_plan(request,active)
            return path
        if (request.get('config',{}).get('path')!=str(path)
                or request.get('config',{}).get('sha256')!=intent.get('before',{}).get('config',{}).get('sha256')):
            raise ValueError('retry intent belongs to another canonical config path')
        if not isinstance(request.get('controller'),dict) or not request.get('owner_census'):
            raise ValueError('retry intent omitted its captured owner authority')
        for name, target in paths.items():
            snap=intent.get('snapshots',{}).get(name)
            candidate=intent.get('candidates',{}).get(name)
            before=intent.get('before',{}).get(name)
            if (not isinstance(snap,dict) or binding(snap['path'])!=snap
                    or not isinstance(candidate,dict) or binding(candidate['path'])!=candidate
                    or not isinstance(before,dict)):
                raise ValueError('retry intent snapshot/candidate inventory is incomplete')
        if (set(intent['snapshots'])!=set(paths) or set(intent['candidates'])!=set(paths)
                or set(intent['before'])!=set(paths)):
            raise ValueError('retry intent snapshot/candidate inventory has extra or missing controls')
        for name,target in paths.items():
            snap=intent['snapshots'][name];candidate=intent['candidates'][name]
            if snap['sha256']!=intent['before'][name].get('sha256'):
                raise ValueError('retry snapshot does not preserve its recorded predecessor bytes')
            actual=binding(target)
            if actual!=intent['before'][name] and actual['sha256']!=candidate['sha256']:
                raise ValueError('retry found an unknown controller successor')
        expected={name:json.loads(_bound(ref).read_text()) for name,ref in intent['candidates'].items()}
        for name,target in paths.items():
            if binding(target).get('sha256')!=intent['candidates'][name].get('sha256'):
                write_bytes_durable(target,_bound(intent['candidates'][name]).read_bytes())
        from .native_bootstrap_metadata import save
        save(root/'bootstrap-verification.json',plan_path,declaration,bootstrap,expected['config'])
        _install_retry_resident_plan(request,expected['config'])
    return path


def _install_retry_resident_plan(request,config):
    """Install the retry's authentic lineage plan through the shared verifier."""
    from .native_completed_supervised import _install_resident_plan
    proof={'recipe':request['recipe'],'lineage_handoff':request['lineage_handoff'],
        'future_plan':request['future_plan'],'lineage':_read(request['lineage_handoff'])['recipe_sha256'],
        'code':_read(request['future_plan'])['code'],'package':request['package']}
    return _install_resident_plan(proof,config)


def _verify_lineage_replay(request,original_config):
    """Replay the shared lineage mechanism against the sealed old/future plans."""
    from .native_raw_handoff import _recipe_plan,_stable_handoff_plan
    from ..train.native_resident import code_inventory
    from ..train.recipe import lineage_handoff,resolve_recipe_data
    package=Path(request['package']).resolve();code=code_inventory(package)
    old_plan=_recipe_plan(Path(original_config['production_recipe']),'cuda')
    old_plan.update(code=code,recipe_sha256=original_config['recipe_lineage_sha256'])
    future=_read(request['future_plan']);lineage=_read(request['lineage_handoff'])
    if (future.get('code')!=code or future.get('recipe')!=resolve_recipe_data(_bound(request['recipe']))
            or _digest(lineage)!=request.get('lineage_handoff_sha256')):
        raise ValueError('retry lineage plan changed its exact package or recipe input')
    replay=lineage_handoff(old_plan,future,package,package,
        Path(request['lineage_handoff']['path']).parent/'configure-lineage-replay.json',
        out=Path(request['lineage_handoff']['path']).parent,dry_run=True,
        started_stage_ids=request.get('started_stage_ids'))
    if _stable_handoff_plan(replay)!=_stable_handoff_plan(lineage):
        raise ValueError('retry lineage proof does not replay from its exact old/future recipe plans')


def verify_release(reference,declaration):
    """Verify the durable no-credit retry authority for RAW resident release."""
    if isinstance(reference,str):reference=binding(reference)
    _artifact_binding(reference)
    receipt=_read(reference);intent_ref=receipt.get('intent')
    if not isinstance(intent_ref,dict):
        raise ValueError('retry release intent binding changed')
    _artifact_binding(intent_ref)
    intent=_read(intent_ref);request=intent.get('request') or {}
    declared=declaration.get('failed_supervised_retry')
    if isinstance(declared,str):declared=binding(declared)
    recipe=declaration.get('production_recipe')
    if isinstance(recipe,str):recipe=binding(recipe)
    if (receipt.get('schema')!=SCHEMA or intent.get('schema')!=SCHEMA
            or receipt.get('request_sha256')!=_digest(request)
            or intent.get('training_credit')!=0 or receipt.get('training_credit')!=0
            or intent.get('readiness_granted') is not False or receipt.get('readiness_granted') is not False
            or declared!=reference or _artifact_binding(request.get('recipe'))!=_artifact_binding(recipe)
            or not isinstance(request.get('lineage_handoff'),dict)
            or not isinstance(request.get('future_plan'),dict)):
        raise ValueError('RAW release lost the exact no-credit retry authority')
    lineage=_read(request['lineage_handoff']);future=_read(request['future_plan'])
    _artifact_binding(request['lineage_handoff']);_artifact_binding(request['future_plan'])
    if (lineage.get('recipe')!=future.get('recipe')
            or lineage.get('stage_inputs')!=future.get('stage_inputs')
            or intent.get('failure_log_source') is None
            or intent.get('failure_log_snapshot') is None):
        raise ValueError('RAW release retry lineage/source proof changed')
    from .native_bridge import verify_result_inventory
    from .native_raw_rounds import _supervised_stage_inventory,cycle_work_directory
    from ..train.recipe import resolve_recipe_data
    failed_request,failed_result=verify_result_inventory(_bound(request['failed_request']),
        _bound(request['failed_result']))
    source=request.get('failure_log',{}).get('source')
    snapshot=intent.get('failure_log_snapshot')
    if (not isinstance(source,dict) or intent.get('failure_log_source')!=source
            or request.get('failure_log_snapshot')!=snapshot or not isinstance(snapshot,dict)
            or binding(snapshot['path'])!=snapshot or snapshot['sha256']!=source.get('sha256')
            or Path(source.get('path','')).name!='worker.log'
            or Path(source['path']).parent.name!=failed_request.get('request_id')):
        raise ValueError('RAW release retry lost its exact source worker failure log')
    worker_log=Path(snapshot['path']).read_text(errors='replace')
    if ('ValueError: '+FAILURE not in worker_log or 'run_prepared' not in worker_log
            or '_run_one_prepared' not in worker_log
            or failed_result.get('status')!='failed'
            or not (failed_result.get('error') or '').startswith('CalledProcessError: Command ')
            or 'returned non-zero exit status 1.' not in failed_result.get('error','')
            or any(failed_result.get(key)is not None for key in ('checkpoint','checkpoint_publication','deployment'))
            or any(Path(item['path']).name in ('checkpoint.pt','native-training-receipt.json',
                   'training-progress.json','progress.json')for item in failed_result.get('artifacts',[]))):
        raise ValueError('RAW release retry no longer proves the exact zero-update failure')
    original_cursor=_read(intent['snapshots']['cursor']);original_state=_read(intent['snapshots']['state'])
    original_config=_read(intent['snapshots']['config'])
    index=original_cursor.get('next_stage')
    stages=_supervised_stage_inventory(resolve_recipe_data(original_config['production_recipe']),
        original_config['production_stage'])
    if (original_cursor.get('phase')!='sft' or original_state.get('phase')!='collect'
            or original_cursor.get('pending_rl_round') is not None or type(index)is not int
            or index!=request.get('next_stage') or not 0<=index<len(stages)
            or stages[index]['id']!=request.get('stage_id')
            or failed_request.get('parent_checkpoint',{}).get('sha256')!=original_cursor.get('current_checkpoint',{}).get('sha256')
            or failed_request.get('parent_checkpoint',{}).get('path')!=original_cursor.get('current_checkpoint',{}).get('path')
            or str(Path(original_state.get('checkpoint','')).resolve())!=str(Path(request['parent']['path']).resolve())
            or failed_request.get('parent_checkpoint',{}).get('sha256')!=request['parent'].get('sha256')
            or Path(failed_request.get('out','')).resolve()!=cycle_work_directory(
                Path(original_config['out']).parent,original_cursor)/'supervised'/f'{index:03d}'):
        raise ValueError('RAW release retry changed its saved parent/stage/zero-update boundary')
    if request.get('future_global_plan'):
        from .native_completed_supervised import _future_global
        original_plan=_read(request['plan'])
        original_config=_read(intent['snapshots']['config'])
        _future_global(request['future_global_plan'],original_plan,request['role'],request['recipe'],
            original_config['raw_model_id'],Path(request['package']))
    if (declaration.get('runtime_handoff') is not None
            and original_config.get('runtime_handoff')!=declaration.get('runtime_handoff')):
        raise ValueError('RAW retry does not inherit the declared prior runtime authority')
    from ..train.recipe import recipe_identity_sha256
    predecessor_cursor=_read(intent['snapshots']['cursor'])
    return {'recipe_sha256':recipe_identity_sha256(_bound(recipe)),
        'recipe_lineage_sha256':lineage['recipe_sha256'],'package':request['package'],
        'prior_runtime_handoff':original_config.get('runtime_handoff'),
        'prior_resident_current':predecessor_cursor.get('resident_current'),
        'resident_plan':{'recipe':request['recipe'],'lineage_handoff':request['lineage_handoff'],
            'future_plan':request['future_plan']}}


def dependencies(reference):
    """Exact files required to preserve and replay the retry control authority."""
    receipt=_read(reference);intent_ref=receipt['intent'];intent=_read(intent_ref);request=intent['request']
    refs=[reference,intent_ref,request['config'],request['plan'],request['recipe'],
        request['failed_request'],request['failed_result'],request['corrected_request'],
        request['lineage_handoff'],request['future_plan'],intent['failure_log_snapshot']]
    if request.get('future_global_plan'):refs.append(request['future_global_plan'])
    failure=request.get('failure_log')
    refs.extend(intent.get('snapshots',{}).values());refs.extend(intent.get('candidates',{}).values())
    unique={}
    for ref in refs:
        if isinstance(ref,dict) and ref.get('path') and ref.get('sha256'):
            unique[ref['path']]=ref
    for ref in unique.values():
        _artifact_binding(ref)
    return unique


def consume_boundary(reference, config, cursor, state, root):
    """Verify and mark one fresh SFT boundary before student_cycle launches it."""
    if binding(reference['path'])!=reference:raise ValueError('retry receipt binding changed')
    receipt=_read(reference)
    if not isinstance(receipt.get('intent'),dict) or binding(receipt['intent']['path'])!=receipt['intent']:
        raise ValueError('retry durable intent binding changed')
    intent=_read(receipt['intent'])
    if config.get('failed_supervised_retry',{}).get('receipt_path')!=str(Path(reference['path']).resolve()):
        raise ValueError('retry receipt is not the active config boundary authority')
    if (receipt.get('schema')!=SCHEMA or intent.get('schema')!=SCHEMA
            or receipt.get('request_sha256')!=_digest(intent.get('request',{}))
            or binding(receipt['intent']['path'])!=receipt['intent']):
        raise ValueError('retry receipt or durable intent was changed')
    marker=cursor.get('failed_supervised_retry') or {}
    identity={'receipt':reference,'stage_id':receipt['stage_id'],'checkpoint':receipt['checkpoint'],
        'training_credit':0,'readiness_granted':False}
    consumed=cursor.get('failed_supervised_retry_consumed')
    history=cursor.get('failed_supervised_retry_history') or []
    if not marker and identity in history:return cursor
    receipt_path=marker.get('receipt_path')
    if not receipt_path or Path(receipt_path).resolve()!=Path(reference['path']).resolve():
        raise ValueError('failed-stage retry cursor does not consume the configured receipt')
    if marker.get('request_sha256')!=receipt.get('request_sha256'):
        raise ValueError('failed-stage retry marker differs from durable request')
    if (receipt.get('schema')!=SCHEMA or receipt.get('whole_recipe_completed') is not False
            or receipt.get('readiness_granted') is not False or state.get('phase')!='collect'
            or cursor.get('phase')!='sft' or cursor.get('next_stage') is None
            or cursor['next_stage']!=intent.get('failed_stage_index')
            or cursor.get('current_checkpoint',{}).get('sha256')!=receipt.get('checkpoint',{}).get('sha256')
            or cursor.get('current_checkpoint',{}).get('path')!=receipt.get('checkpoint',{}).get('path')):
        raise ValueError('failed-stage retry is not at its exact mandatory same-stage boundary')
    if consumed is not None and consumed!=identity:raise ValueError('retry receipt was consumed by another boundary')
    updated=copy.deepcopy(cursor)
    if consumed is None:
        from .native_raw_rounds import cycle_work_directory
        if cycle_work_directory(root,cursor).exists():raise ValueError('retry namespace already contains stale operation output')
        updated['failed_supervised_retry_consumed']=identity
    return updated
