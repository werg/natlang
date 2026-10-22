"""Authenticate and replay a no-credit retry after a pre-update SFT failure."""
from __future__ import annotations

import copy
import hashlib
import json
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


def _verify_failure(request_ref, result_ref, *, config, cursor, state, verified_checkpoint):
    from .native_bridge import verify_result_inventory
    from .native_raw_rounds import _supervised_stage_inventory, cycle_work_directory
    from ..train.recipe import resolve_recipe_data
    request, result = verify_result_inventory(_bound(request_ref), _bound(result_ref))
    if (result.get('status') != 'failed' or result.get('error') != FAILURE
            or any(result.get(key) is not None for key in ('checkpoint','checkpoint_publication','deployment'))
            or any(Path(item['path']).name in ('checkpoint.pt','native-training-receipt.json')
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
        declaration=declaration,plan_reference=future_global_plan)
    return future,lineage_proof,future_plan


def _corrected_request(reference, *, config, verified_checkpoint, package, stage_id):
    from ..common.hashing import canonical_json_sha256_hex
    from ..train.native_resident import code_inventory
    request=_read(reference); selected,saved=verified_checkpoint
    if request.get('schema')=='natlang.native-preference-capacity-case/1':
        if not isinstance(request.get('cases'),list) or not request['cases']:
            raise ValueError('corrected live capacity descriptor has no exact cases')
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
            failed_result, corrected_request, out, *, package, future_global_plan,
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
            'failed_request':failed_request,'failed_result':failed_result,'corrected_request':corrected_request,
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
            failed_request,failed_result,config=config,cursor=cursor,state=state,
            verified_checkpoint=verified_checkpoint)
        old_target=_request_target(old_memory);new_target=_request_target(corrected_request)
        if new_target is None or new_target==old_target:
            raise ValueError('corrected capacity input must be a distinct versioned request/descriptor')
        future,lineage_proof,future_plan=_future(config,plan_path,role,recipe_reference,
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
        request_obj={'schema':SCHEMA,'config':bind(config_path),'plan':bind(plan_path),
            'recipe':recipe_reference,'future_global_plan':future_global_plan,'role':role,
            'package':str(Path(package).resolve()),'failed_request':failed_request,
            'failed_result':failed_result,'corrected_request':corrected_request,
            'controller':controller,'owner_census':owner_census,'parent':verified_checkpoint[0],
            'stage_id':stage['id'],'next_stage':cursor['next_stage'],
            'lineage_handoff_sha256':_digest(lineage_proof),
            'training_credit':0}
        if out.exists():
            existing=_read(bind(out));intent=_read(existing['intent'])
            if existing.get('schema')!=SCHEMA or intent.get('request')!=request_obj:
                raise ValueError('failed retry receipt exists for another exact request')
            return bind(out)
        if intent_path.exists():
            intent=json.loads(intent_path.read_text())
            if intent.get('request')!=request_obj:raise ValueError('failed retry interrupted intent changed')
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
        history_complete=(identity in (live_cursor.get('failed_supervised_retry_history') or [])
            and live_cursor.get('next_stage',0)>intent.get('failed_stage_index',-1))
        if history_complete:
            retry_marker=active.get('failed_supervised_retry') or {}
            if (retry_marker.get('receipt_path')!=str(Path(reference['path']).resolve())
                    or active.get('production_recipe')!=expected_config.get('production_recipe')
                    or active.get('runtime_binding')!=expected_config.get('runtime_binding')
                    or (active.get('trainer_parameters') or {}).get('memory_qualification')!=
                       (expected_config.get('trainer_parameters') or {}).get('memory_qualification')):
                raise ValueError('completed retry metadata cache changed its future invocation identity')
            from .native_bootstrap_metadata import save
            save(root/'bootstrap-verification.json',plan_path,declaration,bootstrap,active)
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
    return path


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
    if not marker and identity in history:
        if cursor.get('next_stage',0)<=intent.get('failed_stage_index',-1):
            raise ValueError('retry history was recorded before its actual stage boundary advanced')
        return cursor
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
    if consumed is None:
        from .native_raw_rounds import cycle_work_directory
        if cycle_work_directory(root,cursor).exists():raise ValueError('retry namespace already contains stale operation output')
        cursor['failed_supervised_retry_consumed']=identity
    return cursor
