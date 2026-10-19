"""Exact operational recipe commits between sealed supervised workers."""
from __future__ import annotations
import copy
import fcntl
import json
from pathlib import Path
from ..common.hashing import canonical_json_sha256_hex as digest
from ..common.jsonio import write_json_durable
from ..train.native_resident import binding,bound,code_inventory,current

SCHEMA='natlang.native-supervised-boundary-continuation/1'
# Allocation, math, objective and sampling changes need their existing separate
# qualification/handoff. These keys affect monitoring and finite handback only.
PARAMETERS=frozenset({'checkpoint_seconds','checkpoint_every','eval_every','task_eval',
    'task_eval_every','refresh_after_updates','refresh_foundation_readiness'})

def _same(first,second):
    return isinstance(second,dict)and all(first.get(k)==second.get(k)for k in('path','sha256'))

def _file(reference):
    path=bound(reference)
    if 'bytes'in reference and path.stat().st_size!=reference['bytes']:
        raise ValueError('supervised boundary physical artifact size changed')
    return path

def _read(reference):
    return json.loads(_file(reference).read_text())

def _comparable(recipe):
    value=copy.deepcopy(recipe)
    for name in ('id','description'):value.pop(name,None)
    for parameters in value.get('stage_parameter_defaults',{}).values():
        for key in PARAMETERS:parameters.pop(key,None)
    value['stage_parameter_defaults']={k:v for k,v in value.get('stage_parameter_defaults',{}).items()if v}
    for stage in value.get('stages',[]):
        for key in PARAMETERS:stage.get('parameters',{}).pop(key,None)
    return value

def _validate(old,new):
    from ..train.recipe import OPERATIONAL_STAGE_PARAMETERS,validate_stage_parameters,effective_stage_parameters
    if not PARAMETERS<=OPERATIONAL_STAGE_PARAMETERS or _comparable(old)!=_comparable(new):
        raise ValueError('supervised boundary must preserve stages, sources, objective and numerical execution')
    for stage in new['stages']:
        parameters=effective_stage_parameters(stage,[],new)
        validate_stage_parameters(stage['kind'],parameters)
        task=parameters.get('task_eval')
        if task:
            from ..eval.task_cadence import parse_task_eval,require_native_transport
            from ..common.jsonio import iter_jsonl_rows
            spec=require_native_transport(parse_task_eval(task));packet=bound(spec['packet'])
            rows=[row for _,row in iter_jsonl_rows(packet)]
            if max(spec['cases'])>=len(rows)or any(rows[i].get('split')!='test'for i in spec['cases']):
                raise ValueError('supervised boundary task cases must remain protected held data')

def _immutable(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('supervised boundary immutable intent changed')
    else:write_json_durable(path,value)
    return binding(path)

def _stable_declaration(declaration):
    return {k:v for k,v in declaration.items()if k!='supervised_boundary_continuation'}

def _intent_identity(intent):
    value=copy.deepcopy(intent);value['new_config'].pop('supervised_boundary_configuration',None)
    return digest(value)

def _intent(path,selector):
    path=Path(path);value=json.loads(path.read_text())
    if (not path.is_absolute()or path.is_symlink()or selector.get('content_sha256')!=_intent_identity(value)
            or selector.get('request')!=value['request']):
        raise ValueError('supervised boundary intent identity changed')
    return value

def _finish(config_path,cursor_path,intent_path,*,plan_path,declaration,handoff):
    intent=json.loads(intent_path.read_text());_intent(intent_path,intent['new_config']['supervised_boundary_configuration']);request=_read(intent['request'])
    if (request.get('schema')!=SCHEMA or request!=intent['request_value']
            or _stable_declaration(declaration)!=intent['declaration']):
        raise ValueError('supervised boundary recovery authority changed')
    if code_inventory(Path(request['package']['path']))!=request['package']['code']:
        raise ValueError('supervised boundary Python package changed')
    _validate(intent.get('source_plan',intent['old_plan'])['recipe'],intent['new_plan']['recipe'])
    config=json.loads(config_path.read_text());cursor=json.loads(cursor_path.read_text())
    if config not in (intent['old_config'],intent['new_config']) or cursor not in(intent['old_cursor'],intent['new_cursor']):
        raise ValueError('supervised boundary compare-and-swap changed')
    _immutable(intent_path.parent/'recipe-plan.json',intent['new_plan'])
    if config!=intent['new_config']:write_json_durable(config_path,intent['new_config'])
    if cursor!=intent['new_cursor']:write_json_durable(cursor_path,intent['new_cursor'])
    from .native_bootstrap_metadata import save
    save(config_path.parent/'bootstrap-verification.json',plan_path,declaration,handoff,intent['new_config'])
    _immutable(intent_path.parent/'completed.json',{'schema':SCHEMA,'request':intent['request'],
        'intent':binding(intent_path),'config':binding(config_path),'recipe':request['recipe'],
        'training_credit':0,'readiness_granted':False,'whole_recipe_completed':False})
    return intent['new_config'],intent['new_cursor']

def recover_configuration(config_path,plan_path,declaration,handoff):
    """Finish only our own interrupted commit before bootstrap can decode."""
    config_path=Path(config_path);config=json.loads(config_path.read_text())
    transition=config.get('supervised_boundary_configuration')
    if not transition:return False
    intent_path=Path(transition['path']);intent=_intent(intent_path,transition)
    if config!=intent.get('new_config'):return False  # A later normal source adoption owns it.
    if _stable_declaration(declaration)!=intent.get('declaration'):return False
    request=_read(intent['request'])
    if code_inventory(Path(request['package']['path']))!=request['package']['code']:
        raise ValueError('supervised boundary recovery Python package changed')
    _validate(intent.get('source_plan',intent['old_plan'])['recipe'],intent['new_plan']['recipe'])
    cursor_path=config_path.parent/'cycle.json';completed=intent_path.parent/'completed.json'
    if completed.exists():
        receipt=json.loads(completed.read_text());_read(receipt['request'])
        if receipt['intent']!=binding(intent_path)or receipt['config']!=binding(config_path):
            raise ValueError('supervised boundary committed receipt changed')
        # Cursor may have genuinely advanced since this commit. Never replay it.
        from .native_bootstrap_metadata import save
        save(config_path.parent/'bootstrap-verification.json',plan_path,declaration,handoff,config)
        return True
    state=json.loads((Path(config['out'])/'state.json').read_text())
    if state.get('phase')!='collect'or state.get('pending_native_checkpoint')or state.get('pending_native_round_dir'):
        raise ValueError('supervised boundary recovery encountered a sealed operation')
    with (Path(config['out'])/'controller.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _finish(config_path,cursor_path,intent_path,plan_path=plan_path,declaration=declaration,handoff=handoff)
    return True

def apply(plan_path,declaration,handoff,config,cursor,directory):
    """Latest declared successor affects the next worker, never the active one."""
    if not declaration.get('model'):
        if declaration.get('supervised_boundary_continuation')is not None:
            raise ValueError('supervised boundary successor needs an owned model declaration')
        return config,cursor
    directory=Path(directory);config_path=directory/'config.json';cursor_path=directory/'cycle.json'
    live=json.loads(Path(plan_path).read_text());rows=[row for row in live['models']if row['model']==declaration['model']]
    if len(rows)!=1 or _stable_declaration(rows[0])!=_stable_declaration(declaration):
        raise ValueError('supervised boundary live declaration changed outside its continuation')
    declared=rows[0];reference=declared.get('supervised_boundary_continuation')
    selector_path=declared.get('supervised_boundary_continuation_pointer',config.get('supervised_boundary_continuation_pointer'))
    if selector_path is not None:
        selector_path=Path(selector_path)
        if reference is not None or not selector_path.is_absolute()or '..'in selector_path.parts or selector_path.is_symlink():
            raise ValueError('supervised boundary needs one explicit owned continuation selector')
        if not selector_path.exists():return config,cursor
        reference=json.loads(selector_path.read_text())
    if reference is None:return config,cursor
    request=_read(reference)
    if request.get('schema')!=SCHEMA or request.get('model_id')!=config.get('raw_model_id'):
        raise ValueError('supervised boundary successor belongs to another model')
    actual=binding(config['production_recipe'])
    if _same(actual,request.get('recipe')):return config,cursor
    if not _same(actual,request.get('parent_recipe')):
        # An admitted source generation may have advanced while CPU preparation
        # ran. Preserve useful current work; the supervisor must rebase its intent.
        cursor['supervised_boundary_deferred']={'request':reference,'actual_recipe':actual,'training_credit':0}
        write_json_durable(cursor_path,cursor);return config,cursor
    package=Path(request['package']['path']);code=code_inventory(package)
    if code!=request['package']['code']:raise ValueError('supervised boundary package inventory changed')
    from ..train.recipe import load_recipe,native_recipe_binding,resolve_stage_inputs
    old=load_recipe(_file(request['parent_recipe']));new=load_recipe(_file(request['recipe']))
    source=Path(config['resident_data_directory']);resident=current(source)
    old_plan=(resident['payload']['plan']if resident else json.loads((source.parent/'recipe-plan.json').read_text()))
    if old_plan['recipe']!=old or old_plan['code']!=code:
        raise ValueError('supervised boundary resident plan differs from its exact current recipe/runtime')
    source_plan=old_plan
    prepared=request.get('source_preparation')
    if prepared is not None:
        from ..train import native_resident as res
        preparation=_read(prepared);payload=preparation.get('payload')or{}
        predecessor=resident['generation']if resident else digest(old_plan)
        if (preparation.get('schema')!=res.SCHEMA or digest(payload)!=preparation.get('generation')
                or payload.get('predecessor')!=predecessor or payload.get('stage_id')!=config['production_stage']):
            raise ValueError('supervised boundary prepared source has another predecessor/stage')
        source_plan=payload['plan'];res._contract(old_plan,source_plan,config['production_stage'])
        if source_plan['code']!=code or load_recipe(Path(payload['recipe']['path']))!=source_plan['recipe']:
            raise ValueError('supervised boundary source preparation changed recipe/runtime')
        res.bound({'path':payload['recipe']['path'],'sha256':payload['recipe']['file_sha256']})
        for role,key in (('native_graphs','prior_train'),('held_native_graphs','prior_held')):
            original=old_plan['stage_inputs'][config['production_stage']][role]
            if payload.get(key)!={k:original[k]for k in('path','sha256')}:
                raise ValueError('supervised boundary source preparation lost its exact prior inputs')
        res._check_indexes(payload['indexes'])
        limits=res.geometry_capacity(payload['geometry_qualification'],identity=payload['geometry_identity'],code=code)
        if any(payload['geometry'][key]>limit for key,limit in limits.items()):
            raise ValueError('supervised boundary prepared source exceeds qualified whole geometry')
    _validate(source_plan['recipe'],new)
    proposed=copy.deepcopy(source_plan);proposed['recipe']=new
    stage_inputs={stage['id']:({}if stage.get('admitted')is False else resolve_stage_inputs(new,stage,{}))for stage in new['stages']}
    if stage_inputs!=source_plan['stage_inputs']:raise ValueError('supervised boundary cannot substitute source inputs')
    root=directory/'boundary-continuations'/reference['sha256'];root.mkdir(parents=True,exist_ok=True)
    new_config=copy.deepcopy(config);new_config['production_recipe']=request['recipe']['path'];new_config['resident_data_directory']=str(root/'resident-data')
    from .native_supervised import bind_recipe_lineage,continuation_config
    bind_recipe_lineage(new_config,old_plan['recipe_sha256'])
    stage=next(row for row in new['stages']if row['id']==config['production_stage'])
    refreshed,_=continuation_config(new_config,stage,checkpoint=config['checkpoint'],adapter=None,out=config['out'])
    for key in ('seed_command','seed_native_graphs','held_native_graphs','pending_native_sources','continued_native_graphs','continued_held_native_graphs','objective_spec'):
        if key in refreshed:new_config[key]=refreshed[key]
        else:new_config.pop(key,None)
    new_config['seed_trainer_parameters']={**refreshed['trainer_parameters'],'resident_data_directory':new_config['resident_data_directory']}
    publication=request.get('publication')
    if publication is not None:
        from ..model.native_checkpoint_control import validate_publication_selection
        validate_publication_selection(publication['native_checkpoint_publication'])
        pointer=Path(publication['native_checkpoint_publication_owner_pointer'])
        if not pointer.is_absolute()or'..'in pointer.parts:raise ValueError('supervised publication pointer must be absolute')
        if not {'native_checkpoint_publication','native_checkpoint_publication_owner_pointer','native_checkpoint_publication_role','native_checkpoint_publication_runtime_root'}<=set(publication)or set(publication)-{'native_checkpoint_publication','native_checkpoint_publication_owner_pointer','native_checkpoint_publication_role','native_checkpoint_publication_runtime_root','native_checkpoint_publication_reserve_bytes'}:
            raise ValueError('supervised boundary publication cannot change other configuration')
        if publication['native_checkpoint_publication_role']!=declared['model']or Path(publication['native_checkpoint_publication_runtime_root']).resolve()!=package.parents[2].resolve():
            raise ValueError('supervised publication role/Python runtime changed')
        reserve=publication.get('native_checkpoint_publication_reserve_bytes',0)
        if type(reserve)is not int or reserve<0:raise ValueError('supervised publication reserve must be nonnegative bytes')
        new_config.update(publication)
    if publication is not None:
        refreshed,_=continuation_config(new_config,stage,checkpoint=config['checkpoint'],adapter=None,out=config['out'])
        new_config['seed_trainer_parameters']={**refreshed['trainer_parameters'],'resident_data_directory':new_config['resident_data_directory']}
    new_config['runtime_binding']={'plan_sha256':binding(plan_path)['sha256'],'handoff':handoff}
    intent_path=root/'intent.json'
    next_cursor=copy.deepcopy(cursor);next_cursor.pop('readiness',None);next_cursor.pop('foundation_qualified',None);next_cursor.pop('supervised_boundary_deferred',None)
    if next_cursor.get('resident_current'):
        next_cursor.setdefault('prior_resident_authorities',[]).append(next_cursor.pop('resident_current'))
    next_cursor['supervised_boundary_continuation']={'request':reference,'recipe':request['recipe'],'lineage':old_plan['recipe_sha256'],'intent':str(intent_path),'training_credit':0,'readiness_granted':False}
    intent={'schema':SCHEMA,'request':reference,'request_value':request,'declaration':_stable_declaration(declared),'old_config':config,'new_config':new_config,'old_cursor':cursor,'new_cursor':next_cursor,'old_plan':old_plan,'source_plan':source_plan,'new_plan':proposed}
    new_config['supervised_boundary_configuration']={'path':str(intent_path),'content_sha256':_intent_identity(intent),'request':reference}
    with (Path(config['out'])/'controller.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((Path(config['out'])/'state.json').read_text())
        if cursor.get('phase')!='sft'or cursor.get('pending_rl_round')or state.get('phase')!='collect'or state.get('pending_native_checkpoint')or state.get('pending_native_round_dir'):
            raise ValueError('supervised boundary cannot cross a sealed operation')
        if selector_path is not None and json.loads(selector_path.read_text())!=reference:
            raise ValueError('supervised boundary selector advanced before commit; retry current intent')
        _immutable(intent_path,intent)
        return _finish(config_path,cursor_path,intent_path,plan_path=plan_path,declaration=declared,handoff=handoff)


def dependencies(config):
    transition=config.get('supervised_boundary_configuration')
    if not transition:return []
    intent_path=Path(transition['path']);intent=_intent(intent_path,transition)
    request=_read(intent['request'])
    refs=[intent['request'],request['parent_recipe'],request['recipe'],binding(intent_path),
        binding(intent_path.parent/'recipe-plan.json'),binding(Path(__file__))]
    if request.get('source_preparation'):refs.append(request['source_preparation'])
    return refs


def release_requirement(requirement,cursor):
    transition=cursor.get('supervised_boundary_continuation')
    if not transition:return requirement
    intent_path=Path(transition['intent']);intent=json.loads(intent_path.read_text())
    selector=intent['new_config']['supervised_boundary_configuration'];_intent(intent_path,selector)
    receipt=json.loads((intent_path.parent/'completed.json').read_text())
    if (receipt['intent']!=binding(intent_path)or receipt['request']!=transition['request']
            or transition['recipe']!=intent['request_value']['recipe']
            or transition['lineage']!=intent['old_plan']['recipe_sha256']):
        raise ValueError('supervised boundary release changed its committed authority')
    from ..train.recipe import recipe_identity_sha256
    return {**requirement,'recipe_sha256':recipe_identity_sha256(_file(transition['recipe'])),
        'recipe_lineage_sha256':transition['lineage']}
