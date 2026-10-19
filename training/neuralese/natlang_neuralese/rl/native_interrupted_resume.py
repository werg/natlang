"""Leased publication of interrupted earned state, without successful handoff credit."""
import copy
import fcntl
import json
from contextlib import ExitStack
from pathlib import Path
from ..common.hashing import canonical_json_sha256_hex as digest
from ..common.jsonio import write_json_durable as write, write_bytes_durable
from ..train.native_readiness import _bound
from ..train.native_resident import binding

SCHEMA = 'natlang.native-interrupted-supervised-resume/1'


def _candidates(config, cursor, state, chain):
    from . import native_raw_rounds as raw
    from ..train.recipe import resolve_recipe_data
    if (cursor.get('phase') != 'sft' or cursor.get('pending_rl_round') is not None
            or state.get('phase') != 'collect'
            or any(state.get(k) is not None for k in ('pending_native_checkpoint','pending_native_round_dir'))):
        raise ValueError('interrupted resume cannot bypass a sealed policy or other central operation')
    stages = raw._supervised_stage_inventory(resolve_recipe_data(config['production_recipe']),config['production_stage'])
    index = cursor.get('next_stage',0)
    if type(index) is not int or not 0 <= index < len(stages) or chain[-1]['stage_id'] != stages[index]['id']:
        raise ValueError('interrupted resume cannot advance or substitute its current supervised stage')
    selected = chain[-1]['checkpoint']
    new_state = raw._published_checkpoint_state(state, selected)
    new_state.pop('readiness',None)
    new_cursor = copy.deepcopy(cursor)
    for key in ('readiness','foundation_qualified'):
        new_cursor.pop(key,None)
    generation = cursor.get('raw_retry_generation',0)
    if type(generation) is not int or generation < 0:
        raise ValueError('interrupted resume lost its exact retry namespace')
    new_cursor.update(current_checkpoint=selected,raw_retry_generation=generation+1)
    new_cursor.setdefault('interrupted_supervised_operations',[]).extend(
        {'saved_update_recovery':item['saved_update_recovery'],'receipt':item['receipt']}
        for item in chain if item.get('saved_update_recovery'))
    new_config = copy.deepcopy(config)
    from .native_raw_handoff import verify_saved_update_acknowledgement
    record = verify_saved_update_acknowledgement(chain[-1]['saved_update_recovery'],receipt=chain[-1]['receipt'])
    evidence = record.get('resident') or {}
    if evidence.get('current'):
        current = json.loads(_bound(evidence['current']).read_text())
        raw._resident_configuration(new_config,current,Path(evidence['directory']),selected)
        new_cursor['resident_current'] = evidence['current']
    return {'config':new_config,'cursor':new_cursor,'state':new_state}


def publish_interrupted_resume(config_path, operations, out, *, reason):
    """Publish an authenticated failed tail into a fresh same-stage retry namespace."""
    from . import native_raw_rounds as raw
    from . import native_interrupted_update as interrupted
    from .native_raw_handoff import verify_saved_update_acknowledgement
    if not isinstance(reason,str) or not reason.strip() or not operations:
        raise ValueError('interrupted resume needs its actual operations and practical reason')
    config_path=Path(config_path).resolve(strict=True);out=Path(out).resolve()
    config=json.loads(config_path.read_text());root=config_path.parent
    if Path(config['out']).resolve().parent != root:
        raise ValueError('interrupted resume lost its exact RAW namespace')
    paths={'config':config_path,'cursor':root/'cycle.json','state':Path(config['out'])/'state.json'}
    request={'schema':SCHEMA,'config':str(config_path),'operations':operations,'reason':reason}
    directory=out.parent/(out.stem+'-control');intent_path=directory/'intent.json'
    with ExitStack() as stack:
        for path in sorted({root.parent/'controller.lock',Path(config['out'])/'controller.lock'}):
            lease=stack.enter_context(path.open('a'));fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists():
            committed=json.loads(out.read_text())
            intent=json.loads(_bound(committed['intent']).read_text())
            if (committed.get('schema')!=SCHEMA or committed.get('request_sha256')!=digest(request)
                    or committed['intent']!=binding(intent_path) or intent.get('request')!=request
                    or committed.get('published_checkpoint')!=intent.get('selected_checkpoint')
                    or committed.get('training_credit')!=0 or committed.get('readiness_granted') is not False
                    or committed.get('whole_recipe_completed') is not False or committed.get('same_stage_retry') is not True):
                raise ValueError('interrupted resume historical receipt changed')
            for ref in intent['candidates'].values():_bound(ref)
            for item in operations:
                if item.get('saved_update_recovery'):
                    verify_saved_update_acknowledgement(item['saved_update_recovery'],receipt=item['receipt'])
            return binding(out)  # Never reopen pruned payloads or rewind later earned progress.
        if intent_path.exists():
            intent=json.loads(intent_path.read_text())
            if intent.get('schema')!=SCHEMA or intent.get('request')!=request:
                raise ValueError('interrupted resume retry changed its exact transaction')
            originals={name:json.loads(_bound(ref).read_text()) for name,ref in intent['snapshots'].items()}
            if set(originals)!=set(paths) or any(binding(_bound(intent['snapshots'][name]))['sha256']!=intent['before'][name]['sha256'] for name in paths):
                raise ValueError('interrupted resume lost its exact predecessor snapshots')
        else:
            if out.exists():raise ValueError('interrupted resume receipt lost its durable intent')
            originals={name:json.loads(path.read_text()) for name,path in paths.items()}
        original=originals['config'];state=originals['state'];cursor=originals['cursor']
        published=binding(state['checkpoint'])
        if {k:cursor['current_checkpoint'].get(k) for k in ('path','sha256')}!=published:
            raise ValueError('interrupted resume predecessor state and cursor disagree')
        record=verify_saved_update_acknowledgement(operations[-1]['saved_update_recovery'],receipt=operations[-1]['receipt'])
        recovery=json.loads(_bound(record['recovery']).read_text())
        if recovery.get('schema')!=interrupted.SCHEMA:
            raise ValueError('interrupted resume requires an actual performed operational stop')
        chain=raw._owned_operation_chain(operations,published,original['raw_model_id'],original['recipe_lineage_sha256'],inspect_interrupted_tail=True)
        candidates=_candidates(original,cursor,state,chain)
        if intent_path.exists():
            if (intent.get('operations')!=chain or intent.get('selected_checkpoint')!=chain[-1]['checkpoint']
                    or set(intent.get('candidates',{}))!=set(paths)
                    or any(json.loads(_bound(intent['candidates'][name]).read_text())!=value for name,value in candidates.items())):
                raise ValueError('interrupted resume changed its authenticated candidate files')
        else:
            if raw.cycle_work_directory(root,candidates['cursor']).exists():
                raise ValueError('interrupted resume retry namespace already has another operation')
            directory.mkdir(parents=True,exist_ok=True)
            before={name:binding(path) for name,path in paths.items()};snapshots={};refs={}
            for name,path in paths.items():
                snapshot=directory/('original-'+name+'.json');write_bytes_durable(snapshot,_bound(before[name]).read_bytes());snapshots[name]=binding(snapshot)
                candidate=directory/('candidate-'+name+'.json');write(candidate,candidates[name]);refs[name]=binding(candidate)
            if any(binding(path)!=before[name] for name,path in paths.items()):
                raise ValueError('interrupted resume predecessor changed before its durable intent')
            intent={'schema':SCHEMA,'request':request,'before':before,'snapshots':snapshots,'candidates':refs,
                'operations':chain,'selected_checkpoint':chain[-1]['checkpoint']}
            write(intent_path,intent)
        expected={'schema':SCHEMA,'request_sha256':digest(request),'intent':binding(intent_path),
            'published_checkpoint':chain[-1]['checkpoint'],'training_credit':0,'readiness_granted':False,
            'whole_recipe_completed':False,'same_stage_retry':True}
        if out.exists():
            if json.loads(out.read_text())!=expected:raise ValueError('interrupted resume receipt changed')
            return binding(out)  # Historical completion must never rewind subsequent progress.
        for name,path in paths.items():
            actual=binding(path)
            if actual!=intent['before'][name] and actual['sha256']!=intent['candidates'][name]['sha256']:
                raise ValueError('interrupted resume found an unknown successor during CAS')
        for name,path in paths.items():
            candidate=intent['candidates'][name]
            if binding(path)['sha256']!=candidate['sha256']:
                write_bytes_durable(path,_bound(candidate).read_bytes())
        write(out,expected)
        return binding(out)
