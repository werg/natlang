"""Prospective custody checks for a resident source append.

These checks are admission evidence only.  They never manufacture a successor
checkpoint, grant exposure, or replace the independent restore qualification.
"""
from __future__ import annotations
from copy import deepcopy
from .checkpoint_offload import (RecoveryExportQueue,validate_recovery_plan_shape,
    validate_recovery_plan_dependencies)
from .native_resident import digest,physical


def _ref(value):
    if value is None:return None
    return {key:value[key] for key in ('path','sha256')}


def validate_committed_record(committed,payload,generation,global_step):
    import re
    validation=committed.get('validation') if isinstance(committed,dict) else None
    if (not isinstance(committed,dict) or committed.get('schema')!='natlang.native-resident-adoption/1'
            or committed.get('generation')!=generation or committed.get('payload')!=payload
            or committed.get('at_global_step')!=global_step or not isinstance(validation,dict)
            or set(validation)!={'entries_sha256','held_entries_sha256'}
            or any(not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{64}',value) for value in validation.values())
            or (payload.get('delta_validation') is None and any(validation.get(key)!=payload.get(key)
                for key in ('entries_sha256','held_entries_sha256')))):
        raise ValueError('resident append has not been committed for this actual training state')


def validate_append_intent(predecessor, successor, payload, *, stage_id, predecessor_generation,
                           predecessor_validator=validate_recovery_plan_dependencies,
                           successor_validator=lambda plan:validate_recovery_plan_dependencies(plan,require_restore=False)):
    """Authenticate a future recipe without claiming it has been adopted."""
    validate_recovery_plan_shape(predecessor);validate_recovery_plan_shape(successor)
    predecessor_validator(predecessor);successor_validator(successor)
    if payload.get('stage_id')!=stage_id or payload.get('predecessor')!=predecessor_generation:
        raise ValueError('resident append targets a different predecessor')
    if _ref(physical(payload['prior_train']))!=predecessor['inputs']['train'] or _ref(physical(payload['prior_held']))!=predecessor['inputs']['held']:
        raise ValueError('resident append predecessor sources differ from recovery plan')
    roles=physical(payload['plan']['stage_inputs'][stage_id]);candidate={
        'train':_ref(roles['native_graphs']),'foundation':_ref(roles.get('foundation_native_graphs')),
        'held':_ref(roles['held_native_graphs']),'ce':_ref(roles.get('ce_supervision_policy'))}
    if candidate!=successor['inputs'] or _ref(physical(payload['train']))!=candidate['train'] or _ref(physical(payload['held']))!=candidate['held']:
        raise ValueError('resident append successor roles differ from recovery plan')
    if candidate['held']!=predecessor['inputs']['held']:
        raise ValueError('resident append changes held source')
    old_foundation=predecessor['inputs']['foundation'];new_foundation=candidate['foundation']
    if new_foundation!=old_foundation and not (old_foundation==predecessor['inputs']['train'] and new_foundation==candidate['train']):
        raise ValueError('resident append changes fixed foundation')
    if predecessor['session']!=successor['session'] or predecessor['destination']!=successor['destination'] or predecessor['journal_directory']!=successor['journal_directory'] or predecessor['controller']!=successor['controller']:
        raise ValueError('resident append changes recovery owner or controller')
    fixed={'runtime','environment','system-python','system-libraries','node','interpreter','base','tokenizer','task-assets'}
    def bindings(plan):return {(x['role'],x['path'],x['sha256']) for x in plan['closure'] if x['role'] in fixed}
    if bindings(predecessor)!=bindings(successor) or predecessor['restore']!=successor['restore'] or predecessor['control_snapshot']!=successor['control_snapshot']:
        raise ValueError('resident append changes independently restored execution closure')
    if any(item not in successor.get('historical_programs',[]) for item in predecessor.get('historical_programs',[])):
        raise ValueError('resident append drops historical source program custody')
    declared={(x['kind'],x['path'],x['sha256']) for x in successor.get('additional_inputs',[])}
    for role in ('train','held','ce'):
        ref=predecessor['inputs'][role]
        if ref and ref!=candidate[role] and ('ce' if role=='ce' else 'source',ref['path'],ref['sha256']) not in declared:
            raise ValueError('resident append omits predecessor recovery custody')
    return {'schema':'natlang.local-recovery-resident-intent/1','predecessor_plan_sha256':RecoveryExportQueue._plan_digest(predecessor),
            'successor_draft_sha256':RecoveryExportQueue._plan_digest(successor),'predecessor_generation':predecessor_generation,
            'generation':digest(payload),'stage_id':stage_id,'inputs':deepcopy(candidate),
            'scope':'prospective only; resident commit and actual successor state still required'}


def seal_append_transition(predecessor_ref, successor, payload, committed, state, *,
                           stage_id, predecessor_generation, proof_directory,
                           extra_controls=(),
                           predecessor_validator=validate_recovery_plan_dependencies,
                           successor_validator=lambda plan:validate_recovery_plan_dependencies(plan,require_restore=False)):
    """Seal an actually committed resident transition without claiming a new restore.

    The resulting proof inherits the predecessor's actual restore and binds the
    real successor metadata.  The first save/export must still validate and bind
    its own optimizer, RNG, exposure and checkpoint bytes.
    """
    import json
    from pathlib import Path
    from .checkpoint_offload import (file_sha256,recovery_static_sft_contract,
        recovery_closure_identity,_plan_matches_checkpoint)
    from ..common.jsonio import write_json_durable
    predecessor_path=Path(predecessor_ref['path']).resolve(strict=True)
    if file_sha256(predecessor_path)!=predecessor_ref['sha256']:
        raise ValueError('predecessor recovery plan binding changed')
    predecessor=json.loads(predecessor_path.read_text())
    intent=validate_append_intent(predecessor,successor,payload,stage_id=stage_id,
        predecessor_generation=predecessor_generation,predecessor_validator=predecessor_validator,
        successor_validator=successor_validator)
    validate_committed_record(committed,payload,intent['generation'],(state.get('training') or {}).get('global_step'))
    result=deepcopy(successor);result['require_static_contract']=True
    result['sft_static_contract']=recovery_static_sft_contract(state['training'])
    if not _plan_matches_checkpoint(result,state):
        raise ValueError('actual successor state differs from append source or quota contract')
    directory=Path(proof_directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    if len(extra_controls)!=4:raise ValueError('resident transition needs intent, staging receipt, staged draft and committed record bindings')
    stem=intent['generation']+'-'+intent['predecessor_plan_sha256'][:16]+'-'+extra_controls[1]['sha256'][:16]
    evidence_path=directory/(stem+'-resident-transition.json')
    draft_path=directory/(stem+'-resident-draft.json')
    proof_path=directory/(stem+'-resident-qualification.json')
    plan_path=directory/(stem+'-resident-plan.json')
    def write_once(path,value):
        if path.exists():
            if json.loads(path.read_text())!=value:raise ValueError('resident transition output differs on retry')
        else:write_json_durable(path,value)
    if json.loads(Path(extra_controls[3]['path']).read_text())!=committed or file_sha256(extra_controls[3]['path'])!=extra_controls[3]['sha256']:
        raise ValueError('resident committed record snapshot differs')
    write_once(draft_path,successor);draft_ref={'path':str(draft_path),'sha256':file_sha256(draft_path)}
    evidence={'schema':'natlang.local-recovery-resident-transition/1','intent':intent,
              'successor_draft':draft_ref,
              'staged_intent':extra_controls[0],'staged_closure':extra_controls[1],'staged_draft':extra_controls[2],
              'resident_current':extra_controls[3],
              'committed':committed,'actual_static_contract':result['sft_static_contract'],
              'actual_global_step':state['training']['global_step'],
              'scope':'actual metadata transition; no new optimizer update or successor restore claimed'}
    write_once(evidence_path,evidence);evidence_ref={'path':str(evidence_path),'sha256':file_sha256(evidence_path)}
    closure=list(result['closure']);seen={(x['role'],x['path'],x['sha256']) for x in closure}
    for item in [*predecessor['closure'],*({'role':'controls',**ref} for ref in extra_controls),{'role':'controls',**predecessor_ref},{'role':'controls',**draft_ref},{'role':'controls',**evidence_ref}]:
        key=(item['role'],item['path'],item['sha256'])
        if key not in seen:closure.append(item);seen.add(key)
    result['restore_qualification']={'path':str(proof_path),'sha256':'0'*64}
    closure.append({'role':'controls',**result['restore_qualification']});result['closure']=closure
    proof={'schema':'natlang.local-recovery-restore-qualification/1','mode':'resident-append/1','status':'passed',
           'inputs':result['inputs'],'restore':result['restore'],'additional_inputs':result.get('additional_inputs',[]),
           'controller_config':result['controller']['config'],'require_static_contract':True,
           'sft_static_contract':result['sft_static_contract'],'control_snapshot':result['control_snapshot'],
           'historical_programs':result.get('historical_programs',[]),
           'closure_identity':recovery_closure_identity(result),'predecessor_plan':predecessor_ref,
           'transition_evidence':evidence_ref,
           'checks':{'predecessor_restore_anchor':True,'resident_commit_validated':True,'successor_metadata_validated':True},
           'scope':'prospective source transition anchored to a real predecessor restore; successor checkpoint/export must be independently verified'}
    write_once(proof_path,proof);result['restore_qualification']['sha256']=file_sha256(proof_path)
    result['closure'][-1]['sha256']=result['restore_qualification']['sha256']
    write_once(plan_path,result)
    return {'path':str(plan_path),'sha256':file_sha256(plan_path)}


def validate_staged_closure(plan, receipt):
    """Check a previously read-back independent closure without rescanning bytes."""
    from pathlib import Path
    if receipt.get('schema')!='natlang.local-recovery-staged-closure/1' or receipt.get('plan_sha256')!=RecoveryExportQueue._plan_digest(plan):
        raise ValueError('staged recovery closure does not bind the successor draft')
    entries=receipt.get('objects')
    if not isinstance(entries,list):raise ValueError('staged recovery closure lacks objects')
    by_digest={}
    root=Path(plan['destination']).resolve(strict=True)/'closure'
    for item in entries:
        if not isinstance(item,dict) or set(item)!={'sha256','path','identity'} or item['sha256'] in by_digest:
            raise ValueError('invalid staged recovery object')
        path=Path(item['path'])
        if path!=root/item['sha256'] or path.is_symlink():raise ValueError('staged recovery object path differs')
        s=path.stat();identity=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
        if identity!=item['identity']:raise ValueError('staged recovery object changed since independent readback')
        by_digest[item['sha256']]=item
    if {item['sha256'] for item in plan['closure']}!=set(by_digest):
        raise ValueError('staged recovery object inventory differs')


def find_staged_append_intent(catalog_path, active_plan, payload, resident_directory):
    """Find one hash-bound prospective intent for a manager-validated append."""
    from pathlib import Path
    from .checkpoint_offload import _read_bound_recovery_bytes,_read_bound_recovery_json
    catalog_path=Path(catalog_path).resolve(strict=True);s=catalog_path.stat()
    catalog=_read_bound_recovery_json({'path':str(catalog_path),'sha256':__import__('hashlib').sha256(catalog_path.read_bytes()).hexdigest()})
    after=catalog_path.stat()
    if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('recovery catalog changed during append admission')
    if catalog.get('schema')!='natlang.local-full-recovery-catalog/1':raise ValueError('append requires a recovery catalog')
    matches=[]
    for reference in catalog.get('append_intents',[]):
        intent=_read_bound_recovery_json(reference)
        if intent.get('generation')!=digest(payload):continue
        if (intent.get('schema')!='natlang.local-recovery-resident-append-intent/1'
            or intent.get('resident_directory')!=str(Path(resident_directory).resolve())
            or intent.get('predecessor_generation')!=payload.get('predecessor')):
            raise ValueError('recovery append intent has a different resident owner')
        predecessor=_read_bound_recovery_json(intent['predecessor_plan']);draft=_read_bound_recovery_json(intent['successor_draft'])
        if RecoveryExportQueue._plan_digest(predecessor)!=RecoveryExportQueue._plan_digest(active_plan):
            raise ValueError('recovery append intent does not descend from active plan')
        validated=validate_append_intent(predecessor,draft,payload,stage_id=payload['stage_id'],
            predecessor_generation=intent['predecessor_generation'])
        if validated['generation']!=intent['generation']:raise ValueError('recovery append generation differs')
        staged=_read_bound_recovery_json(intent['staged_closure']);validate_staged_closure(draft,staged)
        matches.append((intent,draft,reference))
    if len(matches)!=1:raise ValueError('resident append has no unique staged recovery intent')
    return matches[0]


def finalize_staged_append(catalog_path, active_plan, state, resident_directory):
    """Seal a committed append before its first successor checkpoint write."""
    import fcntl,json,os
    from pathlib import Path
    from .checkpoint_offload import _read_bound_recovery_json,validate_restore_qualification
    from ..common.jsonio import write_json_durable,write_bytes_durable
    from .native_resident import current
    directory=Path(resident_directory).resolve(strict=True);current_path=directory/'current.json'
    before=current_path.stat();raw=current_path.read_bytes();after=current_path.stat()
    ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if ident(before)!=ident(after):raise ValueError('resident commit changed during recovery capture')
    committed=current(directory)
    if committed!=json.loads(raw) or ident(current_path.stat())!=ident(after):raise ValueError('resident commit changed during recovery validation')
    if not isinstance(committed,dict):raise ValueError('resident append has no committed record')
    payload=committed.get('payload');generation=committed.get('generation')
    adoption=((state.get('training') or {}).get('options') or {}).get('resident_adoption') or {}
    if adoption.get('generation')!=generation:raise ValueError('actual checkpoint state has another resident generation')
    intent,draft,intent_ref=find_staged_append_intent(catalog_path,active_plan,payload,directory)
    snapshot_dir=Path(active_plan['journal_directory'])/'resident-transitions';snapshot_dir.mkdir(parents=True,exist_ok=True)
    snapshot=snapshot_dir/(generation+'-'+RecoveryExportQueue._plan_digest(active_plan)[:16]+'-current.json')
    if snapshot.exists():
        if snapshot.read_bytes()!=raw:raise ValueError('resident commit snapshot differs on retry')
    else:write_bytes_durable(snapshot,raw)
    current_ref={'path':str(snapshot),'sha256':__import__('hashlib').sha256(raw).hexdigest()}
    if ident(current_path.stat())!=ident(after):raise ValueError('resident commit changed before recovery sealing')
    result=seal_append_transition(intent['predecessor_plan'],draft,payload,committed,state,
        stage_id=payload['stage_id'],predecessor_generation=intent['predecessor_generation'],
        proof_directory=Path(active_plan['journal_directory'])/'resident-transitions',
        extra_controls=(intent_ref,intent['staged_closure'],intent['successor_draft'],current_ref))
    plan=_read_bound_recovery_json(result);validate_restore_qualification(plan)
    catalog_path=Path(catalog_path).resolve(strict=True);fd=os.open(str(catalog_path)+'.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX);catalog=json.loads(catalog_path.read_text())
        if catalog.get('schema')!='natlang.local-full-recovery-catalog/1':raise ValueError('recovery catalog changed schema')
        if intent_ref not in catalog.get('append_intents',[]):raise ValueError('recovery append intent was withdrawn')
        if result not in catalog['plans']:
            catalog['plans'].append(result);write_json_durable(catalog_path,catalog)
    finally:os.close(fd)
    return result


def derive_append_draft(predecessor,payload,stage_id):
    """Derive the closure for a not-yet-adopted resident recipe."""
    from .recipe import native_source_dependencies,recipe_file_dependencies
    from .native_supervision import ce_policy_dependencies
    result=deepcopy(predecessor);roles=physical(payload['plan']['stage_inputs'][stage_id])
    result['inputs']={'train':_ref(roles['native_graphs']),'foundation':_ref(roles.get('foundation_native_graphs')),
        'held':_ref(roles['held_native_graphs']),'ce':_ref(roles.get('ce_supervision_policy'))}
    result['sft_static_contract']={};result['require_static_contract']=True
    closure=list(result['closure']);seen={(x['role'],x['path'],x['sha256']) for x in closure}
    def add(role,ref):
        item={'role':role,'path':ref['path'],'sha256':ref['sha256']};key=(role,item['path'],item['sha256'])
        if key not in seen:closure.append(item);seen.add(key)
    additional=list(result.get('additional_inputs',[]));declared={(x['kind'],x['path'],x['sha256']) for x in additional}
    for role in ('train','held','ce'):
        ref=predecessor['inputs'][role];kind='ce' if role=='ce' else 'source'
        if ref and ref!=result['inputs'][role] and (kind,ref['path'],ref['sha256']) not in declared:
            additional.append({'kind':kind,**ref});declared.add((kind,ref['path'],ref['sha256']))
    result['additional_inputs']=additional
    for role in ('train','foundation','held'):
        ref=result['inputs'][role]
        if ref:
            add('sources',ref)
            for item in native_source_dependencies(ref['path']):add('sources',item)
    if result['inputs']['ce']:
        for item in ce_policy_dependencies(result['inputs']['ce']):add('controls',item)
    recipe=_ref(physical(payload['recipe']));add('controls',recipe)
    for item in recipe_file_dependencies(recipe['path']):add('controls',item)
    result['closure']=closure
    return result


def stage_append_intent(queue,catalog_path,payload,resident_directory,*,stop_requested=lambda:False):
    """Independently stage one prospective append without blocking SGD."""
    import fcntl,json,os
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
    from .checkpoint_offload import (_read_bound_recovery_json,verified_recovery_blob,file_sha256)
    from ..common.jsonio import write_json_durable
    catalog_path=Path(catalog_path).resolve(strict=True);catalog=json.loads(catalog_path.read_text());predecessor=deepcopy(queue.plan)
    candidates=[ref for ref in catalog['plans'] if RecoveryExportQueue._plan_digest(_read_bound_recovery_json(ref))==RecoveryExportQueue._plan_digest(predecessor)]
    if len(candidates)!=1:raise ValueError('active recovery predecessor has no unique catalog binding')
    parent_ref=candidates[0];directory=Path(predecessor['journal_directory'])/'append-intents';directory.mkdir(parents=True,exist_ok=True)
    generation=digest(payload);draft=derive_append_draft(predecessor,payload,payload['stage_id'])
    validate_append_intent(predecessor,draft,payload,stage_id=payload['stage_id'],predecessor_generation=payload['predecessor'])
    stem=generation+'-'+RecoveryExportQueue._plan_digest(predecessor)[:16]
    draft_path=directory/(stem+'-draft.json')
    def once(path,value):
        if path.exists():
            if json.loads(path.read_text())!=value:raise ValueError('append staging retry differs')
        else:write_json_durable(path,value)
    once(draft_path,draft);draft_ref={'path':str(draft_path),'sha256':file_sha256(draft_path)}
    unique={item['sha256']:item for item in draft['closure']};objects={};store=Path(draft['destination']).resolve(strict=True)/'closure'
    def copy(row):
        if stop_requested():raise RuntimeError('recovery append staging stopped')
        digest_value,item=row;target=store/digest_value
        with queue.cond:cached=queue.state.get('closure_cache',{}).get(digest_value)
        identity=verified_recovery_blob(item['path'],target,digest_value,cached_identity=cached,owner_fd=queue._owner.fileno())
        return digest_value,{'path':str(target),'sha256':digest_value,'identity':identity}
    with ThreadPoolExecutor(max_workers=4) as pool:
        iterator=iter(unique.items());pending=set()
        for _ in range(8):
            row=next(iterator,None)
            if row is None:break
            pending.add(pool.submit(copy,row))
        while pending:
            completed,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                key,value=future.result();objects[key]=value
                if stop_requested():raise RuntimeError('recovery append staging stopped')
                row=next(iterator,None)
                if row is not None:pending.add(pool.submit(copy,row))
    if stop_requested():raise RuntimeError('recovery append staging stopped')
    staged={'schema':'natlang.local-recovery-staged-closure/1','plan_sha256':RecoveryExportQueue._plan_digest(draft),
            'objects':[objects[key] for key in sorted(objects)]}
    stage_path=directory/(stem+'-staged-'+digest(staged)[:16]+'.json')
    once(stage_path,staged);stage_ref={'path':str(stage_path),'sha256':file_sha256(stage_path)}
    intent={'schema':'natlang.local-recovery-resident-append-intent/1','predecessor_plan':parent_ref,
            'successor_draft':draft_ref,'staged_closure':stage_ref,'predecessor_generation':payload['predecessor'],
            'generation':generation,'resident_directory':str(Path(resident_directory).resolve())}
    intent_path=directory/(stem+'-intent-'+stage_ref['sha256'][:16]+'.json')
    once(intent_path,intent);intent_ref={'path':str(intent_path),'sha256':file_sha256(intent_path)}
    fd=os.open(str(catalog_path)+'.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX);catalog=json.loads(catalog_path.read_text())
        with queue.cond:
            if stop_requested():raise RuntimeError('recovery append staging stopped before publication')
            if parent_ref not in catalog.get('plans',[]):raise ValueError('recovery predecessor was withdrawn while staging')
            refs=catalog.setdefault('append_intents',[]);retained=[];changed=False
            for prior_ref in refs:
                prior=_read_bound_recovery_json(prior_ref)
                if (prior.get('generation')==generation and prior.get('predecessor_plan')==parent_ref
                        and prior_ref!=intent_ref):
                    catalog.setdefault('append_intent_history',[]).append({'binding':prior_ref,'disposition':'superseded-staging-revalidation'})
                    changed=True
                else:retained.append(prior_ref)
            catalog['append_intents']=refs=retained
            if intent_ref not in refs:refs.append(intent_ref);changed=True
            if changed:write_json_durable(catalog_path,catalog)
    finally:os.close(fd)
    return intent_ref
