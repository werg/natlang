"""Source-host role ownership and exact small readiness exchange.

The unchanged local validators own qualification. These receipts attest their
actual source-host execution; remote checkpoint paths are never local assets.
"""
from __future__ import annotations
import json
import math
import fcntl
import re
import os
import time
from pathlib import Path
import socket
import importlib
from ..common.hashing import canonical_json_sha256_hex,sha256_file_hex_cached,sha256_hex
from ..common.jsonio import write_bytes_durable,write_json_durable
from .native_raw_handoff import binding as _binding

SCHEMA='natlang.native-source-host-role-proof/1'
EXCHANGE='natlang.native-source-host-exchange/1'
MECHANISM_FILES=('rl/native_role_release.py','rl/native_raw_rounds.py','train/native_readiness.py',
                 'eval/native_gradients.py','rl/native_raw_policy.py','rl/native_raw_handoff.py')


def binding(path):
    value=_binding(path)
    return {**value,'bytes':Path(value['path']).stat().st_size}


def ownership(plan):
    if type(plan.get('consumer_preparation_handoff',False)) is not bool:
        raise ValueError('consumer preparation ownership must be explicit')
    if type(plan.get('snapshot_preparation',False)) is not bool:
        raise ValueError('snapshot preparation must be explicitly selected')
    names={row['model']for row in plan['models']}
    owned=plan.get('owned_models',sorted(names))
    if (not isinstance(owned,list)or not owned or any(not isinstance(name,str)for name in owned)
            or len(set(owned))!=len(owned)or not set(owned)<=names):
        raise ValueError('raw source host needs distinct declared owned models')
    exchange=plan.get('role_exchange')
    if exchange is None:
        if set(owned)!=names:raise ValueError('partial ownership requires the complete explicit role exchange')
        return set(owned),None
    if (not isinstance(exchange,dict)or exchange.get('schema')!=EXCHANGE
            or not isinstance(exchange.get('directory'),str)or not Path(exchange['directory']).is_absolute()
            or '..'in Path(exchange['directory']).parts
            or not isinstance(exchange.get('roles'),dict)or not exchange['roles']
            or exchange.get('semantic_consumer') not in exchange['roles']):
        raise ValueError('source-host exchange needs its explicit role assignment and directory')
    all_names=[];source_hosts=[]
    for role,row in exchange['roles'].items():
        values=row.get('owned_models')if isinstance(row,dict)else None
        if (not isinstance(role,str)or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',role)is None
                or not isinstance(values,list)or not values
                or any(not isinstance(value,str)for value in values)or len(set(values))!=len(values)
                or not isinstance(row.get('source_host_id'),str)or not row['source_host_id']):
            raise ValueError('source-host role has invalid ownership or host identity')
        all_names.extend(values);source_hosts.append(row['source_host_id'])
    declared=[row.get('verification_mechanism') for row in exchange['roles'].values()]
    if any('verification_mechanism' in row for row in exchange['roles'].values()):
        if any(not isinstance(value,dict) or set(value)!=set(MECHANISM_FILES)
               or any(not isinstance(digest,str) or re.fullmatch(r'[0-9a-f]{64}',digest) is None
                      for digest in value.values()) for value in declared):
            raise ValueError('source-host mechanism authority needs every exact file hash for every role')
    role=exchange.get('role')
    if (len(set(source_hosts))!=len(source_hosts)or set(all_names)!=names or len(all_names)!=len(set(all_names))
            or role not in exchange['roles']or set(exchange['roles'][role]['owned_models'])!=set(owned)):
        raise ValueError('source-host roles must partition the actual model owners exactly')
    return set(owned),exchange


def contract(plan):
    _,exchange=ownership(plan)
    if exchange is None:raise ValueError('source-host proofs require an explicit exchange')
    keys=('model_id','stage_ids','foundation_epochs','readiness_boundary','forward_atol','gradient_atol')
    models={}
    for row in plan['models']:
        requirement=row['handoff_requirement']
        models[row['model']]={key:requirement.get(key)for key in keys}
        models[row['model']]['foundation_sources_sha256']=(requirement.get('foundation_sources')or{}).get('sha256')
    if len({row['model_id']for row in models.values()})!=len(models):
        raise ValueError('source-host roles need distinct actual model identities')
    return {'models':models,'roles':exchange['roles'],'semantic_consumer':exchange['semantic_consumer']}


def mechanism():
    package=Path(__file__).resolve().parents[1]
    result={}
    for name in MECHANISM_FILES:
        module=importlib.import_module('natlang_neuralese.'+name[:-3].replace('/','.'))
        external=getattr(module,'__native_external_control_source__',None)
        if external is not None:
            if (not isinstance(external,dict) or not isinstance(external.get('path'),str)
                    or not Path(external['path']).is_absolute()):
                raise ValueError('external role mechanism needs its exact source binding')
            source=Path(external['path']);actual=sha256_file_hex_cached(source)
            if actual!=external.get('sha256') or ('bytes' in external and source.stat().st_size!=external['bytes']):
                raise ValueError('external role mechanism source binding changed')
            result[name]=actual
        else:
            if Path(module.__file__).resolve()!=package/name:
                raise ValueError('role mechanism module escaped its installed runtime')
            result[name]=sha256_file_hex_cached(package/name)
    return result


def _expected_mechanism(plan,role):
    _,exchange=ownership(plan)
    if exchange is None or role not in exchange['roles']:
        raise ValueError('mechanism authority requires its declared source-host role')
    declared=exchange['roles'][role].get('verification_mechanism')
    return dict(declared) if declared is not None else mechanism()


def _publishing_mechanism(plan):
    _,exchange=ownership(plan)
    actual=mechanism()
    if exchange is None or actual!=_expected_mechanism(plan,exchange['role']):
        raise ValueError('actual source-host mechanism differs from its declared authority')
    return actual


def _inline(reference):
    from ..train.native_readiness import _bound
    raw=_bound(reference).read_bytes()
    if sha256_hex(raw)!=reference['sha256']:raise ValueError('role evidence changed while reading')
    return {'source':reference,'text':raw.decode('utf-8')}


def _evidence(item):
    raw=item['text'].encode('utf-8');reference=item['source']
    if (sha256_hex(raw)!=reference['sha256']or('bytes'in reference and len(raw)!=reference['bytes'])):
        raise ValueError('role inline source evidence changed')
    return json.loads(raw)


def _immutable(path,value):
    raw=(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('immutable source-host role proof changed')
    else:
        import tempfile
        fd,name=tempfile.mkstemp(prefix='.role-proof-',dir=path.parent)
        temporary=Path(name)
        try:
            with os.fdopen(fd,'wb')as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            try:os.link(temporary,path)
            except FileExistsError:
                if path.read_bytes()!=raw:raise ValueError('immutable source-host role proof changed')
            directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:temporary.unlink(missing_ok=True)
    return binding(path)


def publish(plan_path,state_path,*,kind):
    """Run FULL local validation; final releases require a dead raw owner."""
    if kind not in ('eligible','released'):raise ValueError('unknown source-host role proof kind')
    from .native_raw_rounds import validate_release
    from ..train.native_resident import process_owner
    plan=json.loads(Path(plan_path).read_text());owned,exchange=ownership(plan)
    if exchange is None:raise ValueError('role publication requires explicit ownership')
    state=json.loads(Path(state_path).read_text())
    ready=validate_release(state_path,plan_path,allow_live=kind=='eligible',eligibility=kind=='eligible')
    if ready is None:return None
    evidence={}
    for name in sorted(owned):
        value=state['models'][name];checked=ready[plan_model(plan,name)['handoff_requirement']['model_id']]
        receipt_item=_inline(checked['readiness']);receipt=_evidence(receipt_item)
        evidence[name]={'ready':checked,'readiness':receipt_item,
            'report':_inline(receipt['training_receipt']),
            'runtime':_inline(receipt['runtime_certificate']),
            'task':_inline(receipt['task_execution_receipt'])}
        if plan.get('snapshot_preparation'):
            evidence[name]['snapshot_checkpoint']=pin_checkpoint(checked['checkpoint'],
                Path(exchange['directory'])/'qualified-checkpoints')
        task=_evidence(evidence[name]['task'])
        if task.get('schema')=='natlang.native-task-execution-receipt/2':
            evidence[name]['task_score']=_inline({'path':task['score_path'],'sha256':task['score_sha256']})
    proof={'schema':SCHEMA,'kind':kind,'role':exchange['role'],
        'source_host_id':exchange['roles'][exchange['role']]['source_host_id'],
        'source_host':{'hostname':socket.gethostname(),'validator':process_owner()},
        'contract':contract(plan),'mechanism':_publishing_mechanism(plan),'owned_models':sorted(owned),
        'plan':_inline(binding(plan_path)),'raw_state':_inline(binding(state_path)),
        'controller':state['owner'],'models':evidence,
        'optimizer_released':kind=='released','production_complete':False}
    from ..train.native_readiness import validate_foundation_candidate,_bound
    proof['foundation_candidates']={}
    for name in sorted(owned):
        requirement=ready[plan_model(plan,name)['handoff_requirement']['model_id']]['readiness_requirement']
        if requirement.get('readiness_boundary')=='communication-foundation':
            proof['foundation_candidates'][name]=validate_foundation_candidate(
                _bound(ready[requirement['model_id']]['readiness']),requirement)
    proof['foundation_attempts']={name:value['foundation_qualification_attempt']
        for name,value in state['models'].items() if value.get('foundation_qualification_attempt')}
    return _commit(exchange,proof)


def plan_model(plan,name):
    return next(row for row in plan['models']if row['model']==name)


def validate(reference,plan,*,kind):
    """Validate transported attestation; never bind its remote checkpoint path."""
    from ..train.native_readiness import _bound
    proof=json.loads(_bound(reference).read_text());_,exchange=ownership(plan)
    expected=contract(plan);role=proof.get('role')
    if (proof.get('schema')!=SCHEMA or proof.get('kind')!=kind
            or type(proof.get('revision'))is not int or proof['revision']<1
            or proof.get('optimizer_released')is not(kind=='released')
            or proof.get('production_complete')is not False
            or proof.get('contract')!=expected or role not in exchange['roles']
            or proof.get('mechanism')!=_expected_mechanism(plan,role)
            or proof.get('source_host_id')!=exchange['roles'][role]['source_host_id']
            or proof.get('owned_models')!=sorted(exchange['roles'][role]['owned_models'])
            or set(proof.get('models')or{})!=set(proof['owned_models'])):
        raise ValueError('source-host role proof changed ownership, cohort, mechanism or release scope')
    source_plan=_evidence(proof['plan']);source_state=_evidence(proof['raw_state']);source_owned,source_exchange=ownership(source_plan)
    if (contract(source_plan)!=expected or source_exchange['role']!=role or source_owned!=set(proof['owned_models'])
            or source_state.get('owner')!=proof['controller']
            or source_state.get('plan_sha256')!=proof['plan']['source']['sha256']
            or set(source_state.get('models')or{})!=set(proof['owned_models'])
            or (kind=='released'and source_state.get('released')is not True)):
        raise ValueError('role proof omitted its actual controller/drained source state')
    from ..eval.native_gradients import validate_complete_proof
    result={}
    for name,item in proof['models'].items():
        checked=item['ready'];receipt=_evidence(item['readiness']);report=_evidence(item['report'])
        runtime=_evidence(item['runtime']);task=_evidence(item['task']);requirement=checked['readiness_requirement']
        cohort=expected['models'][name]
        if (checked['model_id']!=cohort['model_id']or receipt['model_id']!=checked['model_id']
                or checked['checkpoint']!=receipt['checkpoint']or checked['native_identity']!=receipt['native_identity']
                or checked['checkpoint']['sha256']!=checked['native_identity']['checkpoint_sha256']
                or checked['recipe']!=receipt['recipe']or checked['stage_id']!=receipt['stage_id']
                or requirement.get('recipe_sha256')!=checked['recipe']['sha256']
                or any(requirement.get(key)!=cohort[key]for key in
                    ('model_id','stage_ids','foundation_epochs','readiness_boundary','forward_atol','gradient_atol'))
                or(requirement.get('foundation_sources')or{}).get('sha256')!=cohort['foundation_sources_sha256']
                or any(receipt[key]!=item[field]['source']for key,field in
                    (('training_receipt','report'),('runtime_certificate','runtime'),('task_execution_receipt','task')))
                or report.get('checkpoint_sha256')!=checked['checkpoint']['sha256']
                or runtime.get('native_identity')!=checked['native_identity']
                or runtime.get('passed')is not True or runtime.get('backbone_optimizer_tested')is not True
                or runtime.get('backbone_optimizer_updates',0)<=0
                or runtime.get('qualification_profile')!=report.get('qualification_profile')
                or task.get('native_identity')!=checked['native_identity']
                or any(task.get(key)is not True for key in('passed','complete','eligible'))):
            raise ValueError('source-host role evidence substituted its actual qualified participant')
        source_value=source_state['models'][name]
        if (any(checked['checkpoint'].get(key)!=source_value['checkpoint'].get(key)for key in('path','sha256'))
                or checked['readiness']!=item['readiness']['source']
                or not isinstance(report.get('qualification_profile'),dict)
                or receipt.get('exposure',{}).get('consumed_updates',0)<=0
                or receipt.get('exposure',{}).get('supervised_token_visits',0)<=0):
            raise ValueError('role release omitted actual checkpoint/readiness/optimizer exposure')
        foundation=cohort['readiness_boundary']=='communication-foundation'
        communication=receipt.get('communication_exposure')or{}
        if (communication!=report.get('communication_exposure')
                or any(type(communication.get(key))is not int or communication[key]<=0
                    for key in('graphs','required_graph_visits','graph_visits'))
                or communication['required_graph_visits']<communication['graphs']
                or(not foundation and(communication.get('complete')is not True
                    or communication['graph_visits']<communication['required_graph_visits']))):
            raise ValueError('role release lacks genuine complete communication exposure')
        if foundation:
            from ..train.native_continuation import validate_communication_foundation
            completed=receipt.get('communication_foundation')or{}
            validate_communication_foundation(completed, epochs=(cohort['foundation_epochs']
                if cohort['foundation_epochs'] is not None else 1.),
                source_sha256=cohort['foundation_sources_sha256'])
            if ((receipt.get('completed')is not True and receipt.get('foundation_ready')is not True)
                    or(report.get('completed')is not True and report.get('foundation_ready')is not True)
                    or completed.get('complete')is not True or not completed.get('rows')
                    or completed!=report.get('communication_foundation')
                    or checked.get('communication_foundation')!=completed
                    or receipt.get('mandatory_sft')!=report.get('mandatory_sft')
                    or not(receipt.get('mandatory_sft')or{}).get('stages')):
                raise ValueError('role release omitted its complete fixed communication cohort and carried debts')
        elif receipt.get('completed')is not True or report.get('completed')is not True:
            raise ValueError('role release did not complete its declared native stage')
        for metric,limit in(('forward_max_abs','forward_atol'),('incremental_max_abs','forward_atol'),
                ('consumer_ce_abs','forward_atol'),('consumer_gradient_max_abs','gradient_atol')):
            value,tolerance=runtime.get(metric),runtime.get(limit)
            if (type(value)not in(int,float)or type(tolerance)not in(int,float)
                    or not math.isfinite(value)or not math.isfinite(tolerance)
                    or not 0<=value<=tolerance<=cohort[limit]):
                raise ValueError('role runtime exceeded its declared numerical contract')
        if task.get('schema')=='natlang.native-task-execution-receipt/2':
            from ..eval.native_state import _execution_receipt
            score=_evidence(item['task_score'])
            if item['task_score']['source']!={'path':task['score_path'],'sha256':task['score_sha256']}:
                raise ValueError('role task score binding changed')
            summary={key:value for key,value in score.items()if key!='cases_detail'}
            expected_task=_execution_receipt(summary,checked['native_identity'],
                checkpoint_sha256=checked['checkpoint']['sha256'],score=score,path=None,digest=task['score_sha256'])
            if(not expected_task['passed']or any(task.get(key)!=value for key,value in expected_task.items()if key!='score_path')
                    or str(Path(score['run'])/'task-score.json')!=task['score_path']):
                raise ValueError('role task score did not actually qualify its released checkpoint')
        elif task.get('schema')!='natlang.native-task-execution-receipt/1':
            raise ValueError('unsupported role task execution evidence')
        validate_complete_proof(runtime.get('complete_gradient_qualification'),checked['native_identity'],
            gradient_atol=cohort['gradient_atol'],execution_profile=report.get('qualification_profile'))
        result[checked['model_id']]={**checked,'source_role_release':reference,
            'source_host_id':proof['source_host_id']}
        if item.get('snapshot_checkpoint'):
            snapshot=item['snapshot_checkpoint']
            if (snapshot.get('sha256')!=checked['checkpoint']['sha256']
                    or not snapshot.get('path') or type(snapshot.get('bytes'))is not int or snapshot['bytes']<=0
                    or ('bytes'in checked['checkpoint'] and checked['checkpoint']['bytes']!=snapshot['bytes'])):
                raise ValueError('qualified snapshot pin changed the actual trained checkpoint')
            result[checked['model_id']]['snapshot_checkpoint']=snapshot
    return role,result


def _status(exchange,role):
    path=Path(exchange['directory'])/(role+'-status.json')
    if not path.exists():return None,None
    from ..train.native_readiness import _bound
    reference=json.loads(path.read_text())
    return reference,json.loads(_bound(reference).read_text())


def _commit(exchange,proof):
    root=Path(exchange['directory']);root.mkdir(parents=True,exist_ok=True);role=exchange['role']
    with (root/(role+'.lock')).open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        previous_ref,previous=_status(exchange,role)
        if previous is not None:
            if previous['contract']!=proof['contract']:
                raise ValueError('source-host exchange directory belongs to another fixed cohort')
            comparison={key:value for key,value in previous.items()if key!='revision'}
            if comparison==proof:return previous_ref
        proof={**proof,'revision':1 if previous is None else previous['revision']+1}
        digest=canonical_json_sha256_hex(proof)
        reference=_immutable(root/'objects'/(digest+'.json'),proof)
        write_json_durable(root/(role+'-'+proof['kind']+'.json'),reference)
        write_json_durable(root/(role+'-status.json'),reference)
        return reference


def withdraw(plan,state_path):
    """Publish pending status; it cannot authorize a remote participant."""
    _,exchange=ownership(plan)
    if exchange is None:return None
    state=json.loads(Path(state_path).read_text())
    candidates={}
    from ..train.native_readiness import validate_foundation_candidate,_bound
    from .native_raw_rounds import _release_requirement
    for name,value in state.get('models',{}).items():
        if value.get('foundation_candidate'):
            requirement=_release_requirement(plan_model(plan,name),value)
            candidate=validate_foundation_candidate(_bound(value['foundation_candidate']),requirement)
            if candidate is not None:
                if candidate['checkpoint']['sha256']!=value['checkpoint']['sha256']:
                    raise ValueError('foundation candidate is not the current published checkpoint')
                candidates[name]=candidate
    return _commit(exchange,{'schema':SCHEMA,'kind':'pending','optimizer_released':False,
        'role':exchange['role'],'source_host_id':exchange['roles'][exchange['role']]['source_host_id'],
        'contract':contract(plan),'mechanism':_publishing_mechanism(plan),'controller':state['owner'],
        'raw_state':_inline(binding(state_path)),'production_complete':False,
        'foundation_candidates':candidates,
        'foundation_attempts':{name:value['foundation_qualification_attempt']
            for name,value in state.get('models',{}).items() if value.get('foundation_qualification_attempt')}})


def foundation_candidates(plan):
    """Read source-host exposure attestations; these never grant readiness."""
    from ..train.native_readiness import validate_foundation_candidate_document
    _,exchange=ownership(plan)
    if exchange is None:return {},{}
    expected=contract(plan);candidates={};attempts={}
    for role,assignment in exchange['roles'].items():
        _,proof=_status(exchange,role)
        if proof is None:continue
        if (proof.get('schema')!=SCHEMA or proof.get('contract')!=expected
                or proof.get('mechanism')!=_expected_mechanism(plan,role)
                or proof.get('role')!=role or proof.get('source_host_id')!=assignment['source_host_id']):
            raise ValueError('foundation rendezvous changed source-host authority')
        rows=proof.get('foundation_candidates',{})
        if set(rows)-set(assignment['owned_models']):
            raise ValueError('foundation candidate came from another optimizer owner')
        for name,candidate in rows.items():
            validate_foundation_candidate_document(candidate,expected['models'][name])
            candidates[candidate['model_id']]=candidate
        for name,attempt in proof.get('foundation_attempts',{}).items():
            if name not in assignment['owned_models']:
                raise ValueError('foundation attempt came from another optimizer owner')
            attempts[name]=attempt
    return candidates,attempts


def foundation_request(plan,out,*,generation):
    """Seal one full check per declared repair generation, after two exposures."""
    if not isinstance(generation,str) or not generation:
        raise ValueError('foundation qualification needs its explicit scheduling generation')
    expected=contract(plan)
    foundation_generation=generation
    handoff=handoff_active(plan)
    if handoff:
        generation=canonical_json_sha256_hex({'foundation_generation':generation,'handoff':handoff['attempt_sha256']})
    elif (preparation:=preparation_handoff(plan)):
        generation=canonical_json_sha256_hex({'foundation_generation':generation,
            'preparation':preparation['source']['sha256']})
    signature=canonical_json_sha256_hex({'contract':expected,'generation':generation})
    path=Path(out)/(signature+'.json')
    if path.exists():
        request=json.loads(path.read_text())
        if request.get('contract')!=expected or request.get('generation')!=generation:
            raise ValueError('sealed foundation rendezvous changed its contract')
        return binding(path)
    candidates,_=foundation_candidates(plan)
    if len(candidates)<2:return None
    return _immutable(path,{'schema':'natlang.native-foundation-qualification-request/1',
        'contract':expected,'generation':generation,'candidates':candidates,'readiness_granted':False,
        **({'foundation_generation':foundation_generation,
            'handoff_request_sha256':handoff['request_sha256'],
            'rendezvous_attempt_sha256':handoff['attempt_sha256']}if handoff else{})})


def import_proof(reference,plan,*,snapshot_only=False):
    """Commit a verified transported object monotonically; never roll back status."""
    from ..train.native_readiness import _bound
    proof=json.loads(_bound(reference).read_text());_,exchange=ownership(plan);role=proof.get('role')
    if(proof.get('schema')!=SCHEMA or proof.get('contract')!=contract(plan)
            or role not in exchange['roles']or proof.get('mechanism')!=_expected_mechanism(plan,role)
            or proof.get('source_host_id')!=exchange['roles'][role]['source_host_id']
            or type(proof.get('revision'))is not int or proof['revision']<1):
        raise ValueError('imported source-host role status has invalid authority')
    if proof.get('production_complete')is not False:raise ValueError('role status cannot declare whole production complete')
    if proof['kind']in('eligible','released'):validate(reference,plan,kind=proof['kind'])
    elif proof['kind']!='pending' or proof.get('optimizer_released')is not False:
        raise ValueError('imported source-host role status has an unknown qualification scope')
    from ..train.native_readiness import validate_foundation_candidate_document
    for name,candidate in proof.get('foundation_candidates',{}).items():
        if name not in exchange['roles'][role]['owned_models']:
            raise ValueError('foundation candidate came from another optimizer owner')
        validate_foundation_candidate_document(candidate,contract(plan)['models'][name])
    root=Path(exchange['directory']);root.mkdir(parents=True,exist_ok=True)
    with (root/(role+'.lock')).open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        if snapshot_only:
            if proof['kind'] not in ('eligible','released'):
                raise ValueError('snapshot transport needs a genuinely qualified immutable proof')
            pointer=root/(role+'-eligible.json')
            if pointer.is_file():
                previous_ref=json.loads(pointer.read_text());previous=json.loads(_bound(previous_ref).read_text())
                if previous['revision']>proof['revision']:return previous_ref
                if previous['revision']==proof['revision']:
                    if previous_ref['sha256']!=reference['sha256']:raise ValueError('snapshot role revision equivocation')
                    return previous_ref
            local=_immutable(root/'objects'/(canonical_json_sha256_hex(proof)+'.json'),proof)
            write_json_durable(pointer,local)
            return local  # Historical snapshots NEVER roll back current owner status.
        previous_ref,previous=_status(exchange,role)
        if previous is not None:
            if proof['revision']<previous['revision']:raise ValueError('source-host role status rollback')
            if proof['revision']==previous['revision']:
                if reference['sha256']!=previous_ref['sha256']:raise ValueError('source-host role revision equivocation')
                return previous_ref
        local=_immutable(root/'objects'/(canonical_json_sha256_hex(proof)+'.json'),proof)
        write_json_durable(root/(role+'-'+proof['kind']+'.json'),local)
        write_json_durable(root/(role+'-status.json'),local)
        return local


def exchanged(plan,*,kind):
    from ..train.native_readiness import _bound
    _,exchange=ownership(plan)
    if exchange is None:return {},[]
    frozen={}
    if kind=='snapshot' and plan.get('consumer_preparation_handoff') and exchange['role']==exchange['semantic_consumer']:
        pointer=Path(exchange['directory'])/'preparation-handoff.json'
        if pointer.is_file():
            preparation=json.loads(_bound(json.loads(pointer.read_text())).read_text())
            if preparation.get('contract')!=contract(plan):raise ValueError('frozen snapshot request changed its contract')
            current=Path(exchange['directory'])/'preparation-work.json'
            consumed=Path(exchange['directory'])/'preparation-consumed'/(json.loads(pointer.read_text())['sha256']+'.json')
            if (not consumed.is_file() and current.is_file()
                    and json.loads(current.read_text())['identity']==preparation['work_identity']):
                frozen={row['source_host_id']:row['role_release']for row in preparation['snapshots']['participants'].values()}
    found={};pending=[]
    for role in exchange['roles']:
        reference,proof=_status(exchange,role)
        if kind=='snapshot':
            pointer=Path(exchange['directory'])/(role+'-eligible.json')
            if exchange['roles'][role]['source_host_id']in frozen:
                reference=frozen[exchange['roles'][role]['source_host_id']]
                proof=json.loads(_bound(reference).read_text())
            elif pointer.is_file():
                reference=json.loads(pointer.read_text());proof=json.loads(_bound(reference).read_text())
            if proof is not None and proof.get('kind') in ('eligible','released'):
                actual,models=validate(reference,plan,kind=proof['kind'])
                if actual!=role or set(found)&set(models):raise ValueError('snapshot exchange repeated a participant')
                if plan.get('snapshot_preparation'):
                    for model in models.values():
                        if not model.get('snapshot_checkpoint'):
                            raise ValueError('live snapshot preparation needs a genuinely pinned checkpoint bundle')
                        model['checkpoint']={**model['checkpoint'],**model['snapshot_checkpoint']}
                        model['source_role_plan']=plan
                found.update(models);continue
            pending.append({'role':role,'reason':'source host has no genuinely qualified snapshot'});continue
        if proof is None or proof.get('kind')not in(('eligible','released')if kind=='eligible'else('released',)):
            pending.append({'role':role,'reason':'source host has not published '+kind});continue
        actual,models=validate(reference,plan,kind=proof['kind'])
        if actual!=role or set(found)&set(models):raise ValueError('role exchange repeated a participant owner')
        found.update(models)
    return found,pending


def eligible(plan):
    if preparation_handoff(plan):
        _,exchange=ownership(plan);reference,proof=_status(exchange,exchange['role'])
        if proof is not None and proof.get('kind') in ('eligible','released'):
            validate(reference,plan,kind=proof['kind'])
            return True  # Only the consumer owns this request; peers keep SGD.
    if plan.get('snapshot_preparation') and not handoff_active(plan):return False
    found,_=exchanged(plan,kind='eligible')
    if plan.get('snapshot_preparation'):
        _,exchange=ownership(plan)
        for role in exchange['roles']:
            reference,proof=_status(exchange,role)
            if not _handoff_proof_matches(plan,proof,reference):return False
    return len(found)>=2


def aggregate(plan,*,assets=None,snapshot=None):
    """Qualified roles are distinct from locally consumable semantic assets."""
    found,pending=exchanged(plan,kind='snapshot' if _snapshot_mode(plan,snapshot) else 'released')
    if len(found)<2:return [],pending
    result=[];assets=assets or {}
    from ..train.native_readiness import _bound
    for model_id,ready in sorted(found.items()):
        asset=assets.get(model_id)
        if asset is None:
            pending.append({'model_id':model_id,'source_host_id':ready['source_host_id'],
                'checkpoint':ready['checkpoint'],'role_release':ready['source_role_release'],
                'reason':'qualified source checkpoint needs actual local semantic assets',
                'semantic_assets_required':True});continue
        path=_bound(asset)
        if (asset['sha256']!=ready['checkpoint']['sha256']
                or('bytes'in ready['checkpoint']and path.stat().st_size!=ready['checkpoint']['bytes'])):
            raise ValueError('semantic asset substituted a source-host released checkpoint')
        result.append({**ready,'checkpoint':{**ready['checkpoint'],'path':str(path)}})
    return result,pending


def receive(raw,plan_path,*,snapshot_only=False):
    """Import exact transported bytes through the same strict proof validator."""
    plan=json.loads(Path(plan_path).read_text());proof=json.loads(raw)
    _,exchange=ownership(plan)
    reference=_immutable(Path(exchange['directory'])/'incoming'/
        (canonical_json_sha256_hex(proof)+'.json'),proof)
    return import_proof(reference,plan,snapshot_only=snapshot_only)


def _snapshot_mode(plan,snapshot):
    if snapshot is not None and type(snapshot)is not bool:
        raise ValueError('semantic asset mode must be explicit')
    return plan.get('snapshot_preparation',False) if snapshot is None else snapshot


def semantic_request(plan,*,snapshot=None):
    """Only the designated semantic owner requests assets after genuine releases."""
    _,exchange=ownership(plan)
    if exchange is None or exchange['role']!=exchange['semantic_consumer']:return None
    found,pending=exchanged(plan,kind='snapshot' if _snapshot_mode(plan,snapshot) else 'released')
    if len(found)<2:return None
    request={'schema':'natlang.native-role-semantic-assets-request/1','contract':contract(plan),
        'participants':{name:{'role_release':row['source_role_release'],
            'checkpoint':row['checkpoint'],'source_host_id':row['source_host_id']}
            for name,row in sorted(found.items())}}
    if _snapshot_mode(plan,snapshot):
        request['snapshot_preparation']=True
    write_json_durable(Path(exchange['directory'])/'semantic-assets-request.json',request)
    return request


def ready_participants(plan,*,snapshot=None):
    _,exchange=ownership(plan)
    if exchange['role']!=exchange['semantic_consumer']:
        return [],[{'reason':'shared stage belongs to its explicit semantic consumer',
                    'semantic_consumer':exchange['semantic_consumer']}]
    request=semantic_request(plan,snapshot=snapshot)
    if request is None:return aggregate(plan,snapshot=snapshot)
    path=Path(exchange['directory'])/'semantic-assets.json'
    if not path.exists():return aggregate(plan,snapshot=snapshot)
    manifest=json.loads(path.read_text())
    if manifest.get('schema')!='natlang.native-role-semantic-assets/1':
        raise ValueError('unsupported semantic assets manifest')
    if manifest.get('request')!=request:return aggregate(plan,snapshot=snapshot)
    return aggregate(plan,assets=manifest['checkpoints'],snapshot=snapshot)


def seal_local(config):
    """Called after the owned raw child exits, never as a live release shortcut."""
    from ..train.native_readiness import _bound
    from .native_raw_rounds import raw_out
    plan_path=_bound(config['raw_rl_bootstrap']);plan=json.loads(plan_path.read_text())
    if ownership(plan)[1]is None:return None
    return finalize(plan_path,raw_out(config)/'state.json')



def finalize(plan_path,state_path):
    """Source-host finalizer for a naturally exited raw owner, never a stop API."""
    plan=json.loads(Path(plan_path).read_text());_,exchange=ownership(plan)
    path=Path(state_path)
    if not path.exists():return None
    state=json.loads(path.read_text())
    if state.get('released')is not True:return None
    from ..train.native_resident import owner_alive
    if owner_alive(state.get('owner')or{}):return None
    reference,proof=_status(exchange,exchange['role'])
    if (proof is not None and proof.get('kind')=='released'
            and proof['raw_state']['source']==binding(path)):
        validate(reference,plan,kind='released');return reference
    return publish(plan_path,path,kind='released')

def local_released(config):
    """Check actual local dead-owner release, so peer/asset waits cannot relaunch it."""
    from ..train.native_readiness import _bound
    from .native_raw_rounds import raw_out,validate_release
    plan_path=_bound(config['raw_rl_bootstrap']);plan=json.loads(plan_path.read_text())
    if ownership(plan)[1]is None:return False
    path=raw_out(config)/'state.json'
    if not path.exists()or json.loads(path.read_text()).get('released')is not True:return False
    # Existing proof was created only by full local validation after owner exit.
    # Revalidate the exact source state binding; changing it requires a fresh seal.
    _,exchange=ownership(plan);reference,proof=_status(exchange,exchange['role'])
    if proof is None or proof.get('kind')!='released':return seal_local(config)is not None
    from ..train.native_resident import owner_alive
    if owner_alive(proof['controller']):raise ValueError('released source-host raw owner is still live')
    if proof['raw_state']['source']!=binding(path):raise ValueError('released source state changed')
    validate(reference,plan,kind='released')
    return True



def pin_checkpoint(reference,directory):
    """Reuse the immutable portable checkpoint bundle, including composite roles."""
    from ..train.native_readiness import _bound
    from ..model.native_bundle import export_bundle
    source=_bound(reference);before=binding(source)
    bundle=export_bundle(source,Path(directory)/reference['sha256'],participant=reference.get('participant'))
    if binding(source)!=before:raise ValueError('source checkpoint changed during semantic bundle')
    checkpoint=binding(bundle['checkpoint']['resolved'])
    if checkpoint['sha256']!=reference['sha256']:raise ValueError('semantic bundle changed checkpoint identity')
    return checkpoint


def consumed_metadata(ready):
    """Exact files read by the unchanged semantic participant/readiness loader."""
    from ..train.native_readiness import _bound
    from ..train.recipe import native_source_dependencies,recipe_file_dependencies,recipe_identity_sha256
    if ready.get('source_role_plan'):
        # All readiness/report/task/numerical bytes are inline in the exact
        # source-host proof. Do not reread a rolling historical source path.
        return [binding(_bound(ready['source_role_release']))]
    receipt=json.loads(_bound(ready['readiness']).read_text())
    recipe=ready['recipe']
    recipe_path=_bound({'path':recipe['path'],'sha256':recipe.get('file_sha256',recipe['sha256'])})
    if recipe_identity_sha256(recipe_path)!=recipe['sha256']:
        raise ValueError('semantic metadata inherited recipe identity changed')
    references=[*recipe_file_dependencies(recipe_path),ready['readiness'],receipt['training_receipt'],
        receipt['runtime_certificate'],receipt['task_execution_receipt']]
    references.extend(row['exposure_receipt']for row in receipt.get('prerequisite_stages',[]))
    task=json.loads(_bound(receipt['task_execution_receipt']).read_text())
    if task.get('schema')=='natlang.native-task-execution-receipt/2':
        references.append({'path':task['score_path'],'sha256':task['score_sha256']})
    requirement=ready.get('readiness_requirement')or{}
    if requirement.get('trainer_exit'):references.append(binding(requirement['trainer_exit']))
    for key in('trainer_launch','resident_adoption'):
        if requirement.get(key):references.append(requirement[key])
    # validate_ready also checks the current resident generation beside its
    # readiness output, even when no explicit launch adoption guard is present.
    implicit=_bound(ready['readiness']).parent.parent/'resident-data'/'current.json'
    if implicit.exists():references.append(binding(implicit))
    foundation=requirement.get('foundation_sources')
    if foundation:
        path=_bound(foundation);references.append(foundation)
        references.extend(native_source_dependencies(path))
    return list({item['path']:item for item in(binding(_bound(reference))for reference in references)}.values())


def install_assets(plan_path,request,checkpoints):
    """Commit only current joint-consumption assets after exact local verification."""
    plan=json.loads(Path(plan_path).read_text())
    snapshot=request.get('snapshot_preparation',False)
    if semantic_request(plan,snapshot=snapshot)!=request:raise ValueError('semantic asset request was superseded')
    ready,pending=aggregate(plan,assets=checkpoints,snapshot=snapshot)
    if (len(ready)<2 or any(row.get('semantic_assets_required')for row in pending)
            or set(checkpoints)!=set(request['participants'])):
        raise ValueError('semantic participant asset closure is incomplete')
    _,exchange=ownership(plan)
    manifest={'schema':'natlang.native-role-semantic-assets/1','request':request,'checkpoints':checkpoints}
    write_json_durable(Path(exchange['directory'])/'semantic-assets.json',manifest)
    return manifest


def recipe_recovery_needed(ready):
    from ..train.native_readiness import _bound
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..train.recipe import training_recipe_binding
    saved=CheckpointPolicy.load(_bound(ready['checkpoint']))
    return training_recipe_binding(saved)is None


def install_checkpoint_alias(ready,checkpoint):
    """Restore an actual legacy readiness path using verified local bytes.

    No receipt or identity is rewritten. An existing different artifact is an
    explicit conflict, never overwritten. The alias references the independently
    copied immutable bundle, not a mutable source-host file.
    """
    from ..train.native_readiness import _bound
    target=_bound(checkpoint)
    if checkpoint['sha256']!=ready['checkpoint']['sha256']:
        raise ValueError('legacy readiness alias changed the actual checkpoint')
    path=Path(ready['checkpoint']['path'])
    if not path.is_absolute()or'..'in path.parts:raise ValueError('legacy readiness alias escaped its path')
    path.parent.mkdir(parents=True,exist_ok=True)
    if os.path.lexists(path):
        _bound(ready['checkpoint']);return binding(path)
    try:os.symlink(str(target),path)
    except FileExistsError:_bound(ready['checkpoint'])
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    _bound(ready['checkpoint'])
    return binding(path)


def missing_metadata(references):
    """Never overwrite a consumer's existing different authority file."""
    from ..train.native_readiness import _bound
    missing=[]
    for reference in references:
        path=Path(reference['path'])
        if os.path.lexists(path):_bound(reference)
        else:missing.append(reference)
    return missing


def metadata_package(references,directory):
    from ..train.native_readiness import _bound
    from ..model.native_bundle import _copy_immutable
    result=[]
    for reference in references:
        source=_bound(reference);target=Path(directory)/reference['sha256']
        _copy_immutable(source,target,reference['sha256'])
        result.append({'source':reference,'transport':binding(target)})
    return result


def install_metadata(package):
    from ..train.native_readiness import _bound
    from ..model.native_bundle import _copy_immutable
    for item in package:
        source,target=item['transport'],item['source']
        if source['sha256']!=target['sha256']or source['bytes']!=target['bytes']:
            raise ValueError('semantic metadata changed during transport')
        _copy_immutable(_bound(source),target['path'],target['sha256'])
    return [item['source']for item in package]


def post_vae_request(plan,completion,controller):
    """Exact frozen preparation requests fresh descendant work, never release."""
    from ..train.native_readiness import _bound,validate_snapshot_preparation
    _,exchange=ownership(plan)
    if exchange['role']!=exchange['semantic_consumer']:
        raise ValueError('only the semantic owner can request descendant adoption')
    path=_bound(completion);proof=json.loads(path.read_text())
    published={(proof.get('participants_signature'),proof.get('curriculum_signature')):
        {'execution_signature':canonical_json_sha256_hex(controller)}}
    validate_snapshot_preparation(path,published)
    full=json.loads(_bound(controller['curriculum']).read_text())
    from ..train.native_semantic_stage import validate_config
    validate_config(full)
    from ..train.native_deployment import validate_joint_task_specs
    validate_joint_task_specs(full,[row['model_id']for row in contract(plan)['models'].values()])
    if full.get('snapshot_preparation') or not full.get('task_eval') or not full.get('policy_schedule'):
        raise ValueError('post-VAE adoption needs actual joint/task/policy work')
    request={'schema':'natlang.native-post-vae-handoff/1','contract':contract(plan),
        'role':exchange['role'],'mechanism':_publishing_mechanism(plan),
        'preparation':_inline(completion),'joint_controller':controller,
        'joint_curriculum':_inline(controller['curriculum']),'readiness_granted':False}
    active=preparation_handoff(plan,check_assets=False)
    if active:
        # The VAE completion is already fully authenticated above. Consume only
        # the exact allocation request whose curriculum and snapshots it used.
        expected=json.loads(_bound(active['value']['curriculum']).read_text())
        identities=(proof.get('snapshot_preparation')or{}).get('participants',{})
        snapshots=active['value']['snapshots']['participants']
        if (canonical_json_sha256_hex(expected)!=proof['curriculum_signature']
                or set(identities)!=set(snapshots) or any(
                    identities[name].get('checkpoint_sha256')!=value['checkpoint']['sha256']
                    for name,value in snapshots.items())):
            raise ValueError('completed VAE did not consume the requested snapshots/curriculum')
        _immutable(Path(exchange['directory'])/'preparation-consumed'/
            (active['source']['sha256']+'.json'),{'request':active['source'],
            'completion':completion,'post_vae_request':canonical_json_sha256_hex(request)})
        _immutable(Path(exchange['directory'])/'preparation-completed-work'/
            (active['value']['work_identity']+'.json'),{'identity':active['value']['work_identity'],
            'request':active['source'],'completion':completion})
    return _immutable(Path(exchange['directory'])/'handoff-requests'/
        (canonical_json_sha256_hex(request)+'.json'),request)


def validate_handoff_request(plan,request):
    _,exchange=ownership(plan);consumer=exchange['semantic_consumer']
    if (request.get('schema')!='natlang.native-post-vae-handoff/1'
            or request.get('contract')!=contract(plan) or request.get('role')!=consumer
            or request.get('mechanism')!=_expected_mechanism(plan,consumer)
            or request.get('readiness_granted') is not False):
        raise ValueError('descendant handoff changed actual role authority')
    proof=_evidence(request['preparation']);curriculum=_evidence(request['joint_curriculum'])
    if (proof.get('preparation_complete') is not True or proof.get('production_ready') is not False
            or (proof.get('snapshot_preparation')or{}).get('participant_optimizer_updates')!=0
            or curriculum.get('snapshot_preparation') or not curriculum.get('task_eval')
            or not curriculum.get('policy_schedule')
            or request['joint_controller'].get('curriculum')!=request['joint_curriculum']['source']):
        raise ValueError('descendant handoff requires frozen VAE and executable joint work')
    return request


def handoff_offer(plan,state_path,request):
    """Probe the actual owner on its own host before arming a pause."""
    from ..train.native_resident import owner_alive
    validate_handoff_request(plan,request);owned,exchange=ownership(plan)
    state=json.loads(Path(state_path).read_text())
    reference,proof=_status(exchange,exchange['role'])
    if state.get('released') is True:
        if proof is None or proof.get('kind')!='released':return None
        if proof['raw_state']['source']!=binding(state_path):
            raise ValueError('handoff offer changed released source state')
        validate(reference,plan,kind='released')
    elif not owner_alive(state.get('owner')or{}):return None
    if state.get('configuration_errors'):return None
    candidates,_=foundation_candidates(plan)
    if not {plan_model(plan,name)['handoff_requirement']['model_id']for name in owned}<=set(candidates):
        return None
    attempts={name:value.get('foundation_qualification_attempt') for name,value in state.get('models',{}).items()}
    from ..train.native_readiness import _bound
    from ..train.native_deployment import binding as checked
    metadata={};checkpoints={};generations={}
    for name in owned:
        row=plan_model(plan,name);value=state['models'][name]
        checkpoints[name]=value['checkpoint'];generations[name]=row['foundation_qualification_generation']
        # Both providers must prove their actual current config and immutable
        # recipe/task/source inputs exist before any GPU is held. No Torch load.
        refs={key:row[key]for key in ('production_recipe','cases','seed_native_graphs','held_native_graphs')}
        refs['config']=value['config']
        metadata[name]={key:checked(ref['path'],{k:ref[k]for k in ('path','sha256','bytes')if k in ref})
            for key,ref in refs.items()}
        attempt=attempts.get(name)
        if attempt:
            selected=json.loads(_bound(attempt['request']).read_text())
            scope=selected.get('rendezvous_attempt_sha256')
            expected=canonical_json_sha256_hex({'foundation_generation':generations[name],'handoff':scope})
            if (selected.get('handoff_request_sha256')==canonical_json_sha256_hex(request)
                    and selected.get('generation')==expected and attempt.get('status')!='passed'):return None
    qualification=None
    if proof is not None and proof.get('kind')in('eligible','released'):
        _,ready=validate(reference,plan,kind=proof['kind'])
        if all(ready[plan_model(plan,name)['handoff_requirement']['model_id']]['checkpoint']['sha256']==
                checkpoints[name]['sha256']for name in owned):qualification=reference
    return {'role':exchange['role'],'mechanism':_publishing_mechanism(plan),
        'contract':contract(plan),'request_sha256':canonical_json_sha256_hex(request),
        'owner':state['owner'],'released':state.get('released')is True,'attempts':attempts,
        'checkpoints':checkpoints,'qualification_generations':generations,
        'provider_inputs':metadata,'current_qualification':qualification,
        'active_attempt':(handoff_active(plan)or{}).get('attempt_sha256')}


def _handoff_offers(plan,request,offers):
    _,exchange=ownership(plan);digest=canonical_json_sha256_hex(request)
    if set(offers)!=set(exchange['roles']):raise ValueError('handoff omitted an actual provider')
    for role,offer in offers.items():
        names=set(exchange['roles'][role]['owned_models'])
        if (not offer or offer.get('role')!=role or offer.get('contract')!=contract(plan)
                or offer.get('mechanism')!=_expected_mechanism(plan,role)
                or offer.get('request_sha256')!=digest or set(offer.get('checkpoints',{}))!=names
                or set(offer.get('qualification_generations',{}))!=names
                or set(offer.get('provider_inputs',{}))!=names):
            raise ValueError('descendant handoff provider offer changed')
    return offers


def _offer_progress(offers):
    return {role:{key:offer.get(key)for key in ('owner','checkpoints','qualification_generations','current_qualification')}
        for role,offer in offers.items()}


def select_handoff_attempt(plan,request,offers,*,lease_seconds):
    """Seal renewed rendezvous after actual progress, under the same VAE request."""
    from ..train.native_readiness import _bound
    validate_handoff_request(plan,request);_,exchange=ownership(plan)
    _handoff_offers(plan,request,offers)
    if exchange['role']!=exchange['semantic_consumer']:
        raise ValueError('only actual semantic owner seals a rendezvous attempt')
    if type(lease_seconds)not in(int,float)or not math.isfinite(lease_seconds)or lease_seconds<=0:
        raise ValueError('handoff attempt needs its explicit renewable lease')
    root=Path(exchange['directory']);root.mkdir(parents=True,exist_ok=True)
    with (root/'handoff-attempt.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        path=root/'handoff-attempt.json';previous=None;ordinal=1;now=time.time()
        if path.is_file():
            pointer=json.loads(path.read_text());previous=pointer['attempt']
            old=json.loads(_bound(previous).read_text());ordinal=old['ordinal']+1
            if old['request_sha256']==canonical_json_sha256_hex(request):
                live_path=root/'joint-handoff-live.json'
                live=json.loads(live_path.read_text())if live_path.is_file()else None
                same_live=live is not None and live.get('attempt_sha256')==previous['sha256']
                expiry=live['expires_at']if same_live else pointer['issued_until']
                same_owners=all(all(old['offers'][role][key]==offer[key]
                    for key in ('owner','qualification_generations'))for role,offer in offers.items())
                if expiry>now and same_owners and (not same_live or all(offer.get('active_attempt')==previous['sha256']
                        for offer in offers.values())):return _inline(previous)
                if _offer_progress(old['offers'])==_offer_progress(offers):return None
        attempt={'schema':'natlang.native-post-vae-rendezvous-attempt/1',
            'request_sha256':canonical_json_sha256_hex(request),'contract':contract(plan),
            'consumer':exchange['semantic_consumer'],'mechanism':_publishing_mechanism(plan),
            'ordinal':ordinal,'superseded_attempt':previous,'offers':offers,'readiness_granted':False}
        reference=_immutable(root/'handoff-attempts'/(canonical_json_sha256_hex(attempt)+'.json'),attempt)
        write_json_durable(path,{'attempt':reference,'issued_until':now+lease_seconds})
        return _inline(reference)


def _validate_handoff_attempt(plan,request,envelope):
    attempt=_evidence(envelope);_,exchange=ownership(plan)
    if (attempt.get('schema')!='natlang.native-post-vae-rendezvous-attempt/1'
            or attempt.get('request_sha256')!=canonical_json_sha256_hex(request)
            or attempt.get('contract')!=contract(plan) or attempt.get('consumer')!=exchange['semantic_consumer']
            or attempt.get('mechanism')!=_expected_mechanism(plan,exchange['semantic_consumer'])
            or type(attempt.get('ordinal'))is not int or attempt['ordinal']<1
            or attempt.get('readiness_granted')is not False):
        raise ValueError('rendezvous attempt changed authenticated VAE/provider authority')
    _handoff_offers(plan,request,attempt['offers'])
    predecessor=attempt.get('superseded_attempt')
    if (attempt['ordinal']==1)!=(predecessor is None):
        raise ValueError('rendezvous attempt lost its superseded attempt')
    return attempt


def renew_handoff(plan,request,offers,*,expires_at,attempt):
    """Existing relay grants a renewable ownership handshake, not quality credit."""
    validate_handoff_request(plan,request);_,exchange=ownership(plan)
    if (set(offers)!=set(exchange['roles']) or type(expires_at)not in(int,float)
            or not math.isfinite(expires_at) or expires_at<=time.time()):
        raise ValueError('descendant handoff needs both current providers and a live lease')
    digest=canonical_json_sha256_hex(request)
    _handoff_offers(plan,request,offers)
    _validate_handoff_attempt(plan,request,attempt)
    own=offers[exchange['role']]
    from ..train.native_resident import owner_alive
    if not own['released'] and not owner_alive(own['owner']):
        raise ValueError('descendant handoff owner ended before arm')
    value={'request':request,'request_sha256':digest,'offers':offers,'expires_at':expires_at,
        'attempt':attempt,'attempt_sha256':attempt['source']['sha256']}
    directory=Path(exchange['directory']);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'handoff-lease.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        path=directory/'joint-handoff-live.json'
        if path.is_file():
            old=json.loads(path.read_text())
            if old['request_sha256']==digest:
                prior=_validate_handoff_attempt(plan,request,old['attempt'])
                current=_evidence(attempt)
                if current['ordinal']<prior['ordinal'] or (current['ordinal']==prior['ordinal']
                        and old['attempt_sha256']!=attempt['source']['sha256']):
                    raise ValueError('handoff cannot roll back or equivocate a rendezvous attempt')
                if current['ordinal']==prior['ordinal']+1 and (
                        current['superseded_attempt']['sha256']!=old['attempt_sha256']):
                    raise ValueError('handoff changed its superseded rendezvous attempt')
        write_json_durable(path,value)
    return value


def handoff_active(plan):
    _,exchange=ownership(plan)
    if exchange is None:return None
    path=Path(exchange['directory'])/'joint-handoff-live.json'
    if not path.is_file():return None
    value=json.loads(path.read_text());validate_handoff_request(plan,value['request'])
    if canonical_json_sha256_hex(value['request'])!=value.get('request_sha256'):
        raise ValueError('descendant handoff request changed')
    _validate_handoff_attempt(plan,value['request'],value['attempt'])
    if value['attempt_sha256']!=value['attempt']['source']['sha256']:
        raise ValueError('handoff lease changed its rendezvous attempt')
    if value['expires_at']<=time.time():return None
    return value


def _handoff_proof_matches(plan,proof,reference):
    active=handoff_active(plan)
    if not active or proof is None or proof.get('kind') not in ('eligible','released'):return False
    carried=_evidence(active['attempt'])['offers'][proof['role']].get('current_qualification')
    if carried and carried['sha256']==reference['sha256']:
        offered=_evidence(active['attempt'])['offers'][proof['role']]['checkpoints']
        return all(offered[name]['sha256']==proof['models'][name]['ready']['checkpoint']['sha256']
            for name in proof['owned_models'])
    from ..train.native_readiness import _bound
    if set(proof.get('foundation_attempts',{}))!=set(proof['owned_models']):return False
    for name,attempt in proof['foundation_attempts'].items():
        expected=canonical_json_sha256_hex({'foundation_generation':plan_model(plan,name)['foundation_qualification_generation'],
            'handoff':active['attempt_sha256']})
        item=proof['models'][name];schedule=_evidence(item['report']).get('foundation_qualification_schedule')or{}
        if (attempt.get('status')!='passed'
                or json.loads(_bound(attempt['request']).read_text())['generation']!=expected
                or attempt.get('checkpoint',{}).get('sha256')!=item['ready']['checkpoint']['sha256']
                or attempt.get('report')!=item['report']['source']
                or schedule.get('request')!=attempt['request'] or schedule.get('requested')is not True
                or schedule.get('readiness_granted')is not True):return False
    return True


def cancel_handoff(plan,request):
    validate_handoff_request(plan,request);_,exchange=ownership(plan)
    path=Path(exchange['directory'])/'joint-handoff-live.json'
    if path.is_file():
        value=json.loads(path.read_text())
        if value['request_sha256']==canonical_json_sha256_hex(request):
            write_json_durable(path,{**value,'expires_at':0})


def handoff_hold(plan):
    """Pause only a genuinely qualified current descendant under the live lease."""
    active=handoff_active(plan)
    if not active:return False
    _,exchange=ownership(plan);reference,proof=_status(exchange,exchange['role'])
    if proof is None or proof.get('kind') not in ('eligible','released'):return False
    validate(reference,plan,kind=proof['kind'])
    return _handoff_proof_matches(plan,proof,reference)


def _preparation_work(plan,curriculum):
    from ..train.native_readiness import _bound
    from ..train.native_semantic_stage import validate_config
    _,exchange=ownership(plan)
    if exchange is None or exchange['role']!=exchange['semantic_consumer']:
        raise ValueError('only the semantic consumer selects preparation work')
    config=validate_config(json.loads(_bound(curriculum).read_text()))
    generation=config.get('snapshot_work_generation','initial')
    if config.get('snapshot_preparation')is not True or not isinstance(generation,str)or not generation:
        raise ValueError('preparation work needs frozen curriculum and explicit valid generation')
    return {'identity':canonical_json_sha256_hex({'contract':contract(plan),'curriculum':config}),
        'curriculum':curriculum,'contract':contract(plan)}


def select_preparation_work(plan,curriculum):
    """Explicit declared work may advance; model progress alone is not new work."""
    from ..train.native_readiness import _bound
    work=_preparation_work(plan,curriculum);identity=work['identity']
    _,exchange=ownership(plan)
    root=Path(exchange['directory']);root.mkdir(parents=True,exist_ok=True)
    pointer=root/'preparation-handoff.json'
    if pointer.is_file():
        reference=json.loads(pointer.read_text());previous=json.loads(_bound(reference).read_text())
        cancelled=root/'preparation-cancelled.json'
        ended=((root/'preparation-consumed'/(reference['sha256']+'.json')).is_file()or
            (cancelled.is_file()and json.loads(cancelled.read_text()).get('request')==reference))
        if previous['work_identity']!=identity and not ended:
            raise ValueError('checkpoint/cancel the active consumer preparation before changing its work')
    write_json_durable(root/'preparation-work.json',work)
    return work


def preparation_work_complete(plan,curriculum):
    work=_preparation_work(plan,curriculum)
    _,exchange=ownership(plan)
    path=Path(exchange['directory'])/'preparation-completed-work'/(work['identity']+'.json')
    if not path.is_file():return False
    from ..train.native_readiness import _bound
    value=json.loads(path.read_text());_bound(value['completion'])
    if value.get('identity')!=work['identity']:raise ValueError('completed preparation work identity changed')
    return True


def handoff_matches_preparation_work(plan,request,curriculum):
    validate_handoff_request(plan,request)
    from ..train.native_readiness import _bound
    from ..train.native_semantic_stage import validate_config
    config=validate_config(json.loads(_bound(curriculum).read_text()))
    return _evidence(request['preparation'])['curriculum_signature']==canonical_json_sha256_hex(config)


def arm_preparation_handoff(plan,request,curriculum):
    """Stage-before-release: the consumer lends its existing allocation to VAE."""
    from ..train.native_readiness import _bound
    from ..train.native_semantic_stage import validate_config
    _,exchange=ownership(plan)
    if (not plan.get('consumer_preparation_handoff') or not plan.get('snapshot_preparation')
            or exchange['role']!=exchange['semantic_consumer']):
        raise ValueError('frozen preparation may borrow only its declared consumer allocation')
    config=validate_config(json.loads(_bound(curriculum).read_text()))
    if config.get('snapshot_preparation') is not True:
        raise ValueError('consumer allocation request must be frozen VAE only')
    work=select_preparation_work(plan,curriculum)
    if preparation_work_complete(plan,curriculum):return None
    if request!=semantic_request(plan,snapshot=True):
        raise ValueError('preparation ownership changed its qualified snapshots')
    ready,pending=ready_participants(plan,snapshot=True)
    if len(ready)<2 or pending:raise ValueError('preparation assets must exist before consumer release')
    generations={row['model']:row.get('foundation_qualification_generation')for row in plan['models']}
    value={'schema':'natlang.native-consumer-preparation-handoff/1','contract':contract(plan),
        'consumer':exchange['role'],'snapshots':request,'curriculum':curriculum,
        'work_identity':work['identity'],
        'repair_generations':generations,'backbone_updates':0,'readiness_granted':False}
    root=Path(exchange['directory'])
    cancelled=root/'preparation-cancelled.json'
    if cancelled.is_file():
        cancellation=json.loads(cancelled.read_text())
        previous=json.loads(_bound(cancellation['request']).read_text())
        if {key:item for key,item in previous.items()if key!='ownership_attempt'}==value:
            if cancellation.get('retry_requires_repair',True):return None
            predecessor=_immutable(root/'preparation-cancellations'/
                (canonical_json_sha256_hex(cancellation)+'.json'),cancellation)
            value['ownership_attempt']={'ordinal':previous.get('ownership_attempt',{}).get('ordinal',1)+1,
                'superseded_attempt':cancellation['request'],'cancellation':predecessor}
    reference=_immutable(root/'preparation-requests'/(canonical_json_sha256_hex(value)+'.json'),value)
    if (root/'preparation-consumed'/(reference['sha256']+'.json')).is_file():return None
    write_json_durable(root/'preparation-handoff.json',reference)
    return reference


def preparation_handoff(plan,*,check_assets=True):
    from ..train.native_readiness import _bound
    _,exchange=ownership(plan)
    if (not plan.get('consumer_preparation_handoff') or not plan.get('snapshot_preparation')
            or exchange is None or exchange['role']!=exchange['semantic_consumer']):return None
    root=Path(exchange['directory']);pointer=root/'preparation-handoff.json'
    if not pointer.is_file():return None
    reference=json.loads(pointer.read_text());value=json.loads(_bound(reference).read_text())
    if (root/'preparation-consumed'/(reference['sha256']+'.json')).is_file():return None
    if (value.get('schema')!='natlang.native-consumer-preparation-handoff/1'
            or value.get('contract')!=contract(plan) or value.get('consumer')!=exchange['role']
            or value.get('backbone_updates')!=0 or value.get('readiness_granted')is not False):
        raise ValueError('consumer preparation request changed its actual ownership')
    if 'ownership_attempt'in value:
        attempt=value['ownership_attempt'];previous=json.loads(_bound(attempt['superseded_attempt']).read_text())
        cancellation=json.loads(_bound(attempt['cancellation']).read_text())
        if (type(attempt.get('ordinal'))is not int or attempt['ordinal']<2
                or attempt['ordinal']!=previous.get('ownership_attempt',{}).get('ordinal',1)+1
                or cancellation.get('request')!=attempt['superseded_attempt']
                or cancellation.get('retry_requires_repair')is not False
                or {key:item for key,item in previous.items()if key!='ownership_attempt'}!=
                   {key:item for key,item in value.items()if key!='ownership_attempt'}):
            raise ValueError('consumer ownership attempt changed its cancellation/predecessor')
    if check_assets and value.get('repair_generations')!={row['model']:row.get('foundation_qualification_generation')for row in plan['models']}:
        return None  # An explicit hot repair needs a freshly staged request.
    if (root/'preparation-cancelled.json').is_file():
        if json.loads((root/'preparation-cancelled.json').read_text()).get('request')==reference:return None
    # Reauthenticate all staged assets each time; a stale pointer cannot release.
    if check_assets:
        _bound(value['curriculum'])
        ready,pending=ready_participants(plan,snapshot=True)
        if len(ready)<2 or pending:return None
    return {'source':reference,'value':value}


def cancel_preparation_handoff(config,*,reason,repair_required=True):
    """Reopen the genuine released RAW cursor after the semantic child drains."""
    from ..train.native_readiness import _bound
    if type(repair_required)is not bool:raise ValueError('preparation cancellation repair policy must be explicit')
    from ..train.native_resident import owner_alive
    from .native_raw_rounds import raw_out
    plan_path=_bound(config['raw_rl_bootstrap']);plan=json.loads(plan_path.read_text())
    active=preparation_handoff(plan,check_assets=False)
    if active is None:return None
    path=raw_out(config)/'state.json';state=json.loads(path.read_text())
    if owner_alive(state.get('owner')or{}):raise ValueError('cannot reopen a live RAW owner')
    if state.get('released')is not True or not local_released(config):
        raise ValueError('preparation recovery requires its genuine consumer release')
    _,exchange=ownership(plan);root=Path(exchange['directory'])
    # Keep the original released state as immutable recovery/ownership provenance.
    before=_immutable(root/'preparation-recovery'/(canonical_json_sha256_hex(state)+'.json'),state)
    cancellation={'request':active['source'],'released_state':before,'reason':reason,
        'schema':'natlang.native-consumer-preparation-cancellation/1','optimizer_reset':False,
        'retry_requires_repair':repair_required}
    write_json_durable(root/'preparation-cancelled.json',cancellation)
    write_json_durable(path,{**state,'released':False,'preparation_cancellation':cancellation})
    withdraw(plan,path)
    return cancellation


def preparation_raw_recovery(config):
    from ..train.native_readiness import _bound
    plan=json.loads(_bound(config['raw_rl_bootstrap']).read_text());_,exchange=ownership(plan)
    if exchange is None or not plan.get('consumer_preparation_handoff'):return False
    path=Path(exchange['directory'])/'preparation-cancelled.json'
    if not path.is_file() or preparation_handoff(plan)is not None:return False
    cancellation=json.loads(path.read_text())
    request=json.loads(_bound(cancellation['request']).read_text())
    if (cancellation.get('schema')!='natlang.native-consumer-preparation-cancellation/1'
            or request.get('contract')!=contract(plan) or request.get('consumer')!=exchange['role']
            or cancellation.get('optimizer_reset')is not False):
        raise ValueError('RAW recovery cancellation changed its ownership')
    from .native_raw_rounds import raw_out
    from ..train.native_resident import owner_alive
    state_path=raw_out(config)/'state.json';state=json.loads(state_path.read_text())
    if state.get('released')is True:
        original=json.loads(_bound(cancellation['released_state']).read_text())
        if state!=original or owner_alive(state.get('owner')or{}):
            raise ValueError('cannot reconcile cancellation over a changed/live RAW owner')
        write_json_durable(state_path,{**state,'released':False,'preparation_cancellation':cancellation})
        withdraw(plan,state_path)
    return True
