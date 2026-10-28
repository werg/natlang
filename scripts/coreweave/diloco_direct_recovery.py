"""Validate direct DiLoCo launch custody for full-cohort recovery."""
import json
import hashlib
import os
import re
from pathlib import Path
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex

REQUIRED=('plan','recipe','worker_argv','coordinator_argv','settings','owner_journal','launch_code','recovery_code',
          'pin_code','direct_code','dependencies_code','archive_code','guard_code','readiness_code','inputs_code','stage_code',
          'private_export_code','verified_download_code')
SOURCE_ARGUMENTS=frozenset({'--native-graphs','--held-native-graphs','--foundation-native-graphs',
    '--continued-native-graphs','--continued-held-native-graphs','--source-correction',
    '--source-classification-migration','--ce-supervision-policy','--sft-packed-geometry',
    '--memory-qualification','--recipe-binding','--pending-native-sources','--withdrawn-native-graphs',
    '--retired-native-withdrawal'})


def _flag(argv,name):
    if argv.count(name)!=1:raise ValueError('direct argv must contain exactly one '+name)
    return argv[argv.index(name)+1]


def _identity_argv(argv):
    """Match native_diloco.trainer_argv without importing torch in the operator."""
    result=[];i=0
    while i<len(argv):
        value=argv[i]
        if value=='--diloco':
            if i+1>=len(argv):raise ValueError('missing DiLoCo argument')
            i+=2;continue
        if not value.startswith('--diloco='):result.append(value)
        i+=1
    return result


def _fixed_identity_argv(argv):
    result=[];i=0
    while i<len(argv):
        value=argv[i];name=value.split('=',1)[0]
        if name in SOURCE_ARGUMENTS:
            if '=' not in value:
                if i+1>=len(argv) or argv[i+1].startswith('--'):raise ValueError('source identity argument lacks a value')
                i+=1
        else:result.append(value)
        i+=1
    return result


def load_control(path, record, *, control_bytes=None, read_bound=None, verify_helpers=True):
    path=Path(path);raw=path.read_bytes() if control_bytes is None else control_bytes;control=json.loads(raw)
    if control.get('schema')!='natlang.diloco-direct-recovery-control/1':raise ValueError('invalid direct control schema')
    bindings=control.get('bindings') or {}
    if not set(REQUIRED)<=set(bindings):raise ValueError('direct control lacks launch, recovery or custody bindings')
    def read(item):
        data=read_bound(item) if read_bound else Path(item['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('direct control binding differs: '+item['path'])
        return data
    for name,item in bindings.items():
        if not Path(item['path']).is_absolute():raise ValueError('direct control binding path is not absolute: '+name)
        try:read(item)
        except (OSError,KeyError,ValueError) as error:raise ValueError('direct control binding differs: '+name) from error
    local=Path(__file__).parent
    for key,name in [('recovery_code','diloco_periodic_recovery.py'),('pin_code','diloco_recovery_pin.py'),
                     ('direct_code','diloco_direct_recovery.py'),('dependencies_code','diloco_recovery_dependencies.py'),
                     ('archive_code','cache_bounded_archive.py'),('guard_code','private_export_guard.py'),
                     ('readiness_code','recovery_readiness.py'),('inputs_code','recovery_inputs.py'),
                     ('stage_code','diloco_stage_recovery.py')]:
        if verify_helpers and sha256_file_hex(local/name,drop_cache=True)!=bindings[key]['sha256']:
            raise ValueError('executing recovery helper differs: '+key)
    def data(name):return json.loads(read(bindings[name]))
    plan=data('plan');argv=data('worker_argv');coord=data('coordinator_argv');settings=data('settings');owner=data('owner_journal')
    paths=control['paths'];out=paths['worker_out'];root=paths['coordinator_root'];runtime=paths['runtime_repo']
    reserve=control.get('storage_reserve_bytes')
    if type(reserve) is not int or reserve<0:raise ValueError('direct storage needs an explicit nonnegative reserve')
    placement=control.get('component_placement',{})
    if placement and set(placement)!={'coordinator',*(f'worker-{i}' for i in range(len(record['learners'])))}:
        raise ValueError('direct storage placement must declare every component')
    for item in placement.values():
        if (not isinstance(item,dict) or not isinstance(item.get('root'),str) or not item['root'].startswith('/')
                or os.path.normpath(item['root'])!=item['root'] or type(item.get('reserve_bytes')) is not int or item['reserve_bytes']<0):
            raise ValueError('invalid direct component storage placement')
    for value in (out,root,runtime):
        if not isinstance(value,str) or not value.startswith('/work/') or os.path.normpath(value)!=value:
            raise ValueError('direct recovery paths must be canonical /work paths')
    original_argv=json.loads(_flag(coord,'--trainer-argv'))
    if _flag(argv,'--out')!=out or _flag(coord,'--root')!=root:
        raise ValueError('direct coordinator and worker paths differ')
    if _fixed_identity_argv(_identity_argv(original_argv))!=_fixed_identity_argv(_identity_argv(argv)):
        raise ValueError('direct source transition changes fixed trainer arguments')
    if json.loads(_flag(coord,'--settings'))!=settings or json.loads(_flag(argv,'--diloco'))!=settings:
        raise ValueError('direct coordinator and learner settings differ')
    if settings.get('workers')!=len(record['learners']) or settings.get('run_id')!=record['run']['settings']['run_id']:
        raise ValueError('direct run settings differ from actual cohort')
    if _flag(coord,'--parent')!=_flag(argv,'--continue-from'):
        raise ValueError('direct coordinator and worker parent differ')
    recipe=json.loads(_flag(argv,'--recipe-binding'))
    if (recipe.get('path')!=bindings['recipe']['path'] or recipe.get('file_sha256')!=bindings['recipe']['sha256']
            or recipe.get('sha256')!=bindings['recipe']['sha256'] or recipe.get('lineage_sha256')!=plan['recipe_sha256']):
        raise ValueError('direct recipe binding differs from plan')
    if canonical_json_sha256_hex(plan['code'])!=control['runtime_code_sha256']:
        raise ValueError('direct plan runtime inventory differs')
    stage=plan['stage_inputs'][_flag(argv,'--stage-id')]
    roles={'native_graphs':'--native-graphs','held_native_graphs':'--held-native-graphs',
           'foundation_native_graphs':'--foundation-native-graphs','ce_supervision_policy':'--ce-supervision-policy'}
    for key,flag in roles.items():
        if stage[key]['path']!=_flag(argv,flag):raise ValueError('direct source role differs: '+key)
    if stage['native_graphs']['sha256']!=stage['foundation_native_graphs']['sha256']:
        raise ValueError('direct foundation and train source roles differ')
    workers=owner.get('workers') or {}
    if (owner.get('run_id')!=settings['run_id'] or owner.get('worker_out')!=out or owner.get('coordinator_root')!=root
            or (owner.get('runtime') or {}).get('package')!=runtime+'/training/neuralese/natlang_neuralese'
            or (owner.get('plan') or {})!=bindings['plan'] or (owner.get('coordinator') or {}).get('sandbox_id')!=record['coordinator']):
        raise ValueError('direct owner journal differs from launch controls')
    if not all((owner.get('coordinator') or {}).get(k) for k in ('pid','boot_id','start_ticks','owner_id')):
        raise ValueError('direct owner journal lacks coordinator identity')
    for i,ident in enumerate(record['learners']):
        role=workers.get(str(i)) or {}
        if (role.get('sandbox_id')!=ident or not all(role.get(k) for k in ('pid','boot_id','start_ticks'))):
            raise ValueError('direct owner journal lacks actual worker identity')
    manifest_checks={'workers':settings['workers']}
    for flag,key,parse in [('--quorum','min_quorum',int),('--max-staleness','max_staleness',int),
                           ('--ema','ema',json.loads),('--outer-schedule','outer_schedule',json.loads)]:
        if flag in coord:manifest_checks[key]=parse(_flag(coord,flag))
    result={'paths':paths,'storage_reserve_bytes':reserve,'component_placement':placement,'control_sha256':hashlib.sha256(raw).hexdigest(),'manifest_checks':manifest_checks,
            'original_identity_argv':_identity_argv(original_argv),'current_identity_argv':_identity_argv(argv),
            'transition_required':_identity_argv(original_argv)!=_identity_argv(argv),
            'helper_sha256':{key:bindings[key]['sha256'] for key in ('pin_code','archive_code','guard_code','readiness_code','inputs_code')},
            'plan_path':bindings['plan']['path'],'parent':_flag(coord,'--parent'),
            'parent_transfer':control.get('parent_transfer'),
            'external_dependencies':control.get('external_dependencies',[]),
            'runtime_code_sha256':control['runtime_code_sha256'],'recipe_binding':recipe,
            'source_receipt_sha256':[stage['native_graphs']['sha256'],stage['held_native_graphs']['sha256']],
            'ce_sha256':stage['ce_supervision_policy']['sha256'],'workers':workers,'coordinator':owner['coordinator'],
            'bindings':list(bindings.values())+[{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}]}
    return result


def record_from_control(path, run_id, *, control_bytes=None, read_bound=None, verify_helpers=True):
    raw=Path(path).read_bytes() if control_bytes is None else control_bytes
    control=json.loads(raw);bindings=control['bindings']
    read=read_bound or (lambda item:Path(item['path']).read_bytes())
    owner=json.loads(read(bindings['owner_journal']))
    settings=json.loads(read(bindings['settings']))
    if settings.get('run_id')!=run_id:raise ValueError('direct run ID differs from settings')
    workers=owner['workers']
    record={'run':{'id':run_id,'settings':settings},'coordinator':owner['coordinator']['sandbox_id'],
            'learners':[workers[str(i)]['sandbox_id'] for i in range(settings['workers'])]}
    load_control(path,record,control_bytes=raw,read_bound=read_bound,verify_helpers=verify_helpers)
    return record


def worker_matches(worker, control):
    binding=worker.get('recipe_binding')
    if isinstance(binding,str):
        try:binding=json.loads(binding)
        except ValueError:return False
    ce=worker.get('ce_supervision_policy_binding') or {}
    return (worker.get('resident_generation') is None and worker.get('runtime_code_sha256')==control['runtime_code_sha256']
            and binding==control['recipe_binding'] and worker.get('source_receipt_sha256')==control['source_receipt_sha256']
            and ce.get('sha256')==control['ce_sha256'])


def transition_matches(worker,index,coordinator,control):
    """Bind a resumed checkpoint to the coordinator's committed source transition."""
    if not control.get('transition_required'):return True
    original=(coordinator.get('manifest') or {}).get('identity') or {}
    if original.get('trainer_argv')!=control['original_identity_argv']:return False
    histories=(coordinator.get('identity_transitions') or {}).get(str(index)) or []
    worker_history=worker.get('identity_transition_history') or []
    if not histories or not worker_history:return False
    proof=histories[-1];local=worker_history[-1]
    if not isinstance(proof,dict) or not isinstance(local,dict):return False
    if {k:v for k,v in proof.items() if k not in ('worker','fingerprint')}!=local:return False
    current=proof.get('current_identity') or {}
    if (proof.get('schema')!='natlang.diloco-source-identity-transition/1' or proof.get('correction_verified') is not True
            or proof.get('worker')!=index or proof.get('fingerprint')!=coordinator.get('fingerprint')
            or proof.get('original_identity')!=original or proof.get('source_hash')!=worker.get('source_hash')
            or current!={**original,'trainer_argv':control['current_identity_argv']}):return False
    if not all(isinstance(proof.get(key),str) and re.fullmatch('[0-9a-f]{64}',proof[key])
               for key in ('checkpoint_sha256','correction_sha256','previous_source_hash','source_hash')):return False
    return proof['previous_source_hash']!=proof['source_hash'] or proof.get('transition_kind')=='consumed-proof-retirement'
