"""Source-sealed, relocation-safe evidence for external native task evaluation.

Seal on the evaluation host while its original score path exists. The raw source
files remain byte-for-byte unchanged and their bindings travel with the receipt.
This is custody evidence, not cryptographic proof that a remote host ran a model.
"""
from __future__ import annotations
import hashlib, json, os, re
from pathlib import Path, PurePosixPath
from .native_state import _execution_receipt
from .task_cadence import parse_task_eval

SCHEMA='natlang.portable-native-task-evidence/1'
FILES=('completed.json','intent.json','spec.json','task-score.json','launch.json','actor-info.json',
       'transport.json','view-receipt.json','runtime-closure.json')

def digest(data):return hashlib.sha256(data).hexdigest()
def signature(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def checked_read(path):
    path=Path(path);before=signature(path);data=path.read_bytes()
    if signature(path)!=before:raise ValueError('task evidence changed while reading')
    return data

def _validate(parts, *, identity, packet):
    completed=json.loads(parts['completed.json']);intent=json.loads(parts['intent.json'])
    spec=parse_task_eval(parts['spec.json'].decode());score=json.loads(parts['task-score.json'])
    contract=completed.get('contract') or {};info=completed.get('actor_info') or {}
    native=info.get('native_state') or {};summary=completed.get('summary') or {}
    launch=json.loads(parts['launch.json']);before=json.loads(parts['actor-info.json'])
    transport=json.loads(parts['transport.json'])
    actor_source=(info.get('rl_policy_binding') or {}).get('inference_source') or {}
    if (completed.get('status')!='finished' or completed.get('training_admission') is not False
            or completed.get('separate_held_outputs') is not True or contract!=intent
            or contract.get('native_identity')!=info.get('rl_policy_binding')
            or (info.get('rl_policy_binding') or {}).get('native_identity')!=identity
            or contract.get('checkpoint_sha256')!=identity.get('checkpoint_sha256')
            or native.get('checkpoint_sha256')!=identity.get('checkpoint_sha256') or native.get('enabled') is not True
            or not isinstance(contract.get('model'),str) or not contract['model']
            or not isinstance(contract.get('endpoint'),str) or not contract['endpoint']
            or contract.get('spec_sha256')!=digest(parts['spec.json'])
            or contract.get('step')!=summary.get('step')
            or info.get('prompt_rendering')!=spec['prompt_rendering']
            or summary.get('prompt_rendering')!=spec['prompt_rendering']
            or spec.get('neuralese_server') is not True or spec.get('async')):
        raise ValueError('portable task actor, intent or source specification differs')
    if (before.get('rl_policy_binding')!=info.get('rl_policy_binding')
            or (before.get('native_state') or {}).get('checkpoint_sha256')!=identity.get('checkpoint_sha256')
            or (before.get('native_state') or {}).get('enabled') is not True
            or before.get('prompt_rendering')!=spec['prompt_rendering']
            or launch.get('checkpoint_sha256')!=identity.get('checkpoint_sha256')
            or launch.get('step')!=contract.get('step')
            or launch.get('task_spec_sha256')!=contract.get('spec_sha256')
            or not isinstance(launch.get('container_id'),str) or not launch['container_id']
            or not isinstance(launch.get('image_id'),str) or not launch['image_id']
            or any(not isinstance(launch.get(key),str) or re.fullmatch('[0-9a-f]{64}',launch[key]) is None
                   for key in ('runtime_receipt_sha256','transport_sha256','inference_view_sha256'))):
        raise ValueError('portable task launch or observed actor contract differs')
    if (digest(parts['transport.json'])!=launch['transport_sha256']
            or digest(parts['runtime-closure.json'])!=launch['runtime_receipt_sha256']
            or (transport.get('source') or {}).get('sha256')!=identity.get('checkpoint_sha256')
            or (transport.get('view') or {}).get('sha256')!=launch['inference_view_sha256']
            or digest(parts['view-receipt.json'])!=(transport.get('view_receipt') or {}).get('sha256')
            or actor_source.get('schema')!='natlang.native-inference-source/1'
            or actor_source.get('training_checkpoint_sha256')!=identity.get('checkpoint_sha256')
            or actor_source.get('step')!=contract.get('step')
            or transport.get('scope')!='diagnostic-local-pin'):
        raise ValueError('portable task runtime or inference transport source differs')
    if (digest(packet)!=spec['packet']['sha256'] or summary.get('packet_sha256')!=digest(packet)
            or summary.get('case_indices')!=spec['cases']):
        raise ValueError('portable task packet or case selection differs')
    rows=[json.loads(line) for line in packet.splitlines() if line.strip()]
    try: ids=[rows[index]['id'] for index in spec['cases']]
    except (IndexError,KeyError,TypeError):raise ValueError('portable task case indices are outside packet') from None
    if [row.get('id') for row in score.get('cases_detail',[])]!=ids:
        raise ValueError('portable task outcomes do not match selected packet cases')
    run=summary.get('run')
    if not isinstance(run,str) or not PurePosixPath(run).is_absolute():raise ValueError('missing original task run path')
    original=str(PurePosixPath(run)/'task-score.json')
    receipt=_execution_receipt(summary,identity,checkpoint_sha256=identity['checkpoint_sha256'],
        score=score,path=Path(original),digest=digest(parts['task-score.json']))
    if not receipt['passed']:raise ValueError('portable task evidence is incomplete: '+str(receipt['execution_errors']))
    return {'score':score,'original_score_path':original,'receipt':receipt}

def seal(evaluation_dir,spec_path,packet_path,out,*,source_root,container_root,transport_path,view_receipt_path,runtime_closure_path):
    """Seal original evaluator files through an explicit container mount mapping."""
    directory=Path(evaluation_dir).resolve(strict=True);spec_path=Path(spec_path).resolve(strict=True)
    source_root=Path(source_root).resolve(strict=True);container_root=PurePosixPath(container_root)
    if spec_path != (source_root/'spec.json').resolve(strict=True):raise ValueError('spec does not belong to source evaluation')
    if not container_root.is_absolute() or '..' in container_root.parts:
        raise ValueError('container root must be an absolute canonical path')
    completed_path=directory/'completed.json';intent_path=directory/'intent.json'
    completed=checked_read(completed_path);obj=json.loads(completed)
    run=(obj.get('summary') or {}).get('run')
    if not isinstance(run,str) or not Path(run).is_absolute():raise ValueError('missing original task run path')
    original=PurePosixPath(run)/'task-score.json'
    try:relative=original.relative_to(container_root)
    except ValueError:raise ValueError('task score is outside the declared container mount') from None
    if '..' in relative.parts:raise ValueError('task score escapes the declared container mount')
    score_path=(source_root/str(relative)).resolve(strict=True)
    if not score_path.is_relative_to(source_root):raise ValueError('mapped score escapes source root')
    if directory!=score_path.parent.parent.parent.resolve():
        raise ValueError('mapped score does not belong to the source evaluation directory')
    sources={'completed.json':completed_path,'intent.json':intent_path,'spec.json':spec_path,'task-score.json':score_path,
             'launch.json':source_root/'launch.json','actor-info.json':source_root/'actor-info.json',
             'transport.json':Path(transport_path),'view-receipt.json':Path(view_receipt_path),
             'runtime-closure.json':Path(runtime_closure_path)}
    parts={'completed.json':completed,**{name:checked_read(path) for name,path in sources.items() if name!='completed.json'}}
    packet=checked_read(packet_path);identity=((obj.get('contract') or {}).get('native_identity') or {}).get('native_identity')
    if not isinstance(identity,dict):raise ValueError('missing observed native identity')
    result=_validate(parts,identity=identity,packet=packet)
    out=Path(out).absolute();out.mkdir(parents=True,exist_ok=True)
    bindings={}
    for name,data in parts.items():
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        bindings[name]={'sha256':digest(data),'bytes':len(data),'original_path':str(sources[name].resolve())}
    bindings['task-score.json']['original_path']=result['original_score_path']
    manifest={'schema':SCHEMA,'files':bindings,'original_score_path':result['original_score_path'],
              'source_mount':{'container_root':str(container_root),'source_root':str(source_root)},
              'packet_sha256':digest(packet),'actor_identity':identity}
    with (out/'evidence.json').open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return out/'evidence.json'

def validate(reference, *, identity, packet_path):
    """Recheck transported bytes against the independently bound manifest."""
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    path=_verify_historical_file_ref(reference);manifest=json.loads(checked_read(path))
    if manifest.get('schema')!=SCHEMA or set(manifest.get('files',{}))!=set(FILES):raise ValueError('invalid portable task evidence')
    parts={}
    for name in FILES:
        ref=manifest['files'][name];data=checked_read(path.parent/name)
        if digest(data)!=ref.get('sha256') or len(data)!=ref.get('bytes') or not isinstance(ref.get('original_path'),str):
            raise ValueError('portable task evidence file binding differs')
        parts[name]=data
    packet=checked_read(packet_path)
    result=_validate(parts,identity=identity,packet=packet)
    mount=manifest.get('source_mount') or {}
    source=PurePosixPath(mount.get('source_root',''));container=PurePosixPath(mount.get('container_root',''))
    if (not source.is_absolute() or not container.is_absolute()
            or '..' in source.parts or '..' in container.parts):
        raise ValueError('invalid portable source mount')
    try:mapped=source/PurePosixPath(result['original_score_path']).relative_to(container)
    except ValueError:raise ValueError('portable score is outside source mount') from None
    evaluation=mapped.parent.parent.parent
    for name,original in (('completed.json',evaluation/'completed.json'),('intent.json',evaluation/'intent.json'),
                          ('spec.json',source/'spec.json'),('launch.json',source/'launch.json'),
                          ('actor-info.json',source/'actor-info.json')):
        if manifest['files'][name]['original_path']!=str(original):
            raise ValueError('portable source path does not match its mount')
    if (manifest.get('actor_identity')!=identity or manifest.get('packet_sha256')!=digest(packet)
            or manifest.get('original_score_path')!=result['original_score_path']
            or manifest['files']['task-score.json']['original_path']!=result['original_score_path']):
        raise ValueError('portable task provenance differs')
    _verify_historical_file_ref(reference)
    return result
