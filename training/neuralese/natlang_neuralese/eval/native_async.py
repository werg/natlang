"""Verified native LFM exports in the existing asynchronous task-eval protocol."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ..common.hashing import canonical_json_sha256_hex,sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from ..common.runtime_inventory import task_runtime_inventory


def file_binding(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256_file_hex_cached(path),'bytes':path.stat().st_size}


def verified_export(state):
    receipt=state['receipt']; identity=receipt['identity']
    proof=receipt.get('independent_verification') or {}
    if (receipt.get('role')!='lfm' or receipt.get('private') is not True
            or proof.get('sha256')!=identity.get('sha256') or proof.get('bytes')!=identity.get('bytes')
            or not isinstance(state.get('last_verified_step'),int)):
        raise ValueError('native eval requires the existing independently verified private LFM export')
    return receipt


def verify_environment(provenance):
    runtime=task_runtime_inventory(provenance['runtime']['path'])
    if canonical_json_sha256_hex(runtime)!=provenance['runtime']['sha256']:
        raise ValueError('native async eval runtime/protection closure changed')
    packet=provenance['packet']
    if sha256_file_hex_cached(packet['path'])!=packet['sha256']:
        raise ValueError('native async eval held packet changed')
    return provenance


def verify_request(request):
    provenance=request['evaluation_provenance']
    if (canonical_json_sha256_hex(request['spec'])!=provenance['evaluation_spec_sha256']
            or request['prompt_rendering']!=request['spec']['prompt_rendering']):
        raise ValueError('native async eval sampling/task specification changed after sealing')
    execution=provenance['execution']
    if any(request.get(key,'default' if key=='sdpa_backend' else None)!=execution[key]
           for key in ('device','sdpa_backend')):
        raise ValueError('native async eval execution specification changed after sealing')
    return verify_environment(provenance)


def seal_request(checkpoint,*,identity,step,spec,out,base,queue,source_export=None,device='cuda'):
    """Seal exact original bytes; no pruning, projection heads or guessed source dtype."""
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..model.native_model import CHECKPOINT_SCHEMA,base_inventory
    from .task_cadence import ASYNC_SCHEMA,TaskEval,parse_task_eval
    checkpoint=Path(checkpoint).resolve(strict=True);spec=parse_task_eval(spec)
    if file_binding(checkpoint)['sha256']!=identity['sha256'] or checkpoint.stat().st_size!=identity['bytes']:
        raise ValueError('native eval checkpoint differs from independently verified export')
    state=CheckpointPolicy.load(checkpoint)
    config=state.get('model_config') or {}
    if state.get('schema')!=CHECKPOINT_SCHEMA or config.get('architecture') not in {'lfm2','lfm2_moe'}:
        raise ValueError('native async export path accepts full native LFM only; no legacy or Mellum fallback')
    if state.get('step')!=step:
        raise ValueError('native eval export step differs from actual saved checkpoint')
    actual_base=base_inventory(base)
    if actual_base!=config['base_inventory']:
        raise ValueError('native eval relocated base differs from actual pinned inventory')
    runtime=Path(spec.get('runtime') or Path(__file__).resolve().parents[4]/'ts-host').resolve(strict=True)
    spec['runtime']=str(runtime)
    stage=Path(out).resolve()/'checkpoints'/identity['sha256']
    evaluator=TaskEval(spec,stage,None,prompt_rendering=spec['prompt_rendering'])
    request_dir=evaluator.root/'async'/f'step-{step:07d}'
    request_dir.mkdir(parents=True,exist_ok=True)
    weights=request_dir/'weights.pt'
    if not weights.exists():
        try:os.link(checkpoint,weights)
        except OSError:
            from ..train.checkpoint_offload import verified_copy
            verified_copy(checkpoint,weights)
    if file_binding(weights)['sha256']!=identity['sha256']:
        raise ValueError('native eval immutable request snapshot changed')
    runtime_files=task_runtime_inventory(runtime)
    packet=Path(spec['packet']['path']);packet=packet if packet.is_absolute() else evaluator.repo/packet
    provenance={'schema':'natlang.native-async-task-eval-binding/1',
        'historical':True,'qualifies_current_weights':False,'checkpoint':file_binding(weights),
        'source_checkpoint_sha256':identity['sha256'],'source_export':source_export,
        'model_config':config,'execution':{'device':device,'sdpa_backend':'default'},'runtime':{'path':str(runtime),'sha256':canonical_json_sha256_hex(runtime_files),
                                      'inventory':runtime_files},
        'packet':file_binding(packet),'case_indices':spec['cases'],
        'evaluation_spec_sha256':canonical_json_sha256_hex(spec),
        'sampler':{'temperature':spec['temperature'],'seed_base':400000,'repeat_seed_stride':100000,
                   'noise_repeats':spec['noise_repeats'],'neuralese_server':spec.get('neuralese_server',False)}}
    verify_environment(provenance)
    request={'schema':ASYNC_SCHEMA,'step':step,'stage_out':str(stage),'spec':spec,
        'prompt_rendering':spec['prompt_rendering'],'device':device,'native_state':True,
        'base':str(Path(base).resolve()),'weights':str(weights),'checkpoint':str(checkpoint),
        'requested_at':time.time(),'evaluation_provenance':provenance,
        'trainer_code_commit':(source_export or {}).get('source_code_commit')}
    write_json_durable(request_dir/'request.json',request)
    queue=Path(queue);queue.mkdir(parents=True,exist_ok=True)
    write_json_durable(queue/('native-'+identity['sha256']+'.json'),{'request':str(request_dir)})
    return {'step':step,'checkpoint_sha256':identity['sha256'],'request':str(request_dir),
            'stage_out':str(stage),'spec':spec,'pending':True,'eligible':None}


def poll_latest(export_state,*,out,spec,base,queue,fetch_command,local_checkpoint=None,device='cuda'):
    """One immutable admitted request at a time; exports coalesce while it drains."""
    from .task_cadence import TaskEval
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'export-ingest.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=out/'native-async-state.json'
        prior=json.loads(state_path.read_text()) if state_path.exists() else {}
        active=prior.get('active')
        if active:
            evaluator=TaskEval(active['spec'],Path(active['stage_out']),None,
                               prompt_rendering=active['spec']['prompt_rendering'])
            result=evaluator.landed(active['step'])
            # Keep references until run_request/dispatcher has actually drained.
            if result is None or Path(active['request']).exists():
                return {'event':'native_eval_pending','active_step':active['step'],'eligible':None}
            write_json_durable(out/'latest-result.json',{'checkpoint_sha256':active['checkpoint_sha256'],
                'step':active['step'],'result':result,'historical':True,'qualifies_current_weights':False})
            prior['last_completed_sha256']=active['checkpoint_sha256'];prior.pop('active')
            write_json_durable(state_path,prior)
        current=json.loads(Path(export_state).read_text());receipt=verified_export(current)
        identity=receipt['identity']
        if identity['sha256']==prior.get('last_completed_sha256'):
            return {'event':'native_eval_up_to_date','step':current['last_verified_step']}
        destination=out/'incoming'/('checkpoint-'+identity['sha256']+'.pt')
        destination.parent.mkdir(parents=True,exist_ok=True)
        receipt_path=destination.with_suffix('.receipt.json');write_json_durable(receipt_path,receipt)
        if local_checkpoint:
            checkpoint=Path(local_checkpoint)
        else:
            argv=[part.replace('{receipt}',str(receipt_path)).replace('{out}',str(destination)) for part in fetch_command]
            subprocess.run(argv,check=True,stdout=subprocess.DEVNULL)
            checkpoint=destination
        active=seal_request(checkpoint,identity=identity,step=current['last_verified_step'],
            spec=spec,out=out,base=base,queue=queue,source_export=receipt,device=device)
        write_json_durable(state_path,{**prior,'active':active,'last_enqueued_at':time.time()})
        if not local_checkpoint:destination.unlink(missing_ok=True)  # sealed request now owns immutable bytes
        receipt_path.unlink(missing_ok=True)
        return {'event':'native_eval_enqueued',**active}
