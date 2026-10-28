"""Verified native student exports in the existing asynchronous task-eval protocol."""
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
from ..common.line_backbones import all_architectures
from ..common.runtime_inventory import task_runtime_inventory, snapshot_task_runtime


def file_binding(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256_file_hex_cached(path),'bytes':path.stat().st_size}


def verified_export(state):
    receipt=state['receipt']; identity=receipt['identity']
    proof=receipt.get('independent_verification') or {}
    if (receipt.get('role') not in {'lfm','mellum'} or receipt.get('private') is not True
            or proof.get('sha256')!=identity.get('sha256') or proof.get('bytes')!=identity.get('bytes')
            or not isinstance(state.get('last_verified_step'),int)):
        raise ValueError('native eval requires an independently verified private student export')
    return receipt


def selected_ingest_export(export_state, prior):
    """An interrupted transfer retains its verified receipt before newer exports."""
    pending=prior.get('fetching_export')
    if 'fetching_export'not in prior:
        selected=json.loads(Path(export_state).read_text())
    else:
        if (not isinstance(pending,dict)
                or pending.get('schema')!='natlang.native-eval-transfer/1'
                or not isinstance(pending.get('export'),dict)
                or canonical_json_sha256_hex(pending['export'])!=pending.get('sha256')):
            raise ValueError('native eval interrupted transfer binding changed')
        selected=pending['export']
    if 'fetching_export' not in prior and selected.get('schema') == 'natlang.owned-publication-owner-index/1':
        selected = selected_owner_generation_export(export_state)
    verified_export(selected)
    return selected


def selected_owner_generation_export(index_path):
    """Resolve only the index's exact current owner generation for task evaluation."""
    index_path=Path(index_path)
    if index_path.is_symlink():raise ValueError('native eval owner index may not be a symlink')
    index_path=index_path.resolve(strict=True)
    raw_index=index_path.read_bytes();index=json.loads(raw_index)
    if index.get('schema')!='natlang.owned-publication-owner-index/1' or index.get('role') not in {'lfm','mellum'}:
        raise ValueError('native eval owner index has the wrong schema or role')
    current=index.get('current')
    generation=current.get('generation_sha256') if isinstance(current,dict) else None
    if not isinstance(generation,str) or len(generation)!=64 or any(c not in '0123456789abcdef' for c in generation):
        raise ValueError('native eval owner index has no exact current generation')
    owner_root=index_path.parent/'owner-generations'
    if owner_root.is_symlink():raise ValueError('native eval owner journal may not be a symlink')
    generation_dir=owner_root/generation
    if generation_dir.is_symlink() or not generation_dir.resolve(strict=True).is_relative_to(owner_root.resolve(strict=True)):
        raise ValueError('native eval owner generation escaped its journal')
    records=index.get('generations')
    if not isinstance(records,list):raise ValueError('native eval owner index has no generation records')
    record=next((row for row in records if isinstance(row,dict)
                 and row.get('generation_sha256')==generation),None)
    selection_path=generation_dir/'owner-selection.json'
    state_path=generation_dir/f"coreweave-native-{index['role']}-periodic-export-state-20261015.json"
    if selection_path.is_symlink() or state_path.is_symlink():
        raise ValueError('native eval owner generation journal may not use symlinks')
    raw_selection=selection_path.read_bytes();selection=json.loads(raw_selection)
    raw_state=state_path.read_bytes();state=json.loads(raw_state)
    expected={'schema':'natlang.owned-publication-owner-selection/1','role':index['role'],
        'generation_sha256':generation,'manifest':current.get('manifest')}
    if (current!=expected or record!=expected or selection!=expected
            or state.get('last_verified_step') is None):
        raise ValueError('native eval current owner selection differs from its generation journal')
    verified_export(state)
    if state['receipt'].get('role')!=index['role']:
        raise ValueError('native eval owner receipt role differs')
    # Reject a journal swap during selection; the current generation may advance
    # immediately afterward, but this selected full-read receipt remains exact.
    if (index_path.read_bytes()!=raw_index or selection_path.read_bytes()!=raw_selection
            or state_path.read_bytes()!=raw_state):
        raise ValueError('native eval owner generation changed during selection')
    return {**state,'owner_generation_sha256':generation}


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


def _active_request(directory,request):
    return {'step':request['step'],'checkpoint_sha256':request['evaluation_provenance']['checkpoint']['sha256'],
            'request':str(directory),'stage_out':request['stage_out'],'spec':request['spec'],
            'pending':True,'eligible':None}


def _publish_request(active,queue):
    queue=Path(queue);queue.mkdir(parents=True,exist_ok=True)
    write_json_durable(queue/('native-'+active['checkpoint_sha256']+'.json'),{'request':active['request']})


def _recover_request(*,out,identity,step,spec,base,source_export,device,prepared=None):
    """Authenticate a sealed request without rewriting it or downloading again."""
    from .task_cadence import ASYNC_SCHEMA,parse_task_eval,require_native_transport
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..model.native_model import CHECKPOINT_SCHEMA,base_inventory
    stage=Path(out).resolve()/'checkpoints'/identity['sha256']
    directory=stage/'task-eval/async'/f'step-{step:07d}';path=directory/'request.json'
    if not path.exists()and prepared is None:return None
    if (not isinstance(prepared,dict)or prepared.get('schema')!='natlang.native-eval-seal/1'
            or prepared.get('directory')!=str(directory)or not isinstance(prepared.get('request'),dict)
            or canonical_json_sha256_hex(prepared['request'])!=prepared.get('sha256')):
        raise ValueError('native eval sealed request differs from durable preparation')
    request=json.loads(path.read_text())if path.exists()else prepared['request']
    if canonical_json_sha256_hex(request)!=prepared['sha256']:
        raise ValueError('native eval sealed request differs from durable preparation')
    provenance=request['evaluation_provenance']
    expected=require_native_transport(parse_task_eval(spec))
    # The frozen runtime is authoritative after sealing, even after a live build.
    expected['runtime']=request['spec']['runtime']
    weights=directory/'weights.pt'
    packet=Path(expected['packet']['path'])
    if not packet.is_absolute():
        from .task_cadence import TaskEval
        packet=TaskEval(expected,stage,None,prompt_rendering=expected['prompt_rendering']).repo/packet
    if (request.get('schema')!=ASYNC_SCHEMA or request.get('step')!=step
            or request.get('stage_out')!=str(stage) or request.get('spec')!=expected
            or request.get('base')!=str(Path(base).resolve()) or request.get('device')!=device
            or request.get('native_state')is not True or request.get('weights')!=str(weights)
            or provenance.get('checkpoint')!=file_binding(weights)
            or provenance['checkpoint']['sha256']!=identity['sha256']
            or provenance['checkpoint']['bytes']!=identity['bytes']
            or provenance.get('source_checkpoint_sha256')!=identity['sha256']
            or provenance.get('source_export')!=source_export
            or provenance.get('case_indices')!=expected['cases']
            or provenance.get('packet')!=file_binding(packet)
            or provenance.get('sampler')!={'temperature':expected['temperature'],'seed_base':400000,
                'repeat_seed_stride':100000,'noise_repeats':expected['noise_repeats'],
                'neuralese_server':expected.get('neuralese_server',False)}):
        raise ValueError('native eval sealed request differs from admitted export')
    state=CheckpointPolicy.load(weights)
    if (state.get('schema')!=CHECKPOINT_SCHEMA or state.get('step')!=step
            or state.get('model_config')!=provenance.get('model_config')
            or state.get('inference_source')!=provenance.get('inference_source')
            or (state.get('inference_source')is not None and
                state['inference_source']['training_checkpoint_sha256']!=provenance.get('source_training_checkpoint_sha256'))
            or state['model_config']['base_inventory']!=base_inventory(base)):
        raise ValueError('native eval sealed request differs from saved checkpoint')
    del state
    verify_request(request)
    if not path.exists():write_json_durable(path,request)
    return _active_request(directory,request)


def seal_request(checkpoint,*,identity,step,spec,out,base,queue,source_export=None,device='cuda',consume_export=False,on_sealed=None,on_prepared=None):
    """Seal verified export bytes; keep a derived view's original parent separate."""
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..model.native_model import CHECKPOINT_SCHEMA,base_inventory
    from .task_cadence import ASYNC_SCHEMA,TaskEval,parse_task_eval,require_native_transport
    checkpoint_argument=Path(checkpoint)
    checkpoint=checkpoint_argument.resolve(strict=True);spec=require_native_transport(parse_task_eval(spec))
    if file_binding(checkpoint)['sha256']!=identity['sha256'] or checkpoint.stat().st_size!=identity['bytes']:
        raise ValueError('native eval checkpoint differs from independently verified export')
    state=CheckpointPolicy.load(checkpoint)
    config=state.get('model_config') or {}
    if state.get('schema')!=CHECKPOINT_SCHEMA or config.get('architecture') not in set(all_architectures()):
        raise ValueError('native async export path requires a supported native student checkpoint')
    inference_source=state.get('inference_source')
    if state.get('optimizer_pruned'):
        import re
        if (not isinstance(inference_source,dict) or inference_source.get('schema')!='natlang.native-inference-source/1'
                or not isinstance(inference_source.get('training_checkpoint_sha256'),str)
                or not re.fullmatch('[0-9a-f]{64}',inference_source['training_checkpoint_sha256'])
                or type(inference_source.get('training_checkpoint_bytes')) is not int
                or inference_source['training_checkpoint_bytes']<=identity['bytes']
                or inference_source.get('step')!=step):
            raise ValueError('native inference export lacks its exact original full-parent provenance')
    if state.get('step')!=step:
        raise ValueError('native eval export step differs from actual saved checkpoint')
    # Large training schedule dictionaries are not needed while copying runtime
    # assets or scheduling the actor. Tensor data stays memory mapped.
    del state
    actual_base=base_inventory(base)
    if actual_base!=config['base_inventory']:
        raise ValueError('native eval relocated base differs from actual pinned inventory')
    runtime=Path(spec.get('runtime') or Path(__file__).resolve().parents[4]/'ts-host').resolve(strict=True)
    runtime=snapshot_task_runtime(runtime,Path(out).resolve()/'task-runtimes')
    spec['runtime']=str(runtime)
    stage=Path(out).resolve()/'checkpoints'/identity['sha256']
    evaluator=TaskEval(spec,stage,None,prompt_rendering=spec['prompt_rendering'])
    request_dir=evaluator.root/'async'/f'step-{step:07d}'
    request_dir.mkdir(parents=True,exist_ok=True)
    weights=request_dir/'weights.pt'
    if not weights.exists():
        if consume_export:
            import errno
            try:
                # A verified external payload is presented through a logical
                # symlink in the incoming directory. Move that symlink within
                # the request filesystem; moving its resolved target would
                # cross filesystems and copy the whole checkpoint back to RAM.
                source=checkpoint_argument if checkpoint_argument.is_symlink() else checkpoint
                if checkpoint_argument.is_symlink() and source.parent.stat().st_dev!=weights.parent.stat().st_dev:
                    raise ValueError('logical export symlink is not colocated with the sealed request')
                os.replace(source,weights)
            except OSError as error:
                if error.errno!=errno.EXDEV:raise
                from ..train.checkpoint_offload import verified_copy
                verified_copy(checkpoint,weights)
        else:
            from ..train.checkpoint_offload import verified_copy
            verified_copy(checkpoint,weights)
    if file_binding(weights)['sha256']!=identity['sha256']:
        raise ValueError('native eval immutable request snapshot changed')
    runtime_files=task_runtime_inventory(runtime)
    packet=Path(spec['packet']['path']);packet=packet if packet.is_absolute() else evaluator.repo/packet
    provenance={'schema':'natlang.native-async-task-eval-binding/1',
        'historical':True,'qualifies_current_weights':False,'checkpoint':file_binding(weights),
        'source_checkpoint_sha256':identity['sha256'],'source_export':source_export,
        **({'inference_source':inference_source,
            'source_training_checkpoint_sha256':inference_source['training_checkpoint_sha256']}
           if inference_source is not None else {}),
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
    if on_prepared is not None:on_prepared(request_dir,request)
    write_json_durable(request_dir/'request.json',request)
    active=_active_request(request_dir,request)
    if on_sealed is not None:on_sealed(active)
    _publish_request(active,queue)
    return active


def poll_latest(export_state,*,out,spec,base,queue,fetch_command,local_checkpoint=None,device='cuda',export_admission=None):
    """One immutable admitted request at a time; exports coalesce while it drains."""
    from .task_cadence import TaskEval
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'export-ingest.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=out/'native-async-state.json'
        prior=json.loads(state_path.read_text()) if state_path.exists() else {}
        active=prior.get('active')
        if active:
            if active.get('publication_pending')and Path(active['request']).exists():
                current=selected_ingest_export(export_state,prior);receipt=verified_export(current)
                recovered=_recover_request(out=out,identity=receipt['identity'],step=current['last_verified_step'],
                    spec=spec,base=base,source_export=receipt,device=device,prepared=prior.get('sealing_request'))
                if (recovered is None or {k:active[k]for k in recovered}!=recovered
                        or file_binding(Path(active['request'])/'request.json')['sha256']!=active['request_sha256']):
                    raise ValueError('native eval pending publication changed')
                _publish_request(recovered,queue)
                active=recovered;prior['active']=active;prior.pop('fetching_export',None);prior.pop('sealing_request',None)
                write_json_durable(state_path,prior)
            evaluator=TaskEval(active['spec'],Path(active['stage_out']),None,
                               prompt_rendering=active['spec']['prompt_rendering'])
            result=evaluator.landed(active['step'])
            # Keep references until run_request/dispatcher has actually drained.
            if result is None or Path(active['request']).exists():
                return {'event':'native_eval_pending','active_step':active['step'],'eligible':None}
            write_json_durable(out/'latest-result.json',{'checkpoint_sha256':active['checkpoint_sha256'],
                'step':active['step'],'result':result,'historical':True,'qualifies_current_weights':False})
            prior['last_completed_sha256']=active['checkpoint_sha256'];prior.pop('active');prior.pop('fetching_export',None);prior.pop('sealing_request',None)
            write_json_durable(state_path,prior)
        current=selected_ingest_export(export_state,prior);receipt=verified_export(current)
        identity=receipt['identity']
        if identity['sha256']==prior.get('last_completed_sha256'):
            return {'event':'native_eval_up_to_date','step':current['last_verified_step']}
        def journal_sealed(sealed):
            prior['active']={**sealed,'publication_pending':True,
                'request_sha256':file_binding(Path(sealed['request'])/'request.json')['sha256']}
            write_json_durable(state_path,prior)
        def journal_prepared(directory,request):
            prior['sealing_request']={'schema':'natlang.native-eval-seal/1','directory':str(directory),
                'request':request,'sha256':canonical_json_sha256_hex(request)}
            write_json_durable(state_path,prior)
        recovered=_recover_request(out=out,identity=identity,step=current['last_verified_step'],
            spec=spec,base=base,source_export=receipt,device=device,prepared=prior.get('sealing_request'))
        if recovered is not None:
            journal_sealed(recovered);_publish_request(recovered,queue)
            prior.pop('fetching_export',None);prior.pop('sealing_request',None);prior['active']=recovered
            write_json_durable(state_path,prior)
            return {'event':'native_eval_enqueued',**recovered}
        destination=out/'incoming'/('checkpoint-'+identity['sha256']+'.pt')
        if export_admission is not None:
            admission=export_admission(current,destination)
            if admission is not None:return admission
        destination.parent.mkdir(parents=True,exist_ok=True)
        receipt_path=destination.with_suffix('.receipt.json');write_json_durable(receipt_path,receipt)
        sealed_weights=out/'checkpoints'/identity['sha256']/'task-eval/async'/f"step-{current['last_verified_step']:07d}"/'weights.pt'
        if 'fetching_export'not in prior:
            selected={'last_verified_step':current['last_verified_step'],'receipt':receipt}
            prior['fetching_export']={'schema':'natlang.native-eval-transfer/1',
                'export':selected,'sha256':canonical_json_sha256_hex(selected)}
            write_json_durable(state_path,prior)
        if sealed_weights.exists():
            checkpoint=sealed_weights
        elif local_checkpoint:
            checkpoint=Path(local_checkpoint)
        else:
            argv=[part.replace('{receipt}',str(receipt_path)).replace('{out}',str(destination)) for part in fetch_command]
            subprocess.run(argv,check=True,stdout=subprocess.DEVNULL)
            checkpoint=destination
        active=seal_request(checkpoint,identity=identity,step=current['last_verified_step'],
            spec=spec,out=out,base=base,queue=queue,source_export=receipt,device=device,
            consume_export=not local_checkpoint,on_sealed=journal_sealed,on_prepared=journal_prepared)
        prior.pop('fetching_export',None);prior.pop('sealing_request',None)
        write_json_durable(state_path,{**prior,'active':active,'last_enqueued_at':time.time()})
        if not local_checkpoint:destination.unlink(missing_ok=True)  # sealed request now owns immutable bytes
        receipt_path.unlink(missing_ok=True)
        return {'event':'native_eval_enqueued',**active}
