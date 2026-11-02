"""Authenticated source correction without an optimizer step or completion claim.

This operation uses the trainer's correction and CE admission functions.  Its
verifier independently replays those functions against the retained full parent
and compares every value in the resulting full checkpoint to the saved child.
"""
from __future__ import annotations
import json
import os
import shutil
from pathlib import Path

SCHEMA = 'natlang.native-zero-sgd-source-adoption/1'


def _roles(recipe_path, stage_id):
    from .recipe import load_recipe, resolve_stage_inputs
    recipe=load_recipe(recipe_path)
    stages=[stage for stage in recipe['stages'] if stage['id']==stage_id]
    if len(stages)!=1:
        raise ValueError('source adoption requires the exact single SFT stage')
    return resolve_stage_inputs(recipe,stages[0],{})


def _stage_parameters(recipe_path, stage_id):
    from .recipe import load_recipe, effective_stage_parameters
    recipe=load_recipe(recipe_path)
    stages=[stage for stage in recipe['stages'] if stage['id']==stage_id]
    if len(stages)!=1 or stages[0].get('kind')!='native_state_sft':
        raise ValueError('source adoption requires the exact single native SFT stage')
    parameters=effective_stage_parameters(stages[0],[],recipe)
    return {'epochs':parameters.get('epochs',1.),
            'communication_min_visits':parameters.get('communication_min_visits')}


def admitted_checkpoint(parent, *, prior_recipe, successor_recipe, stage_id):
    """Derive the sole allowed state using the same admission as the trainer."""
    from ..data.native_sources import NativeGraphSource
    from .native_continuation import (collect_sft_progress, restore_sft_progress, stage_key,
        apply_communication_quota_options, _retain_pending_sources, active_exposure_quotas)
    from .native_operation_report import source_receipt
    from .native_source_correction import apply_source_correction
    from .native_supervision import load_policy
    from .own_output_supervision import prepare_policy, apply_ce_admission_options
    from .recipe import _source_correction_continuation
    from .native_quota_aware_ce import validate_admission as validate_quota_aware_ce
    from ..model.native_model import CHECKPOINT_SCHEMA
    if parent.get('schema')!=CHECKPOINT_SCHEMA or not isinstance(parent.get('training'),dict):
        raise ValueError('adoption requires an actual full native checkpoint')
    training=parent['training']; config=parent['model_config']
    if not training.get('optimizer') or not training.get('rng'):
        raise ValueError('adoption requires retained optimizer and RNG state')
    old,new=_roles(prior_recipe,stage_id),_roles(successor_recipe,stage_id)
    parameters=_stage_parameters(successor_recipe,stage_id)
    if not _source_correction_continuation(old,new):
        raise ValueError('adoption recipes do not prove this exact source and CE correction')
    if old.get('held_native_graphs') != new.get('held_native_graphs'):
        raise ValueError('zero-SGD adoption cannot change held data')
    correction=new['source_correction']; proof=json.loads(Path(correction['path']).read_text())
    if proof.get('protected_held_correction') is not None:
        raise ValueError('zero-SGD adoption cannot change held data')
    current=NativeGraphSource(new['native_graphs']['path'],config['tokenizer_fingerprint'],
                              required_split='train',role_ids=config['marker_ids'])
    source=source_receipt(current)
    is_flat=((training.get('identity') or {}).get('objective') is None and stage_key(training)==stage_id)
    progress=collect_sft_progress(training)
    previous=training if is_flat else restore_sft_progress(training,stage_id,collected_progress=progress)
    if previous is None:
        raise ValueError('parent checkpoint does not carry the requested earned SFT frame')
    prior_receipts=previous.get('source_receipts') or []
    if (len(prior_receipts)<2 or prior_receipts[1].get('sha256') != old['held_native_graphs']['sha256']
            or (previous.get('options', {}).get('ce_supervision_policy_binding') or {}).get('sha256')
                != old['ce_supervision_policy']['sha256']):
        raise ValueError('earned SFT frame differs from the held source or prior CE recipe')
    policy_ref={k:v for k,v in new['ce_supervision_policy'].items() if k in ('path','sha256')}
    policy=prepare_policy(load_policy(policy_ref),current)
    prior_options=previous.get('options') or {}
    quota_record=validate_quota_aware_ce(policy,previous,stage_id=stage_id,
        candidate_sha256=current.file_sha256,epochs=parameters['epochs'],
        communication_min_visits=parameters['communication_min_visits'],
        previous_options=prior_options,policy_binding=policy_ref)
    updated, summary=apply_source_correction(previous,correction['path'],
        model_id=config['architecture']+':'+config['base_inventory_sha256'],stage_id=stage_id,
        current_source_receipt=source,current_bindings=current.bindings,current_order=current.ordered_ids(),
        current_communication_ids=[entry['id'] for entry in current.entries if entry.get('communication')])
    if summary.get('changed') is not True or not any(summary.get(key) for key in ('corrected_rows','deferred_rows','added_rows')):
        raise ValueError('zero-SGD adoption requires an actual source change')
    step=int(training.get('global_step',training.get('step',0)))
    old_frames=collect_sft_progress(training)
    new_frames=updated['sft_progress']
    for name,frame in new_frames.items():
        prior=old_frames.get(name)
        if not prior:
            raise ValueError('adoption unexpectedly created a supervised frame')
        if (frame.get('source_receipts') or [{}])[0].get('sha256') == new['native_graphs']['sha256']:
            frame['options']=apply_ce_admission_options(frame.get('options') or {},prior.get('options') or {},
                                                        policy_ref,policy,step)
        old_counts=prior.get('exposure_counts') or {}
        next_counts=frame.get('exposure_counts') or {}
        corrected={row['id'] for row in ((frame.get('options') or {}).get('source_correction_history') or [])[-1].get('corrected_rows',[])} if (frame.get('options') or {}).get('source_correction_history') else set()
        if any(next_counts.get(key,0)!=value for key,value in old_counts.items() if key not in corrected):
            raise ValueError('zero-SGD adoption changed earned source exposure')
        if any(value!=0 for key,value in next_counts.items() if key not in old_counts):
            raise ValueError('zero-SGD adoption credited a new row')
    def finish_active(frame):
        options=apply_ce_admission_options(dict(frame.get('options') or {}),prior_options,policy_ref,policy,step)
        if quota_record is not None:
            history=list(prior_options.get('quota_aware_ce_history') or [])
            if quota_record not in history:history.append(quota_record)
            options['quota_aware_ce_history']=history
        options['epochs']=parameters['epochs']
        corrected_minima=dict(options.get('_objective_required_visits') or {})
        apply_communication_quota_options(options,prior_options,train=current,ce_policy=policy,
            stage_id=stage_id,requested_visits=parameters['communication_min_visits'])
        if corrected_minima:
            minima=options.setdefault('_objective_required_visits',{})
            for ident,visits in corrected_minima.items():minima[ident]=max(minima.get(ident,0),visits)
        result=_retain_pending_sources({**frame,'options':options,
            'exposure_counts':dict(frame.get('exposure_counts') or {})},frame)
        result['required_visits'].update(active_exposure_quotas(current.entries,epochs=parameters['epochs'],
                                                                 minimum_visits=result['required_visits']))
        return result
    if stage_id in new_frames:
        new_frames[stage_id]=finish_active(new_frames[stage_id])
    if is_flat:
        updated=finish_active(updated)
        updated['sft_progress']=new_frames
        result={**parent,'training':updated}
    else:
        # The policy frame and all its optimizer/RNG/state fields stay as earned.
        replacement=dict(training); replacement['sft_progress']=new_frames
        result={**parent,'training':replacement}
    return result, summary


def _same(a,b, path='checkpoint'):
    """Compare all tensor bytes and every metadata field without float tolerance."""
    import torch
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape or a.stride()!=b.stride() or a.requires_grad!=b.requires_grad:
            raise ValueError('adoption changed tensor shape or dtype at '+path)
        if a.layout!=torch.strided or b.layout!=torch.strided:
            raise ValueError('unsupported tensor layout in full checkpoint comparison')
        aa=a.contiguous().reshape(-1).view(torch.uint8);bb=b.contiguous().reshape(-1).view(torch.uint8)
        for offset in range(0,aa.numel(),4*1024*1024):
            if not torch.equal(aa[offset:offset+4*1024*1024],bb[offset:offset+4*1024*1024]):
                raise ValueError('adoption changed tensor bytes at '+path)
        return
    if type(a)!=type(b):raise ValueError('adoption changed value type at '+path)
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('adoption changed keys at '+path)
        for k in a:_same(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('adoption changed sequence length at '+path)
        for i,(x,y) in enumerate(zip(a,b)):_same(x,y,path+'.'+str(i))
    elif type(a).__module__.startswith('numpy'):
        import numpy as np
        equal_nan=np.issubdtype(np.asarray(a).dtype,np.inexact)
        if not np.array_equal(a,b,equal_nan=equal_nan):raise ValueError('adoption changed numpy state at '+path)
    elif isinstance(a,float):
        import struct
        if struct.pack('!d',a)!=struct.pack('!d',b):raise ValueError('adoption changed float at '+path)
    elif a!=b:raise ValueError('adoption changed value at '+path)


def verify(parent_ref, child_ref, *, prior_recipe, successor_recipe, stage_id):
    import torch
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref, _strong_binding
    def signature(path):
        st=path.stat();return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    parent_path=Path(parent_ref['path']).resolve(strict=True)
    child_path=Path(child_ref['path']).resolve(strict=True)
    recipe_paths=(Path(prior_recipe).resolve(strict=True),Path(successor_recipe).resolve(strict=True))
    parent_stamp,child_stamp=signature(parent_path),signature(child_path)
    recipe_stamps=tuple(signature(path) for path in recipe_paths)
    recipe_bindings=tuple(_strong_binding(path) for path in recipe_paths)
    if (_verify_historical_file_ref(parent_ref)!=parent_path or _verify_historical_file_ref(child_ref)!=child_path
            or parent_path==child_path or parent_ref['sha256']==child_ref['sha256']):
        raise ValueError('adoption needs distinct retained parent and child')
    before=torch.load(parent_path,map_location='cpu',weights_only=False,mmap=True)
    child=torch.load(child_path,map_location='cpu',weights_only=False,mmap=True)
    expected,summary=admitted_checkpoint(before,prior_recipe=prior_recipe,successor_recipe=successor_recipe,stage_id=stage_id)
    _same(expected,child)
    # Reload separately: correction metadata must never mutate the actual
    # retained parent or change any tensor/optimizer/RNG through aliasing.
    original=torch.load(parent_path,map_location='cpu',weights_only=False,mmap=True)
    _same({k:v for k,v in original.items() if k!='training'},
          {k:v for k,v in child.items() if k!='training'})
    for key in ('optimizer','rng','fp8','global_step','step'):
        if key in original['training'] or key in child['training']:
            _same(original['training'].get(key),child['training'].get(key),'training.'+key)
    if (Path(parent_ref['path']).resolve(strict=True)!=parent_path
            or Path(child_ref['path']).resolve(strict=True)!=child_path
            or signature(parent_path)!=parent_stamp or signature(child_path)!=child_stamp
            or tuple(signature(path) for path in recipe_paths)!=recipe_stamps
            or tuple(_strong_binding(path) for path in recipe_paths)!=recipe_bindings):
        raise ValueError('checkpoint or recipe changed while verifying zero-SGD adoption')
    step=int(original['training'].get('global_step',original['training'].get('step',0)))
    return {'schema':SCHEMA,'parent':parent_ref,'checkpoint':child_ref,'stage_id':stage_id,
            'global_step':step,'optimizer_updates':0,'graph_visits':0,'stage_completed':False,
            'source_correction':summary,'scope':'exact full-state rederived metadata adoption; no behavior qualification'}


def prepare(parent_ref, *, prior_recipe, successor_recipe, stage_id, output):
    import torch
    from ..rl.native_direct_sft_recovery import _strong_binding, _verify_historical_file_ref
    from ..common.jsonio import write_json_durable
    parent_path=_verify_historical_file_ref(parent_ref)
    prior_ref=_strong_binding(prior_recipe); successor_ref=_strong_binding(successor_recipe)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(output).free < parent_path.stat().st_size + 40*1024**3:
        raise ValueError('full checkpoint adoption would breach the 40 GiB floor')
    parent=torch.load(parent_path,map_location='cpu',weights_only=False,mmap=True)
    child,_=admitted_checkpoint(parent,prior_recipe=prior_recipe,successor_recipe=successor_recipe,stage_id=stage_id)
    pending=output/'checkpoint.pt.pending'; final=output/'checkpoint.pt'
    with pending.open('xb') as stream:
        torch.save(child,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(pending,final)
    os.chmod(final,0o444)
    directory=os.open(output,os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    child_ref=_strong_binding(final)
    checked=verify(parent_ref,child_ref,prior_recipe=prior_recipe,successor_recipe=successor_recipe,stage_id=stage_id)
    if _strong_binding(prior_recipe)!=prior_ref or _strong_binding(successor_recipe)!=successor_ref:
        raise ValueError('adoption recipe changed while deriving the full checkpoint')
    proof={**checked,'prior_recipe':prior_ref,'successor_recipe':successor_ref}
    write_json_durable(output/'proof.json',proof)
    return proof
