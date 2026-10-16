"""Exact origin and immutable numeric-update checks for optimized native state.

This is a protocol verifier, not a learned policy or a recapture claim. The
producer identity describes the original basis; derivation names changed state.
"""
from __future__ import annotations
import torch
from .native_derivative_contract import DERIVATION,optimizer_binding
from .native_artifact import NativeTensorArtifact


def state_values(block):
    return {name:value for name,value in block.tensors().items() if name.startswith('layers.')}


def replace_state_values(block,values,*,derivation=None):
    tensors={**block.tensors(),**values}
    metadata=block.metadata()
    if derivation is not None:metadata={**metadata,'derivation':derivation}
    from .state_block import NativeStateBlock
    return NativeStateBlock.from_tensors(metadata,tensors)


def origins(store,ident):
    record=store.get_native_state(ident)
    if record is None:raise ValueError('native derivation origin is unavailable')
    derivation=record.block.derivation
    from .native_derivative_contract import validate_derivation
    validate_derivation(derivation)
    return {ident} if derivation is None else set(derivation['origin_state_ids'])


def verify_state_derivation(store,block,seen=None):
    """Verify full immutable ancestry iteratively, with no Python recursion cap."""
    visited=set() if seen is None else set(seen)
    current=block
    while current.derivation is not None:
        ident=current.derivation.get('parent_state_id')
        if ident in visited:raise ValueError('native derivation ancestry is cyclic')
        visited.add(ident)
        current=(_verify_optimizer_derivation(store,current) if current.derivation['kind']=='optimizer'
                 else _verify_semantic_derivation(store,current))


def _verify_optimizer_derivation(store,block):
    """One exact byte/provenance edge; the caller verifies every parent."""
    value=block.derivation
    if value is None:return
    from .native_derivative_contract import validate_derivation
    validate_derivation(value)
    parent_id=value['parent_state_id']
    parent=store.get_native_state(parent_id)
    if parent is None:raise ValueError('optimized native state needs its exact archived parent')
    if sorted(origins(store,parent_id))!=value['origin_state_ids']:
        raise ValueError('optimized native state changed its original producer provenance')
    old_metadata={key:item for key,item in parent.block.metadata().items() if key!='derivation'}
    if {key:item for key,item in block.metadata().items() if key!='derivation'}!=old_metadata:
        raise ValueError('optimized native state changed token text, native basis or adapter provenance')
    records={key:store.get_native_tensor_artifact(value[key]) for key in ('gradient_id','update_artifact_id')}
    if any(record is None for record in records.values()):raise ValueError('optimized native state needs its actual gradient and numeric update')
    gradient,update=(records[key].artifact for key in ('gradient_id','update_artifact_id'))
    from .native_derivative_contract import verify_derivative_gradient_origin
    verify_derivative_gradient_origin(store,gradient)
    names=value['updated_tensor_names']
    provenance=update.provenance or {}
    expected_base=parent.block.adapter_binding['base_native_identity'] if parent.block.adapter_binding is not None else {key:getattr(parent.block,key) for key in ('checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256','architecture')}
    if (gradient.native_identity!=expected_base or gradient.kind!='state-gradient' or gradient.source_artifact_id!=parent_id
            or update.kind!='parameter-update' or update.source_artifact_id!=value['gradient_id']
            or sorted(gradient.tensors)!=names or sorted(update.tensors)!=names
            or update.native_identity!=gradient.native_identity
            or provenance!={'schema':DERIVATION,'parent_state_id':parent_id,'optimizer':value['optimizer'],
                'step':value['step'],'optimizer_state_parent_id':value['optimizer_state_parent_id'],
                'derivative_program_id':value['program_id']}):
        raise ValueError('optimized native state update, gradient, optimizer or program binding differs')
    if (gradient.provenance or {}).get('derivative_program_id')!=value['program_id']:
        raise ValueError('optimized native state gradient belongs to another derivative program')
    before,after=parent.block.tensors(),block.tensors()
    if set(before)!=set(after) or any(before[name].shape!=after[name].shape or before[name].dtype!=after[name].dtype
            or not torch.equal(after[name].detach().cpu(),(update.tensors[name] if name in names else before[name]).detach().cpu())
            for name in before):
        raise ValueError('optimized native state changed an unselected tensor or differs from its numeric update')
    prior_id=value['optimizer_state_parent_id']
    if prior_id is None:
        if value['step']!=1:raise ValueError('new optimized native state history must begin at step one')
    else:
        prior=store.get_native_tensor_artifact(prior_id)
        if (prior is None or prior.artifact.kind!='optimizer-state' or prior.artifact.source_artifact_id!=parent_id
                or prior.artifact.native_identity!=gradient.native_identity
                or prior.artifact.provenance.get('scope')!=DERIVATION
                or prior.artifact.provenance.get('optimizer')!=value['optimizer']['name']
                or prior.artifact.provenance.get('step')!=value['step']-1):
            raise ValueError('optimized native state lacks its exact previous optimizer history')
    if value['program_id'] is not None:
        program=store.get_native_tensor_artifact(value['program_id'])
        if program is None or program.artifact.kind!='derivative-program':
            raise ValueError('optimized native state lacks its executable derivative program')
    return parent.block


def semantic_derivation(block,parent_id,adapter_sha256,source_identity,target_identity,kind):
    from .native_derivative_contract import validate_derivation
    validate_derivation(block.derivation)
    result={'schema':DERIVATION,'kind':kind,'parent_state_id':parent_id,'adapter_sha256':adapter_sha256,
        'origin_state_ids':block.derivation['origin_state_ids'] if block.derivation else [parent_id],
        'source_native_identity':dict(source_identity),'target_native_identity':dict(target_identity)}
    validate_derivation(result)
    return result


def _verify_semantic_derivation(store,block):
    from .native_derivative_contract import validate_derivation
    value=block.derivation;validate_derivation(value)
    parent=store.get_native_state(value['parent_state_id'])
    if parent is None:raise ValueError('derived semantic state needs its exact archived source')
    identity=lambda item:{key:getattr(item,key) for key in ('checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256','architecture')}
    if (value['origin_state_ids']!=sorted(origins(store,parent.id)) or value['source_native_identity']!=identity(parent.block)
            or value['target_native_identity']!=identity(block) or block.text!=parent.block.text):
        raise ValueError('derived semantic state changed its source, target or original producer provenance')
    if value['kind']=='semantic-encode':
        if (block.metadata()['schema']!='neuralese.semantic-state/1' or block.adapter_sha256!=value['adapter_sha256']
                or identity(block)!=identity(parent.block) or any(not torch.equal(block.tensors()[name].detach().cpu(),
                    parent.block.tensors()[name].detach().cpu()) for name in ('token_ids','positions','embeddings'))):
            raise ValueError('derived semantic encoding changed source text, tokens, embeddings or adapter')
    elif (block.metadata()['schema']!='neuralese.native-state/1' or parent.block.metadata()['schema']=='neuralese.semantic-state/1'
            and parent.block.adapter_sha256!=value['adapter_sha256']):
        raise ValueError('derived semantic decode changed its trained adapter or native target format')
    return parent.block


def optimize_state(engine,gradient_record,options,state_id,owner=None,*,values=None,gradients=None,moments=None,program_id=None):
    """One immutable shared-optimizer update, retaining optional live derivative DAG."""
    from ..train.optim import functional_named_service_step
    from .state_block import SCHEMA
    gradient=gradient_record.artifact
    if gradient.kind!='state-gradient' or gradient.native_identity!=engine.native_identity:
        raise ValueError('native state update requires an exact live-base state gradient')
    parent_id=gradient.source_artifact_id
    record=engine.store.get_native_state(parent_id)
    if record is None or record.block.metadata()['schema']!=SCHEMA:
        raise ValueError('native state optimizer requires an archived raw KV/hybrid state')
    parent=record.block
    verify_state_derivation(engine.store,parent)
    available=state_values(parent)
    if not gradient.tensors or not set(gradient.tensors)<=available.keys():raise ValueError('native state gradient selects unknown state tensors')
    for name,value in gradient.tensors.items():
        if value.shape!=available[name].shape:raise ValueError('native state gradient geometry differs')
    binding=optimizer_binding(options)
    prior=engine.store.get_native_tensor_artifact(state_id) if state_id else None
    if state_id and (prior is None or prior.artifact.kind!='optimizer-state' or prior.artifact.source_artifact_id!=parent_id
            or prior.artifact.native_identity!=engine.native_identity or (prior.artifact.provenance or {}).get('scope')!=DERIVATION
            or prior.artifact.provenance.get('optimizer')!=options['name']):
        raise ValueError('native state optimizer needs its exact prior state and unchanged algorithm')
    stored_moments=prior.artifact.tensors if prior else {}
    old_step=prior.artifact.provenance['step'] if prior else 0
    prefixes=('m.','v.','step.') if options['name']=='adam' else ('momentum.',)
    if any(key!='optimizer.step' and not any(key.startswith(prefix) and key[len(prefix):] in available for prefix in prefixes)
           for key in stored_moments):
        raise ValueError('native state optimizer history has unknown named tensors')
    if prior and ('optimizer.step' not in stored_moments or stored_moments['optimizer.step'].item()!=old_step):
        raise ValueError('native state optimizer counter differs from its exact history')
    source_values=available if values is None else values
    live_values={name:value.to(device=engine.device,dtype=torch.float32 if value.dtype in (torch.float16,torch.bfloat16) else value.dtype).clone() for name,value in source_values.items()}
    source_gradients=gradient.tensors if gradients is None else gradients
    live_gradients={name:value.to(device=engine.device,dtype=live_values[name].dtype) for name,value in source_gradients.items()}
    if set(live_values)!=set(available) or set(live_gradients)!=set(gradient.tensors):raise ValueError('native state live update tensor inventory differs')
    selected_values={name:live_values[name] for name in live_gradients}
    updated,next_moments=functional_named_service_step(selected_values,live_gradients,
        stored_moments if moments is None else moments,**options)
    updated={**live_values,**updated}
    selected={name:updated[name].to(available[name].dtype) for name in gradient.tensors}
    step=old_step+1
    update=engine.store.put_native_tensor_artifact(NativeTensorArtifact('parameter-update',engine.native_identity,
        selected,gradient_record.id,{'schema':DERIVATION,'parent_state_id':parent_id,'optimizer':binding,'step':step,
            'optimizer_state_parent_id':state_id,'derivative_program_id':program_id}),owner)
    derivation={'schema':DERIVATION,'kind':'optimizer','origin_state_ids':sorted(origins(engine.store,parent_id)),
        'parent_state_id':parent_id,'gradient_id':gradient_record.id,'update_artifact_id':update.id,'program_id':program_id,
        'optimizer_state_parent_id':state_id,'step':step,'optimizer':binding,'updated_tensor_names':sorted(selected)}
    block=replace_state_values(parent,selected,derivation=derivation)
    verify_state_derivation(engine.store,block)
    result=engine.store.put_native_state(block,owner)
    history=engine.store.put_native_tensor_artifact(NativeTensorArtifact('optimizer-state',engine.native_identity,
        next_moments,result.id,{'scope':DERIVATION,'optimizer':options['name'],'step':step,
            'gradient_id':gradient_record.id,'parent_state_id':parent_id,'derivative_program_id':program_id}),owner)
    live_block=replace_state_values(block,{name:updated[name].to(available[name].dtype) for name in available})
    return {'native_state_id':result.id,'optimizer_state_id':history.id,'update_artifact_id':update.id,'step':step},live_block,next_moments
