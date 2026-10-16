"""Exact portable coefficient-policy values; actor choice belongs to natlang.

A policy changes coefficient artifacts, never its native base or semantic VAE.
Its existing RL bundle retains named moments and frozen bases on every reload.
"""
from __future__ import annotations
import copy
import json
import re
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex,sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from .native_composite import resolve_composite_artifact
from .native_adapters import resolve_adapters

SCHEMA='natlang.native-coefficient-policy/1'
_ID=re.compile(r'nnt1_[a-z2-7]{52}')
_CORE={'schema','base_checkpoint','base_native_identity','handles','defaults','state_transport'}


def policy_core(policy):
    if not isinstance(policy,dict) or set(policy)-(_CORE|{'bundle','training_state'}) or not _CORE<=set(policy) or policy.get('schema')!=SCHEMA:
        raise ValueError('native coefficient policy requires its exact typed base, handles and transport')
    identity=policy['base_native_identity']
    if (not isinstance(identity,dict) or set(identity)!={'checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256','architecture'}
            or any(not isinstance(identity[key],str) or not re.fullmatch('[0-9a-f]{64}',identity[key])
                   for key in ('checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256'))
            or not isinstance(identity['architecture'],str) or not identity['architecture']
            or not isinstance(policy['base_checkpoint'],dict) or set(policy['base_checkpoint'])!={'path','sha256'}
            or not isinstance(policy['base_checkpoint']['path'],str)
            or policy['base_checkpoint'].get('sha256')!=identity['checkpoint_sha256']
            or policy['state_transport'] not in {'strict','same-base-adapter-state'}
            or not isinstance(policy['handles'],dict) or not policy['handles']
            or not isinstance(policy['defaults'],list)):
        raise ValueError('native coefficient policy changed its exact base or declared transport')
    for handle,entry in policy['handles'].items():
        if (not isinstance(handle,str) or not _ID.fullmatch(handle) or not isinstance(entry,dict)
                or set(entry)!={'original_id','current_id','optimizer_state_id','parameter_names'}
                or not isinstance(entry['original_id'],str) or not _ID.fullmatch(entry['original_id'])
                or not isinstance(entry['current_id'],str) or not _ID.fullmatch(entry['current_id'])
                or entry['optimizer_state_id'] is not None and not _ID.fullmatch(entry['optimizer_state_id'])
                or not isinstance(entry['parameter_names'],list) or not entry['parameter_names']
                or any(not isinstance(name,str) or not name for name in entry['parameter_names'])
                or len(set(entry['parameter_names']))!=len(entry['parameter_names'])):
            raise ValueError('native coefficient policy requires exact unique named coefficient handles and moments')
    for entry in policy['defaults']:
        if (not isinstance(entry,dict) or set(entry)!={'id','scale_hex'} or entry['id'] not in policy['handles']
                or not isinstance(entry['scale_hex'],str)):
            raise ValueError('native coefficient defaults must bind declared handles with exact scales')
        scale=float.fromhex(entry['scale_hex'])
        import math
        if not math.isfinite(scale) or scale.hex()!=entry['scale_hex']:
            raise ValueError('native coefficient policy scale must retain exact finite hex encoding')
    result={key:copy.deepcopy(policy[key]) for key in _CORE}
    if policy.get('training_state') is not None:
        binding=policy['training_state']
        if (not isinstance(binding,dict) or set(binding)!={'path','sha256'} or not isinstance(binding['path'],str)
                or not isinstance(binding['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',binding['sha256'])):
            raise ValueError('coefficient policy training state requires its exact immutable sidecar binding')
        result['training_state']=copy.deepcopy(binding)
    return result


def actor_policy_sha256(policy):
    core=policy_core(policy)
    return canonical_json_sha256_hex({key:value for key,value in core.items() if key!='training_state'})


def runtime_binding(policy,path):
    core=policy_core(policy)
    return {'schema':'natlang.native-coefficient-runtime/1','base_native_identity':core['base_native_identity'],
        'policy_sha256':actor_policy_sha256(policy),'replacements':{handle:entry['current_id'] for handle,entry in core['handles'].items()},
        'defaults':[{'id':entry['id'],'scale':float.fromhex(entry['scale_hex'])} for entry in core['defaults']],
        'state_transport':core['state_transport']}


def validate_loaded_policy(engine,policy):
    core=policy_core(policy)
    if core['base_native_identity']!=engine.native_identity:
        raise ValueError('native coefficient policy belongs to another exact native base')
    for handle,entry in core['handles'].items():
        original=resolve_adapters(engine,[entry['original_id']])[0]
        current=resolve_adapters(engine,[entry['current_id']])[0]
        before=engine.store.get_native_tensor_artifact(entry['original_id']).artifact
        after=engine.store.get_native_tensor_artifact(entry['current_id']).artifact
        if before.source_artifact_id!=after.source_artifact_id or original[2]!=current[2] or not set(entry['parameter_names'])<=current[3].keys():
            raise ValueError('native coefficient continuation changed its original frozen basis or named targets')
        if entry['optimizer_state_id'] is not None:
            record=engine.store.get_native_tensor_artifact(entry['optimizer_state_id'])
            if (record is None or record.artifact.kind!='optimizer-state' or record.artifact.native_identity!=engine.native_identity
                    or record.artifact.source_artifact_id!=entry['current_id']
                    or record.artifact.provenance.get('scope')!='natlang.native-adapter/1'):
                raise ValueError('native coefficient policy lost its exact current optimizer state')
    resolve_adapters(engine,[{'id':core['handles'][item['id']]['current_id'],'scale':float.fromhex(item['scale_hex'])}
                            for item in core['defaults']])
    return core


def restore_policy(engine,path):
    from ..rl.graph_replay import restore_bundle
    path=Path(path).resolve(strict=True);policy=json.loads(path.read_text());core=policy_core(policy)
    resolve_composite_artifact(core['base_checkpoint'],path)
    bundle=resolve_composite_artifact(policy['bundle'],path)
    restore_bundle(engine,bundle,policy_id=canonical_json_sha256_hex(core))
    validate_loaded_policy(engine,policy)
    if core.get('training_state'):
        import torch
        state=torch.load(resolve_composite_artifact(core['training_state'],path),map_location='cpu',weights_only=False)
        if (state.get('schema')!='natlang.native-coefficient-training/1' or 'policy' in state
                or state.get('base_native_identity')!=core['base_native_identity']
                or state.get('actor_policy_sha256')!=actor_policy_sha256(core)):
            raise ValueError('coefficient policy progress changed its exact actor, base or acyclic schema')
    return policy


def publish_policy(engine,policy,out):
    from ..rl.graph_replay import save_bundle
    from .native_artifact import encode_native_tensor_artifact
    core=validate_loaded_policy(engine,policy)
    owner=canonical_json_sha256_hex(core);out=Path(out).resolve()/owner
    out.mkdir(parents=True,exist_ok=True)
    path=out/'policy.json'
    base=Path(core['base_checkpoint']['path'])
    if not base.is_absolute() or sha256_file_hex_cached(base)!=core['base_checkpoint']['sha256']:
        raise ValueError('coefficient policy base must retain its exact absolute pinned checkpoint')
    # Stable program handles are names, not old-base executable artifacts.
    # Archive only the actual current-base originals, current values/moments.
    roots=[{key:entry[key] for key in ('original_id','current_id','optimizer_state_id')} for entry in core['handles'].values()]
    save_bundle(roots,out/'blocks',policy_id=owner,
                fetch=lambda ident:encode_native_tensor_artifact(engine.store.get_native_tensor_artifact(ident).artifact))
    bundle=out/'blocks/manifest.json'
    complete={**core,'bundle':{'path':'blocks/manifest.json','sha256':sha256_file_hex_cached(bundle)}}
    if path.exists() and json.loads(path.read_text())!=complete:
        raise ValueError('immutable native coefficient policy destination has different bytes')
    write_json_durable(path,complete)
    return path


def initialize_policy(engine,handles,*,defaults,out,state_transport='strict'):
    """Caller declares actual coefficient IDs and named targets; no actor chooser."""
    checkpoint=Path(engine.native_checkpoint).resolve(strict=True)
    policy={'schema':SCHEMA,'base_native_identity':dict(engine.native_identity),
        'base_checkpoint':{'path':str(checkpoint),'sha256':sha256_file_hex_cached(checkpoint)},
        'handles':{handle:{'original_id':handle,'current_id':handle,'optimizer_state_id':None,'parameter_names':names}
                   for handle,names in handles.items()},
        'defaults':[{'id':item if isinstance(item,str) else item['id'],
                     'scale_hex':float(1 if isinstance(item,str) else item.get('scale',1)).hex()} for item in defaults],
        'state_transport':state_transport}
    return publish_policy(engine,policy,out)


def _migrate_training_state(training,*,source_sha256,source_identity,target_identity,actor_sha256,cancel_stale_reason=None):
    """Preserve credited work; explicitly cancel untrained stale behavior only."""
    result=copy.deepcopy(training)
    pending=[entry for entry in result['obligations']
             if not entry.get('cancelled') and entry['visits']<entry['required_visits']]
    if pending:
        if not isinstance(cancel_stale_reason,str) or not cancel_stale_reason.strip():
            raise ValueError('owed old-base policy rows require explicit supervised cancellation and recollection before migration')
        receipt={'schema':'natlang.native-coefficient-stale-cancellation/1',
            'source_policy_sha256':source_sha256,'source_native_identity':source_identity,
            'target_native_identity':target_identity,'reason':cancel_stale_reason,
            'rows':[{'key':entry['key'],'visits':entry['visits'],'required_visits':entry['required_visits'],
                     'uncredited_visits':entry['required_visits']-entry['visits'],'objective':entry['objective']}
                    for entry in pending]}
        receipt_sha=canonical_json_sha256_hex(receipt)
        result.setdefault('cancellations',[]).append(receipt)
        for entry in pending:
            entry['cancelled']=receipt_sha
            entry.pop('graph',None)
    result['base_native_identity']=dict(target_identity)
    result['actor_policy_sha256']=actor_sha256
    return result


def rebase_policy(engine,source,out,*,cancel_stale_reason=None):
    """Explicit exact-value migration after a real backbone continuation.

    Frozen basis values, coefficient values and every named moment are retained.
    New immutable artifacts bind the new base; old KV and behavior are not
    relabeled or admitted. Callers must qualify the new actual actor policy
    before deployment. This function provides no task-quality certificate.
    """
    from .native_artifact import NativeTensorArtifact
    from .native_adapters import target_parameters
    from ..rl.graph_replay import restore_bundle
    source=Path(source).resolve(strict=True);old=json.loads(source.read_text());core=policy_core(old)
    if core['base_native_identity']==engine.native_identity:
        restore_policy(engine,source)
        return source
    if any(core['base_native_identity'][key]!=engine.native_identity[key] for key in ('architecture','tokenizer_fingerprint')):
        raise ValueError('coefficient base migration requires the same exact architecture and tokenizer')
    resolve_composite_artifact(core['base_checkpoint'],source)
    bundle=resolve_composite_artifact(old['bundle'],source)
    restore_bundle(engine,bundle,policy_id=canonical_json_sha256_hex(core))
    available=target_parameters(engine.shared_model)
    transition={'schema':'natlang.native-coefficient-base-transition/1','source_policy_sha256':sha256_file_hex_cached(source),
        'source_native_identity':core['base_native_identity'],'target_native_identity':dict(engine.native_identity),
        'numeric_values':'unchanged','qualified':False}
    cache={}
    def migrate_coefficient(ident):
        if ident in cache:return cache[ident]
        record=engine.store.get_native_tensor_artifact(ident)
        if record is None:raise ValueError('coefficient base migration lost an exact source coefficient artifact')
        original=record.artifact
        if original.kind!='adapter-coefficients' or original.native_identity!=core['base_native_identity']:
            raise ValueError('coefficient base migration source is bound to a different native actor')
        basis_record=engine.store.get_native_tensor_artifact(original.source_artifact_id)
        if basis_record is None:raise ValueError('coefficient base migration lost its frozen basis')
        basis=basis_record.artifact;spec=original.provenance['adapter_spec'];rank=spec['rank']
        if basis.kind!='adapter-basis' or basis.native_identity!=core['base_native_identity'] or basis.provenance.get('adapter_spec')!=spec:
            raise ValueError('coefficient base migration changed its exact source basis specification')
        for name in spec['parameter_names']:
            weight=available.get(name)
            if weight is None:
                raise ValueError('coefficient base migration requires the exact named matrix layouts')
        migrated_basis=engine.store.put_native_tensor_artifact(NativeTensorArtifact('adapter-basis',dict(engine.native_identity),
            basis.tensors,provenance={'adapter_spec':spec,'base_transition':{**transition,'source_artifact_id':original.source_artifact_id}}))
        migrated=engine.store.put_native_tensor_artifact(NativeTensorArtifact('adapter-coefficients',dict(engine.native_identity),
            original.tensors,migrated_basis.id,{'adapter_spec':spec,'coefficient_lineage':original.provenance.get('coefficient_lineage',ident),
                'base_transition':{**transition,'source_artifact_id':ident}}))
        # The existing one geometry/specification verifier checks exact left,
        # right, projection and expert-batch layouts against the NEW weights.
        resolve_adapters(engine,[migrated.id])
        cache[ident]=migrated.id
        return migrated.id
    next_core=copy.deepcopy(core)
    for handle,entry in core['handles'].items():
        changed=next_core['handles'][handle]
        changed['original_id']=migrate_coefficient(entry['original_id'])
        changed['current_id']=migrate_coefficient(entry['current_id'])
        if entry['optimizer_state_id'] is not None:
            state_record=engine.store.get_native_tensor_artifact(entry['optimizer_state_id'])
            if state_record is None:raise ValueError('coefficient base migration lost named optimizer moments')
            state=state_record.artifact
            if (state.kind!='optimizer-state' or state.native_identity!=core['base_native_identity']
                    or state.source_artifact_id!=entry['current_id'] or state.provenance.get('scope')!='natlang.native-adapter/1'):
                raise ValueError('coefficient base migration optimizer does not bind its actual source coefficients')
            provenance={key:copy.deepcopy(value) for key,value in state.provenance.items()
                        if key in {'scope','optimizer','step','optimizer_options'}}
            provenance['base_transition']={**transition,'source_artifact_id':entry['optimizer_state_id']}
            migrated_state=engine.store.put_native_tensor_artifact(NativeTensorArtifact('optimizer-state',dict(engine.native_identity),
                state.tensors,changed['current_id'],provenance))
            changed['optimizer_state_id']=migrated_state.id
    checkpoint=Path(engine.native_checkpoint).resolve(strict=True)
    next_core['base_checkpoint']={'path':str(checkpoint),'sha256':sha256_file_hex_cached(checkpoint)}
    next_core['base_native_identity']=dict(engine.native_identity)
    if core.get('training_state'):
        import torch
        training_path=resolve_composite_artifact(core['training_state'],source)
        training=torch.load(training_path,map_location='cpu',weights_only=False)
        training=_migrate_training_state(training,source_sha256=sha256_file_hex_cached(source),
            source_identity=core['base_native_identity'],target_identity=engine.native_identity,
            actor_sha256=actor_policy_sha256(next_core),cancel_stale_reason=cancel_stale_reason)
        next_core['training_state']=write_training_state(training,Path(out)/'training-state')
    return publish_policy(engine,next_core,out)


def _same_training_state(left,right):
    """Exact recursive verifier for the RNG/exposure sidecar (including dtype)."""
    import torch
    if type(left) is not type(right):return False
    if isinstance(left,torch.Tensor):
        return left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(_same_training_state(left[key],right[key]) for key in left)
    if isinstance(left,(tuple,list)):
        return len(left)==len(right) and all(_same_training_state(a,b) for a,b in zip(left,right))
    return left==right


def verify_base_transition(source,target,*,identity,cancel_stale_reason=None):
    """CPU verifier of an actor's exact-value migration, independent of its claim."""
    from ..rl.graph_replay import verified_bundle_bytes
    from .native_artifact import decode_native_tensor_artifact
    source,target=Path(source).resolve(strict=True),Path(target).resolve(strict=True)
    old,new=json.loads(source.read_text()),json.loads(target.read_text())
    before,after=policy_core(old),policy_core(new)
    if after['base_native_identity']!=identity:
        raise ValueError('coefficient actor migration did not bind the requested actual new base')
    if source==target:
        if before['base_native_identity']!=identity:raise ValueError('coefficient actor still serves the old base')
        return after
    if (before['base_native_identity']==identity or before['defaults']!=after['defaults']
            or before['state_transport']!=after['state_transport'] or set(before['handles'])!=set(after['handles'])):
        raise ValueError('coefficient actor base transition changed its declared policy program')
    for core,path in ((before,source),(after,target)):
        resolve_composite_artifact(core['base_checkpoint'],path)
    bundles=[resolve_composite_artifact(policy['bundle'],path) for policy,path in ((old,source),(new,target))]
    owners=[canonical_json_sha256_hex(core) for core in (before,after)]
    def artifact(index,ident):
        raw=verified_bundle_bytes(bundles[index],ident,policy_id=owners[index])
        if raw is None:raise ValueError('coefficient actor base transition omitted an exact required artifact')
        value,actual=decode_native_tensor_artifact(raw)
        if actual!=ident:raise ValueError('coefficient actor base transition changed artifact bytes')
        return value
    def pair(old_id,new_id):
        left,right=artifact(0,old_id),artifact(1,new_id)
        transition=right.provenance.get('base_transition') or {}
        if (left.kind!=right.kind or left.native_identity!=before['base_native_identity'] or right.native_identity!=identity
                or set(left.tensors)!=set(right.tensors) or any(left.tensors[name].dtype!=right.tensors[name].dtype
                    or left.tensors[name].shape!=right.tensors[name].shape
                    or not __import__('torch').equal(left.tensors[name],right.tensors[name]) for name in left.tensors)
                or transition.get('schema')!='natlang.native-coefficient-base-transition/1'
                or transition.get('source_policy_sha256')!=sha256_file_hex_cached(source)
                or transition.get('source_artifact_id')!=old_id or transition.get('source_native_identity')!=before['base_native_identity']
                or transition.get('target_native_identity')!=identity or transition.get('numeric_values')!='unchanged'
                or transition.get('qualified') is not False):
            raise ValueError('coefficient actor base migration changed exact values, moments or provenance')
        return left,right
    for handle,entry in before['handles'].items():
        changed=after['handles'][handle]
        if entry['parameter_names']!=changed['parameter_names']:
            raise ValueError('coefficient actor base migration changed named gradient targets')
        for role in ('original_id','current_id'):
            left,right=pair(entry[role],changed[role])
            if (left.provenance.get('adapter_spec')!=right.provenance.get('adapter_spec')
                    or right.provenance.get('coefficient_lineage')!=left.provenance.get('coefficient_lineage',entry[role])):
                raise ValueError('coefficient actor base migration changed exact adapter specification or lineage')
            old_basis,new_basis=pair(left.source_artifact_id,right.source_artifact_id)
            if old_basis.provenance.get('adapter_spec')!=new_basis.provenance.get('adapter_spec'):
                raise ValueError('coefficient actor base migration changed its frozen basis specification')
        if (entry['optimizer_state_id'] is None)!=(changed['optimizer_state_id'] is None):
            raise ValueError('coefficient actor base migration reset optimizer history')
        if entry['optimizer_state_id'] is not None:
            left,right=pair(entry['optimizer_state_id'],changed['optimizer_state_id'])
            if (left.source_artifact_id!=entry['current_id'] or right.source_artifact_id!=changed['current_id']
                    or any(left.provenance.get(key)!=right.provenance.get(key) for key in ('scope','optimizer','step','optimizer_options'))):
                raise ValueError('coefficient actor base migration changed actual optimizer continuation')
    if bool(before.get('training_state'))!=bool(after.get('training_state')):
        raise ValueError('coefficient actor base migration dropped training progress or RNG')
    if before.get('training_state'):
        import torch
        old_state=torch.load(resolve_composite_artifact(before['training_state'],source),map_location='cpu',weights_only=False)
        new_state=torch.load(resolve_composite_artifact(after['training_state'],target),map_location='cpu',weights_only=False)
        expected=_migrate_training_state(old_state,source_sha256=sha256_file_hex_cached(source),
            source_identity=before['base_native_identity'],target_identity=identity,
            actor_sha256=actor_policy_sha256(after),cancel_stale_reason=cancel_stale_reason)
        if not _same_training_state(expected,new_state):
            raise ValueError('coefficient actor base migration changed exact exposure, RNG or training progress')
    return after


def write_training_state(state,out):
    """Immutable acyclic state sidecar using the existing exact artifact writer."""
    import io
    import torch
    from .native_composite import _durable_immutable
    body=io.BytesIO();torch.save(state,body)
    payload=body.getvalue()
    import hashlib
    digest=hashlib.sha256(payload).hexdigest()
    path=Path(out).resolve()/(digest+'.pt')
    _durable_immutable(path,payload)
    return {'path':str(path),'sha256':digest}


def policy_dependencies(path):
    """Exact executable closure, excluding completed historical objective inputs."""
    import torch
    from .native_bundle import composite_dependencies
    from ..rl.graph_replay import verified_bundle_bytes
    path=Path(path).resolve(strict=True);policy=json.loads(path.read_text());core=policy_core(policy)
    result={}
    def add(source,digest=None):
        source=Path(source).resolve(strict=True)
        actual=sha256_file_hex_cached(source)
        if digest is not None and actual!=digest:
            raise ValueError('coefficient dependency bytes changed')
        result[str(source)]={'path':str(source),'sha256':actual,'bytes':source.stat().st_size}
    add(path)
    checkpoint=resolve_composite_artifact(core['base_checkpoint'],path)
    for digest,source in composite_dependencies(checkpoint).items():add(source,digest)
    visited=set()
    def archive(manifest,owner):
        manifest=Path(manifest).resolve(strict=True)
        if manifest in visited:raise ValueError('coefficient archive provenance contains a cycle')
        visited.add(manifest)
        value=json.loads(manifest.read_text());add(manifest)
        if value.get('schema')!='natlang.rl-block-bundle/1' or value.get('policy_id')!=owner:
            raise ValueError('coefficient dependency archive differs from its exact policy owner')
        original=value.get('source_collection_manifest')
        if original is not None:
            source=resolve_composite_artifact(original,manifest)
            source_value=json.loads(source.read_text())
            if source_value.get('blocks')!=value['blocks']:
                raise ValueError('coefficient admission archive changed original collector artifact inventory')
            archive(source,core['base_native_identity']['checkpoint_sha256'])
        for ident,entry in value['blocks'].items():
            if verified_bundle_bytes(manifest,ident,policy_id=owner) is None:
                raise ValueError('coefficient dependency archive lost an actual executable artifact')
            artifact=(manifest.parent/entry['path']).resolve(strict=True)
            add(artifact,entry['sha256'])
        visited.remove(manifest)
    archive(resolve_composite_artifact(policy['bundle'],path),canonical_json_sha256_hex(core))
    if core.get('training_state'):
        frame=resolve_composite_artifact(core['training_state'],path)
        add(frame)
        state=torch.load(frame,map_location='cpu',weights_only=False)
        if state.get('schema')!='natlang.native-coefficient-training/1' or state['base_native_identity']!=core['base_native_identity']:
            raise ValueError('coefficient dependency sidecar changed its actual base or schema')
        for entry in state['obligations']:
            if entry.get('cancelled') or entry['visits']>=entry['required_visits']:continue
            objective=resolve_composite_artifact(entry['objective'],frame)
            add(objective,entry['objective']['sha256'])
            from ..rl.native_replay import NativePolicyObjective
            admitted=NativePolicyObjective(objective)
            if admitted.identity!=core['base_native_identity']:
                raise ValueError('owed coefficient objective belongs to a different base')
            archive(admitted.bundle[0],admitted.archive_owner)
    return [result[path] for path in sorted(result)]
