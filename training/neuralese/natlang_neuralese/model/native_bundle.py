"""Portable immutable native checkpoint dependencies, with no identity rewriting.

HF model source files remain separately pinned by the native checkpoint. This
bundle transports the updated actor state, parameter/optimizer artifacts and
training RNG; the standard loader accepts an explicit identical HF base path.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import stat
import tempfile

from ..common.hashing import sha256_file_hex, sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from .native_composite import SCHEMA, resolve_composite_artifact

BUNDLE_SCHEMA = 'natlang.native-checkpoint-bundle/1'
ACTOR_VIEW_SCHEMA = 'natlang.native-actor-view/1'
TRANSPORT_SCHEMA = 'natlang.native-actor-transport/1'


def _file_binding(path):
    path = Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256_file_hex_cached(path),'bytes':path.stat().st_size}


def _fresh_file_binding(path):
    path=Path(path).resolve(strict=True);before=path.stat()
    digest=sha256_file_hex(path);after=path.stat()
    signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if signature(before)!=signature(after):raise ValueError('actor view input changed during hash verification')
    return {'path':str(path),'sha256':digest,'bytes':after.st_size}


def _same_values_bounded(a,b,where='state',*,max_bytes=4*1024*1024):
    """Bitwise compare tensors with bounded contiguous temporaries, including strided views."""
    import torch
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape:
            raise RuntimeError(f'actor view differs at {where}')
        def compare(x,y):
            if x.numel()*x.element_size()>max_bytes and x.ndim and x.numel():
                axis=max(range(x.ndim),key=lambda i:x.shape[i])
                if x.shape[axis]>1:
                    middle=x.shape[axis]//2
                    compare(x.narrow(axis,0,middle),y.narrow(axis,0,middle))
                    compare(x.narrow(axis,middle,x.shape[axis]-middle),y.narrow(axis,middle,x.shape[axis]-middle))
                    return
            left=x.detach().to('cpu').contiguous().reshape(-1);right=y.detach().to('cpu').contiguous().reshape(-1)
            try:left,right=left.view(torch.uint8),right.view(torch.uint8)
            except RuntimeError:pass
            if not torch.equal(left,right):raise RuntimeError(f'actor view differs at {where}')
        compare(a,b)
    elif isinstance(a,dict):
        if not isinstance(b,dict) or list(a)!=list(b):raise RuntimeError(f'actor view differs at {where}')
        for key in a:_same_values_bounded(a[key],b[key],f'{where}.{key}',max_bytes=max_bytes)
    elif isinstance(a,(list,tuple)):
        if not isinstance(b,(list,tuple)) or len(a)!=len(b):raise RuntimeError(f'actor view differs at {where}')
        for i,(left,right) in enumerate(zip(a,b,strict=True)):_same_values_bounded(left,right,f'{where}[{i}]',max_bytes=max_bytes)
    elif a!=b and not (a!=a and b!=b):raise RuntimeError(f'actor view differs at {where}')


def _cached_actor_view(source, directory):
    """Find only a content-addressed receipt, rechecking its source and view bytes."""
    for proof in sorted(directory.iterdir()):
        if not re.fullmatch('[0-9a-f]{64}\\.json',proof.name):continue
        if proof.is_symlink() or not proof.is_file():raise ValueError('actor view receipt must be a regular file')
        reference=_fresh_file_binding(proof)
        if proof.stem!=reference['sha256']:raise ValueError('cached actor view receipt digest differs')
        metadata=json.loads(proof.read_text());name=(metadata.get('view') or {}).get('path')
        if not isinstance(name,str) or Path(name).name!=name or (directory/name).is_symlink():
            raise ValueError('cached actor view must be a regular sibling')
        _,_,view=_actor_view_files(source,reference)
        if view.is_symlink() or view.name!=_fresh_file_binding(view)['sha256']+'.pt':
            raise ValueError('cached actor view is not content addressed')
        return view
    return None


def write_snapshot_actor_view(checkpoint,out,*,snapshot,decoded_source_stat):
    """Serialize the committed owner's exact selected state without reloading the full source.

    This is the bounded serialization primitive; the caller must hold an exact
    full-source inode pin until its consumer or verified full recovery releases it.
    """
    import math,torch
    from ..train.checkpoint_safety import PersistentCheckpointReservation,checkpoint_size_upper_bound
    from ..train.trajectory_state import atomic_checkpoint,available_system_memory_bytes,page_cache_reclaimable_for_host_allocations
    from .native_model import CHECKPOINT_SCHEMA
    source=Path(checkpoint).resolve(strict=True)
    if snapshot.get('schema')!=CHECKPOINT_SCHEMA or snapshot.get('inference_only') or snapshot.get('optimizer_pruned'):
        raise ValueError('snapshot view requires a complete native checkpoint')
    training=snapshot.get('training') or {}
    if not all(training.get(k) for k in ('optimizer','rng','source_receipts')):
        raise ValueError('snapshot view lacks optimizer, RNG or source custody')
    actual=source.stat();signature=(actual.st_dev,actual.st_ino,actual.st_size,actual.st_mtime_ns)
    if signature!=tuple(decoded_source_stat):raise ValueError('snapshot view source changed after commit')
    binding=_fresh_file_binding(source)
    selected={k:snapshot[k] for k in ('schema','model_config','backbone_state','interface_state','step')}
    if type(selected['step']) is not int or selected['step']<0:raise ValueError('snapshot view has invalid step')
    selected.update(inference_only=True,optimizer_pruned=True,optimizer_pruned_keys=['training'],
        inference_source={'schema':'natlang.native-inference-source/1','training_checkpoint_sha256':binding['sha256'],
                          'training_checkpoint_bytes':binding['bytes'],'step':selected['step']})
    tensors=[]
    def visit(value):
        if isinstance(value,torch.Tensor):
            if value.layout!=torch.strided or value.is_conj() or value.is_neg():
                raise ValueError('snapshot view has unsupported tensor storage')
            tensors.append((value,int(value._version),int(value.untyped_storage().nbytes())))
        elif isinstance(value,dict):
            for item in value.values():visit(item)
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
    visit(selected)
    largest=max((size for tensor,_,size in tensors if tensor.device.type!='cpu'),default=0)
    available=available_system_memory_bytes(reclaimable_page_cache=page_cache_reclaimable_for_host_allocations())
    required=math.ceil(largest*1.25)+16*1024**2
    if available is None or available<required:
        raise ValueError('snapshot view storage transfer exceeds actual host headroom')
    def unchanged():
        if any(tensor._version!=version for tensor,version,_ in tensors):
            raise ValueError('snapshot tensor changed while preparing its view')
    directory=Path(out).resolve()/binding['sha256'];directory.mkdir(parents=True,exist_ok=True)
    if _cached_actor_view(source,directory) is not None:
        result=write_actor_view(source,out,validated_state=snapshot,decoded_source_stat=signature,validate_before_publish=unchanged)
    else:
        staged=directory/'.actor-view-snapshot.pt'
        if not staged.exists():
            reservation=PersistentCheckpointReservation(staged)
            try:
                reservation.acquire(checkpoint_size_upper_bound(selected))
                atomic_checkpoint(staged,selected,reservation=reservation)
            finally:reservation.close()
        unchanged()
        result=write_actor_view(source,out,validated_state=snapshot,decoded_source_stat=signature,prepared_view=staged,
                                validate_before_publish=unchanged)
    return {'view_receipt':result,'source':binding,'maximum_cuda_storage_bytes':largest,
            'required_host_headroom_bytes':required,'available_host_bytes':available,
            'tensor_versions_unchanged':True,'training_resume':False}


def write_actor_view(checkpoint, out, *, validated_state, decoded_source_stat=None,
                     prepared_view=None, validate_before_publish=None):
    """Extract exact actor tensors from the exporter's already-validated decode.

    The caller must pass the actual state decoded from this immutable source,
    after its normal full-state validation. Unsupported source forms retain the
    existing full-checkpoint path by returning None, without writing a view.
    An explicitly named interrupted temporary output is never trusted: it must
    pass the same full safe reread, exact comparison and immutable publication.
    """
    from .native_model import CHECKPOINT_SCHEMA
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..train.trajectory_state import atomic_checkpoint
    from .native_composite import _durable_immutable
    source = Path(checkpoint).resolve(strict=True)
    if validated_state.get('schema') != CHECKPOINT_SCHEMA:
        return None
    with source.open('rb') as stream:
        if stream.read(1) == b'{':
            return None  # Existing composite/shared closure remains the full actor path.
    if validated_state.get('optimizer_pruned') or validated_state.get('inference_only'):
        raise ValueError('actor view requires its complete resumable native source')
    training = validated_state.get('training') or {}
    if not all(training.get(key) for key in ('optimizer','rng','source_receipts')):
        raise ValueError('actor view source lacks its actual optimizer/RNG/source validation')
    before = source.stat()
    signature = (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
    if signature != decoded_source_stat:
        raise ValueError('actor view source is not the unchanged file used for the validated decode')
    source_binding = _fresh_file_binding(source)
    selected = {key:validated_state[key] for key in
                ('schema','model_config','backbone_state','interface_state','step')}
    if type(selected['step']) is not int or selected['step'] < 0:
        raise ValueError('actor view source has no actual saved integer step')
    selected.update(inference_only=True,optimizer_pruned=True,
        optimizer_pruned_keys=['training'],inference_source={
            'schema':'natlang.native-inference-source/1',
            'training_checkpoint_sha256':source_binding['sha256'],
            'training_checkpoint_bytes':source_binding['bytes'],'step':selected['step']})
    mechanism = _file_binding(__file__)
    out = Path(out).resolve()/source_binding['sha256']
    out.mkdir(parents=True,exist_ok=True)
    recovered = prepared_view is not None
    cached = None if recovered else _cached_actor_view(source,out)
    if recovered:
        pending = Path(prepared_view).absolute()
        if (pending.is_symlink() or pending.parent.resolve(strict=True) != out
                or not pending.name.startswith('.actor-view-') or pending.suffix != '.pt'
                or not stat.S_ISREG(pending.lstat().st_mode)):
            raise ValueError('recovered actor view must be a regular temporary file in its exact source directory')
    elif cached is not None:
        pending=cached
    else:
        fd,name = tempfile.mkstemp(prefix='.actor-view-',suffix='.pt',dir=out)
        os.close(fd)
        pending = Path(name)
    try:
        if not recovered and cached is None:
            atomic_checkpoint(pending,selected)
        candidate_before = pending.stat()
        candidate_signature = (candidate_before.st_dev,candidate_before.st_ino,
            candidate_before.st_size,candidate_before.st_mtime_ns,candidate_before.st_ctime_ns)
        written = CheckpointPolicy.load(pending)
        _same_values_bounded(selected,written)
        del written
        digest = sha256_file_hex_cached(pending)
        candidate_after = pending.stat()
        if (candidate_after.st_dev,candidate_after.st_ino,candidate_after.st_size,
                candidate_after.st_mtime_ns,candidate_after.st_ctime_ns) != candidate_signature:
            raise ValueError('actor view temporary output changed during verification')
        if validate_before_publish is not None:validate_before_publish()
        target = out/(digest+'.pt')
        if cached is None:
            try:os.link(pending,target)
            except FileExistsError:
                if sha256_file_hex_cached(target) != digest:
                    raise ValueError('immutable actor view collision')
        directory = os.open(out,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if not recovered and cached is None:
            pending.unlink(missing_ok=True)
    after = source.stat()
    if ((after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) != signature
            or after.st_ctime_ns != before.st_ctime_ns):
        raise ValueError('actor view source changed during extraction')
    if validate_before_publish is not None:validate_before_publish()
    receipt = {'schema':ACTOR_VIEW_SCHEMA,'source':source_binding,'view':{
        **_file_binding(target),'path':target.name},'step':selected['step'],
        'model_config':selected['model_config'],'mechanism':mechanism,
        'verification':'safe reread and bitwise comparison of every selected value',
        'training_resume':False,'logical_identity':'original full source checkpoint'}
    raw = (json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
    import hashlib
    proof = out/(hashlib.sha256(raw).hexdigest()+'.json')
    _durable_immutable(proof,raw)
    if recovered:
        pending.unlink()
    return _file_binding(proof)


def prepare_actor_view(checkpoint, out):
    """Decode a complete native checkpoint once and write (or reuse) its verified actor view.

    Returns the receipt binding, or None for source forms that keep the full-checkpoint path.
    The view is read back through load_actor_view before it is returned.
    """
    from ..train.checkpoint_policy import CheckpointPolicy
    source = Path(checkpoint).resolve(strict=True)
    before = source.stat()
    signature = (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
    state = CheckpointPolicy.load(source)
    reference = write_actor_view(source,out,validated_state=state,decoded_source_stat=signature)
    del state
    if reference is not None:
        load_actor_view(source,reference)
    return reference


# diagnostic-local-pin: evaluation evidence (refused as an RL policy); rl-controller-pin: the same local-pin verification for
# the controller's own policy checkpoint, whose actor a remote host serves for RL collection (no remote recovery claim).
TRANSPORT_SCOPES = ('independent-recovery', 'diagnostic-local-pin', 'rl-controller-pin')


def prepare_actor_transport(checkpoint, out, recovery_receipt=None, *, scope='independent-recovery'):
    """Seal an inference view to a separately verified private full-state recovery."""
    from ..common.runtime_inventory import python_runtime_binding
    from ..common.code_provenance import code_provenance
    from .native_composite import _durable_immutable
    import hashlib
    if scope not in TRANSPORT_SCOPES:
        raise ValueError('unsupported actor transport custody scope')
    if (scope=='independent-recovery') != (recovery_receipt is not None):
        raise ValueError('actor transport recovery receipt does not match custody scope')
    source=_file_binding(checkpoint);recovery=_file_binding(recovery_receipt) if recovery_receipt is not None else None
    proof=json.loads(Path(recovery_receipt).read_text()) if recovery_receipt is not None else None
    if proof is not None and (proof.get('private') is not True or proof.get('upload_commit_notified') is not True
            or proof.get('upload',{}).get('status_code') not in (200,201)
            or any(proof.get('upload',{}).get(k)!=source.get(k) for k in ('sha256','bytes'))
            or proof.get('identity',{}).get('sha256')!=source['sha256']
            or proof.get('identity',{}).get('bytes')!=source['bytes']
            or proof.get('independent_verification',{}).get('sha256')!=source['sha256']
            or proof.get('independent_verification',{}).get('bytes')!=source['bytes']
            or not isinstance(proof.get('independent_verification',{}).get('readback_location'),str)
            or not proof['independent_verification']['readback_location']):
        raise ValueError('actor transport requires independent complete private full-source verification')
    view_ref=prepare_actor_view(checkpoint,out)
    if view_ref is None:raise ValueError('actor transport requires a single-model native checkpoint')
    view_proof=json.loads(Path(view_ref['path']).read_text())
    value={'schema':TRANSPORT_SCHEMA,'scope':scope,'source':{'sha256':source['sha256'],'bytes':source['bytes']},
        'recovery_receipt':({'sha256':recovery['sha256'],'bytes':recovery['bytes']} if recovery else None),
        'view_receipt':{'sha256':view_ref['sha256'],'bytes':view_ref['bytes']},
        'view':view_proof['view'],'model_config':view_proof['model_config'],
        'step':view_proof['step'],'python_runtime':python_runtime_binding(),'code':code_provenance(),
        'training_resume':False,'verification':('source tensor equality and independent full-source readback'
            if recovery else 'source tensor equality against local immutable pin; no remote recovery claim')}
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    target=Path(view_ref['path']).parent/(hashlib.sha256(raw).hexdigest()+'.transport.json')
    _durable_immutable(target,raw)
    return _file_binding(target)


def load_transported_actor_view(view, *, transport, view_receipt, recovery_receipt, source_sha256, scope):
    """Verify an out-of-band trusted transport without needing the full source locally."""
    from ..train.checkpoint_policy import CheckpointPolicy
    from .native_model import CHECKPOINT_SCHEMA
    def bound(reference):
        if not isinstance(reference,dict) or set(reference)!={'path','sha256','bytes'}:
            raise ValueError('transport input requires exact trusted binding')
        if _file_binding(reference['path'])!=reference:raise ValueError('transport input binding changed')
        return json.loads(Path(reference['path']).read_text())
    if scope not in TRANSPORT_SCOPES:
        raise ValueError('unsupported actor transport custody scope')
    if (scope=='independent-recovery') != (recovery_receipt is not None):
        raise ValueError('actor transport recovery receipt does not match custody scope')
    manifest=bound(transport);proof=bound(view_receipt);recovery=bound(recovery_receipt) if recovery_receipt else None
    source=manifest.get('source') or {}; actual=_file_binding(view)
    if (manifest.get('schema')!=TRANSPORT_SCHEMA or manifest.get('scope')!=scope or manifest.get('training_resume') is not False
            or source.get('sha256')!=source_sha256
            or any(manifest.get('view',{}).get(k)!=actual.get(k) for k in ('sha256','bytes'))
            or any(manifest.get('view_receipt',{}).get(k)!=view_receipt.get(k) for k in ('sha256','bytes'))
            or (recovery is None and manifest.get('recovery_receipt') is not None)
            or (recovery is not None and any((manifest.get('recovery_receipt') or {}).get(k)!=recovery_receipt.get(k) for k in ('sha256','bytes')))
            or proof.get('schema')!=ACTOR_VIEW_SCHEMA or proof.get('training_resume') is not False
            or proof.get('logical_identity')!='original full source checkpoint'
            or any(proof.get('source',{}).get(k)!=source.get(k) for k in ('sha256','bytes'))
            or any(proof.get('view',{}).get(k)!=actual.get(k) for k in ('sha256','bytes'))
            or (recovery is not None and (recovery.get('private') is not True
            or recovery.get('upload_commit_notified') is not True
            or recovery.get('upload',{}).get('status_code') not in (200,201)
            or any(recovery.get('upload',{}).get(k)!=source.get(k) for k in ('sha256','bytes'))
            or not isinstance(recovery.get('independent_verification',{}).get('readback_location'),str)
            or not recovery['independent_verification']['readback_location']
            or any(recovery.get('identity',{}).get(k)!=source.get(k) for k in ('sha256','bytes'))
            or any(recovery.get('independent_verification',{}).get(k)!=source.get(k) for k in ('sha256','bytes'))))):
        raise ValueError('transport source, view, recovery, or runtime binding mismatch')
    state=CheckpointPolicy.load(view)
    if (state.get('schema')!=CHECKPOINT_SCHEMA or state.get('inference_only') is not True
            or state.get('optimizer_pruned') is not True or 'training' in state
            or state.get('model_config')!=manifest.get('model_config') or state.get('model_config')!=proof.get('model_config')
            or state.get('step')!=manifest.get('step') or state.get('step')!=proof.get('step')
            or state.get('inference_source')!={'schema':'natlang.native-inference-source/1',
                'training_checkpoint_sha256':source_sha256,'training_checkpoint_bytes':source['bytes'],'step':state.get('step')}):
        raise ValueError('transported actor payload differs from authenticated source proof')
    state['actor_view']={'transport':transport,'view_receipt':view_receipt,'view':actual,
        'source':source,'recovery_receipt':recovery_receipt,'scope':scope,'training_resume':False}
    return state


def _actor_view_files(checkpoint, reference):
    """One exact file/receipt contract shared by transport and tensor loading."""
    if not isinstance(reference,dict) or set(reference) != {'path','sha256','bytes'}:
        raise ValueError('actor view needs an explicit exact receipt binding')
    if type(reference['bytes']) is not int or reference['bytes'] <= 0:
        raise ValueError('actor view receipt requires its exact integer byte count')
    proof = Path(reference['path']).resolve(strict=True)
    if _file_binding(proof) != reference:
        raise ValueError('actor view receipt binding changed')
    receipt = json.loads(proof.read_text())
    if (receipt.get('schema') != ACTOR_VIEW_SCHEMA or receipt.get('training_resume') is not False
            or receipt.get('logical_identity') != 'original full source checkpoint'
            or type(receipt.get('step')) is not int
            or type(receipt.get('source',{}).get('bytes')) is not int
            or type(receipt.get('view',{}).get('bytes')) is not int):
        raise ValueError('actor view receipt is not a read-only full-source binding')
    source = _file_binding(checkpoint)
    if any(source[key] != receipt['source'].get(key) for key in ('sha256','bytes')):
        raise ValueError('actor view belongs to another complete checkpoint')
    view = resolve_composite_artifact(receipt['view'],proof,sibling=True)
    if view.stat().st_size != receipt['view']['bytes']:
        raise ValueError('actor view byte length differs from receipt')
    return receipt, source, view


def load_actor_view(checkpoint, reference):
    """Verify an explicitly bound actor view while keeping full-source identity."""
    from ..train.checkpoint_policy import CheckpointPolicy
    from .native_model import CHECKPOINT_SCHEMA
    receipt, source, view = _actor_view_files(checkpoint, reference)
    state = CheckpointPolicy.load(view)
    if (state.get('schema') != CHECKPOINT_SCHEMA or state.get('inference_only') is not True
            or state.get('optimizer_pruned') is not True or 'training' in state
            or state.get('model_config') != receipt['model_config']
            or state.get('step') != receipt['step']
            or state.get('inference_source') != {
                'schema':'natlang.native-inference-source/1',
                'training_checkpoint_sha256':source['sha256'],
                'training_checkpoint_bytes':source['bytes'],'step':receipt['step']}):
        raise ValueError('actor view payload differs from its exact source receipt')
    state['actor_view'] = {'receipt':reference,'view':_file_binding(view),'source':source,
                           'training_resume':False}
    return state


def actor_view_dependencies(reference, *, metadata_only=False, checkpoint=None):
    """Exact explicitly consumed proof/view closure; full source stays separate."""
    proof = Path(reference['path']).resolve(strict=True)
    if _file_binding(proof) != reference:
        raise ValueError('actor view receipt binding changed')
    receipt = json.loads(proof.read_text())
    if metadata_only:
        # Dependency enumeration authenticates physical bytes. The actual
        # actor/raw-policy consumer still performs the unchanged tensor gates.
        _, source, view = _actor_view_files(checkpoint or receipt['source']['path'], reference)
        return [reference, _file_binding(view), source]
    state = load_actor_view(checkpoint or receipt['source']['path'],reference)
    return [reference,state['actor_view']['view'],state['actor_view']['source']]


def composite_dependencies(checkpoint, *, participant=None, checkpoint_loader=None):
    """Verify every parent/parameter/optimizer/training role without loading weights."""
    current = Path(checkpoint).resolve(strict=True)
    result, seen = {}, set()
    while current not in seen:
        seen.add(current)
        digest = sha256_file_hex_cached(current)
        result[digest] = current
        with current.open('rb') as stream:
            prefix = stream.read(1)
        if prefix != b'{':
            from .native_shared_checkpoint import shared_manifest, shared_dependencies
            manifest = shared_manifest(current,load=checkpoint_loader)
            if manifest is not None:
                for source in shared_dependencies(manifest, current, participant=participant).values():
                    result[sha256_file_hex_cached(source)] = source
            return result
        manifest = json.loads(current.read_text())
        if manifest.get('schema') != SCHEMA:
            raise ValueError('native bundle JSON checkpoint must be an exact native composite')
        for role in ('parameter_artifact','optimizer_state_artifact','training_state'):
            binding = manifest.get(role)
            if binding is not None:
                artifact = resolve_composite_artifact(binding,current,sibling=True)
                result[binding['sha256']] = artifact
        selected = manifest['base_checkpoint'].get('participant')
        if selected is not None:
            if participant is not None and participant != selected:
                raise ValueError('native composite selected participant differs from requested worker')
            participant = selected
        current = resolve_composite_artifact(manifest['base_checkpoint'],current)
    raise ValueError('native bundle checkpoint ancestry contains a cycle')


def _copy_immutable(source,destination,digest):
    """Bounded-memory durable transport copy; never hardlink a mutable source."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if sha256_file_hex_cached(destination) != digest:
            raise ValueError('native bundle immutable destination contains different bytes')
        return
    fd,pending = tempfile.mkstemp(prefix='.native-bundle-',dir=destination.parent)
    try:
        with os.fdopen(fd,'wb') as target,Path(source).open('rb') as original:
            shutil.copyfileobj(original,target,length=8*1024*1024)
            target.flush();os.fsync(target.fileno())
        if sha256_file_hex_cached(pending) != digest:
            raise ValueError('native bundle source changed while copying')
        try:
            os.link(pending,destination)
        except FileExistsError:
            if sha256_file_hex_cached(destination) != digest:
                raise ValueError('native bundle immutable destination collision')
        descriptor = os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        Path(pending).unlink(missing_ok=True)


def export_bundle(checkpoint,out,*,participant=None):
    checkpoint,out = Path(checkpoint).resolve(strict=True),Path(out).resolve()
    dependencies = composite_dependencies(checkpoint, participant=participant)
    digest = sha256_file_hex_cached(checkpoint)
    out.mkdir(parents=True,exist_ok=True)
    with checkpoint.open('rb') as stream:
        filename = 'checkpoint.json' if stream.read(1) == b'{' else 'objects/'+digest
    files = []
    for expected,source in sorted(dependencies.items()):
        target = out/filename if expected == digest else out/'objects'/expected
        _copy_immutable(source,target,expected)
        if target != out/filename:
            files.append({'path':str(target.relative_to(out)),'sha256':expected,'bytes':target.stat().st_size})
    entry = out/filename
    _copy_immutable(checkpoint,entry,digest)
    receipt = {'schema':BUNDLE_SCHEMA,'participant':participant,
        'checkpoint':{'path':filename,'sha256':digest},
        'files':files,'hf_base':'separately pinned; standard loader --base may relocate identical files',
        'identity_rewritten':False}
    write_json_durable(out/'bundle.json',receipt)
    return verify_bundle(out)


def verify_bundle(path):
    root = Path(path).resolve(strict=True)
    receipt = json.loads((root/'bundle.json').read_text())
    if receipt.get('schema') != BUNDLE_SCHEMA or receipt.get('identity_rewritten') is not False:
        raise ValueError('unsupported native checkpoint bundle')
    if receipt.get('semantic_adapter') or receipt.get('semantic_files'):
        raise ValueError('native bundle carries a removed cross-model semantic adapter')
    native_entries = [receipt['checkpoint'],*receipt['files']]
    seen = set()
    for binding in native_entries:
        artifact = (root/binding['path']).resolve(strict=True)
        if root not in artifact.parents or str(artifact) in seen or sha256_file_hex_cached(artifact) != binding['sha256']:
            raise ValueError('native bundle inventory changed, repeated or escaped its root')
        if 'bytes' in binding and artifact.stat().st_size != binding['bytes']:
            raise ValueError('native bundle artifact length changed')
        seen.add(str(artifact))
    checkpoint = root/receipt['checkpoint']['path']
    # Verify declared closure uses supplied objects even if original absolute
    # paths happen to remain available on this machine.
    actual = composite_dependencies(checkpoint, participant=receipt.get('participant'))
    declared = {value['sha256'] for value in native_entries}
    if set(actual) != declared:
        raise ValueError('native bundle inventory differs from complete checkpoint dependency closure')
    return {**receipt,'checkpoint':{**receipt['checkpoint'],'resolved':str(checkpoint)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action',required=True)
    export = commands.add_parser('export');export.add_argument('--checkpoint',required=True);export.add_argument('--out',required=True);export.add_argument('--participant')
    verify = commands.add_parser('verify');verify.add_argument('--bundle',required=True)
    view = commands.add_parser('actor-view');view.add_argument('--checkpoint',required=True);view.add_argument('--out',required=True)
    transport = commands.add_parser('actor-transport');transport.add_argument('--checkpoint',required=True);transport.add_argument('--out',required=True);transport.add_argument('--recovery-receipt');transport.add_argument('--scope',choices=TRANSPORT_SCOPES,required=True)
    args = parser.parse_args()
    if args.action == 'actor-view':
        print(json.dumps(prepare_actor_view(args.checkpoint,args.out),sort_keys=True))
        return
    if args.action == 'actor-transport':
        print(json.dumps(prepare_actor_transport(args.checkpoint,args.out,args.recovery_receipt,scope=args.scope),sort_keys=True))
        return
    print(json.dumps(export_bundle(args.checkpoint,args.out,participant=args.participant) if args.action=='export' else verify_bundle(args.bundle),sort_keys=True))


if __name__ == '__main__':
    main()
