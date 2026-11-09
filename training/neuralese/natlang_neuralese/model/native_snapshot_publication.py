"""Bounded full-source pin and leased actor-view publication at a save boundary."""
from __future__ import annotations
import fcntl,json,os,stat,time
from pathlib import Path
from ..common.jsonio import write_json_durable
from ..train.checkpoint_safety import checkpoint_target_lock
from .native_bundle import _fresh_file_binding,write_snapshot_actor_view


def selected_versions(state):
    """Capture selected tensor version counters across the primary write."""
    import torch
    if not isinstance(state,dict) or state.get('schema')!='natlang.neuralese-native-checkpoint/1':return ()
    result=[]
    def walk(value):
        if isinstance(value,torch.Tensor):result.append((value,int(value._version)))
        elif isinstance(value,dict):
            for item in value.values():walk(item)
        elif isinstance(value,(list,tuple)):
            for item in value:walk(item)
    for key in ('backbone_state','interface_state'):walk(state.get(key,{}))
    return tuple(result)


def require_versions_unchanged(versions):
    if any(tensor._version!=version for tensor,version in versions):
        raise ValueError('actor snapshot source changed during the primary checkpoint write')


def publish_saved_snapshot(queue,slot,snapshot,*,reservation=None):
    """Use one durable pending owner pin and the existing recovery lease ledger.

    A ready/pending pin remains leased until explicit, independently verified
    recovery and consumer release. No committed full checkpoint is retired here.
    """
    training=snapshot.get('training') or {}
    if not all(training.get(k) for k in ('optimizer','rng','source_receipts')):
        return {'status':'deferred','reason':'checkpoint is not a complete resumable native source'}
    slot=Path(slot).resolve(strict=True)
    if queue is None:
        root=slot.parent/'actor-view-owner';root.mkdir(parents=True,exist_ok=True);root=root.resolve(strict=True)
    else:root=Path(queue.directory).resolve(strict=True)
    if root.stat().st_dev!=slot.parent.stat().st_dev:raise ValueError('snapshot pin must share the checkpoint filesystem')
    pointer=root/'actor-view-pending.json';lock_path=root/'actor-view-owner.lock'
    from scripts.coreweave.retention_leases import change_lease
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if pointer.exists():
            current=json.loads(pointer.read_text())
            if current.get('schema')!='natlang.actor-view-owner-pin/1':raise ValueError('invalid pending actor view owner pin')
            if current.get('full_source_pinned') is not True or not Path(current['pin']).is_file():
                raise ValueError('incomplete actor-view source pin requires recovery before another update')
            return {'status':'deferred','reason':'bounded actor-view owner pin is occupied','pending':str(pointer)}
        with checkpoint_target_lock(slot,reservation=reservation):
            st=slot.stat();signature=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
            if not stat.S_ISREG(st.st_mode):raise ValueError('full checkpoint must be a regular file')
            source=_fresh_file_binding(slot);sha=source['sha256']
            role=f'actor-view-source:{sha}:{st.st_dev}:{st.st_ino}:{time.time_ns()}'
            consumer='actor-view-owner:'+sha
            identity={'sha256':sha,'bytes':source['bytes'],'step':snapshot.get('step')}
            lease={'lock_path':root/'standalone-cloud-retention.lock',
                'state_path':root/'standalone-retention-consumer-leases.json',
                'retirement_state_path':root/'local-recovery-retirement.json',
                'role':role,'consumer_id':consumer,'identity':identity,'write_durable':write_json_durable}
            pin=slot.with_name('.'+slot.name+'.actor-view-source-'+sha)
            record={'schema':'natlang.actor-view-owner-pin/1','status':'acquiring','source':source,
                    'pin':str(pin),'step':snapshot.get('step'),'lease':{'role':role,'consumer_id':consumer,'identity':identity},
                    'source_identity':{'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns},
                    'training_resume':False,'full_source_pinned':False}
            write_json_durable(pointer,record)
            try:
                change_lease(**lease,release=False)
                if not pin.exists():os.link(slot,pin,follow_symlinks=False)
                p=pin.stat();current=slot.stat()
                if ((p.st_dev,p.st_ino,p.st_size,p.st_mtime_ns)!=signature
                        or (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)!=signature):
                    raise ValueError('full checkpoint changed while acquiring its view pin')
                fd=os.open(pin.parent,os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
                record.update(status='pending',full_source_pinned=True)
                write_json_durable(pointer,record)
            except BaseException as error:
                record['last_error']={'type':type(error).__name__,'message':str(error)[:1000]}
                write_json_durable(pointer,record)
                raise
        try:
            result=write_snapshot_actor_view(pin,root/'actor-views',snapshot=snapshot,decoded_source_stat=signature)
            record.update(status='ready',view=result)
        except Exception as error:
            record.update(status='pending',last_error={'type':type(error).__name__,'message':str(error)[:1000]})
        write_json_durable(pointer,record)
        return {'status':record['status'],'pending':str(pointer),'source_sha256':sha,
                **({'view_receipt':record['view']['view_receipt']} if record['status']=='ready' else {'reason':record['last_error']})}


def retry_pending(root):
    """Retry a pinned failed view from the exact full source under operator admission."""
    from ..train.checkpoint_policy import CheckpointPolicy
    root=Path(root).resolve(strict=True);pointer=root/'actor-view-pending.json'
    with (root/'actor-view-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=json.loads(pointer.read_text());pin=Path(record['pin']).resolve(strict=True)
        if (record.get('schema')!='natlang.actor-view-owner-pin/1' or record.get('status')!='pending'
                or record.get('full_source_pinned') is not True):
            raise ValueError('actor view has no retryable pinned source')
        actual=_fresh_file_binding(pin)
        if any(actual[k]!=record['source'][k] for k in ('sha256','bytes')):raise ValueError('actor view source pin changed')
        st=pin.stat()
        if record.get('source_identity')!={'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns}:
            raise ValueError('actor view source pin inode changed')
        signature=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        state=CheckpointPolicy.load(pin)
        try:result=write_snapshot_actor_view(pin,root/'actor-views',snapshot=state,decoded_source_stat=signature)
        except Exception as error:
            record['last_error']={'type':type(error).__name__,'message':str(error)[:1000]};write_json_durable(pointer,record);raise
        record.update(status='ready',view=result);record.pop('last_error',None);write_json_durable(pointer,record)
        return record


def release_pending(root,*,recovery_ref,transport_ref,delivery_ref,recovery_export_ref=None):
    """Release only after actual full recovery and exact actor transport readback."""
    import hashlib
    from .native_composite import _durable_immutable
    root=Path(root).resolve(strict=True);pointer=root/'actor-view-pending.json'
    def read(ref):
        actual=_fresh_file_binding(ref['path'])
        if actual!=ref:raise ValueError('actor owner release evidence changed')
        return json.loads(Path(ref['path']).read_text())
    with (root/'actor-view-owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=json.loads(pointer.read_text());recovery=read(recovery_ref);transport=read(transport_ref);delivery=read(delivery_ref)
        source=record['source'];view=record.get('view',{}).get('view_receipt',{})
        saved=recovery.get('checkpoint') or {}
        private=recovery.get('schema')=='natlang.private-exact-earned-SFT-recovery/1'
        if private:
            _verify_private_reconstruction(recovery,recovery_ref,recovery_export_ref,source)
        elif (recovery.get('schema')!='natlang.local-full-recovery-export/1'
                or recovery.get('full_state') is not True or recovery.get('independent_readback') is not True
                or saved.get('sha256')!=source['sha256'] or saved.get('bytes')!=source['bytes']
                or _fresh_file_binding(saved['path'])['sha256']!=source['sha256']):
            raise ValueError('actor owner release lacks verified independent full recovery')
        if (record.get('status')!='ready' or record.get('full_source_pinned') is not True
                or transport.get('schema')!='natlang.native-actor-transport/1'
                or transport.get('source',{}).get('sha256')!=source['sha256']
                or transport.get('view_receipt',{}).get('sha256')!=view.get('sha256')
                or delivery.get('schema')!='natlang.actor-transfer-readback/1'
                or delivery.get('transport_sha256')!=transport_ref['sha256']
                or delivery.get('verified',{}).get('source_checkpoint_sha256')!=source['sha256']
                or delivery.get('verified',{}).get('sha256')!=transport.get('view',{}).get('sha256')):
            raise ValueError('actor owner release requires exact independent full recovery and actor readback')
        lease=record['lease'];role=lease['role']
        release={'schema':'natlang.actor-view-owner-release/1','source_sha256':source['sha256'],
                 'view_receipt':view,'recovery':recovery_ref,'transport':transport_ref,'delivery':delivery_ref,
                 'source_pin_identity':record.get('source_identity'),
                 **({'recovery_export':recovery_export_ref} if private else {})}
        raw=(json.dumps(release,sort_keys=True,indent=2)+'\n').encode();digest=hashlib.sha256(raw).hexdigest()
        proof_path=root/'actor-view-releases'/(digest+'.json')
        pin=Path(record['pin'])
        if pin.exists():
            actual=_fresh_file_binding(pin)
            if actual['sha256']!=source['sha256'] or actual['bytes']!=source['bytes']:
                raise ValueError('actor owner full-source pin changed')
            ps=pin.stat()
            if record.get('source_identity')!={'device':ps.st_dev,'inode':ps.st_ino,'size':ps.st_size,'mtime_ns':ps.st_mtime_ns}:
                raise ValueError('actor owner source inode identity differs')
            if not private and (Path(saved['path']).stat().st_dev==ps.st_dev or recovery.get('source_identity')!={
                    'device':ps.st_dev,'inode':ps.st_ino,'size':ps.st_size,'mtime_ns':ps.st_mtime_ns}):
                raise ValueError('actor owner release recovery is not independent of its exact source inode')
        elif not proof_path.is_file() or proof_path.read_bytes()!=raw:
            raise ValueError('actor owner full-source pin disappeared before durable release')
        with (root/'standalone-cloud-retention.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            leases_path=root/'standalone-retention-consumer-leases.json';ledger=json.loads(leases_path.read_text())
            retirement_path=root/'local-recovery-retirement.json'
            retirement=json.loads(retirement_path.read_text()) if retirement_path.exists() else {'retired_roles':[],'operations':[]}
            matching=[item for item in ledger.get('leases',[]) if item.get('role')==role]
            if matching!=[lease] and not (not matching and role in retirement['retired_roles'] and proof_path.exists()):
                raise ValueError('actor owner pin has missing or additional consumer leases')
            _durable_immutable(proof_path,raw)
            if role not in retirement['retired_roles']:retirement['retired_roles'].append(role)
            write_json_durable(retirement_path,retirement)
            if matching:ledger['leases'].remove(lease);write_json_durable(leases_path,ledger)
            pin.unlink(missing_ok=True);pointer.unlink()
            for directory in (pin.parent,root):
                fd=os.open(directory,os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
        return release


def prepare_pending_transport(root):
    """Bind a ready owner view to this actual Python runtime without a full-state decode."""
    import hashlib
    from ..common.runtime_inventory import python_runtime_binding
    from ..common.code_provenance import code_provenance
    from .native_bundle import _actor_view_files,_file_binding,TRANSPORT_SCHEMA
    from .native_composite import _durable_immutable
    root=Path(root).resolve(strict=True)
    with (root/'actor-view-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=json.loads((root/'actor-view-pending.json').read_text())
        if record.get('status')!='ready' or record.get('full_source_pinned') is not True:
            raise ValueError('actor view owner has no ready pinned view')
        ledger=json.loads((root/'standalone-retention-consumer-leases.json').read_text())
        if record['lease'] not in ledger.get('leases',[]):raise ValueError('actor view owner lease is missing')
        st=Path(record['pin']).stat()
        if record.get('source_identity')!={'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns}:
            raise ValueError('actor view owner source inode changed')
        reference=record['view']['view_receipt'];proof,source,view=_actor_view_files(record['pin'],reference)
        if any(source[k]!=record['source'][k] for k in ('sha256','bytes')):
            raise ValueError('actor view owner source changed')
        value={'schema':TRANSPORT_SCHEMA,'scope':'diagnostic-local-pin','source':{'sha256':source['sha256'],'bytes':source['bytes']},
            'recovery_receipt':None,'view_receipt':{'sha256':reference['sha256'],'bytes':reference['bytes']},
            'view':proof['view'],'model_config':proof['model_config'],'step':proof['step'],
            'python_runtime':python_runtime_binding(),'code':code_provenance(),'training_resume':False,
            'verification':'source tensor equality against local immutable pin; no remote recovery claim'}
        raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
        target=Path(reference['path']).parent/(hashlib.sha256(raw).hexdigest()+'.transport.json')
        _durable_immutable(target,raw)
        return _file_binding(target)


def _verify_private_reconstruction(capsule, capsule_ref, export_ref, source):
    """Rejoin the private publisher journal, decoded readback, and base/codec closure."""
    import math
    if export_ref is None:raise ValueError('private reconstruction requires its bound publisher journal')
    if _fresh_file_binding(export_ref['path'])!=export_ref:raise ValueError('private recovery publisher journal changed')
    state=json.loads(Path(export_ref['path']).read_text())
    target={'sha256':source['sha256'],'bytes':source['bytes']}
    if (capsule.get('schema')!='natlang.private-exact-earned-SFT-recovery/1'
            or capsule.get('complete_target_stream_verified') is not True
            or capsule.get('training_restore_qualified') is not False
            or capsule.get('target')!=target or not isinstance(capsule.get('run'),str) or not capsule['run']
            or state.get('schema')!='natlang.private-native-admission-evidence/1'
            or state.get('complete') is not True or state.get('run')!=capsule['run'] or state.get('target')!=target
            or (state.get('capsule') or {}).get('sha256')!=capsule_ref['sha256']
            or (state.get('capsule') or {}).get('bytes')!=capsule_ref['bytes']
            or (state.get('capsule') or {}).get('role')!='earned-SFT-exact-full-state-private-capsule'):
        raise ValueError('private reconstruction capsule or publisher binding differs')
    objects=state.get('objects') or {}
    def verified(ref):
        obj=objects.get(ref['sha256']) or {};get=obj.get('owner_get') or {};parts=obj.get('parts')
        if (obj.get('phase')!='independently-verified' or obj.get('bytes')!=ref['bytes']
                or get.get('sha256')!=ref['sha256'] or get.get('bytes')!=ref['bytes']
                or get.get('mechanism')!='complete private GET on DGX; no sandbox'
                or type(get.get('unix')) not in (int,float) or not math.isfinite(get['unix'])
                or not isinstance(parts,list) or not parts):
            raise ValueError('private reconstruction dependency lacks independent GET evidence')
        offset=0
        for index,part in enumerate(parts):
            if (part.get('index')!=index or part.get('offset')!=offset
                    or part.get('name')!=f"blobs/{ref['sha256']}.p{index:04d}"
                    or type(part.get('bytes')) is not int or part['bytes']<=0
                    or not isinstance(part.get('sha256'),str) or len(part['sha256'])!=64):
                raise ValueError('private reconstruction multipart custody differs')
            offset+=part['bytes']
        if offset!=ref['bytes']:raise ValueError('private reconstruction multipart length differs')
    verified(capsule_ref)
    files=capsule.get('files')
    if not isinstance(files,list):raise ValueError('private reconstruction lacks dependency inventory')
    def role(name,*,multiple=False):
        names={name} if isinstance(name,str) else set(name)
        found=[item for item in files if isinstance(item,dict) and item.get('role') in names]
        if not found or (not multiple and len(found)!=1):raise ValueError('private reconstruction dependency role differs')
        for item in found:
            verified(item)
            obj=objects[item['sha256']]
            if obj.get('source')!=item.get('original_path') or obj.get('role')!=item.get('role'):
                raise ValueError('private reconstruction producer lineage differs')
        return found if multiple else found[0]
    local=role('local-full-state-reconstruction-capsule');delta=role('exact-full-state-delta-receipt')
    payload=role('exact-full-state-delta-payload');readback=role('independent-private-full-target-stream-readback')
    # The base capsule is bound by content below; its role names the actual base.
    parent=role(('retained-private-BF95-base-capsule','retained-private-internal21b4-base-capsule'));decoders=role('exact-reconstruction-decoder',multiple=True)
    if len(decoders)<2:raise ValueError('private reconstruction lacks its complete decoder closure')
    def local_json(item):
        path=Path(item['original_path']).resolve(strict=True);actual=_fresh_file_binding(path)
        if actual['sha256']!=item['sha256'] or actual['bytes']!=item['bytes']:
            raise ValueError('private reconstruction local evidence changed')
        return json.loads(path.read_text())
    local_capsule=local_json(local);delta_receipt=local_json(delta);stream=local_json(readback);parent_capsule=local_json(parent)
    for item in decoders:
        path=Path(item['original_path']).resolve(strict=True);actual=_fresh_file_binding(path)
        if actual['sha256']!=item['sha256'] or actual['bytes']!=item['bytes']:
            raise ValueError('private reconstruction decoder changed')
    base=capsule.get('base')
    if (local_capsule.get('schema')!='natlang.local-exact-checkpoint-reconstruction/1'
            or local_capsule.get('all_full_state_bytes_verified') is not True or local_capsule.get('target')!=target
            or {k:(local_capsule.get('base') or {}).get(k) for k in ('sha256','bytes')}!=base
            or (local_capsule.get('receipt') or {}).get('sha256')!=delta['sha256']
            or (local_capsule.get('payload') or {}).get('sha256')!=payload['sha256']
            or (local_capsule.get('retained_independent_private_base_capsule') or {}).get('sha256')!=parent['sha256']
            or {(x.get('sha256'),x.get('bytes')) for x in local_capsule.get('decoders',[])}
                !={(x['sha256'],x['bytes']) for x in decoders}
            or delta_receipt.get('schema') not in ('natlang.artifact-zip-delta/1','natlang.artifact-zip-delta/2')
            or delta_receipt.get('target')!=target or delta_receipt.get('base')!=base
            or (delta_receipt.get('payload') or {}).get('sha256')!=payload['sha256']
            or (delta_receipt.get('payload') or {}).get('bytes')!=payload['bytes']
            or stream.get('sha256')!=target['sha256'] or stream.get('bytes')!=target['bytes']
            or stream.get('path') is not None or stream.get('receipt_sha256')!=delta['sha256']
            or stream.get('payload_get_sha256')!=payload['sha256'] or stream.get('payload_get_bytes')!=payload['bytes']
            or type(stream.get('verified_unix')) not in (int,float) or not math.isfinite(stream['verified_unix'])
            or parent_capsule.get('schema')!='natlang.exact-native-execution-recovery-capsule/1'
            or parent_capsule.get('run')!=capsule['run']
            or not any(item.get('target')==base for item in (parent_capsule.get('targets') or {}).values())):
        raise ValueError('private reconstruction target, base, decoder or streamed readback differs')
    base_target=next(item for item in parent_capsule['targets'].values() if item.get('target')==base)
    required=[parent_capsule['base'],base_target['payload'],base_target['receipt']]
    presence=stream.get('base_private_presence')
    if not isinstance(presence,list):raise ValueError('private reconstruction lacks base presence')
    observed={item.get('name'):(item.get('bytes'),item.get('http_status')) for item in presence if isinstance(item,dict)}
    for ref in required:
        remaining=ref['bytes'];index=0
        while remaining:
            size=min(1024**3,remaining);name=f"blobs/{ref['sha256']}.p{index:04d}"
            if observed.get(name) not in ((size,200),(size,206)):
                raise ValueError('private reconstruction base dependency was not observed')
            remaining-=size;index+=1
    if _fresh_file_binding(export_ref['path'])!=export_ref or _fresh_file_binding(capsule_ref['path'])!=capsule_ref:
        raise ValueError('private reconstruction evidence changed during verification')
