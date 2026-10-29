"""Exact coherent replica recovery closure: current state plus witnessed lineage."""
from __future__ import annotations
import hashlib,json,os,re
from pathlib import Path
from ...common.jsonio import write_json_durable
from .replica_baseline import capture,verify,reconstruct,finish_pinned
from .replica_journal import ReplicaJournal,_identity
from .replica_state import replay


def _digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def capture_recovery(dispatch,files,store,destination,*,fence_id,workers,owner_journal_sha256,on_pinned=None):
    """Pin a sealed coordinator, its exact baseline/WAL, and committed uploads.

    The caller must hold the real worker fence until this returns. Uploads in
    flight fail closed; workers' immutable checkpoint identities are bound.
    """
    if not re.fullmatch('[A-Za-z0-9_-]{1,128}',fence_id) or not re.fullmatch('[0-9a-f]{64}',owner_journal_sha256):raise ValueError('invalid recovery fence identity')
    if (not isinstance(workers,dict) or sorted(workers)!=[str(i) for i in range(len(workers))] or not workers or
            any(not isinstance(x,dict) or x.get('id')!=fence_id or x.get('worker')!=int(rank) or
                not isinstance(x.get('checkpoint'),str) or not x['checkpoint'].startswith('/') or
                not isinstance(x.get('sequences'),list) or not isinstance(x.get('versions'),list) or type(x.get('global_step')) is not int or not x.get('fingerprint') or
                not isinstance(x.get('deadline'),(int,float)) for rank,x in workers.items())):
        raise ValueError('recovery requires every exact fenced worker checkpoint')
    store=Path(store).resolve();destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    def pin_extra(coordinator):
        if any(__import__('time').time()>=value['deadline'] for value in workers.values()):raise ValueError('worker recovery fence expired before coordinator pin')
        if any(value['fingerprint']!=coordinator.manifest['fingerprint'] for value in workers.values()):raise ValueError('worker recovery fence fingerprint differs')
        if len(workers)!=coordinator.manifest['workers']:raise ValueError('worker recovery fence omits coordinator ranks')
        journal=dispatch.executor.journal
        if journal.sequence%2:raise ValueError('recovery cannot pin an unsealed journal')
        pointer=json.loads((store/'primary-active.json').read_text());baseline=Path(pointer['baseline']).resolve()
        if Path(pointer['journal']).resolve()!=journal.root.resolve() or pointer['baseline_sha256']!=journal.identity['baseline_sha256']:
            raise ValueError('recovery active generation differs')
        target=destination/'lineage';(target/'baseline').mkdir(parents=True);(target/'journal').mkdir()
        old=json.loads((baseline/'replica-baseline.json').read_text())
        entries=[('replica-baseline.json',baseline/'replica-baseline.json')]+[(item['path'],baseline/item['path']) for item in old['files']]
        for name,source in entries:
            dest=target/'baseline'/name;dest.parent.mkdir(parents=True,exist_ok=True);os.link(source,dest)
        os.link(journal.root/'identity.json',target/'journal'/'identity.json')
        for seq in range(1,journal.sequence+1):
            folder=target/'journal'/f'{seq:020d}';folder.mkdir()
            for name in ('record.json','payload'):os.link(journal.root/f'{seq:020d}'/name,folder/name)
        pending=files.root/'.replication-pending'
        if pending.exists() and any(pending.iterdir()):raise ValueError('file replication is in flight')
        partial=files.root/'.partial'
        if partial.exists() and any(partial.iterdir()):raise ValueError('partial file upload is in flight')
        if files.root.exists():
            for source in sorted(files.root.iterdir()):
                if source.name.startswith('.'):continue
                if source.is_symlink() or not source.is_file():raise ValueError('invalid coordinator upload')
                if source.name.endswith('.sha256'):
                    payload=source.with_name(source.name[:-7])
                    if not payload.is_file() or payload.is_symlink():raise ValueError('upload custody differs')
                elif not source.with_name(source.name+'.sha256').is_file():raise ValueError('unsealed coordinator upload')
                out=destination/'uploads'/source.name;out.parent.mkdir(exist_ok=True);os.link(source,out)
        for folder in sorted({destination,*(p.parent for p in destination.rglob('*') if p.is_file())},key=lambda p:len(p.parts),reverse=True):
            fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        return {'sequence':journal.sequence,'head':journal.head,'baseline_sha256':pointer['baseline_sha256']}
    files.snapshot_gate.acquire()
    held=True
    def pinned(value):
        nonlocal held
        write_json_durable(destination/'pin-record.json',{'schema':'natlang.diloco-recovery-pin/1','pinned':value,'fence_id':fence_id,'workers':workers,'owner_journal_sha256':owner_journal_sha256})
        files.snapshot_gate.release();held=False
        if on_pinned:on_pinned(value)
    try:
        current=capture(dispatch,destination/'current',pin_extra=pin_extra,on_pinned=pinned)
    finally:
        if held:files.snapshot_gate.release()
    return finish_recovery(destination,current=current)


def finish_recovery(destination,*,current=None):
    destination=Path(destination)
    record=json.loads((destination/'pin-record.json').read_text())
    if record.get('schema')!='natlang.diloco-recovery-pin/1':raise ValueError('invalid recovery pin record')
    fence_id=record['fence_id'];workers=record['workers'];owner_journal_sha256=record['owner_journal_sha256']
    receipt_path=destination/'recovery-closure.json'
    if receipt_path.exists():
        value=json.loads(receipt_path.read_text())
        return {'path':str(receipt_path.resolve()),'sha256':_digest(receipt_path),'value':value}
    if current is None:current=finish_pinned(destination/'current',record['pinned'])
    head=current['value']['extra']
    verify(destination/'lineage'/'baseline',head['baseline_sha256'])
    identity=json.loads((destination/'lineage'/'journal'/'identity.json').read_text())
    check=ReplicaJournal(destination/'lineage'/'journal',run_identity=identity['run_identity'],baseline_sha256=identity['baseline_sha256'],
        max_payload_bytes=identity['max_payload_bytes'],max_transactions=identity['max_transactions'],previous_generation=identity.get('previous_generation'))
    try:check.verify(expected_sequence=head['sequence'],expected_head=head['head'])
    finally:check.close();(destination/'lineage'/'journal'/'.writer.lock').unlink(missing_ok=True)
    inventory=[]
    for path in sorted(destination.rglob('*')):
        if path.is_symlink():raise ValueError('recovery closure contains a symlink')
        if path.is_file():inventory.append({'path':str(path.relative_to(destination)),'bytes':path.stat().st_size,'sha256':_digest(path)})
    by_name={item['path']:item for item in inventory}
    for name,item in by_name.items():
        if name.startswith('uploads/') and name.endswith('.sha256'):
            payload=name[:-7]
            if payload not in by_name or (destination/name).read_text().strip()!=by_name[payload]['sha256']:
                raise ValueError('pinned upload content differs from its digest')
    from .coordinator import _metadata
    from .protocol import load_tensors
    manifest=json.loads((destination/'current'/'manifest.json').read_text());versions=[];ledgers=[];queued=[]
    for fragment in manifest['fragments']:
        index=fragment['index'];meta=_metadata(destination/'current'/'fragments'/f'{index:05d}.safetensors')
        versions.append(int(meta['version']));ledgers.append(json.loads(meta.get('last_seq','{}')));items={}
        folder=destination/'current'/'pending'/f'{index:05d}'
        if folder.is_dir():
            for path in sorted(folder.glob('*.safetensors')):
                entry=_metadata(path);rank=str(int(entry['worker']))
                if rank in items:raise ValueError('duplicate queued worker')
                delta=load_tensors(path)['delta'].detach().float().cpu().contiguous()
                items[rank]={'sequence':int(entry['sequence']),'base_version':int(entry['base_version']),
                    'steps':int(entry['steps']),'tokens':int(entry['tokens']),'delta_sha256':hashlib.sha256(delta.numpy().tobytes()).hexdigest()}
        queued.append(items)
    coordinator={'fingerprint':manifest['fingerprint'],'versions':versions,'last_seq':ledgers,'queued':queued,'manifest':manifest,
                 'identity_transitions':json.loads((destination/'current'/'progress.json').read_text()).get('identity_transitions') or {},
                 'runtime_code_sha256':current['value']['runtime_code_sha256']}
    receipt={'schema':'natlang.diloco-replicated-recovery/1','run_id':current['value']['run_id'],'fence_id':fence_id,'workers':workers,
        'owner_journal_sha256':owner_journal_sha256,'coordinator':coordinator,'current_baseline_sha256':current['sha256'],'lineage':head,'files':inventory}
    write_json_durable(destination/'recovery-closure.json',receipt)
    return {'path':str((destination/'recovery-closure.json').resolve()),'sha256':_digest(destination/'recovery-closure.json'),'value':receipt}


def restore_recovery(source,destination,*,receipt_sha256,workers,worker_components,owner_journal_sha256,owner_journal_path,worker_paths):
    """Verify complete closure and replay lineage before restoring a new root."""
    source=Path(source).resolve();receipt_path=source/'recovery-closure.json'
    if _digest(receipt_path)!=receipt_sha256:raise ValueError('recovery receipt hash differs')
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('schema')!='natlang.diloco-replicated-recovery/1' or receipt['workers']!=workers or receipt['owner_journal_sha256']!=owner_journal_sha256:
        raise ValueError('recovery worker or owner custody differs')
    if _digest(owner_journal_path)!=owner_journal_sha256:raise ValueError('owner journal bytes differ')
    owner=json.loads(Path(owner_journal_path).read_text())
    if owner.get('phase')!='complete' or owner.get('run_id')!=receipt['run_id'] or sorted(owner.get('workers',{}))!=sorted(workers) or sorted(worker_paths)!=sorted(workers):
        raise ValueError('complete actual worker ownership is required')
    from ..checkpoint_policy import CheckpointPolicy
    from ..native_state_sft import TRAINING_SCHEMA
    from ...model.native_model import CHECKPOINT_SCHEMA
    if sorted(worker_components)!=sorted(workers):raise ValueError('worker component set differs')
    if any(not all((owner['workers'][rank].get('sandbox_id'),owner['workers'][rank].get('boot_id'),owner['workers'][rank].get('owner_id'),owner['workers'][rank].get('pid'),owner['workers'][rank].get('start_ticks'))) for rank in workers):
        raise ValueError('owner journal lacks actual worker process identity')
    worker_states={}
    for rank,item in workers.items():
        component=worker_components[rank]
        if (component.get('source')!=item['checkpoint'] or component.get('fingerprint')!=item['fingerprint'] or
                component.get('versions')!=item['versions'] or component.get('sequences')!=item['sequences'] or component.get('global_step')!=item['global_step'] or
                _digest(worker_paths[rank])!=component.get('sha256')):raise ValueError('worker full checkpoint binding differs')
        state=CheckpointPolicy.load(worker_paths[rank]);training=state.get('training') or {};dl=training.get('diloco') or {}
        if (state.get('schema')!=CHECKPOINT_SCHEMA or training.get('schema')!=TRAINING_SCHEMA or dl.get('worker')!=int(rank) or
                dl.get('fingerprint')!=item['fingerprint'] or dl.get('versions')!=item['versions'] or dl.get('sequences')!=item['sequences'] or
                not training.get('optimizer') or not training.get('rng') or not training.get('exposure_counts') or
                not training.get('source_receipts') or training.get('global_step')!=item['global_step']):
            raise ValueError('worker checkpoint lacks exact full-state custody')
        options=training.get('options') or {}
        if (training.get('source_hash')!=component.get('source_hash') or
                [x.get('sha256') for x in training['source_receipts'][:2]]!=component.get('source_receipt_sha256') or
                options.get('recipe_binding')!=component.get('recipe_binding') or
                options.get('ce_supervision_policy_binding')!=component.get('ce_supervision_policy_binding')):
            raise ValueError('worker checkpoint source or recipe custody differs')
        worker_states[rank]=dl

    seen=set()
    for item in receipt['files']:
        relative=Path(item['path']);path=source/relative
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen or path.is_symlink() or not path.resolve().is_relative_to(source):raise ValueError('invalid recovery closure path')
        seen.add(str(relative))
        if not path.is_file() or path.stat().st_size!=item['bytes'] or _digest(path)!=item['sha256']:raise ValueError('recovery closure file differs')
    actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and p!=receipt_path and p!=source/'lineage'/'journal'/'.writer.lock'}
    if actual!=seen:raise ValueError('recovery closure inventory differs')
    component_values=list(worker_components.values())
    if (any(not item.get('source_hash') or item.get('runtime_code_sha256')!=receipt['coordinator']['runtime_code_sha256'] or not item.get('source_receipt_sha256') for item in component_values) or
            len({item['source_hash'] for item in component_values})!=1 or
            len({json.dumps(item.get('recipe_binding'),sort_keys=True) for item in component_values})!=1 or
            len({json.dumps(item.get('ce_supervision_policy_binding'),sort_keys=True) for item in component_values})!=1):
        raise ValueError('worker source, recipe, CE or runtime custody differs')
    prior=verify(source/'lineage'/'baseline',receipt['lineage']['baseline_sha256']);current=verify(source/'current',receipt['current_baseline_sha256'])
    identity=json.loads((source/'lineage'/'journal'/'identity.json').read_text())
    journal=ReplicaJournal(source/'lineage'/'journal',run_identity=identity['run_identity'],baseline_sha256=identity['baseline_sha256'],
        max_payload_bytes=identity['max_payload_bytes'],max_transactions=identity['max_transactions'],previous_generation=identity.get('previous_generation'))
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    try:
        reconstruct(source/'lineage'/'baseline',destination/'replayed',expected_sha256=receipt['lineage']['baseline_sha256'])
        state=replay(journal,destination/'replayed',expected_sequence=receipt['lineage']['sequence'],expected_head=receipt['lineage']['head'],
            quorum_since={int(k):v for k,v in prior['quorum_since'].items()},clock_anchor={'wall':prior['captured_wall'],'monotonic':prior['captured_monotonic']})
        for item in current['files']:
            if _digest(destination/'replayed'/item['path'])!=item['sha256']:raise ValueError('replayed coordinator differs from captured baseline')
        if {str(k):v for k,v in state.coordinator.quorum_since.items()}!=current['quorum_since']:raise ValueError('replayed quorum timers differ')
        from .coordinator import _metadata
        from .protocol import load_tensors
        for index,fragment in enumerate(state.coordinator.fragments):
            meta=_metadata(state.coordinator._path(index));version=int(meta['version']);ledger=json.loads(meta.get('last_seq','{}'))
            pending={str(int((entry:=_metadata(path))['worker'])):(entry,path) for path in state.coordinator._pending_dir(index).glob('*.safetensors')}
            for rank,dl in worker_states.items():
                if len(dl['versions'])!=len(state.coordinator.fragments) or len(dl['sequences'])!=len(state.coordinator.fragments) or dl['versions'][index]>version:
                    raise ValueError('worker fragment versions do not match recovered coordinator')
                merged=int(ledger.get(rank,0));queued=pending.get(rank);held=max(merged,int(queued[0]['sequence']) if queued else 0)
                if held>dl['sequences'][index] or (index in dl.get('awaiting_global',[]) and held<dl['sequences'][index]):
                    raise ValueError('worker sequence custody does not match recovered coordinator')
                local=(dl.get('pending') or {}).get(str(index))
                if queued and local and int(queued[0]['sequence'])==local.get('sequence'):
                    tensor=(dl.get('pending_payloads') or {}).get(str(index))
                    if tensor is None:raise ValueError('worker pending payload is missing')
                    expected=hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
                    actual=hashlib.sha256(load_tensors(queued[1])['delta'].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
                    if expected!=actual:raise ValueError('pending worker payload differs')

    finally:journal.close()
    reconstruct(source/'current',destination/'coordinator',expected_sha256=receipt['current_baseline_sha256'])
    if (source/'uploads').exists():
        (destination/'coordinator'/'uploads').mkdir()
        for item in (source/'uploads').iterdir():os.link(item,destination/'coordinator'/'uploads'/item.name)
    new=ReplicaJournal(destination/'journal',run_identity=receipt['run_id'],baseline_sha256=receipt['current_baseline_sha256'],
        previous_generation={'sequence':receipt['lineage']['sequence'],'head':receipt['lineage']['head']})
    write_json_durable(destination/'verified-restore.json',{'schema':'natlang.diloco-verified-restore/1','receipt_sha256':receipt_sha256,'owner_journal_sha256':owner_journal_sha256,'worker_sha256':{rank:worker_components[rank]['sha256'] for rank in workers}})
    return {'receipt':receipt,'journal':new,'coordinator':destination/'coordinator'}


def prepare_restored_peer_seed(*, closure, restored, destination, seed, receipt_sha256, freeze_nonce):
    """Prepare an exact frozen peer seed from an already verified staged restore."""
    from .replica_seed import publish_seed
    if not re.fullmatch('[A-Za-z0-9_-]{32,128}',freeze_nonce):raise ValueError('invalid recovery activation nonce')
    closure=Path(closure).resolve();restored=Path(restored).resolve();destination=Path(destination)
    receipt_path=closure/'recovery-closure.json'
    if _digest(receipt_path)!=receipt_sha256:raise ValueError('recovery closure differs')
    receipt=json.loads(receipt_path.read_text());baseline=closure/'current'
    verified=json.loads((restored/'verified-restore.json').read_text())
    if (verified.get('schema')!='natlang.diloco-verified-restore/1' or verified.get('receipt_sha256')!=receipt_sha256 or
        verified.get('owner_journal_sha256')!=receipt['owner_journal_sha256'] or sorted(verified.get('worker_sha256',{}))!=sorted(receipt['workers'])):
        raise ValueError('full-cohort restore was not verified')
    verify(baseline,receipt['current_baseline_sha256'])
    identity=json.loads((restored/'journal'/'identity.json').read_text())
    if (identity.get('baseline_sha256')!=receipt['current_baseline_sha256'] or identity.get('run_identity')!=receipt['run_id'] or
        identity.get('previous_generation')!={'sequence':receipt['lineage']['sequence'],'head':receipt['lineage']['head']}):
        raise ValueError('restored journal ancestry differs')
    check=ReplicaJournal(restored/'journal',run_identity=identity['run_identity'],baseline_sha256=identity['baseline_sha256'],
        max_payload_bytes=identity['max_payload_bytes'],max_transactions=identity['max_transactions'],previous_generation=identity.get('previous_generation'))
    try:
        if check.sequence!=0:raise ValueError('restored journal must be unmodified')
        head=check.head
    finally:check.close()
    destination.mkdir(parents=True,exist_ok=False)
    for source,relative in [(baseline,Path('baseline')),(restored/'journal',Path('journal')),(closure/'uploads',Path('replica-uploads'))]:
        if not source.exists():continue
        for path in sorted(source.rglob('*')):
            if path.is_symlink():raise ValueError('restored seed contains symlink')
            if path.is_file() and path.name!='.writer.lock':
                out=destination/relative/path.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True);os.link(path,out)
    write_json_durable(destination/'replica-active.json',{'id':'seeded','baseline':str((destination/'baseline').resolve()),
        'journal':str((destination/'journal').resolve()),'baseline_sha256':receipt['current_baseline_sha256']})
    write_json_durable(destination/'replica-files-frozen.json',{'schema':'natlang.diloco-replica-file-freeze/1','nonce':freeze_nonce})
    result=publish_seed(destination,seed,expected_sequence=0,expected_head=head,lease_id='restored-closure',freeze_nonce=freeze_nonce)
    return {'seed':str(Path(seed).resolve()),'receipt_sha256':_digest(Path(seed)/'seed-receipt.json'),'head':head,'value':result}
