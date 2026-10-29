"""Explicit bootstrap for the replicated coordinator and its peer.

The supervising controller must independently distribute/verify the baseline,
verify the old primary fence, and preserve independent recovery exports.
"""
from __future__ import annotations
import hashlib
import os
import json
import threading
import uuid
from pathlib import Path
from ...common.jsonio import write_bytes_durable,write_json_durable
from .replica_baseline import verify
from .replica_journal import ReplicaJournal,_identity
from .replica_promotion import recover_receiver,promote
from .replica_generation import load_active,load_prepared,rotate
from .replica_transport import serve as serve_peer_transport,generation_request,witness_head
from .replica_executor import ReplicatedExecutor
from .replica_dispatch import ReplicatedDispatch
from .coordinator import serve_coordinator


def serve_peer(*, baseline, baseline_sha256, store, work_root, host, port, token, max_transaction_bytes, stop, token_path=None):
    store=Path(store);store.mkdir(parents=True,exist_ok=True)
    pointer=store/'replica-active.json'
    if pointer.exists():
        value=json.loads(pointer.read_text())
        expected=verify(value['baseline'],value['baseline_sha256'])['run_id']
        journal,machine,prepared=load_active(store,run_identity=expected,destination=work_root,initial_baseline=baseline,initial_baseline_sha256=baseline_sha256)
    else:
        value=verify(baseline,baseline_sha256);journal=ReplicaJournal(store/'journal0',run_identity=value['run_id'],baseline_sha256=baseline_sha256)
        machine=recover_receiver(baseline=baseline,baseline_sha256=baseline_sha256,journal=journal,destination=work_root)
        prepared=load_prepared(store,journal)
        write_json_durable(pointer,{'schema':'natlang.diloco-replica-active-generation/1','id':'initial',
            'previous':None,'baseline':str(Path(baseline).resolve()),'baseline_sha256':baseline_sha256,
            'journal':str(journal.root.resolve()),'head':_identity(journal.identity),
            'max_payload_bytes':journal.max_payload_bytes,'max_transactions':journal.max_transactions})
    serve_peer_transport(journal,host,port,token,max_bytes=max_transaction_bytes,stop=stop,
                         machine=machine,prepared=prepared,store_root=store,token_path=token_path)


def serve_new_primary(*, baseline, baseline_sha256, store, root, host, port, token, peer_url,
                      supervisor_proof, verify_supervisor_fence, expected_primary, expected_workers, stop, journal, sequence, head, session_transition, previous_token, restart=False, bootstrap=False, file_custody_store=None, file_freeze_nonce=None, gateway=None, allow_http=False, max_generations=3):
    """Start a fresh unique primary only after an explicit supervisor fence.

    After both physical replicas confirm the new generation, local retention
    keeps the active and previous generations plus any leased generations.
    Independent private backup cadence and its receipts are managed separately.
    """
    if bootstrap:
        if restart or not isinstance(session_transition,dict) or session_transition.get('new_token_sha256')!=hashlib.sha256(token.encode()).hexdigest():
            raise ValueError('bootstrap session identity differs')
    elif restart:
        # A restart keeps the exact live session, whether it began as a bootstrap or a promotion; it is compared
        # with the stored start record below and must still name this token.
        if not isinstance(session_transition,dict) or session_transition.get('new_token_sha256')!=hashlib.sha256(token.encode()).hexdigest():
            raise ValueError('restart session identity differs')
    else:
        required={'previous_url','new_url','previous_token_sha256','new_token_sha256','generation_nonce'}
        if (not isinstance(session_transition,dict) or set(session_transition)!=required or
                not all(isinstance(session_transition[k],str) and session_transition[k] for k in required) or
                (not restart and session_transition['previous_url']==session_transition['new_url']) or
                not isinstance(previous_token,str) or not previous_token or
                session_transition['previous_token_sha256']!=hashlib.sha256(previous_token.encode()).hexdigest() or
                (not restart and session_transition['previous_token_sha256']==session_transition['new_token_sha256']) or
                session_transition['new_token_sha256']!=hashlib.sha256(token.encode()).hexdigest() or
                len(session_transition['generation_nonce'])<32):
            raise ValueError('new generation requires distinct bound endpoint and credential')
    # The supervisor supplies the exact independently witnessed sealed head;
    # reconstructing an empty journal here would silently discard earned state.
    value=verify(baseline,baseline_sha256)
    store=Path(store)
    if restart:
        if not store.is_dir():raise ValueError('restart requires the existing primary store')
        active=json.loads((store/'primary-active.json').read_text())
        if (Path(active['journal']).resolve()!=journal.root.resolve() or Path(active['baseline']).resolve()!=Path(baseline).resolve()
                or active['baseline_sha256']!=baseline_sha256):raise ValueError('restart differs from active generation')
        old_session=json.loads((store/'supervisor-start.json').read_text())['session_transition']
        if old_session!=session_transition:raise ValueError('restart changes the live coordinator session')
        peer=witness_head(peer_url,token,allow_http=allow_http,gateway=gateway)
        if peer!={'run_id':value['run_id'],'baseline_sha256':baseline_sha256,'sequence':sequence,'head':head}:
            raise ValueError('restart head differs from current durable peer witness')
        journal.verify(expected_sequence=sequence,expected_head=head)
    else:store.mkdir(parents=True,exist_ok=False)
    if bootstrap:
        evidence=verify_supervisor_fence(supervisor_proof)
        if evidence.get('run_id')!=value['run_id'] or evidence.get('source_verified') is not True:
            raise ValueError('bootstrap source verification differs')
        if value['quorum_since'] or sequence!=0 or journal.sequence!=0 or journal.head!=head:
            raise ValueError('first outer bootstrap requires an empty journal and no pending quorum')
        if witness_head(peer_url,token,allow_http=allow_http,gateway=gateway)!={'run_id':value['run_id'],'baseline_sha256':baseline_sha256,'sequence':0,'head':head}:
            raise ValueError('bootstrap peer does not witness the same empty journal')
        machine=recover_receiver(baseline=baseline,baseline_sha256=baseline_sha256,journal=journal,destination=root)
        receipt={'schema':'natlang.diloco-replica-bootstrap/1','run_id':value['run_id'],'sequence':0,'head':head,
                 'clock_anchor':machine.last_clock,'old_timers':{},'source':evidence}
    else:
        machine,receipt=promote(baseline=baseline,baseline_sha256=baseline_sha256,
            journal=journal,sequence=sequence,head=head,destination=root,
            supervisor_proof=supervisor_proof,verify_supervisor_fence=verify_supervisor_fence,
            expected_primary=expected_primary,expected_workers=expected_workers)
    if restart:
        import shutil
        previous_root=Path(active['work_root']);previous_uploads=previous_root/'uploads'
        if previous_uploads.exists():
            markers=previous_uploads/'.replication-pending'
            if markers.exists():
                from .replica_transport import transmit_file
                for marker in sorted(markers.iterdir()):
                    if marker.is_symlink() or not marker.is_file():raise ValueError('invalid pending file marker')
                    intent=json.loads(marker.read_text());name=intent['name'];sha=intent['sha256']
                    if marker.name!=name or not __import__('re').fullmatch('[A-Za-z0-9][A-Za-z0-9._-]{0,127}',name):raise ValueError('invalid pending file identity')
                    sidecar=previous_uploads/(name+'.sha256');payload=previous_uploads/name
                    if sidecar.is_file() and payload.is_file():
                        if sidecar.read_text().strip()!=sha:raise ValueError('pending file hash binding differs')
                        transmit_file(peer_url,token,name,payload,sha,allow_http=allow_http,gateway=gateway)
                        marker.unlink();fd=os.open(markers,os.O_RDONLY|os.O_DIRECTORY)
                        try:os.fsync(fd)
                        finally:os.close(fd)
            def preserve(source,target):
                if '/.partial/' in str(source) or '/.replication-pending/' in str(source):return shutil.copy2(source,target)
                return os.link(source,target)
            shutil.copytree(previous_uploads,Path(root)/'uploads',copy_function=preserve)
    elif not bootstrap:
        if file_custody_store is None or not isinstance(file_freeze_nonce,str):raise ValueError('promotion requires exact frozen file custody')
        from .replica_seed import restore_replica_uploads
        receipt['restored_uploads']=restore_replica_uploads(file_custody_store,root,freeze_nonce=file_freeze_nonce)
    if restart:
        active['work_root']=str(Path(root).resolve());write_json_durable(store/'primary-active.json',active)
    start_path=store/('supervisor-restart-'+uuid.uuid4().hex+'.json' if restart else 'supervisor-start.json')
    write_json_durable(start_path,{'schema':'natlang.diloco-replica-start/1',
        'baseline_sha256':baseline_sha256,'promotion':receipt,'session_transition':session_transition})
    if not restart:
        write_json_durable(store/'primary-active.json',{'id':'initial','baseline':str(Path(baseline).resolve()),
            'journal':str(journal.root.resolve()),'work_root':str(Path(root).resolve()),'baseline_sha256':baseline_sha256,'head':_identity(journal.identity)})
    empty=store/'empty-payload';write_bytes_durable(empty,b'')
    writer=ReplicatedExecutor(journal,peer_url,token,allow_http=allow_http,gateway=gateway)
    dispatch=ReplicatedDispatch(machine,writer,empty);dispatch.recovery_store=store
    def rollover():
        existing=[p for p in store.glob('generation-*') if p.is_dir()]
        if len(existing)>max_generations:raise ValueError('replica generation retention exceeds its local bound')
        identifier=uuid.uuid4().hex;directory=store/('generation-'+identifier)
        def prepare(previous,clock,wall):
            return generation_request(peer_url,token,'prepare',{'id':identifier,'previous':previous,'captured_monotonic':clock,'captured_wall':wall},allow_http=allow_http,gateway=gateway)
        def commit(intent):
            return generation_request(peer_url,token,'commit',{'id':identifier,'previous':intent['previous'],
                'baseline_sha256':intent['baseline_sha256'],'new_head':intent['new_head']},allow_http=allow_http,gateway=gateway)
        receipt=rotate(dispatch,local_baseline_dir=directory/'baseline',local_journal_dir=directory/'journal',
                      receipt_path=directory/'generation.json',peer_prepare=prepare,peer_commit=commit,within_writer=True)
        pointer=store/'primary-active.json'
        write_json_durable(pointer,{'id':identifier,'baseline':str((directory/'baseline').resolve()),'journal':str((directory/'journal').resolve()),'work_root':str(Path(root).resolve()),'baseline_sha256':receipt['baseline_sha256'],'head':receipt['new_head'],'previous':receipt['previous']})
        confirmation=generation_request(peer_url,token,'retain',{'id':identifier,'previous':receipt['previous'],
            'baseline_sha256':receipt['baseline_sha256'],'new_head':receipt['new_head']},allow_http=allow_http,gateway=gateway)
        from .replica_retention import Retention
        retention=Retention(store);prior=retention.latest()
        retention.register(generation=identifier,baseline_sha256=receipt['baseline_sha256'],head=receipt['new_head'],
            previous=prior,replica_confirmation=confirmation,active_pointer=pointer)
        retention.collect()
        return receipt
    writer.rollover_callback=rollover
    rebase=dispatch.call('clock_rebase',{'old_wall':receipt['clock_anchor']['wall'],'old_monotonic':receipt['clock_anchor']['monotonic'],
        'old_timers':receipt['old_timers']})
    sealed=writer.read(lambda:{'sequence':writer.journal.sequence,'head':writer.journal.head})
    write_json_durable(Path(root)/'replica-session.json',{'run_id':receipt['run_id'],'sequence':receipt['sequence'],
        'head':receipt['head'],'url':session_transition['new_url'],'token_sha256':session_transition['new_token_sha256'],
        'clock_rebase':rebase,'sealed_rebase':sealed,'ready':True})
    try:serve_coordinator(root,host,port,token,stop=stop,replication=dispatch)
    finally:
        writer.close();writer.journal.close()
