"""Explicit bootstrap for the replicated coordinator and its peer.

The supervising controller must independently distribute/verify the baseline,
verify the old primary fence, and preserve independent recovery exports.
"""
from __future__ import annotations
import hashlib
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


def serve_peer(*, baseline, baseline_sha256, store, work_root, host, port, token, max_transaction_bytes, stop):
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
                         machine=machine,prepared=prepared,store_root=store)


def serve_new_primary(*, baseline, baseline_sha256, store, root, host, port, token, peer_url,
                      supervisor_proof, verify_supervisor_fence, expected_primary, expected_workers, stop, journal, sequence, head, session_transition, previous_token, restart=False, gateway=None, allow_http=False, max_generations=3):
    """Start a fresh unique primary only after an explicit supervisor fence.

    After both physical replicas confirm the new generation, local retention
    keeps the active and previous generations plus any leased generations.
    Independent private backup cadence and its receipts are managed separately.
    """
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
    machine,receipt=promote(baseline=baseline,baseline_sha256=baseline_sha256,
        journal=journal,sequence=sequence,head=head,destination=root,
        supervisor_proof=supervisor_proof,verify_supervisor_fence=verify_supervisor_fence,
        expected_primary=expected_primary,expected_workers=expected_workers)
    start_path=store/('supervisor-restart-'+uuid.uuid4().hex+'.json' if restart else 'supervisor-start.json')
    write_json_durable(start_path,{'schema':'natlang.diloco-replica-start/1',
        'baseline_sha256':baseline_sha256,'promotion':receipt,'session_transition':session_transition})
    if not restart:
        write_json_durable(store/'primary-active.json',{'id':'initial','baseline':str(Path(baseline).resolve()),
            'journal':str(journal.root.resolve()),'baseline_sha256':baseline_sha256,'head':_identity(journal.identity)})
    empty=store/'empty-payload';write_bytes_durable(empty,b'')
    writer=ReplicatedExecutor(journal,peer_url,token,allow_http=allow_http,gateway=gateway)
    dispatch=ReplicatedDispatch(machine,writer,empty)
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
        write_json_durable(pointer,{'id':identifier,'baseline':str((directory/'baseline').resolve()),'journal':str((directory/'journal').resolve()),'baseline_sha256':receipt['baseline_sha256'],'head':receipt['new_head'],'previous':receipt['previous']})
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
