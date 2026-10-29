"""Supervisor-controlled, locally pinned replica journal generation change."""
from __future__ import annotations
import json
from pathlib import Path
from ...common.jsonio import write_json_durable
from .replica_baseline import capture
from .replica_journal import ReplicaJournal


def prepare_peer_generation(machine, journal, baseline_dir, journal_dir, *, previous, captured_monotonic, captured_wall):
    """Pin a peer's locally materialized state; retain the previous journal."""
    journal.verify(expected_sequence=previous['sequence'],expected_head=previous['head'])
    if previous['sequence'] % 2:raise ValueError('cannot rotate an unsealed coordinator operation')
    class Immediate:
        def __init__(self):self.machine=machine
        def read(self,fn):return fn()
    baseline=capture(Immediate(),baseline_dir,captured_monotonic=captured_monotonic,captured_wall=captured_wall)
    next_journal=ReplicaJournal(journal_dir,run_identity=journal.identity['run_identity'],
        baseline_sha256=baseline['sha256'],previous_generation=previous,
        max_payload_bytes=journal.max_payload_bytes,max_transactions=journal.max_transactions)
    return baseline,next_journal


def rotate(dispatch, *, local_baseline_dir, local_journal_dir, receipt_path, peer_prepare, peer_commit, within_writer=False):
    """Change generation while the serialized executor prevents concurrent mutation.

    Callbacks are controller/transport operations and run without coordinator
    locks. They must keep old data and fail closed on any ambiguous commit.
    """
    executor=dispatch.executor
    def barrier():
        old=executor.journal;previous={'sequence':old.sequence,'head':old.head}
        old.verify(expected_sequence=previous['sequence'],expected_head=previous['head'])
        if previous['sequence']%2:raise ValueError('cannot rotate unsealed journal')
        clock=__import__('time').monotonic();wall=__import__('time').time()
        if previous['sequence']:
            op=json.loads((old.root/f"{previous['sequence']-1:020d}"/'record.json').read_text())['event']
            clock=op['monotonic'];wall=op['wall']
        local,next_journal=prepare_peer_generation(dispatch.machine,old,local_baseline_dir,local_journal_dir,
            previous=previous,captured_monotonic=clock,captured_wall=wall)
        peer=peer_prepare(previous,clock,wall)
        if peer.get('baseline_sha256')!=local['sha256'] or peer.get('new_head')!=next_journal.head:
            next_journal.close();raise ValueError('peer baseline or new generation differs')
        intent={'schema':'natlang.diloco-replica-generation/1','state':'prepared','previous':previous,
                'baseline_sha256':local['sha256'],'baseline':str(Path(local_baseline_dir).resolve()),
                'journal':str(Path(local_journal_dir).resolve()),'new_head':next_journal.head,'captured_monotonic':clock,'captured_wall':wall}
        write_json_durable(receipt_path,intent)
        confirmed=peer_commit(intent)
        if confirmed != {'baseline_sha256':local['sha256'],'new_head':next_journal.head}:
            next_journal.close();raise ValueError('peer generation commit is ambiguous')
        executor.journal=next_journal
        write_json_durable(receipt_path,{**intent,'state':'committed'})
        old.close()
        return {**intent,'state':'committed'}
    def fenced_barrier():
        try:return barrier()
        except BaseException as exc:
            executor.failure=exc
            raise
    return fenced_barrier() if within_writer else executor.read(fenced_barrier)


def load_prepared(store_root, current):
    """Recover a prepared, uncommitted peer generation for idempotent retry."""
    from .replica_baseline import verify
    path=Path(store_root)/'replica-prepared.json'
    if not path.exists():return None
    record=json.loads(path.read_text())
    if record.get('schema')!='natlang.diloco-replica-prepared-generation/1':raise ValueError('invalid prepared generation receipt')
    request=record['request'];reply=record['reply']
    pointer=Path(store_root)/'replica-active.json'
    if pointer.exists() and json.loads(pointer.read_text()).get('id')==request.get('id'):
        return None  # Committed pointer is authoritative; stale prepare is harmless.
    if request.get('previous')!={'sequence':current.sequence,'head':current.head}:
        raise ValueError('prepared generation no longer follows current head')
    root=Path(store_root).resolve();baseline=Path(record['baseline']).resolve();new_path=Path(record['journal']).resolve()
    if not baseline.is_relative_to(root) or not new_path.is_relative_to(root):raise ValueError('prepared generation escapes storage')
    verify(baseline,reply['baseline_sha256'])
    new=ReplicaJournal(new_path,run_identity=current.identity['run_identity'],baseline_sha256=reply['baseline_sha256'],
        previous_generation=request['previous'],max_payload_bytes=current.max_payload_bytes,max_transactions=current.max_transactions)
    if new.sequence!=0 or new.head!=reply['new_head']:
        new.close();raise ValueError('prepared generation is not its initial head')
    return {'request':request,'reply':reply,'baseline':str(baseline),'journal':new}


def load_active(store_root, *, run_identity, destination, initial_baseline=None, initial_baseline_sha256=None):
    """Recover a committed peer generation without interpreting a timeout as a fence."""
    from .replica_promotion import recover_receiver
    store=Path(store_root).resolve();pointer=json.loads((store/'replica-active.json').read_text())
    if pointer.get('schema')!='natlang.diloco-replica-active-generation/1':raise ValueError('invalid active generation pointer')
    baseline=Path(pointer['baseline']).resolve();path=Path(pointer['journal']).resolve()
    external_initial=(pointer.get('id')=='initial' and pointer.get('previous') is None and initial_baseline is not None
                      and baseline==Path(initial_baseline).resolve() and pointer.get('baseline_sha256')==initial_baseline_sha256)
    if (not baseline.is_relative_to(store) and not external_initial) or not path.is_relative_to(store):
        raise ValueError('active generation escapes storage or differs from the pinned initial baseline')
    journal=ReplicaJournal(path,run_identity=run_identity,baseline_sha256=pointer['baseline_sha256'],previous_generation=pointer['previous'],
        max_payload_bytes=pointer['max_payload_bytes'],max_transactions=pointer['max_transactions'])
    from .replica_journal import _identity
    if _identity(journal.identity)!=pointer['head']:
        journal.close();raise ValueError('active generation initial head differs')
    machine=recover_receiver(baseline=baseline,baseline_sha256=pointer['baseline_sha256'],journal=journal,destination=destination)
    return journal,machine,load_prepared(store,journal)
