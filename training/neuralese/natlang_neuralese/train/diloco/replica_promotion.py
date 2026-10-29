"""Explicit supervisor-fenced promotion; never automatic on a network partition."""
from __future__ import annotations
from pathlib import Path
import json
import time
from ...common.jsonio import write_json_durable
from .replica_baseline import reconstruct
from .replica_state import replay


def validate_fence(supervisor_proof, verify_supervisor_fence, expected_primary, expected_workers):
    if not callable(verify_supervisor_fence):raise ValueError('verified supervisor fence is mandatory')
    evidence=verify_supervisor_fence(supervisor_proof)
    if (not isinstance(expected_primary,dict) or not all(expected_primary.get(k) for k in ('boot_id','pid','start_ticks'))
            or not isinstance(expected_workers,list) or not expected_workers or len(set(expected_workers)) != len(expected_workers)):
        raise ValueError('expected primary and worker identities are required')
    if (not isinstance(evidence,dict) or evidence.get('primary_fenced') is not True
            or evidence.get('all_workers_quiesced') is not True
            or evidence.get('primary') != expected_primary
            or sorted(evidence.get('quiesced_workers') or []) != sorted(expected_workers)
            or not evidence.get('run_id') or not evidence.get('proof_sha256')):
        raise ValueError('primary and worker fence evidence is incomplete')
    return evidence


def promote(*, baseline, baseline_sha256, journal, sequence, head, destination,
            supervisor_proof, verify_supervisor_fence, expected_primary, expected_workers):
    """Require the controller's independently verified old-primary and worker fence.

    `verify_supervisor_fence` must query/validate the actual provider or owned
    supervisor identities. This package cannot infer process death from an RPC
    timeout or a self-asserted local boolean; no default verifier is supplied.
    """
    evidence=validate_fence(supervisor_proof,verify_supervisor_fence,expected_primary,expected_workers)
    if journal.identity.get('baseline_sha256') != baseline_sha256 or journal.identity.get('run_identity') != evidence['run_id']:
        raise ValueError('journal does not bind this baseline and fenced run')
    value=reconstruct(baseline,destination,expected_sha256=baseline_sha256)
    if evidence['run_id']!=value['run_id']:raise ValueError('fence belongs to a different coordinator run')
    if sorted(expected_workers)!=list(range(int(json.loads((Path(destination)/'manifest.json').read_text())['workers']))):
        raise ValueError('fence does not cover every coordinator worker')
    timers={int(k):v for k,v in value['quorum_since'].items()}
    machine=replay(journal,destination,expected_sequence=sequence,expected_head=head,quorum_since=timers,
                   clock_anchor={'wall':value['captured_wall'],'monotonic':value['captured_monotonic']})
    anchor=machine.last_clock
    if anchor is None:raise ValueError('promotion lacks an authenticated clock anchor')
    receipt={'schema':'natlang.diloco-replica-promotion/1','run_id':value['run_id'],
             'clock_anchor':anchor,'old_timers':{str(k):v for k,v in sorted(machine.coordinator.quorum_since.items())},
             'baseline_sha256':baseline_sha256,'sequence':sequence,'head':head,
             'fence':evidence,'scope':'supervisor-fenced exact journal replay; clock rebase must be sealed before publication'}
    write_json_durable(Path(destination)/'replica-promotion.json',receipt)
    return machine,receipt


def recover_receiver(*, baseline, baseline_sha256, journal, destination):
    """Rebuild a replica's sealed state; retain a single unsealed tail for retry.

    This does not authorize coordinator promotion. The returned machine may
    only serve the authenticated replica endpoint until a supervisor fence.
    """
    if journal.identity.get('baseline_sha256') != baseline_sha256:
        raise ValueError('receiver journal does not bind this baseline')
    value=reconstruct(baseline,destination,expected_sha256=baseline_sha256)
    if journal.identity.get('run_identity') != value['run_id']:
        raise ValueError('receiver journal belongs to another run')
    timers={int(k):v for k,v in value['quorum_since'].items()}
    return replay(journal,destination,expected_sequence=journal.sequence,expected_head=journal.head,
                  quorum_since=timers,receiver_restart=True,clock_anchor={'wall':value['captured_wall'],'monotonic':value['captured_monotonic']})


def recover_after_replica_loss(*, baseline, baseline_sha256, journal, sequence, head, destination, verified_owner_evidence, source_work_root, source_primary_store, freeze_nonce):
    """Preserve locally sealed updates after a fenced replica disappears.

    The controller must first verify the old primary owner is stopped, the
    replica host is fenced, and workers are quiesced. The original journal,
    including any unsealed operation, is retained unchanged as custody.
    """
    from .replica_baseline import capture,verify
    from .replica_journal import ReplicaJournal,_identity
    if (not isinstance(verified_owner_evidence,dict) or verified_owner_evidence.get('primary_process_absent') is not True
            or verified_owner_evidence.get('replica_fenced') is not True or verified_owner_evidence.get('all_workers_quiesced') is not True
            or not verified_owner_evidence.get('proof_sha256')):
        raise ValueError('replica loss recovery requires exact fenced owner evidence')
    value=verify(baseline,baseline_sha256)
    if verified_owner_evidence.get('run_id')!=value['run_id'] or journal.identity.get('baseline_sha256')!=baseline_sha256:
        raise ValueError('replica loss custody differs from baseline')
    journal.verify(expected_sequence=sequence,expected_head=head)
    sealed_sequence=sequence-(sequence%2)
    sealed_head=(_identity(journal.identity) if not sealed_sequence else
        _identity(json.loads((journal.root/f'{sealed_sequence:020d}'/'record.json').read_text())))
    target=Path(destination);target.mkdir(parents=True,exist_ok=False)
    if not isinstance(freeze_nonce,str) or len(freeze_nonce)<32:raise ValueError('replica loss file freeze nonce is invalid')
    source_root=Path(source_work_root).resolve(strict=True)
    if not source_root.is_dir():raise ValueError('old primary work root is not a directory')
    source_store=Path(source_primary_store).resolve(strict=True)
    previous_start=json.loads((source_store/'supervisor-start.json').read_text())
    if previous_start.get('schema')!='natlang.diloco-replica-start/1' or previous_start.get('promotion',{}).get('run_id')!=value['run_id'] or not previous_start.get('session_transition'):
        raise ValueError('old primary session custody differs')

    uploads=source_root/'uploads';target_uploads=target/'replica-uploads';files=[]
    if uploads.exists():
        target_uploads.mkdir()
        for item in sorted(uploads.iterdir()):
            if item.name.startswith('.'):continue  # original partial files/markers remain in source custody
            if item.is_symlink() or not item.is_file():raise ValueError('invalid local file custody')
            if item.name.endswith('.sha256'):
                source=item.with_name(item.name[:-7])
                if not source.is_file():raise ValueError('local upload payload missing')
                with source.open('rb') as stream:digest=__import__('hashlib').file_digest(stream,'sha256').hexdigest()
                if digest!=item.read_text().strip():raise ValueError('local upload hash differs')
            elif not item.with_name(item.name+'.sha256').is_file():raise ValueError('local upload is unsealed')
            __import__('os').link(item,target_uploads/item.name);files.append(item.name)
    write_json_durable(target/'replica-files-frozen.json',{'schema':'natlang.diloco-replica-file-freeze/1','nonce':freeze_nonce})
    reconstruct(baseline,target/'recovered',expected_sha256=baseline_sha256)
    machine=replay(journal,target/'recovered',expected_sequence=sequence,expected_head=head,
        quorum_since={int(k):v for k,v in value['quorum_since'].items()},receiver_restart=True,
        clock_anchor={'wall':value['captured_wall'],'monotonic':value['captured_monotonic']})
    class Immediate:
        def __init__(self,machine):self.machine=machine
        def read(self,fn):return fn()
    anchor=machine.last_clock
    captured=capture(Immediate(machine),target/'baseline',captured_wall=anchor['wall'],captured_monotonic=anchor['monotonic'])
    previous={'sequence':sealed_sequence,'head':sealed_head}
    new=ReplicaJournal(target/'journal',run_identity=value['run_id'],baseline_sha256=captured['sha256'],previous_generation=previous,
        max_payload_bytes=journal.max_payload_bytes,max_transactions=journal.max_transactions)
    try:
        new_head=new.head
        receipt={'schema':'natlang.diloco-replica-loss-recovery/1','run_id':value['run_id'],'source_baseline_sha256':baseline_sha256,
            'source_journal':str(journal.root.resolve()),'source_sequence':sequence,'source_head':head,
            'sealed_predecessor':previous,'unsealed_tail_retained':bool(sequence%2),
            'new_baseline_sha256':captured['sha256'],'new_head':new_head,'owner_evidence':verified_owner_evidence,
            'source_work_root':str(source_root),'retained_complete_files':files,'partial_source_retained':True}
        write_json_durable(target/'recovery.json',receipt)
        write_json_durable(target/'replica-active.json',{'schema':'natlang.diloco-replica-active-generation/1','id':'seeded',
            'previous':previous,'baseline':str((target/'baseline').resolve()),'baseline_sha256':captured['sha256'],
            'journal':str((target/'journal').resolve()),'head':new_head,'max_payload_bytes':journal.max_payload_bytes,'max_transactions':journal.max_transactions})
        write_json_durable(target/'primary-active.json',{'id':'seeded','baseline':str((target/'baseline').resolve()),
            'journal':str((target/'journal').resolve()),'work_root':str(source_root),'baseline_sha256':captured['sha256'],'head':new_head,'previous':previous})
        write_json_durable(target/'supervisor-start.json',previous_start)
        return receipt
    finally:new.close()
