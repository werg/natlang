"""Explicit supervisor-fenced promotion; never automatic on a network partition."""
from __future__ import annotations
from pathlib import Path
import json
import time
from ...common.jsonio import write_json_durable
from .replica_baseline import reconstruct
from .replica_state import replay


def promote(*, baseline, baseline_sha256, journal, sequence, head, destination,
            supervisor_proof, verify_supervisor_fence):
    """Require the controller's independently verified old-primary and worker fence.

    `verify_supervisor_fence` must query/validate the actual provider or owned
    supervisor identities. This package cannot infer process death from an RPC
    timeout or a self-asserted local boolean; no default verifier is supplied.
    """
    if not callable(verify_supervisor_fence):raise ValueError('verified supervisor fence is mandatory')
    evidence=verify_supervisor_fence(supervisor_proof)
    if (not isinstance(evidence,dict) or evidence.get('primary_fenced') is not True
            or evidence.get('all_workers_quiesced') is not True
            or not evidence.get('primary_boot_id') or not evidence.get('run_id')
            or not evidence.get('proof_sha256')):
        raise ValueError('primary and worker fence evidence is incomplete')
    if journal.identity.get('baseline_sha256') != baseline_sha256 or journal.identity.get('run_identity') != evidence['run_id']:
        raise ValueError('journal does not bind this baseline and fenced run')
    value=reconstruct(baseline,destination,expected_sha256=baseline_sha256)
    if evidence['run_id']!=value['run_id']:raise ValueError('fence belongs to a different coordinator run')
    timers={int(k):v for k,v in value['quorum_since'].items()}
    machine=replay(journal,destination,expected_sequence=sequence,expected_head=head,quorum_since=timers)
    old_now=value['captured_monotonic']
    if sequence:
        operation=json.loads((journal.root/f'{sequence-1:020d}'/'record.json').read_text())['event']
        old_now=operation['monotonic']
    new_now=time.monotonic();old_timers=dict(machine.coordinator.quorum_since)
    if any(start > old_now for start in old_timers.values()):
        raise ValueError('replica quorum timer is newer than its sealed event clock')
    machine.coordinator.quorum_since={key:new_now-(old_now-start) for key,start in old_timers.items()}
    receipt={'schema':'natlang.diloco-replica-promotion/1','run_id':value['run_id'],
             'timer_rebase':{'old_monotonic':old_now,'new_monotonic':new_now,
                             'old_timers':{str(k):v for k,v in old_timers.items()},
                             'new_timers':{str(k):v for k,v in machine.coordinator.quorum_since.items()},
                             'downtime_counts_toward_grace':False},
             'baseline_sha256':baseline_sha256,'sequence':sequence,'head':head,
             'fence':evidence,'scope':'supervisor-fenced exact journal replay; independent private export remains required'}
    write_json_durable(Path(destination)/'replica-promotion.json',receipt)
    return machine,receipt
