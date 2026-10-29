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
