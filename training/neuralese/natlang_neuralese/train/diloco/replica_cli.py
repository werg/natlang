"""Supervised replica peer and fenced promotion entrypoints.

The operator supplies immutable baseline, journal and owner bindings. A
promotion never creates a substitute journal or initiates a worker stop.
"""
from __future__ import annotations
import argparse
import json
import signal
import threading
from pathlib import Path
from .replica_journal import ReplicaJournal
from .replica_service import serve_new_primary,serve_peer
from .replica_attestation import load_owned_config,verify_attestation


def gateway_credential(config):
    if config.get('gateway_file') and config.get('gateway'):
        raise ValueError('replica gateway must have one credential source')
    value=Path(config['gateway_file']).read_text().strip() if config.get('gateway_file') else config.get('gateway')
    if config.get('gateway_file') and not value:raise ValueError('empty replica gateway token')
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('role',choices=('peer','bootstrap','promote','restart','seed-freeze','seed-publish','seed-serve','seed-fetch','seed-install','seed-release','seed-restored','replica-loss-recover'));parser.add_argument('--config',required=True);parser.add_argument('--config-sha256',required=True)
    args=parser.parse_args();config=load_owned_config(args.config,args.config_sha256);stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    if args.role=='seed-restored':
        from .replica_recovery import prepare_restored_peer_seed
        value=prepare_restored_peer_seed(closure=config['closure'],restored=config['restored'],destination=config['destination'],seed=config['seed'],receipt_sha256=config['receipt_sha256'],freeze_nonce=config['activation_nonce'])
        print(json.dumps(value,sort_keys=True));return
    if args.role.startswith('seed-'):
        from .replica_seed import publish_seed,serve_seed,download_seed,install_seed,release_seed_lease
        if args.role=='seed-publish':
            value=publish_seed(config['store'],config['destination'],expected_sequence=config['sequence'],expected_head=config['head'],lease_id=config['lease_id'],freeze_nonce=config['freeze_nonce'])
            print(json.dumps(value,sort_keys=True));return
        if args.role=='seed-install':
            value=install_seed(config['seed'],config['store'],expected_receipt_sha256=config['receipt_sha256']);print(json.dumps(value,sort_keys=True));return
        if args.role=='seed-freeze':
            from .replica_transport import freeze_replica_files
            seed_token=Path(config['token_file']).read_text().strip()
            print(json.dumps(freeze_replica_files(config['peer_url'],seed_token,config['freeze_nonce'],gateway=(Path(config['gateway_file']).read_text().strip() if config.get('gateway_file') else config.get('gateway')),allow_http=config.get('allow_http',False)),sort_keys=True));return
        if args.role=='seed-release':
            import hashlib
            from .replica_transport import witness_head
            raw=Path(config['seed_receipt']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=config['receipt_sha256']:raise ValueError('seed receipt hash differs')
            record=json.loads(raw);peer_token=Path(config['token_file']).read_text().strip()
            witness=witness_head(config['peer_url'],peer_token,gateway=(Path(config['gateway_file']).read_text().strip() if config.get('gateway_file') else None),allow_http=config.get('allow_http',False))
            release_seed_lease(config['store'],record,witnessed=witness);return
        seed_token=Path(config['token_file']).read_text().strip()
        if args.role=='seed-serve':return serve_seed(config['seed'],config['host'],config['port'],seed_token,stop=stop)
        value=download_seed(config['url'],seed_token,config['destination'],expected_receipt_sha256=config['receipt_sha256'],
            gateway=(Path(config['gateway_file']).read_text().strip() if config.get('gateway_file') else None),allow_http=config.get('allow_http',False));print(json.dumps(value,sort_keys=True));return
    if args.role=='replica-loss-recover':
        from .replica_promotion import recover_after_replica_loss
        evidence=verify_attestation(config,mode=args.role)
        identity=json.loads((Path(config['journal'])/'identity.json').read_text())
        journal=ReplicaJournal(config['journal'],run_identity=identity['run_identity'],baseline_sha256=identity['baseline_sha256'],
            max_payload_bytes=identity['max_payload_bytes'],max_transactions=identity['max_transactions'],previous_generation=identity.get('previous_generation'))
        try:
            value=recover_after_replica_loss(baseline=config['baseline'],baseline_sha256=config['baseline_sha256'],journal=journal,
                sequence=config['sequence'],head=config['head'],destination=config['destination'],verified_owner_evidence=evidence,
                source_work_root=config['source_work_root'],source_primary_store=config['source_primary_store'],freeze_nonce=config['activation_nonce'])
            print(json.dumps(value,sort_keys=True));return
        finally:journal.close()
    token=Path(config['token_file']).read_text().strip()
    if not token:raise ValueError('empty replication token')
    gateway=gateway_credential(config)
    common=dict(baseline=config['baseline'],baseline_sha256=config['baseline_sha256'],store=config['store'],
                host=config['host'],port=config['port'],token=token,stop=stop)
    if args.role=='peer':
        return serve_peer(**common,work_root=config['work_root'],max_transaction_bytes=config['max_transaction_bytes'],token_path=config['token_file'])
    evidence=verify_attestation(config,mode=args.role)
    verify=lambda _proof:evidence
    if args.role=='restart':
        from .replica_generation import reconcile_primary_pointer
        from .replica_transport import witness_head
        witness=witness_head(config['peer_url'],token,allow_http=config.get('allow_http',False),gateway=gateway)
        selected=reconcile_primary_pointer(config['store'],witness)
        if (selected['baseline_sha256']!=config['baseline_sha256'] or Path(selected['baseline']).resolve()!=Path(config['baseline']).resolve()
                or Path(selected['journal']).resolve()!=Path(config['journal']).resolve() or witness['sequence']!=config['sequence'] or witness['head']!=config['head']):
            raise ValueError('restart config does not bind the reconciled latest peer head')
    journal_path=Path(config['journal'])
    identity=json.loads((journal_path/'identity.json').read_text())
    journal=ReplicaJournal(journal_path,run_identity=identity['run_identity'],baseline_sha256=identity['baseline_sha256'],
        max_payload_bytes=identity['max_payload_bytes'],max_transactions=identity['max_transactions'],previous_generation=identity.get('previous_generation'))
    try:
        return serve_new_primary(**common,root=config['work_root'],peer_url=config['peer_url'],
            supervisor_proof=config['supervisor_proof'],verify_supervisor_fence=verify,
            expected_primary=config.get('expected_primary'),expected_workers=config.get('expected_workers'),
            journal=journal,sequence=config['sequence'],head=config['head'],session_transition=config['session_transition'],
            previous_token=(Path(config['previous_token_file']).read_text().strip() if args.role=='promote' else None),
            restart=(args.role=='restart'),bootstrap=(args.role=='bootstrap'),file_custody_store=config.get('file_custody_store'),file_freeze_nonce=config.get('file_freeze_nonce'),gateway=gateway,allow_http=config.get('allow_http',False),max_generations=config.get('max_generations',3))
    finally:journal.close()


if __name__=='__main__':main()
