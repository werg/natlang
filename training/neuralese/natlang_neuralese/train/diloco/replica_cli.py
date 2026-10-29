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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('role',choices=('peer','promote','restart'));parser.add_argument('--config',required=True);parser.add_argument('--config-sha256',required=True)
    args=parser.parse_args();config=load_owned_config(args.config,args.config_sha256);stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    token=Path(config['token_file']).read_text().strip()
    if not token:raise ValueError('empty replication token')
    common=dict(baseline=config['baseline'],baseline_sha256=config['baseline_sha256'],store=config['store'],
                host=config['host'],port=config['port'],token=token,stop=stop)
    if args.role=='peer':
        return serve_peer(**common,work_root=config['work_root'],max_transaction_bytes=config['max_transaction_bytes'])
    evidence=verify_attestation(config,mode=args.role)
    verify=lambda _proof:evidence
    journal_path=Path(config['journal'])
    identity=json.loads((journal_path/'identity.json').read_text())
    journal=ReplicaJournal(journal_path,run_identity=identity['run_identity'],baseline_sha256=identity['baseline_sha256'],
        max_payload_bytes=identity['max_payload_bytes'],max_transactions=identity['max_transactions'],previous_generation=identity.get('previous_generation'))
    try:
        return serve_new_primary(**common,root=config['work_root'],peer_url=config['peer_url'],
            supervisor_proof=config['supervisor_proof'],verify_supervisor_fence=verify,
            expected_primary=config['expected_primary'],expected_workers=config['expected_workers'],
            journal=journal,sequence=config['sequence'],head=config['head'],session_transition=config['session_transition'],
            previous_token=Path(config['previous_token_file']).read_text().strip(),
            restart=(args.role=='restart'),gateway=config.get('gateway'),allow_http=config.get('allow_http',False),max_generations=config.get('max_generations',3))
    finally:journal.close()


if __name__=='__main__':main()
