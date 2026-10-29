"""Validate supervisor-owned, SHA-bound coordinator activation attestations."""
import hashlib
import json
import time
from pathlib import Path


def load_owned_config(path,expected_sha256):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('owned CLI config hash differs')
    return json.loads(raw)


def verify_attestation(config, *, mode, now=None):
    att=config['attestation'];clock=time.time() if now is None else now
    if (att.get('schema')!='natlang.diloco-replica-supervisor-attestation/1' or att.get('mode')!=mode
            or not isinstance(att.get('verified_at'),(float,int)) or not isinstance(att.get('expires_at'),(float,int))
            or not att['verified_at']<=clock<=att['expires_at'] or att['expires_at']-att['verified_at']>300):
        raise ValueError('supervisor attestation is stale or invalid')
    expected={'run_id':config['run_id'],'baseline_sha256':config['baseline_sha256'],
              'sequence':config['sequence'],'head':config['head'],
              'generation_nonce':config['session_transition']['generation_nonce'],
              'new_url':config['session_transition']['new_url'],
              'new_token_sha256':config['session_transition']['new_token_sha256']}
    if att.get('binding')!=expected:raise ValueError('supervisor attestation binding differs')
    evidence=att.get('evidence',{})
    if (evidence.get('run_id')!=expected['run_id'] or evidence.get('primary_fenced') is not True
            or evidence.get('all_workers_quiesced') is not True or not evidence.get('proof_sha256')
            or evidence.get('primary')!=config['expected_primary']
            or sorted(evidence.get('quiesced_workers') or [])!=sorted(config['expected_workers'])):
        raise ValueError('supervisor fence evidence incomplete')
    if mode=='restart':
        owner=att.get('owner_observation',{})
        if (att.get('owner_process_absent') is not True or att.get('port_closed') is not True
                or owner.get('identity')!=config['expected_primary'] or owner.get('nonce')!=expected['generation_nonce']
                or owner.get('process_absent') is not True or owner.get('port_closed') is not True
                or not isinstance(owner.get('observed_at'),(float,int)) or not 0<=att['verified_at']-owner['observed_at']<=120):
            raise ValueError('restart lacks verified owner death and closed port')
    return evidence
