"""Single-owner durable controller journal for fenced replica promotion."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
from natlang_neuralese.common.jsonio import write_json_durable


def run_promotion(directory, *, intent, fence_workers, verify_fence, start_primary, release_workers):
    """Perform one promotion under an exclusive durable controller generation.

    Callback implementations belong to the authorized supervisor. On failure,
    already installed worker fences are deliberately left in place. The new
    primary must acknowledge the exact witnessed WAL head before release.
    """
    required={'run_id','generation_nonce','sequence','head','new_url','new_token_sha256'}
    if not isinstance(intent,dict) or set(intent)!=required or not all(intent.values()) or len(intent['generation_nonce'])<32:
        raise ValueError('invalid promotion intent')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=directory/'promotion.json'
        if journal.exists():raise ValueError('promotion generation already exists; reconcile its journal')
        record={'schema':'natlang.diloco-promotion-controller/1','intent':intent,'phase':'intent'}
        write_json_durable(journal,record)
        fences=fence_workers(intent);record.update(phase='workers-fenced',workers=fences);write_json_durable(journal,record)
        evidence=verify_fence(intent)
        if (evidence.get('primary_fenced') is not True or evidence.get('all_workers_quiesced') is not True
                or evidence.get('run_id')!=intent['run_id'] or not evidence.get('proof_sha256')):
            raise ValueError('invalid supervisor fence evidence')
        record.update(phase='provider-fenced',fence=evidence);write_json_durable(journal,record)
        ready=start_primary(intent,evidence)
        if (ready.get('run_id')!=intent['run_id'] or ready.get('sequence')!=intent['sequence']
                or ready.get('head')!=intent['head'] or ready.get('url')!=intent['new_url']
                or ready.get('token_sha256')!=intent['new_token_sha256'] or ready.get('ready') is not True
                or not isinstance(ready.get('clock_rebase'),dict) or not isinstance(ready.get('sealed_rebase'),dict)
                or type(ready['sealed_rebase'].get('sequence')) is not int or ready['sealed_rebase']['sequence']<2
                or not ready['sealed_rebase'].get('head')):
            raise ValueError('new primary did not prove exact WAL and session identity')
        record.update(phase='primary-ready',primary=ready);write_json_durable(journal,record)
        released=release_workers(intent,fences)
        if (not isinstance(fences,list) or not fences or not isinstance(released,list) or
                sorted(x['sandbox_id'] for x in released)!=sorted(x['sandbox_id'] for x in fences)):
            raise ValueError('worker release did not cover exact fenced owners')
        record.update(phase='released',released=released);write_json_durable(journal,record)
        return record


def run_coreweave_promotion(directory, config):
    """Authorized supervisor adapter using SDK-owned roles and authenticated health.

    The configured role publisher must be the exact SHA-bound installed file;
    this function neither allocates sandboxes nor stops existing learners.
    """
    import time
    try:
        from . import cwctl, diloco_replica_fence as fence_module
    except ImportError:
        import cwctl
        import diloco_replica_fence as fence_module
    from natlang_neuralese.train.diloco.worker import Transport
    intent=config['intent'];workers=config['workers'];nonce=intent['generation_nonce']
    ref=config['supervisor_proof']['journal']
    owner=fence_module.owner_reference(ref['path'],ref['sha256'],config['expected_primary'])
    if owner['run_id']!=intent['run_id']:raise ValueError('owner journal run differs from promotion')
    expected={(w['sandbox_id'],w['boot_id'],w['owner_id']) for w in owner['workers'].values()
              if w['sandbox_id']!=config['expected_primary']['sandbox_id']}
    if {(w['sandbox_id'],w['boot_id'],w['owner_id']) for w in workers}!=expected:
        raise ValueError('worker authority set differs from exact owner journal')
    verifier=fence_module.provider_fence_verifier(expected_primary=config['expected_primary'],
        expected_workers=config['expected_workers'],control_dirs=config['control_dirs'])
    token=Path(config['new_token_file']).read_text().strip()
    if hashlib.sha256(token.encode()).hexdigest()!=intent['new_token_sha256']:
        raise ValueError('new token file does not match promotion intent')
    def install(_):return fence_module.sdk_set_worker_authority(workers,nonce=nonce,action='fence')
    def verify(_):return verifier(config['supervisor_proof'])
    def start(_,evidence):
        role=config['primary_role'];publisher=role['publisher']
        if not Path(publisher['path']).is_absolute() or len(publisher['sha256'])!=64:raise ValueError('invalid role publisher identity')
        program=('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
                 'p=Path(sys.argv[1]);actual=hashlib.sha256(p.read_bytes()).hexdigest()\n'
                 'if actual!=sys.argv[2]:raise ValueError("role publisher hash differs")\n'
                 'm=runpy.run_path(str(p));r=json.loads(sys.argv[4]);'
                 'print(json.dumps(m["submit"](sys.argv[3],r,timeout=60)))\n')
        request=role['request']
        if request.get('action')!='start' or request.get('role')!='coordinator':raise ValueError('promotion requires owned coordinator start')
        fence_module.sdk_authorize_coordinator(role,nonce)
        cwctl.load_env_file();cwctl.require_key();sandbox=cwctl.read_sandbox(role['sandbox_id'],patience=60)
        result=sandbox.exec(['python3','-c',program,publisher['path'],publisher['sha256'],role['directory'],json.dumps(request)],timeout_seconds=90).result(timeout=120)
        if result.returncode!=0 or json.loads(result.stdout).get('accepted') is not True:raise ValueError('coordinator owner did not accept start')
        client=Transport(intent['new_url'],token,5,gateway=config.get('gateway'),allow_http=config.get('allow_http',False))
        deadline=time.monotonic()+config.get('readiness_timeout',120)
        while True:
            try:return client.json('GET','/v1/replica-promotion')
            except OSError:
                if time.monotonic()>=deadline:raise
                time.sleep(1)
    def release(_,fences):return fence_module.sdk_set_worker_authority(workers,nonce=nonce,action='release')
    return run_promotion(directory,intent=intent,fence_workers=install,verify_fence=verify,start_primary=start,release_workers=release)


def prepare_owned_activation(config, *, mode, supervisor_evidence, now, owner_stop=None):
    """Create exact bounded attestation bytes for a SHA-bound owned CLI argv.

    `supervisor_evidence` is the result of the controller's fresh provider and
    worker verifier. Restart additionally requires a fresh nonce-bound owner
    process/port observation; no provider credential is included in the file.
    """
    import copy
    if mode not in ('promote','restart'):raise ValueError('invalid replica activation mode')
    result=copy.deepcopy(config);transition=result['session_transition']
    binding={'run_id':result['run_id'],'baseline_sha256':result['baseline_sha256'],'sequence':result['sequence'],
             'head':result['head'],'generation_nonce':transition['generation_nonce'],'activation_nonce':result['activation_nonce'],'new_url':transition['new_url'],
             'new_token_sha256':transition['new_token_sha256']}
    observed=supervisor_evidence.get('owner_observed_at' if mode=='restart' else 'provider_observed_at')
    if not isinstance(observed,(int,float)) or not 0<=now-observed<=120:raise ValueError('supervisor fence observation is not fresh')
    if not isinstance(binding['activation_nonce'],str) or len(binding['activation_nonce'])<32:raise ValueError('invalid activation nonce')
    if supervisor_evidence.get('run_id')!=binding['run_id'] or supervisor_evidence.get('primary_fenced') is not True or supervisor_evidence.get('all_workers_quiesced') is not True or not supervisor_evidence.get('proof_sha256') or supervisor_evidence.get('primary')!=result['expected_primary'] or sorted(supervisor_evidence.get('quiesced_workers') or [])!=sorted(result['expected_workers']):
        raise ValueError('unverified supervisor fence')
    att={'schema':'natlang.diloco-replica-supervisor-attestation/1','mode':mode,'verified_at':now,'expires_at':now+300,
         'binding':binding,'evidence':supervisor_evidence}
    if mode=='restart':
        if (not isinstance(owner_stop,dict) or owner_stop.get('nonce')!=binding['activation_nonce'] or
                owner_stop.get('process_absent') is not True or owner_stop.get('port_closed') is not True or
                owner_stop.get('identity')!=result['expected_primary'] or not isinstance(owner_stop.get('observed_at'),(int,float)) or
                not 0<=now-owner_stop['observed_at']<=120):raise ValueError('unverified old role owner stop')
        att.update(owner_process_absent=True,port_closed=True,owner_observation=owner_stop)
    result['attestation']=att
    raw=json.dumps(result,sort_keys=True,separators=(',',':')).encode()
    return raw,hashlib.sha256(raw).hexdigest()
