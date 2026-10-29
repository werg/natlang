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
    required={'run_id','generation_nonce','activation_nonce','sequence','head','new_url','new_token_sha256'}
    if (not isinstance(intent,dict) or set(intent)!=required or type(intent.get('sequence')) is not int or intent['sequence']<0 or
            not all(intent.get(k) for k in required-{'sequence'}) or len(intent['generation_nonce'])<32 or len(intent['activation_nonce'])<32):
        raise ValueError('invalid promotion intent')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=directory/'promotion.json'
        if journal.exists():
            record=json.loads(journal.read_text())
            if record.get('schema')!='natlang.diloco-promotion-controller/1' or record.get('intent')!=intent:
                raise ValueError('promotion journal belongs to another generation')
        else:
            record={'schema':'natlang.diloco-promotion-controller/1','intent':intent,'phase':'intent'}
            write_json_durable(journal,record)
        phase=record.get('phase')
        if phase not in ('intent','workers-fenced','provider-fenced','primary-ready','released'):raise ValueError('invalid promotion phase')
        if phase=='released':return record
        if phase=='intent':
            fences=fence_workers(intent);record.update(phase='workers-fenced',workers=fences);write_json_durable(journal,record)
        fences=record['workers']
        if record['phase'] in ('workers-fenced','provider-fenced'):
            evidence=verify_fence(intent)
            if (evidence.get('primary_fenced') is not True or evidence.get('all_workers_quiesced') is not True
                    or evidence.get('run_id')!=intent['run_id'] or not evidence.get('proof_sha256')):
                raise ValueError('invalid supervisor fence evidence')
            record.update(phase='provider-fenced',fence=evidence);write_json_durable(journal,record)
        # The adapter must recognize an already-running exact promoted role and
        # return its authenticated readiness on retries, rather than launch it twice.
        ready=start_primary(intent,record['fence'])
        if (ready.get('run_id')!=intent['run_id'] or ready.get('sequence')!=intent['sequence']
                or ready.get('head')!=intent['head'] or ready.get('url')!=intent['new_url']
                or ready.get('token_sha256')!=intent['new_token_sha256'] or ready.get('ready') is not True
                or not isinstance(ready.get('clock_rebase'),dict) or not isinstance(ready.get('sealed_rebase'),dict)
                or type(ready['sealed_rebase'].get('sequence')) is not int or ready['sealed_rebase']['sequence']<2
                or not ready['sealed_rebase'].get('head')):
            raise ValueError('new primary did not prove exact WAL and session identity')
        if record['phase']=='primary-ready' and record['primary']!=ready:raise ValueError('promoted primary readiness changed')
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
    intent=config['intent'];workers=config['workers'];nonce=intent['activation_nonce']
    ref=config['supervisor_proof']['journal']
    owner=fence_module.owner_reference(ref['path'],ref['sha256'],config['expected_primary'])
    if owner['run_id']!=intent['run_id']:raise ValueError('owner journal run differs from promotion')
    expected={(w['sandbox_id'],w['boot_id'],w['owner_id']) for w in owner['workers'].values()
              if w['sandbox_id']!=config['expected_primary']['sandbox_id']}
    actual={(w['sandbox_id'],w['boot_id'],w['owner_id']) for w in workers}
    if not expected<=actual:raise ValueError('worker authority omits an exact surviving owner')
    extra=actual-expected;replacement_proof=None
    if extra:
        replacement=config.get('replacement_owner')
        if len(extra)!=1 or not isinstance(replacement,dict):raise ValueError('unverified replacement owner')
        target=replacement['target'];identity=(target['sandbox_id'],target['boot_id'],target['owner_id'])
        if extra!={identity}:raise ValueError('replacement owner differs from authority set')
        replacement_proof=fence_module.verify_replacement_owner(replacement['reference'],owner_journal=owner,expected_primary=config['expected_primary'],
            expected_target=target,worker_index=replacement['worker_index'],nonce=nonce,get_sandbox=fence_module.sdk_get_sandbox,observe_owner=fence_module.sdk_observe_replacement_owner)
    verifier=fence_module.provider_fence_verifier(expected_primary=config['expected_primary'],
        expected_workers=config['expected_workers'],control_dirs=config['control_dirs'])
    token=Path(config['new_token_file']).read_text().strip()
    if hashlib.sha256(token.encode()).hexdigest()!=intent['new_token_sha256']:
        raise ValueError('new token file does not match promotion intent')
    def install(_):return fence_module.sdk_set_worker_authority(workers,nonce=nonce,action='fence')
    def verify(_):
        evidence=verifier(config['supervisor_proof'])
        if replacement_proof is not None:
            evidence={**evidence,'replacement_owner':replacement_proof}
            evidence['proof_sha256']=hashlib.sha256(json.dumps({k:v for k,v in evidence.items() if k!='proof_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return evidence
    def start(_,evidence):
        from natlang_neuralese.train.diloco.replica_transport import witness_head,rotate_replica_credential
        from natlang_neuralese.train.diloco.worker import RemoteError
        client=Transport(intent['new_url'],token,5,gateway=config.get('gateway'),allow_http=config.get('allow_http',False))
        try:existing=client.json('GET','/v1/replica-promotion')
        except OSError:existing=None
        except RemoteError as exc:
            if exc.status not in (404,503):raise
            existing=None
        if existing is not None:
            if (existing.get('run_id')!=intent['run_id'] or existing.get('sequence')!=intent['sequence'] or
                    existing.get('head')!=intent['head'] or existing.get('token_sha256')!=intent['new_token_sha256']):
                raise ValueError('existing promoted coordinator identity differs')
            fence_module.sdk_confirm_promoted_role(config['primary_role'],nonce)
            return existing
        replica=config['outgoing_replica'];old_token=Path(replica['old_token_file']).read_text().strip()
        expected={'run_id':intent['run_id'],'baseline_sha256':replica['baseline_sha256'],'sequence':intent['sequence'],'head':intent['head']}
        try:witnessed=witness_head(replica['url'],token,allow_http=config.get('allow_http',False),gateway=config.get('gateway'))
        except RemoteError as exc:
            if exc.status!=403:raise
            witnessed=None
        if witnessed is None:
            witnessed=witness_head(replica['url'],old_token,allow_http=config.get('allow_http',False),gateway=config.get('gateway'))
            if witnessed!=expected:raise ValueError('outgoing replica does not witness exact promoted head')
            rotate_replica_credential(replica['url'],old_token,token,sequence=intent['sequence'],head=intent['head'],nonce=nonce,
                allow_http=config.get('allow_http',False),gateway=config.get('gateway'))
        elif witnessed!=expected:raise ValueError('rotated replica does not witness exact promoted head')
        # The surviving receiver owns the journal lock. Stop only that exact
        # role after its replacement has witnessed the head, before promotion.
        fence_module.sdk_stop_owned_role(config['promoted_replica_role'])
        role=dict(config['primary_role']);publisher=role['publisher']
        if not Path(publisher['path']).is_absolute() or len(publisher['sha256'])!=64:raise ValueError('invalid role publisher identity')
        cwctl.load_env_file();cwctl.require_key();sandbox=cwctl.read_sandbox(role['sandbox_id'],patience=60)
        status=sandbox.exec(['cat',role['directory'].rstrip('/')+'/state.json'],timeout_seconds=30).result(timeout=60)
        if status.returncode!=0:raise ValueError('promoted owner state unavailable')
        owner_state=json.loads(status.stdout)
        if owner_state.get('owner_id')!=role['owner_id'] or owner_state.get('boot_id')!=role['boot_id'] or owner_state.get('pending_start'):
            raise ValueError('promoted owner identity changed')
        activation=dict(config['activation_config'])
        if (activation.get('run_id')!=intent['run_id'] or activation.get('sequence')!=intent['sequence'] or activation.get('head')!=intent['head'] or
                activation.get('activation_nonce')!=nonce or activation.get('file_freeze_nonce')!=nonce):
            raise ValueError('activation does not bind exact promotion custody')
        raw,activation_sha=prepare_owned_activation(activation,mode='promote',supervisor_evidence=evidence,now=time.time())
        import uuid
        directory=role['stage_directory'].rstrip('/')+'/promotion-'+uuid.uuid4().hex
        made=sandbox.exec(['mkdir','-p',directory],timeout_seconds=30).result(timeout=60)
        if made.returncode!=0:raise ValueError('owned activation directory failed')
        activation_path=directory+'/activation.json';sandbox.write_file(activation_path,raw).result(timeout=60)
        request=dict(role['request']);request.update(owner_id=role['owner_id'],expected_generation=owner_state['generation'],promotion_nonce=nonce,
            argv=[role['python'],'-m','natlang_neuralese.train.diloco.replica_cli','promote','--config',activation_path,'--config-sha256',activation_sha])
        if any('COREWEAVE' in key.upper() or key.upper().startswith('CW_') for key in request.get('env',{})):
            raise ValueError('provider credential must not enter owned coordinator environment')
        role['request']=request
        program=('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
                 'p=Path(sys.argv[1]);actual=hashlib.sha256(p.read_bytes()).hexdigest()\n'
                 'if actual!=sys.argv[2]:raise ValueError("role publisher hash differs")\n'
                 'm=runpy.run_path(str(p));r=json.loads(sys.argv[4]);'
                 'print(json.dumps(m["submit"](sys.argv[3],r,timeout=60)))\n')
        if request.get('action')!='start' or request.get('role')!='coordinator':raise ValueError('promotion requires owned coordinator start')
        fence_module.sdk_authorize_coordinator(role,nonce)
        result=sandbox.exec(['python3','-c',program,publisher['path'],publisher['sha256'],role['directory'],json.dumps(request)],timeout_seconds=90).result(timeout=120)
        if result.returncode!=0 or json.loads(result.stdout).get('accepted') is not True:raise ValueError('coordinator owner did not accept start')
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
    if mode not in ('bootstrap','promote','restart','replica-loss-recover'):raise ValueError('invalid replica activation mode')
    result=copy.deepcopy(config);transition=result['session_transition']
    binding={'run_id':result['run_id'],'baseline_sha256':result['baseline_sha256'],'sequence':result['sequence'],
             'head':result['head'],'generation_nonce':transition['generation_nonce'],'activation_nonce':result['activation_nonce'],'new_url':transition['new_url'],
             'new_token_sha256':transition['new_token_sha256']}
    observed=supervisor_evidence.get('source_verified_at' if mode=='bootstrap' else 'owner_observed_at' if mode=='restart' else 'replica_observed_at' if mode=='replica-loss-recover' else 'provider_observed_at')
    if not isinstance(observed,(int,float)) or not 0<=now-observed<=120:raise ValueError('supervisor observation is not fresh')
    if not isinstance(binding['activation_nonce'],str) or len(binding['activation_nonce'])<32:raise ValueError('invalid activation nonce')
    if mode=='bootstrap':
        if (supervisor_evidence.get('run_id')!=binding['run_id'] or supervisor_evidence.get('source_verified') is not True or
                not supervisor_evidence.get('bootstrap_source_sha256') or not supervisor_evidence.get('source_receipt_sha256')):
            raise ValueError('bootstrap source is not verified')
    elif mode=='replica-loss-recover':
        owner=supervisor_evidence.get('owner_evidence',{}).get('owner_observation',{})
        if (supervisor_evidence.get('run_id')!=binding['run_id'] or supervisor_evidence.get('primary_process_absent') is not True or
                supervisor_evidence.get('replica_fenced') is not True or supervisor_evidence.get('all_workers_quiesced') is not True or
                supervisor_evidence.get('primary')!=result['expected_primary'] or owner.get('identity')!=result['expected_primary'] or
                owner.get('nonce')!=binding['activation_nonce'] or owner.get('process_absent') is not True or owner.get('port_closed') is not True or
                sorted(supervisor_evidence.get('quiesced_workers') or [])!=sorted(result['expected_workers']) or not supervisor_evidence.get('proof_sha256')):
            raise ValueError('replica loss fence is not verified')
    elif supervisor_evidence.get('run_id')!=binding['run_id'] or supervisor_evidence.get('primary_fenced') is not True or supervisor_evidence.get('all_workers_quiesced') is not True or not supervisor_evidence.get('proof_sha256') or supervisor_evidence.get('primary')!=result['expected_primary'] or sorted(supervisor_evidence.get('quiesced_workers') or [])!=sorted(result['expected_workers']):
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


def verify_bootstrap_inputs(*, baseline, baseline_sha256, source_manifest, source_sha256, source_receipt, receipt_sha256, expected_parent_sha256):
    """Verify exact earned-parent and source input bytes before first outer staging."""
    import time
    from natlang_neuralese.train.diloco.replica_baseline import verify
    def digest(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    source=Path(source_manifest);receipt=Path(source_receipt)
    if digest(source)!=source_sha256 or digest(receipt)!=receipt_sha256:raise ValueError('bootstrap source or receipt bytes differ')
    value=verify(baseline,baseline_sha256);manifest=json.loads((Path(baseline)/'manifest.json').read_text())
    if manifest.get('seed_source',{}).get('sha256')!=expected_parent_sha256:raise ValueError('bootstrap baseline does not bind the earned parent')
    return {'run_id':value['run_id'],'source_verified':True,'source_verified_at':time.time(),
            'bootstrap_source_sha256':source_sha256,'source_receipt_sha256':receipt_sha256,
            'earned_parent_sha256':expected_parent_sha256,'baseline_sha256':baseline_sha256,
            'scope':'exact source manifest and receipt bytes and earned-parent binding; source admission remains its own evidence'}


def run_coreweave_seed_stage(config):
    """Stage a third peer directly over gateways using exact, owned SDK roles.

    This operates only on the two supplied existing sandboxes. The caller must
    first persist and verify the supervisor fence; incomplete transfers retain
    the source lease and fail closed. No provider credential enters either host.
    """
    import time,uuid
    try:from . import cwctl,diloco_replica_fence as fences
    except ImportError:
        import cwctl
        import diloco_replica_fence as fences
    from natlang_neuralese.train.diloco.replica_transport import freeze_replica_files,witness_head
    witness=config['witness'];nonce=config['activation_nonce']
    if config.get('fence_mode','primary-loss')=='replica-loss':
        owner=config['primary_owner_observer']
        def observe(identity,challenge):
            return fences.sdk_observe_coordinator_stop(identity,challenge,owner['directory'],owner['port'],owner['supervisor_cmdline_sha256'],owner['entrypoint_sha256'])
        owner_evidence=fences.verify_owned_restart_fence(config['supervisor_proof'],expected_primary=config['expected_primary'],
            expected_workers=config['expected_workers'],nonce=nonce,observe_primary=observe,probe_worker=fences.sdk_worker_probe(config['control_dirs']),
            completed_sandbox_ids=[config['lost_replica_sandbox_id']],get_sandbox=fences.sdk_get_sandbox)
        evidence=fences.verify_replica_loss_fence(owner_evidence,replica_sandbox_id=config['lost_replica_sandbox_id'],get_sandbox=fences.sdk_get_sandbox)
        observations=owner_evidence['workers'];primary_stopped=evidence.get('primary_process_absent') is True
    else:
        verifier=fences.provider_fence_verifier(expected_primary=config['expected_primary'],expected_workers=config['expected_workers'],control_dirs=config['control_dirs'])
        evidence=verifier(config['supervisor_proof']);observations=evidence['workers'];primary_stopped=evidence.get('primary_fenced') is True
    if (not primary_stopped or evidence.get('all_workers_quiesced') is not True or
            evidence.get('run_id')!=witness['run_id'] or not evidence.get('proof_sha256')):
        raise ValueError('seed stage requires a verified stopped primary and worker fence')
    if len(nonce)<32 or witness['sequence']%2:raise ValueError('invalid seed boundary')
    source=config['source'];target=config['target']
    authorized={(item['identity']['sandbox_id'],item['identity']['boot_id'],item['identity']['owner_id']) for item in observations}
    source_id=(source['sandbox_id'],source['boot_id'],source['owner_id']);target_id=(target['sandbox_id'],target['boot_id'],target['owner_id'])
    if source_id==target_id or source_id not in authorized:raise ValueError('seed source is not a distinct verified survivor')
    if target_id not in authorized:
        replacement=config.get('replacement_owner')
        if not isinstance(replacement,dict):raise ValueError('seed target is not an authenticated successor owner')
        ref=config['supervisor_proof']['journal'];old_owner=fences.owner_reference(ref['path'],ref['sha256'],config['expected_primary'])
        proof=fences.verify_replacement_owner(replacement['reference'],owner_journal=old_owner,expected_primary=config['expected_primary'],
            expected_target=target,worker_index=replacement['worker_index'],nonce=nonce,get_sandbox=fences.sdk_get_sandbox,observe_owner=fences.sdk_observe_replacement_owner,
            failed_sandbox_id=(config.get('lost_replica_sandbox_id') if config.get('fence_mode')=='replica-loss' else None))
        fences.sdk_set_worker_authority([{'sandbox_id':target['sandbox_id'],'boot_id':target['boot_id'],'owner_id':target['owner_id'],
            'directory':target['directory'],'runtime':target['runtime']}],nonce=nonce,action='fence')
    else:proof=None
    cwctl.load_env_file();cwctl.require_key()
    source_box=cwctl.read_sandbox(source['sandbox_id'],patience=60);target_box=cwctl.read_sandbox(target['sandbox_id'],patience=60)
    source_token=Path(config['source_token_file']).read_text().strip()
    source_gateway=Path(config['source_gateway_file']).read_text().strip() if config.get('source_gateway_file') else None
    target_gateway=Path(config['target_gateway_file']).read_text().strip() if config.get('target_gateway_file') else None
    actual=witness_head(source['replica_url'],source_token,allow_http=config.get('allow_http',False),gateway=source_gateway)
    if actual!=witness:raise ValueError('source peer changed before seed staging')
    freeze_replica_files(source['replica_url'],source_token,nonce,allow_http=config.get('allow_http',False),gateway=source_gateway)
    unique=uuid.uuid4().hex
    def config_argv(box,host,role,value):
        directory=host['stage_directory'].rstrip('/')+'/'+unique
        result=box.exec(['mkdir','-p',directory],timeout_seconds=30).result(timeout=60)
        if result.returncode!=0:raise ValueError('seed staging directory failed')
        path=directory+'/'+role+'.json';raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode();digest=hashlib.sha256(raw).hexdigest()
        box.write_file(path,raw).result(timeout=60)
        return [host['python'],'-m','natlang_neuralese.train.diloco.replica_cli',role,'--config',path,'--config-sha256',digest]
    def execute(box,host,argv,timeout):
        result=box.exec(['env','PYTHONPATH='+host['runtime']+'/training/neuralese',*argv],timeout_seconds=timeout).result(timeout=timeout+30)
        if result.returncode!=0:raise ValueError('seed command '+argv[3]+' failed')
        return result.stdout
    def owned_start(host,argv,request_id):
        request={'schema':'natlang.diloco-role-request/1','id':request_id,'owner_id':host['owner_id'],
            'expected_generation':host['generation'],'action':'start','role':host.get('seed_role','actor'),'promotion_nonce':nonce,
            'argv':argv,'env':{'PYTHONPATH':host['runtime']+'/training/neuralese'}}
        role={'sandbox_id':host['sandbox_id'],'directory':host['directory'],'boot_id':host['boot_id'],
              'runtime':host['runtime'],'request':request}
        fences.sdk_authorize_seed_server(role,nonce)
        publisher=host['publisher'];program=('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
            'p=Path(sys.argv[1]);actual=hashlib.sha256(p.read_bytes()).hexdigest()\n'
            'if actual!=sys.argv[2]:raise ValueError("owner publisher differs")\n'
            'print(json.dumps(runpy.run_path(str(p))["submit"](sys.argv[3],json.loads(sys.argv[4]),timeout=60)))\n')
        box=source_box if host['sandbox_id']==source['sandbox_id'] else target_box
        result=box.exec(['python3','-c',program,publisher['path'],publisher['sha256'],host['directory'],json.dumps(request)],timeout_seconds=90).result(timeout=120)
        if result.returncode!=0:raise ValueError('owned seed role did not start')
        value=json.loads(result.stdout)
        if value.get('accepted') is not True:raise ValueError('owned seed role was rejected')
        return value
    publish=config_argv(source_box,source,'seed-publish',{'store':source['store'],'destination':source['seed_directory'],
        'sequence':witness['sequence'],'head':witness['head'],'lease_id':unique,'freeze_nonce':nonce})
    execute(source_box,source,publish,config.get('publish_timeout',3600))
    sha_program='import hashlib,sys;print(hashlib.file_digest(open(sys.argv[1],"rb"),"sha256").hexdigest())'
    hashed=source_box.exec(['python3','-c',sha_program,source['seed_directory']+'/seed-receipt.json'],timeout_seconds=120).result(timeout=150)
    if hashed.returncode!=0:raise ValueError('seed receipt hash unavailable')
    receipt_sha=hashed.stdout.strip()
    if len(receipt_sha)!=64:raise ValueError('invalid seed receipt hash')
    serve=config_argv(source_box,source,'seed-serve',{'seed':source['seed_directory'],'host':source['seed_host'],'port':source['seed_port'],'token_file':source['seed_token_file']})
    source_role=owned_start(source,serve,'seed-source-'+unique)
    def stop_source():
        stop_request={'schema':'natlang.diloco-role-request/1','id':'seed-stop-'+unique,'owner_id':source_role['owner_id'],
            'expected_generation':source_role['generation'],'role':source.get('seed_role','actor'),'action':'stop','expected_pid':source_role['pid'],
            'expected_process_start_ticks':source_role['process_start_ticks']}
        publisher=source['publisher'];program=('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
            'p=Path(sys.argv[1]);actual=hashlib.sha256(p.read_bytes()).hexdigest()\n'
            'if actual!=sys.argv[2]:raise ValueError("owner publisher differs")\n'
            'print(json.dumps(runpy.run_path(str(p))["submit"](sys.argv[3],json.loads(sys.argv[4]),timeout=60)))\n')
        stopped=source_box.exec(['python3','-c',program,publisher['path'],publisher['sha256'],source['directory'],json.dumps(stop_request)],timeout_seconds=90).result(timeout=120)
        if stopped.returncode!=0:raise ValueError('seed actor stop was not accepted')
        stopped_value=json.loads(stopped.stdout)
        if stopped_value.get('accepted') is not True:raise ValueError('seed actor stop was not accepted')
        probe=('import json,sys,time\nfrom pathlib import Path\n'
            'deadline=time.monotonic()+60\n'
            'while time.monotonic()<deadline:\n'
            ' s=json.loads(Path(sys.argv[1]).read_text());r=s.get("roles",{}).get(sys.argv[3],{})\n'
            ' if r.get("pid")==int(sys.argv[2]) and r.get("alive") is False:break\n'
            ' time.sleep(.2)\n'
            'else:raise SystemExit(1)\n')
        observed=source_box.exec(['python3','-c',probe,source['directory'].rstrip('/')+'/state.json',str(source_role['pid']),source.get('seed_role','actor')],timeout_seconds=70).result(timeout=90)
        if observed.returncode!=0:raise ValueError('seed actor exit not observed')
        return stopped_value
    target_role=None;completed=False
    try:
        fetch=config_argv(target_box,target,'seed-fetch',{'url':source['seed_url'],'token_file':target['seed_token_file'],
            'gateway_file':target.get('source_gateway_file'),'destination':target['seed_directory'],'receipt_sha256':receipt_sha,'allow_http':config.get('allow_http',False)})
        execute(target_box,target,fetch,config.get('transfer_timeout',21600))
        install=config_argv(target_box,target,'seed-install',{'seed':target['seed_directory'],'store':target['store'],'receipt_sha256':receipt_sha})
        execute(target_box,target,install,config.get('install_timeout',3600))
        peer_config={'baseline':target['store']+'/baseline','baseline_sha256':witness['baseline_sha256'],'store':target['store'],
            'work_root':target['work_root'],'host':target['peer_host'],'port':target['peer_port'],'token_file':target['peer_token_file'],'max_transaction_bytes':target['max_transaction_bytes']}
        peer_argv=config_argv(target_box,target,'peer',peer_config);target_role=owned_start(target,peer_argv,'seed-peer-'+unique)
        preserved={'role':target.get('seed_role','actor'),'pid':target_role['pid'],'start_ticks':target_role['process_start_ticks'],
                   'argv_sha256':hashlib.sha256(json.dumps(peer_argv,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()}
        fences.sdk_record_preserved_role(target,nonce,preserved)
        target_token=Path(config['target_token_file']).read_text().strip();deadline=time.monotonic()+config.get('readiness_timeout',300)
        while True:
            try:
                actual=witness_head(target['replica_url'],target_token,allow_http=config.get('allow_http',False),gateway=target_gateway)
                if actual!=witness:raise ValueError('replacement peer witnessed a different head')
                break
            except OSError:
                if time.monotonic()>=deadline:raise
                time.sleep(1)
        release=config_argv(source_box,source,'seed-release',{'store':source['store'],'seed_receipt':source['seed_directory']+'/seed-receipt.json',
            'receipt_sha256':receipt_sha,'peer_url':target['replica_url'],'token_file':source['target_peer_token_file'],
            'gateway_file':source.get('target_gateway_file'),'allow_http':config.get('allow_http',False)})
        execute(source_box,source,release,120);completed=True
    finally:
        try:
            if not completed and target_role is not None:
                request={'schema':'natlang.diloco-role-request/1','id':'seed-target-stop-'+unique,'owner_id':target_role['owner_id'],
                    'expected_generation':target_role['generation'],'role':target.get('seed_role','actor'),'action':'stop',
                    'expected_pid':target_role['pid'],'expected_process_start_ticks':target_role['process_start_ticks']}
                fences.sdk_stop_owned_role({'sandbox_id':target['sandbox_id'],'directory':target['directory'],
                    'publisher':target['publisher'],'request':request})
        finally:stopped_value=stop_source()
    return {'schema':'natlang.diloco-third-peer-stage/1','witness':witness,'receipt_sha256':receipt_sha,
            'source_role':source_role,'target_role':target_role,'replacement_owner_proof':proof,'target_preserve_role':preserved,'source_generation_after_seed_stop':stopped_value['generation'],'source_seed_server_stopped':True}


def run_coreweave_replica_loss_restart(directory, config):
    """Resume the same stopped primary after its lost peer is replaced.

    The recovered local sealed state and the replacement peer must already be
    staged. This path retains the existing coordinator session and never
    pretends the still-running primary sandbox was provider-Completed.
    """
    import time,uuid
    try:from . import cwctl,diloco_replica_fence as fences
    except ImportError:
        import cwctl
        import diloco_replica_fence as fences
    from natlang_neuralese.train.diloco.worker import Transport,RemoteError
    from natlang_neuralese.train.diloco.replica_transport import witness_head
    intent=config['intent'];nonce=intent['activation_nonce'];workers=config['workers'];lost=config['lost_replica_sandbox_id']
    ref=config['supervisor_proof']['journal'];old=fences.owner_reference(ref['path'],ref['sha256'],config['expected_primary'])
    expected={(w['sandbox_id'],w['boot_id'],w['owner_id']) for w in old['workers'].values() if w['sandbox_id']!=lost}
    actual={(w['sandbox_id'],w['boot_id'],w['owner_id']) for w in workers};extra=actual-expected
    if not expected<=actual or len(extra)!=1:raise ValueError('restart owners differ from surviving roster and one successor')
    replacement=config['replacement_owner'];target=replacement['target']
    if extra!={(target['sandbox_id'],target['boot_id'],target['owner_id'])}:raise ValueError('restart replacement owner differs')
    replacement_proof=fences.verify_replacement_owner(replacement['reference'],owner_journal=old,expected_primary=config['expected_primary'],
        expected_target=target,worker_index=replacement['worker_index'],nonce=nonce,get_sandbox=fences.sdk_get_sandbox,
        observe_owner=fences.sdk_observe_replacement_owner,failed_sandbox_id=lost)
    owner=config['primary_owner_observer']
    def observe(identity,challenge):return fences.sdk_observe_coordinator_stop(identity,challenge,owner['directory'],owner['port'],owner['supervisor_cmdline_sha256'],owner['entrypoint_sha256'])
    token=Path(config['token_file']).read_text().strip()
    if hashlib.sha256(token.encode()).hexdigest()!=intent['new_token_sha256']:raise ValueError('restart session token differs')
    def install(_):return fences.sdk_set_worker_authority(workers,nonce=nonce,action='fence')
    def verify(_):
        owner_proof=fences.verify_owned_restart_fence(config['supervisor_proof'],expected_primary=config['expected_primary'],
            expected_workers=config['expected_workers'],nonce=nonce,observe_primary=observe,probe_worker=fences.sdk_worker_probe(config['control_dirs']),
            completed_sandbox_ids=[lost],get_sandbox=fences.sdk_get_sandbox)
        loss_proof=fences.verify_replica_loss_fence(owner_proof,replica_sandbox_id=lost,get_sandbox=fences.sdk_get_sandbox)
        result={**owner_proof,'replica_loss':loss_proof,'replacement_owner':replacement_proof}
        result['proof_sha256']=hashlib.sha256(json.dumps({k:v for k,v in result.items() if k!='proof_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return result
    def start(_,evidence):
        client=Transport(intent['new_url'],token,5,gateway=config.get('gateway'),allow_http=config.get('allow_http',False))
        try:existing=client.json('GET','/v1/replica-promotion')
        except OSError:existing=None
        except RemoteError as exc:
            if exc.status not in (404,503):raise
            existing=None
        if existing is not None:
            if (existing.get('run_id')!=intent['run_id'] or existing.get('sequence')!=intent['sequence'] or
                    existing.get('head')!=intent['head'] or existing.get('token_sha256')!=intent['new_token_sha256']):raise ValueError('existing restart differs')
            fences.sdk_confirm_promoted_role(config['primary_role'],nonce);return existing
        peer=config['outgoing_replica'];witness=witness_head(peer['url'],token,allow_http=config.get('allow_http',False),gateway=config.get('gateway'))
        if witness!={'run_id':intent['run_id'],'baseline_sha256':peer['baseline_sha256'],'sequence':intent['sequence'],'head':intent['head']}:
            raise ValueError('replacement peer does not witness recovered head')
        fences.sdk_stop_owned_role(config['recovered_source_peer_role'])
        role=dict(config['primary_role']);cwctl.load_env_file();cwctl.require_key();box=cwctl.read_sandbox(role['sandbox_id'],patience=60)
        status=box.exec(['cat',role['directory'].rstrip('/')+'/state.json'],timeout_seconds=30).result(timeout=60)
        if status.returncode!=0:raise ValueError('restarting owner state unavailable')
        state=json.loads(status.stdout)
        if state.get('owner_id')!=role['owner_id'] or state.get('boot_id')!=role['boot_id'] or state.get('pending_start'):
            raise ValueError('restarting owner identity changed')
        activation=dict(config['activation_config'])
        if (activation.get('run_id')!=intent['run_id'] or activation.get('sequence')!=intent['sequence'] or activation.get('head')!=intent['head'] or activation.get('activation_nonce')!=nonce):
            raise ValueError('restart activation differs from recovered custody')
        raw,digest=prepare_owned_activation(activation,mode='restart',supervisor_evidence=evidence,now=time.time(),owner_stop=evidence['owner_observation'])
        stage=role['stage_directory'].rstrip('/')+'/restart-'+uuid.uuid4().hex
        made=box.exec(['mkdir','-p',stage],timeout_seconds=30).result(timeout=60)
        if made.returncode!=0:raise ValueError('restart staging directory failed')
        path=stage+'/activation.json';box.write_file(path,raw).result(timeout=60)
        request=dict(role['request']);request.update(owner_id=role['owner_id'],expected_generation=state['generation'],promotion_nonce=nonce,
            argv=[role['python'],'-m','natlang_neuralese.train.diloco.replica_cli','restart','--config',path,'--config-sha256',digest])
        if any('COREWEAVE' in k.upper() or k.upper().startswith('CW_') for k in request.get('env',{})):
            raise ValueError('provider credential must not enter restart environment')
        role['request']=request;fences.sdk_authorize_coordinator(role,nonce)
        publisher=role['publisher'];program=('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
            'p=Path(sys.argv[1]);actual=hashlib.sha256(p.read_bytes()).hexdigest()\n'
            'if actual!=sys.argv[2]:raise ValueError("owner publisher differs")\n'
            'print(json.dumps(runpy.run_path(str(p))["submit"](sys.argv[3],json.loads(sys.argv[4]),timeout=60)))\n')
        started=box.exec(['python3','-c',program,publisher['path'],publisher['sha256'],role['directory'],json.dumps(request)],timeout_seconds=90).result(timeout=120)
        if started.returncode!=0 or json.loads(started.stdout).get('accepted') is not True:raise ValueError('restart owner did not accept coordinator')
        deadline=time.monotonic()+config.get('readiness_timeout',120)
        while True:
            try:return client.json('GET','/v1/replica-promotion')
            except OSError:
                if time.monotonic()>=deadline:raise
                time.sleep(1)
    def release(_,fenced):return fences.sdk_set_worker_authority(workers,nonce=nonce,action='release')
    return run_promotion(directory,intent=intent,fence_workers=install,verify_fence=verify,start_primary=start,release_workers=release)


def run_coreweave_prepare_replica_loss(config):
    """Recover the stopped primary's exact local WAL and publish a temporary owned peer.

    No allocation or GPU work occurs. The caller owns the identified sandbox,
    has quiesced workers, and supplies the actual local sealed/unsealed head.
    """
    import time,uuid
    try:from . import cwctl,diloco_replica_fence as fences
    except ImportError:
        import cwctl
        import diloco_replica_fence as fences
    from natlang_neuralese.train.diloco.replica_transport import witness_head
    nonce=config['activation_nonce'];workers=config['workers'];lost=config['lost_replica_sandbox_id'];owner=config['primary_owner_observer']
    ref=config['supervisor_proof']['journal'];journal=fences.owner_reference(ref['path'],ref['sha256'],config['expected_primary'])
    expected={(w['sandbox_id'],w['boot_id'],w['owner_id']) for w in journal['workers'].values() if w['sandbox_id']!=lost}
    if {(w['sandbox_id'],w['boot_id'],w['owner_id']) for w in workers}!=expected:
        raise ValueError('replica loss fencing does not cover exact surviving owners')
    fences.sdk_set_worker_authority(workers,nonce=nonce,action='fence')
    def observe(identity,challenge):return fences.sdk_observe_coordinator_stop(identity,challenge,owner['directory'],owner['port'],owner['supervisor_cmdline_sha256'],owner['entrypoint_sha256'])
    owner_proof=fences.verify_owned_restart_fence(config['supervisor_proof'],expected_primary=config['expected_primary'],
        expected_workers=config['expected_workers'],nonce=nonce,observe_primary=observe,probe_worker=fences.sdk_worker_probe(config['control_dirs']),
        completed_sandbox_ids=[lost],get_sandbox=fences.sdk_get_sandbox)
    loss=fences.verify_replica_loss_fence(owner_proof,replica_sandbox_id=lost,get_sandbox=fences.sdk_get_sandbox)
    source=config['source'];primary=config['expected_primary']
    if (source['sandbox_id'],source['boot_id'],source['owner_id'])!=(primary['sandbox_id'],primary['boot_id'],primary['owner_id']):
        raise ValueError('recovered source is not the exact stopped primary owner')
    activation=dict(config['activation_config'])
    if activation.get('activation_nonce')!=nonce or activation.get('run_id')!=owner_proof['run_id']:
        raise ValueError('replica-loss activation custody differs')
    raw,digest=prepare_owned_activation(activation,mode='replica-loss-recover',supervisor_evidence=loss,now=time.time())
    cwctl.load_env_file();cwctl.require_key();box=cwctl.read_sandbox(source['sandbox_id'],patience=60);unique=uuid.uuid4().hex
    stage=source['stage_directory'].rstrip('/')+'/peer-loss-'+unique
    made=box.exec(['mkdir','-p',stage],timeout_seconds=30).result(timeout=60)
    if made.returncode!=0:raise ValueError('replica loss staging directory failed')
    path=stage+'/recovery.json';box.write_file(path,raw).result(timeout=60)
    argv=[source['python'],'-m','natlang_neuralese.train.diloco.replica_cli','replica-loss-recover','--config',path,'--config-sha256',digest]
    result=box.exec(['env','PYTHONPATH='+source['runtime']+'/training/neuralese',*argv],timeout_seconds=config.get('recovery_timeout',3600)).result(timeout=config.get('recovery_timeout',3600)+30)
    if result.returncode!=0:raise ValueError('local replica-loss recovery failed; original journal is retained')
    receipt=json.loads(result.stdout)
    if receipt.get('run_id')!=owner_proof['run_id'] or receipt.get('source_head')!=activation['head'] or receipt.get('source_sequence')!=activation['sequence']:
        raise ValueError('recovered local journal receipt differs')
    peer_config={'baseline':activation['destination']+'/baseline','baseline_sha256':receipt['new_baseline_sha256'],'store':activation['destination'],
        'work_root':source['peer_work_root'],'host':source['peer_host'],'port':source['peer_port'],'token_file':source['peer_token_file'],'max_transaction_bytes':source['max_transaction_bytes']}
    peer_raw=json.dumps(peer_config,sort_keys=True,separators=(',',':')).encode();peer_sha=hashlib.sha256(peer_raw).hexdigest();peer_path=stage+'/peer.json';box.write_file(peer_path,peer_raw).result(timeout=60)
    status=box.exec(['cat',source['directory'].rstrip('/')+'/state.json'],timeout_seconds=30).result(timeout=60)
    if status.returncode!=0:raise ValueError('recovered source owner unavailable')
    state=json.loads(status.stdout)
    if state.get('owner_id')!=source['owner_id'] or state.get('boot_id')!=source['boot_id'] or state.get('pending_start'):
        raise ValueError('recovered source owner identity changed')
    peer_argv=[source['python'],'-m','natlang_neuralese.train.diloco.replica_cli','peer','--config',peer_path,'--config-sha256',peer_sha]
    request={'schema':'natlang.diloco-role-request/1','id':'recovered-peer-'+unique,'owner_id':source['owner_id'],
        'expected_generation':state['generation'],'action':'start','role':source.get('peer_role','actor'),'promotion_nonce':nonce,
        'argv':peer_argv,'env':{'PYTHONPATH':source['runtime']+'/training/neuralese'}}
    role={'sandbox_id':source['sandbox_id'],'directory':source['directory'],'boot_id':source['boot_id'],'runtime':source['runtime'],'request':request}
    fences.sdk_authorize_seed_server(role,nonce)
    publisher=source['publisher'];program=('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
        'p=Path(sys.argv[1]);actual=hashlib.sha256(p.read_bytes()).hexdigest()\n'
        'if actual!=sys.argv[2]:raise ValueError("owner publisher differs")\n'
        'print(json.dumps(runpy.run_path(str(p))["submit"](sys.argv[3],json.loads(sys.argv[4]),timeout=60)))\n')
    started=box.exec(['python3','-c',program,publisher['path'],publisher['sha256'],source['directory'],json.dumps(request)],timeout_seconds=90).result(timeout=120)
    if started.returncode!=0:raise ValueError('recovered source peer owner failed')
    peer_role=json.loads(started.stdout)
    if peer_role.get('accepted') is not True:raise ValueError('recovered source peer was not accepted')
    preserved={'role':source.get('peer_role','actor'),'pid':peer_role['pid'],'start_ticks':peer_role['process_start_ticks'],
               'argv_sha256':hashlib.sha256(json.dumps(peer_argv,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()}
    try:
        fences.sdk_record_preserved_role(source,nonce,preserved)
        token=Path(config['peer_token_file']).read_text().strip();deadline=time.monotonic()+config.get('readiness_timeout',300)
        expected={'run_id':receipt['run_id'],'baseline_sha256':receipt['new_baseline_sha256'],'sequence':0,'head':receipt['new_head']}
        while True:
            try:
                actual=witness_head(source['replica_url'],token,gateway=config.get('gateway'),allow_http=config.get('allow_http',False))
                if actual!=expected:raise ValueError('recovered source peer head differs')
                break
            except OSError:
                if time.monotonic()>=deadline:raise
                time.sleep(1)
    except BaseException:
        stop_request={'schema':'natlang.diloco-role-request/1','id':'recovered-peer-stop-'+unique,'owner_id':peer_role['owner_id'],
            'expected_generation':peer_role['generation'],'role':source.get('peer_role','actor'),'action':'stop',
            'expected_pid':peer_role['pid'],'expected_process_start_ticks':peer_role['process_start_ticks']}
        fences.sdk_stop_owned_role({'sandbox_id':source['sandbox_id'],'directory':source['directory'],
            'publisher':source['publisher'],'request':stop_request})
        raise
    return {'schema':'natlang.diloco-owned-replica-loss-stage/1','recovery':receipt,'witness':expected,
            'owner_fence':owner_proof,'lost_replica_fence':loss,'peer_role':peer_role,'preserve_role':preserved}
