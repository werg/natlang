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
    if not isinstance(intent,dict) or set(intent)!=required or not all(intent.values()) or len(intent['generation_nonce'])<32 or len(intent['activation_nonce'])<32:
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
    intent=config['intent'];workers=config['workers'];nonce=intent['activation_nonce']
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
        from natlang_neuralese.train.diloco.replica_transport import witness_head,rotate_replica_credential
        replica=config['outgoing_replica'];old_token=Path(replica['old_token_file']).read_text().strip()
        witnessed=witness_head(replica['url'],old_token,allow_http=config.get('allow_http',False),gateway=config.get('gateway'))
        if witnessed!={'run_id':intent['run_id'],'baseline_sha256':replica['baseline_sha256'],'sequence':intent['sequence'],'head':intent['head']}:
            raise ValueError('outgoing replica does not witness exact promoted head')
        rotate_replica_credential(replica['url'],old_token,token,sequence=intent['sequence'],head=intent['head'],nonce=nonce,
            allow_http=config.get('allow_http',False),gateway=config.get('gateway'))
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
    evidence=config['verified_fence'];witness=config['witness'];nonce=config['activation_nonce']
    if (evidence.get('primary_fenced') is not True or evidence.get('all_workers_quiesced') is not True or
            evidence.get('run_id')!=witness['run_id'] or not evidence.get('proof_sha256')):
        raise ValueError('seed stage requires a verified old-primary and worker fence')
    if len(nonce)<32 or witness['sequence']%2:raise ValueError('invalid seed boundary')
    source=config['source'];target=config['target'];cwctl.load_env_file();cwctl.require_key()
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
            'expected_generation':host['generation'],'action':'start','role':'actor','promotion_nonce':nonce,
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
    fetch=config_argv(target_box,target,'seed-fetch',{'url':source['seed_url'],'token_file':target['seed_token_file'],
        'gateway_file':target.get('source_gateway_file'),'destination':target['seed_directory'],'receipt_sha256':receipt_sha,'allow_http':config.get('allow_http',False)})
    execute(target_box,target,fetch,config.get('transfer_timeout',21600))
    install=config_argv(target_box,target,'seed-install',{'seed':target['seed_directory'],'store':target['store'],'receipt_sha256':receipt_sha})
    execute(target_box,target,install,config.get('install_timeout',3600))
    peer_config={'baseline':target['store']+'/baseline','baseline_sha256':witness['baseline_sha256'],'store':target['store'],
        'work_root':target['work_root'],'host':target['peer_host'],'port':target['peer_port'],'token_file':target['peer_token_file'],'max_transaction_bytes':target['max_transaction_bytes']}
    peer_argv=config_argv(target_box,target,'peer',peer_config);target_role=owned_start(target,peer_argv,'seed-peer-'+unique)
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
    execute(source_box,source,release,120)
    stop_request={'schema':'natlang.diloco-role-request/1','id':'seed-stop-'+unique,'owner_id':source_role['owner_id'],
        'expected_generation':source_role['generation'],'role':'actor','action':'stop','expected_pid':source_role['pid'],
        'expected_process_start_ticks':source_role['process_start_ticks']}
    publisher=source['publisher'];program=('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
        'p=Path(sys.argv[1]);actual=hashlib.sha256(p.read_bytes()).hexdigest()\n'
        'if actual!=sys.argv[2]:raise ValueError("owner publisher differs")\n'
        'print(json.dumps(runpy.run_path(str(p))["submit"](sys.argv[3],json.loads(sys.argv[4]),timeout=60)))\n')
    stopped=source_box.exec(['python3','-c',program,publisher['path'],publisher['sha256'],source['directory'],json.dumps(stop_request)],timeout_seconds=90).result(timeout=120)
    if stopped.returncode!=0 or json.loads(stopped.stdout).get('accepted') is not True:raise ValueError('seed actor stop was not accepted')
    probe=('import json,sys,time\nfrom pathlib import Path\n'
        'deadline=time.monotonic()+60\n'
        'while time.monotonic()<deadline:\n'
        ' s=json.loads(Path(sys.argv[1]).read_text());r=s.get("roles",{}).get("actor",{})\n'
        ' if r.get("pid")==int(sys.argv[2]) and r.get("alive") is False:break\n'
        ' time.sleep(.2)\n'
        'else:raise SystemExit(1)\n')
    observed=source_box.exec(['python3','-c',probe,source['directory'].rstrip('/')+'/state.json',str(source_role['pid'])],timeout_seconds=70).result(timeout=90)
    if observed.returncode!=0:raise ValueError('seed actor exit not observed')
    return {'schema':'natlang.diloco-third-peer-stage/1','witness':witness,'receipt_sha256':receipt_sha,
            'source_role':source_role,'target_role':target_role,'source_seed_server_stopped':True}
