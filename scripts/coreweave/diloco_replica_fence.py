"""Fresh provider/owner observations for explicit, supervisor-fenced replica promotion."""
from __future__ import annotations
import hashlib
import json
import re
import secrets
import time
from pathlib import Path

COMPLETED = 3  # cwsandbox._proto.sandbox_pb2.State.STATE_COMPLETED


def _sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def owner_reference(path, expected_sha256, expected_primary):
    """Bind the old coordinator identity to its immutable owner journal bytes."""
    path=Path(path).resolve(strict=True);before=path.stat();actual=_sha(path);value=json.loads(path.read_text());after=path.stat()
    if actual!=expected_sha256 or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('owner journal changed or its digest differs')
    primary=value.get('coordinator') or (value.get('owner') or {}).get('coordinator') or {}
    fields=('sandbox_id','boot_id','owner_id','pid','start_ticks')
    canonical={key:primary.get(key) for key in fields};canonical['start_ticks']=primary.get('start_ticks') or primary.get('process_start_ticks')
    if canonical!=expected_primary or not all(canonical.values()):raise ValueError('owner journal does not bind the expected old primary')
    if not value.get('run_id'):raise ValueError('owner journal lacks run identity')
    return value


def _top_state(sandbox, sandbox_id):
    if getattr(sandbox,'sandbox_id',None)!=sandbox_id:raise ValueError('provider returned a different sandbox identity')
    status=getattr(sandbox,'status',None)
    if status is None:raise ValueError('provider omitted top-level sandbox status')
    return getattr(status,'state',None)


def verify_provider_fence(proof, *, expected_primary, expected_workers, get_sandbox, probe_worker, now=time.time):
    """Require a fresh top-level provider Completed and fresh, exact worker stops.

    `get_sandbox` must issue GetSandbox for this request. `probe_worker` must
    execute a nonce-bound live process observation in the identified sandbox.
    Nested container status never establishes or defeats the provider fence.
    """
    if not isinstance(proof,dict) or set(proof)!={'journal','expected_run_id'}:raise ValueError('invalid supervisor fence request')
    ref=proof['journal'];path=ref.get('path');sha=ref.get('sha256')
    if not isinstance(path,str) or not re.fullmatch('[0-9a-f]{64}',str(sha)):raise ValueError('fence needs an exact owner journal')
    owner=owner_reference(path,sha,expected_primary)
    if owner['run_id']!=proof['expected_run_id']:raise ValueError('owner journal run differs')
    worker_map=owner.get('workers') or (owner.get('owner') or {}).get('workers') or {}
    if sorted(map(int,worker_map))!=sorted(expected_workers):raise ValueError('owner journal does not bind every expected worker')
    started=now();sandbox=get_sandbox(expected_primary['sandbox_id']);state=_top_state(sandbox,expected_primary['sandbox_id']);completed=now()
    if completed<started or completed-started>120 or state!=COMPLETED:raise ValueError('fresh provider observation did not fence the primary')
    observations=[]
    for worker in sorted(expected_workers):
        old=worker_map[str(worker)];required=('sandbox_id','boot_id','owner_id','pid','start_ticks')
        identity={key:old.get(key) for key in required};identity['start_ticks']=old.get('start_ticks') or old.get('process_start_ticks')
        if not all(identity.values()):raise ValueError('worker owner journal identity incomplete')
        nonce=secrets.token_hex(24);begin=now()
        if identity['sandbox_id']==expected_primary['sandbox_id']:
            if identity['boot_id']!=expected_primary['boot_id'] or identity['owner_id']!=expected_primary['owner_id']:
                raise ValueError('cohost worker does not bind fenced owner boot')
            observed={'identity':identity,'nonce':nonce,'process_absent':True,'owner_quiesced':True,
                      'method':'provider-completed-cohost'}
        else:
            observed=probe_worker(identity,nonce)
        end=now()
        if (end<begin or end-begin>120 or not isinstance(observed,dict) or observed.get('nonce')!=nonce
                or observed.get('identity')!=identity or observed.get('process_absent') is not True
                or observed.get('owner_quiesced') is not True):raise ValueError('fresh worker quiescence observation failed')
        observations.append({'worker':worker,'identity':identity,'observed_at':end,'nonce':nonce,'method':observed.get('method')})
    owner_reference(path,sha,expected_primary)  # detect replacement during observations
    result={'schema':'natlang.diloco-provider-fence/1','run_id':owner['run_id'],'primary':expected_primary,
            'primary_fenced':True,'all_workers_quiesced':True,'quiesced_workers':sorted(expected_workers),
            'owner_journal':{'path':str(Path(path).resolve()),'sha256':sha},'request_started_at':started,
            'provider_observed_at':completed,'provider_state':'STATE_COMPLETED','workers':observations}
    result['proof_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def sdk_get_sandbox(sandbox_id):
    """Issue the authoritative SDK GetSandbox call; failures propagate closed."""
    from cwsandbox._proto import sandbox_pb2
    try:from . import cwctl
    except ImportError:import cwctl
    cwctl.load_env_file();cwctl.require_key();sandbox=cwctl.read_sandbox(sandbox_id,patience=60)
    async def request():
        await sandbox._ensure_client()
        return await sandbox._stub.GetSandbox(sandbox_pb2.GetSandboxRequest(sandbox_id=sandbox_id),timeout=30,metadata=sandbox._auth_metadata)
    return sandbox._loop_manager.run_sync(request())


def observe_local_worker(identity, nonce, control_dir, fence_nonce, supervisor_cmdline_sha256, entrypoint_sha256=None):
    """Observe an exact role owner and its stopped worker on its sandbox host.

    The controlling supervisor must also prevent new role requests through
    promotion; this observation alone is not a distributed lock.
    """
    import os
    directory=Path(control_dir)
    if directory.is_symlink(): raise ValueError('role control path is a symlink')
    directory=directory.resolve(strict=True)
    def ticks(pid):
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            return None if fields[0]=='Z' else fields[19]
        except (OSError,IndexError): return None
    state=json.loads((directory/'state.json').read_text())
    if (state.get('schema')!='natlang.diloco-role-supervisor/1' or
            Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=identity['boot_id'] or
            state.get('boot_id')!=identity['boot_id'] or state.get('owner_id')!=identity['owner_id'] or
            type(state.get('supervisor_pid')) is not int or not state.get('supervisor_start_ticks') or
            ticks(state.get('supervisor_pid'))!=state.get('supervisor_start_ticks')):
        raise ValueError('role supervisor identity is not live and exact')
    import hashlib
    cmdline=Path(f"/proc/{state['supervisor_pid']}/cmdline").read_bytes()
    if hashlib.sha256(cmdline).hexdigest()!=supervisor_cmdline_sha256:
        raise ValueError('role supervisor executable identity differs')
    if entrypoint_sha256 is not None:
        args=cmdline.split(b'\0')
        if b'-c' not in args or args.index(b'-c')+1>=len(args) or hashlib.sha256(args[args.index(b'-c')+1]).hexdigest()!=entrypoint_sha256:
            raise ValueError('role supervisor authority entrypoint differs')
    fence=json.loads((directory/'promotion-fence.json').read_text())
    if fence!={'schema':'natlang.diloco-role-promotion-fence/1','owner_id':identity['owner_id'],'boot_id':identity['boot_id'],'nonce':fence_nonce}:
        raise ValueError('durable promotion authority fence is missing or changed')
    role=state.get('roles',{}).get('worker',{})
    if state.get('pending_start') or (directory/'request.json').exists():
        raise ValueError('role owner has an unprocessed request')
    if ticks(identity['pid'])==str(identity['start_ticks']): raise ValueError('recorded worker is alive')
    if role.get('alive') or (role.get('pid') and ticks(role['pid']) is not None):
        raise ValueError('worker role has a live replacement')
    return {'nonce':nonce,'identity':identity,'process_absent':True,'owner_quiesced':True,'method':'live-role-owner'}


def sdk_worker_probe(control_dirs):
    """Build a nonce-bound worker observer using the existing SDK exec channel.

    control_dirs maps sandbox IDs to the exact role-supervisor directory. The
    caller must hold its controller's exclusive promotion/start authorization.
    """
    import inspect
    try: from . import cwctl
    except ImportError: import cwctl
    source=inspect.getsource(observe_local_worker)
    program=('import json,sys\nfrom pathlib import Path\n'+source+
             '\nprint(json.dumps(observe_local_worker(json.loads(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])))\n')
    def probe(identity,nonce):
        entry=control_dirs.get(identity['sandbox_id'])
        if not isinstance(entry,dict) or set(entry)!={'directory','nonce','supervisor_cmdline_sha256','entrypoint_sha256'}: raise ValueError('missing exact worker promotion authority')
        directory=entry['directory']; fence_nonce=entry['nonce']; command_sha=entry['supervisor_cmdline_sha256']; entrypoint_sha=entry['entrypoint_sha256']
        if not isinstance(directory,str) or not directory.startswith('/') or not isinstance(fence_nonce,str) or len(fence_nonce)<32 or not re.fullmatch('[0-9a-f]{64}',str(command_sha)) or not re.fullmatch('[0-9a-f]{64}',str(entrypoint_sha)): raise ValueError('invalid worker promotion authority')
        cwctl.load_env_file();cwctl.require_key()
        sandbox=cwctl.read_sandbox(identity['sandbox_id'],patience=60)
        result=sandbox.exec(['python3','-c',program,json.dumps(identity,sort_keys=True),nonce,directory,fence_nonce,command_sha,entrypoint_sha],timeout_seconds=30).result(timeout=60)
        if result.returncode!=0: raise ValueError('worker identity observation failed')
        return json.loads(result.stdout)
    return probe


def provider_fence_verifier(*, expected_primary, expected_workers, control_dirs):
    """Concrete verifier for the promotion callback; caller serializes starts."""
    worker_probe=sdk_worker_probe(control_dirs)
    return lambda proof: verify_provider_fence(proof,expected_primary=expected_primary,
        expected_workers=expected_workers,get_sandbox=sdk_get_sandbox,probe_worker=worker_probe)


def sdk_set_worker_authority(workers, *, nonce, action):
    """Fence or release the exact nested owners through the existing SDK.

    The authorized controller calls this; partial failure deliberately leaves
    every already-fenced owner fenced. No operation is executed on import.
    """
    import inspect
    try: from . import cwctl, diloco_promotion_authority as authority
    except ImportError:
        import cwctl
        import diloco_promotion_authority as authority
    if action not in ('fence','release') or not isinstance(nonce,str) or len(nonce)<32:
        raise ValueError('invalid controller authority transition')
    function=getattr(authority,action)
    tail=(',preserve_role=json.loads(sys.argv[6])' if action=='fence' else '')
    program=('import fcntl,json,os,sys\nfrom pathlib import Path\n'
             'sys.path.insert(0,sys.argv[5]+"/training/neuralese")\nfrom natlang_neuralese.common.jsonio import write_json_durable\n'+inspect.getsource(function)+
             '\nprint(json.dumps('+action+'(sys.argv[1],expected_owner=sys.argv[2],expected_boot=sys.argv[3],nonce=sys.argv[4]'+tail+')))\n')
    results=[]
    for worker in workers:
        if set(worker)-{'sandbox_id','directory','owner_id','boot_id','runtime','preserve_role'} or not {'sandbox_id','directory','owner_id','boot_id','runtime'}<=set(worker):raise ValueError('invalid worker owner binding')
        cwctl.load_env_file();cwctl.require_key();sandbox=cwctl.read_sandbox(worker['sandbox_id'],patience=60)
        result=sandbox.exec(['python3','-c',program,worker['directory'],worker['owner_id'],worker['boot_id'],nonce,worker['runtime'],json.dumps(worker.get('preserve_role'))],timeout_seconds=30).result(timeout=60)
        if result.returncode!=0:raise ValueError('worker authority transition failed; previous fences remain in force')
        results.append({'sandbox_id':worker['sandbox_id'],'authority':json.loads(result.stdout) if action=='fence' else None})
    return results


def observe_coordinator_stop(identity, nonce, control_dir, port, supervisor_cmdline_sha256, entrypoint_sha256):
    """Observe stopped exact coordinator and closed loopback port under a durable role fence."""
    import errno,socket
    state=json.loads((Path(control_dir)/'state.json').read_text())
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=identity['boot_id'] or state.get('owner_id')!=identity['owner_id'] or state.get('boot_id')!=identity['boot_id']:
        raise ValueError('role owner identity changed')
    marker=json.loads((Path(control_dir)/'promotion-fence.json').read_text())
    if marker!={'schema':'natlang.diloco-role-promotion-fence/1','owner_id':identity['owner_id'],'boot_id':identity['boot_id'],'nonce':nonce}:
        raise ValueError('role owner is not fenced')
    def ticks(pid):
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            return None if fields[0]=='Z' else fields[19]
        except (OSError,IndexError):return None
    if type(state.get('supervisor_pid')) is not int or not state.get('supervisor_start_ticks') or ticks(state['supervisor_pid'])!=state['supervisor_start_ticks']:
        raise ValueError('role owner is not alive')
    command=Path(f"/proc/{state['supervisor_pid']}/cmdline").read_bytes();parts=command.split(b'\0')
    if (hashlib.sha256(command).hexdigest()!=supervisor_cmdline_sha256 or b'-c' not in parts or
            parts.index(b'-c')+1>=len(parts) or hashlib.sha256(parts[parts.index(b'-c')+1]).hexdigest()!=entrypoint_sha256):
        raise ValueError('role owner authority entrypoint differs')
    role=state.get('roles',{}).get('coordinator',{})
    if ticks(identity['pid'])==str(identity['start_ticks']) or role.get('alive') or (role.get('pid') and ticks(role['pid']) is not None):
        raise ValueError('old or replacement coordinator is alive')
    if state.get('pending_start') or (Path(control_dir)/'request.json').exists() or (Path(control_dir)/'request.claimed.json').exists():
        raise ValueError('role request remains in flight')
    with socket.socket() as sock:
        sock.settimeout(1);closed=sock.connect_ex(('127.0.0.1',int(port)))==errno.ECONNREFUSED
    if not closed:raise ValueError('coordinator port remains open')
    return {'identity':identity,'nonce':nonce,'process_absent':True,'port_closed':True,'observed_at':time.time()}


def sdk_observe_coordinator_stop(identity,nonce,control_dir,port,supervisor_cmdline_sha256,entrypoint_sha256):
    import inspect
    try:from . import cwctl
    except ImportError:import cwctl
    program=('import hashlib,json,sys,time\nfrom pathlib import Path\n'+inspect.getsource(observe_coordinator_stop)+
             '\nprint(json.dumps(observe_coordinator_stop(json.loads(sys.argv[1]),sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5],sys.argv[6])))\n')
    cwctl.load_env_file();cwctl.require_key();sandbox=cwctl.read_sandbox(identity['sandbox_id'],patience=60)
    result=sandbox.exec(['python3','-c',program,json.dumps(identity,sort_keys=True),nonce,control_dir,str(port),supervisor_cmdline_sha256,entrypoint_sha256],timeout_seconds=30).result(timeout=60)
    if result.returncode!=0:raise ValueError('old coordinator death observation failed')
    return json.loads(result.stdout)


def verify_owned_restart_fence(proof, *, expected_primary, expected_workers, nonce, observe_primary, probe_worker, completed_sandbox_ids=(), get_sandbox=None, now=time.time):
    """Fresh role-owner death and worker quiescence without a provider shutdown."""
    if set(proof)!={'journal','expected_run_id'}:raise ValueError('invalid restart fence request')
    ref=proof['journal'];owner=owner_reference(ref['path'],ref['sha256'],expected_primary)
    if owner['run_id']!=proof['expected_run_id']:raise ValueError('owner journal run differs')
    workers=owner['workers']
    if sorted(map(int,workers))!=sorted(expected_workers):raise ValueError('restart fence omits workers')
    started=now();old=observe_primary(expected_primary,nonce);finished=now()
    if (not isinstance(old,dict) or old.get('identity')!=expected_primary or old.get('nonce')!=nonce
            or old.get('process_absent') is not True or old.get('port_closed') is not True or
            finished<started or finished-started>120):raise ValueError('old coordinator stop is unverified')
    observations=[]
    for rank in sorted(expected_workers):
        entry=workers[str(rank)];identity={key:entry.get(key) for key in ('sandbox_id','boot_id','owner_id','pid','start_ticks')}
        identity['start_ticks']=entry.get('start_ticks') or entry.get('process_start_ticks')
        challenge=secrets.token_hex(24);start=now()
        if identity['sandbox_id'] in completed_sandbox_ids:
            if not callable(get_sandbox) or _top_state(get_sandbox(identity['sandbox_id']),identity['sandbox_id'])!=COMPLETED:
                raise ValueError('completed worker provider fence is not proven')
            result={'identity':identity,'nonce':challenge,'process_absent':True,'owner_quiesced':True,'method':'provider-completed'}
        else:result=probe_worker(identity,challenge)
        end=now()
        if (not isinstance(result,dict) or result.get('identity')!=identity or result.get('nonce')!=challenge
                or result.get('process_absent') is not True or result.get('owner_quiesced') is not True or end<start or end-start>120):
            raise ValueError('worker role remains unverified')
        observations.append({'worker':rank,'identity':identity,'observed_at':end,'nonce':challenge})
    owner_reference(ref['path'],ref['sha256'],expected_primary)
    result={'schema':'natlang.diloco-owner-restart-fence/1','run_id':owner['run_id'],'primary':expected_primary,
            'primary_fenced':True,'all_workers_quiesced':True,'quiesced_workers':sorted(expected_workers),
            'owner_journal':ref,'owner_observation':old,'owner_observed_at':finished,'workers':observations}
    result['proof_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def sdk_authorize_coordinator(role, nonce):
    import inspect
    try:from . import cwctl,diloco_promotion_authority as authority
    except ImportError:
        import cwctl
        import diloco_promotion_authority as authority
    request=role['request']
    if request.get('promotion_nonce')!=nonce:raise ValueError('coordinator request nonce differs')
    program=('import fcntl,json,os,sys\nfrom pathlib import Path\n'
             'sys.path.insert(0,sys.argv[5]+"/training/neuralese")\n'
             'from natlang_neuralese.common.jsonio import write_json_durable\n'+inspect.getsource(authority.authorize_coordinator)+
             '\nprint(json.dumps(authorize_coordinator(sys.argv[1],expected_owner=sys.argv[2],expected_boot=sys.argv[3],nonce=sys.argv[4],request=json.loads(sys.argv[6]))))\n')
    cwctl.load_env_file();cwctl.require_key();sandbox=cwctl.read_sandbox(role['sandbox_id'],patience=60)
    result=sandbox.exec(['python3','-c',program,role['directory'],request['owner_id'],role['boot_id'],nonce,role['runtime'],json.dumps(request,sort_keys=True)],timeout_seconds=30).result(timeout=60)
    if result.returncode!=0:raise ValueError('coordinator authorization failed; worker fence remains active')
    return json.loads(result.stdout)


def verify_replica_loss_fence(owner_evidence, *, replica_sandbox_id, get_sandbox, now=time.time):
    """Add a fresh provider fence for the lost peer to owned-primary stop evidence."""
    if (owner_evidence.get('schema')!='natlang.diloco-owner-restart-fence/1' or
            owner_evidence.get('primary_fenced') is not True or owner_evidence.get('all_workers_quiesced') is not True):
        raise ValueError('primary owner and worker fence is incomplete')
    started=now();sandbox=get_sandbox(replica_sandbox_id);state=_top_state(sandbox,replica_sandbox_id);finished=now()
    if state!=COMPLETED or finished<started or finished-started>120:raise ValueError('lost replica provider fence is not proven')
    result={'schema':'natlang.diloco-replica-loss-fence/1','run_id':owner_evidence['run_id'],
            'primary':owner_evidence['primary'],'primary_process_absent':True,'replica_fenced':True,
            'all_workers_quiesced':True,'quiesced_workers':owner_evidence['quiesced_workers'],
            'owner_evidence':owner_evidence,'replica_sandbox_id':replica_sandbox_id,'replica_observed_at':finished}
    result['proof_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def sdk_authorize_seed_server(role, nonce):
    import inspect
    try:from . import cwctl,diloco_promotion_authority as authority
    except ImportError:
        import cwctl
        import diloco_promotion_authority as authority
    request=role['request']
    if request.get('promotion_nonce')!=nonce:raise ValueError('seed request nonce differs')
    program=('import fcntl,json,os,re,sys\nfrom pathlib import Path\n'
             'sys.path.insert(0,sys.argv[5]+"/training/neuralese")\n'
             'from natlang_neuralese.common.jsonio import write_json_durable\n'+inspect.getsource(authority.authorize_seed_server)+
             '\nprint(json.dumps(authorize_seed_server(sys.argv[1],expected_owner=sys.argv[2],expected_boot=sys.argv[3],nonce=sys.argv[4],request=json.loads(sys.argv[6]))))\n')
    cwctl.load_env_file();cwctl.require_key();sandbox=cwctl.read_sandbox(role['sandbox_id'],patience=60)
    result=sandbox.exec(['python3','-c',program,role['directory'],request['owner_id'],role['boot_id'],nonce,role['runtime'],json.dumps(request,sort_keys=True)],timeout_seconds=30).result(timeout=60)
    if result.returncode!=0:raise ValueError('seed server authorization failed; worker fence remains active')
    return json.loads(result.stdout)


def sdk_stop_owned_role(role):
    """Stop and observe one exact role; never infer exit from acceptance."""
    try:from . import cwctl
    except ImportError:import cwctl
    request=role['request'];publisher=role['publisher']
    if request.get('action')!='stop' or request.get('role') not in ('actor','coordinator'):
        raise ValueError('invalid owned role stop')
    program=('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
        'p=Path(sys.argv[1]);actual=hashlib.sha256(p.read_bytes()).hexdigest()\n'
        'if actual!=sys.argv[2]:raise ValueError("owner publisher differs")\n'
        'print(json.dumps(runpy.run_path(str(p))["submit"](sys.argv[3],json.loads(sys.argv[4]),timeout=60)))\n')
    cwctl.load_env_file();cwctl.require_key();box=cwctl.read_sandbox(role['sandbox_id'],patience=60)
    result=box.exec(['python3','-c',program,publisher['path'],publisher['sha256'],role['directory'],json.dumps(request)],timeout_seconds=90).result(timeout=120)
    if result.returncode!=0 or json.loads(result.stdout).get('accepted') is not True:raise ValueError('owned role stop not accepted')
    probe=('import json,sys,time\nfrom pathlib import Path\n'
        'deadline=time.monotonic()+60\n'
        'while time.monotonic()<deadline:\n'
        ' s=json.loads(Path(sys.argv[1]).read_text());r=s.get("roles",{}).get(sys.argv[2],{})\n'
        ' if r.get("pid")==int(sys.argv[3]) and r.get("alive") is False:break\n'
        ' time.sleep(.2)\n'
        'else:raise SystemExit(1)\n')
    observed=box.exec(['python3','-c',probe,role['directory'].rstrip('/')+'/state.json',request['role'],str(request['expected_pid'])],timeout_seconds=70).result(timeout=90)
    if observed.returncode!=0:raise ValueError('owned role exit not observed')
    return {'pid':request['expected_pid'],'start_ticks':request['expected_process_start_ticks'],'exit_observed':True}


def sdk_confirm_promoted_role(role,nonce):
    """Prove an already-running coordinator is the exactly authorized owner child."""
    try:from . import cwctl
    except ImportError:import cwctl
    program='''import hashlib,json,sys
from pathlib import Path
d=Path(sys.argv[1]);owner=sys.argv[2];boot=sys.argv[3];nonce=sys.argv[4]
s=json.loads((d/'state.json').read_text());fp=d/'promotion-fence.json';ap=d/'promotion-coordinator.json'
if not fp.exists():fp=d/('promotion-release-'+nonce+'.json')
if not ap.exists():ap=d/('promotion-coordinator-'+nonce+'.json')
f=json.loads(fp.read_text());a=json.loads(ap.read_text())
if s.get('owner_id')!=owner or s.get('boot_id')!=boot or f.get('owner_id')!=owner or f.get('boot_id')!=boot or f.get('nonce')!=nonce or a.get('nonce')!=nonce:raise ValueError('owner authorization differs')
r=s.get('roles',{}).get('coordinator',{});pid=r.get('pid');ticks=r.get('start_ticks')
if not r.get('alive') or not isinstance(pid,int) or not ticks:raise ValueError('coordinator role is not alive')
parts=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
if parts[0]=='Z' or parts[19]!=ticks:raise ValueError('coordinator process identity changed')
events=[x for x in s.get('events',[]) if x.get('event')=='start' and x.get('role')=='coordinator' and x.get('pid')==pid]
if len(events)!=1:raise ValueError('coordinator start history ambiguous')
publisher=json.loads((d/('publisher-'+events[0]['request_id']+'.json')).read_text())
if publisher.get('request_sha256')!=a.get('request_sha256'):raise ValueError('running coordinator authorization differs')
print(json.dumps({'sandbox_id':sys.argv[5],'pid':pid,'start_ticks':ticks,'owner_id':owner,'boot_id':boot}))
'''
    cwctl.load_env_file();cwctl.require_key();box=cwctl.read_sandbox(role['sandbox_id'],patience=60)
    result=box.exec(['python3','-c',program,role['directory'],role['owner_id'],role['boot_id'],nonce,role['sandbox_id']],timeout_seconds=30).result(timeout=60)
    if result.returncode!=0:raise ValueError('running coordinator owner is not the exact authorized child')
    return json.loads(result.stdout)


def verify_replacement_owner(ref, *, owner_journal, expected_primary, expected_target, worker_index, nonce, get_sandbox, observe_owner, failed_sandbox_id=None, now=time.time):
    """Bind a fresh replacement owner to one exact failed worker lineage."""
    path=Path(ref['path']).resolve(strict=True);before=path.stat();raw=path.read_bytes();after=path.stat()
    if hashlib.sha256(raw).hexdigest()!=ref['sha256'] or (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('replacement owner journal changed or hash differs')
    value=json.loads(raw);old=owner_journal['workers'][str(worker_index)]
    old_identity={k:old.get(k) for k in ('sandbox_id','boot_id','owner_id','pid','start_ticks')};old_identity['start_ticks']=old.get('start_ticks') or old.get('process_start_ticks')
    new_identity={k:expected_target.get(k) for k in ('sandbox_id','boot_id','owner_id')}
    if (value.get('schema')!='natlang.diloco-replacement-owner/1' or value.get('worker_index')!=worker_index or
            value.get('old_identity')!=old_identity or value.get('new_identity')!=new_identity or
            old_identity['sandbox_id']!=(failed_sandbox_id or expected_primary['sandbox_id']) or new_identity['sandbox_id']==old_identity['sandbox_id'] or
            value.get('control_directory')!=expected_target.get('directory') or value.get('owner_runtime')!=expected_target.get('runtime') or
            value.get('supervisor')!=expected_target.get('supervisor')):
        raise ValueError('replacement journal does not bind the failed worker and successor owner')
    started=now();sandbox=get_sandbox(new_identity['sandbox_id'])
    if _top_state(sandbox,new_identity['sandbox_id'])!=2:raise ValueError('replacement sandbox is not top-level Running')
    observed=observe_owner(value,nonce);finished=now()
    if (finished<started or finished-started>120 or not isinstance(observed,dict) or observed.get('nonce')!=nonce or
            observed.get('new_identity')!=new_identity or observed.get('supervisor')!=value.get('supervisor') or
            observed.get('owner_live') is not True or observed.get('worker_absent') is not True):
        raise ValueError('replacement owner is not freshly verified and quiescent')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('replacement journal changed during verification')
    result={'schema':'natlang.diloco-replacement-owner-proof/1','worker_index':worker_index,'old_identity':old_identity,
            'new_identity':new_identity,'journal_sha256':ref['sha256'],'nonce':nonce,'provider_state':'STATE_RUNNING',
            'observed_at':finished,'supervisor':observed['supervisor']}
    result['proof_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def sdk_observe_replacement_owner(journal,nonce):
    """Observe exact inline nested-owner process on the proposed successor box."""
    try:from . import cwctl
    except ImportError:import cwctl
    program='''import hashlib,json,sys
from pathlib import Path
v=json.loads(sys.argv[1]);nonce=sys.argv[2];n=v['new_identity'];sup=v['supervisor'];d=Path(v['control_directory'])
s=json.loads((d/'state.json').read_text());boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if boot!=n['boot_id'] or s.get('boot_id')!=boot or s.get('owner_id')!=n['owner_id'] or s.get('supervisor_pid')!=sup['pid'] or s.get('supervisor_start_ticks')!=sup['start_ticks']:raise ValueError('successor owner identity changed')
p=Path('/proc/'+str(sup['pid']));fields=(p/'stat').read_text().rsplit(')',1)[1].split();command=(p/'cmdline').read_bytes();args=command.split(b'\\0')
if fields[0]=='Z' or fields[19]!=sup['start_ticks'] or hashlib.sha256(command).hexdigest()!=sup['cmdline_sha256'] or b'-c' not in args or hashlib.sha256(args[args.index(b'-c')+1]).hexdigest()!=sup['entrypoint_sha256']:raise ValueError('successor owner executable differs')
r=s.get('roles',{}).get('worker',{})
if r.get('alive') or s.get('pending_start') or (d/'request.json').exists() or (d/'request.claimed.json').exists():raise ValueError('successor worker is not quiescent')
print(json.dumps({'nonce':nonce,'new_identity':n,'supervisor':sup,'owner_live':True,'worker_absent':True}))
'''
    cwctl.load_env_file();cwctl.require_key();box=cwctl.read_sandbox(journal['new_identity']['sandbox_id'],patience=60)
    result=box.exec(['python3','-c',program,json.dumps(journal,sort_keys=True),nonce],timeout_seconds=30).result(timeout=60)
    if result.returncode!=0:raise ValueError('replacement owner observation failed')
    return json.loads(result.stdout)


def sdk_record_preserved_role(host, nonce, role):
    import inspect
    try:from . import cwctl,diloco_promotion_authority as authority
    except ImportError:
        import cwctl
        import diloco_promotion_authority as authority
    program=('import fcntl,json,os,sys\nfrom pathlib import Path\n'
        'sys.path.insert(0,sys.argv[5]+"/training/neuralese")\nfrom natlang_neuralese.common.jsonio import write_json_durable\n'+inspect.getsource(authority.record_preserved_role)+
        '\nprint(json.dumps(record_preserved_role(sys.argv[1],expected_owner=sys.argv[2],expected_boot=sys.argv[3],nonce=sys.argv[4],preserve_role=json.loads(sys.argv[6]))))\n')
    cwctl.load_env_file();cwctl.require_key();box=cwctl.read_sandbox(host['sandbox_id'],patience=60)
    result=box.exec(['python3','-c',program,host['directory'],host['owner_id'],host['boot_id'],nonce,host['runtime'],json.dumps(role)],timeout_seconds=30).result(timeout=60)
    if result.returncode!=0:raise ValueError('replacement peer role custody was not recorded')
    return json.loads(result.stdout)


def sdk_capture_live_worker(host,nonce):
    """Read exact live worker and inline role-owner identities on one sandbox."""
    try:from . import cwctl
    except ImportError:import cwctl
    program='''import hashlib,json,sys,time
from pathlib import Path
v=json.loads(sys.argv[1]);nonce=sys.argv[2];d=Path(v['directory']);s=json.loads((d/'state.json').read_text());sup=v['supervisor']
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if boot!=v['boot_id'] or s.get('boot_id')!=boot or s.get('owner_id')!=v['owner_id'] or s.get('supervisor_pid')!=sup['pid'] or s.get('supervisor_start_ticks')!=sup['start_ticks']:raise ValueError('worker owner identity differs')
p=Path('/proc/'+str(sup['pid']));fields=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes();args=cmd.split(b'\\0')
if fields[0]=='Z' or fields[19]!=sup['start_ticks'] or hashlib.sha256(cmd).hexdigest()!=sup['cmdline_sha256'] or b'-c' not in args or hashlib.sha256(args[args.index(b'-c')+1]).hexdigest()!=sup['entrypoint_sha256']:raise ValueError('worker supervisor executable differs')
r=s.get('roles',{}).get('worker',{});pid=r.get('pid');ticks=r.get('start_ticks')
if not r.get('alive') or not isinstance(pid,int) or not ticks:raise ValueError('worker role is not live')
p=Path('/proc/'+str(pid));fields=(p/'stat').read_text().rsplit(')',1)[1].split()
if fields[0]=='Z' or fields[19]!=ticks:raise ValueError('worker process identity differs')
events=[x for x in s.get('events',[]) if x.get('event')=='start' and x.get('role')=='worker' and x.get('pid')==pid]
if len(events)!=1 or events[0].get('argv_sha256')!=v['worker_argv_sha256']:raise ValueError('worker launch identity differs')
argv=(p/'cmdline').read_bytes().split(b'\\0');argv=[x.decode('utf-8') for x in argv if x]
if hashlib.sha256(json.dumps(argv,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()!=v['worker_argv_sha256']:raise ValueError('live worker argv differs')
print(json.dumps({'nonce':nonce,'identity':{'sandbox_id':v['sandbox_id'],'boot_id':boot,'owner_id':v['owner_id'],'pid':pid,'start_ticks':ticks},'worker_argv_sha256':v['worker_argv_sha256'],'observed_at':time.time()}))
'''
    cwctl.load_env_file();cwctl.require_key();box=cwctl.read_sandbox(host['sandbox_id'],patience=60)
    result=box.exec(['python3','-c',program,json.dumps(host,sort_keys=True),nonce],timeout_seconds=30).result(timeout=60)
    if result.returncode!=0:raise ValueError('live worker owner observation failed')
    return json.loads(result.stdout)


def verify_full_restore_fence(*, journal_ref, expected_primary, expected_workers, replacements, nonce,
                              get_sandbox, observe_owner, now=time.time):
    """Fence every lost owner and authenticate each exact replacement lineage."""
    owner=owner_reference(journal_ref['path'],journal_ref['sha256'],expected_primary)
    if owner.get('phase')!='complete':raise ValueError('full restore requires captured actual owner journal')
    if sorted(map(int,owner.get('workers',{})))!=sorted(expected_workers) or sorted(replacements)!=sorted(expected_workers):
        raise ValueError('full restore must bind every original and replacement rank')
    old_ids={expected_primary['sandbox_id'],*(owner['workers'][str(rank)]['sandbox_id'] for rank in expected_workers)}
    observations=[]
    for ident in sorted(old_ids):
        before=now();result=get_sandbox(ident);after=now()
        if after<before or after-before>120 or _top_state(result,ident)!=COMPLETED:
            raise ValueError('full restore original owner is not provider completed')
        observations.append({'sandbox_id':ident,'observed_at':after,'state':'STATE_COMPLETED'})
    proofs=[];new_ids=set()
    for rank in sorted(expected_workers):
        item=replacements[rank];target=item['target'];ident=(target['sandbox_id'],target['boot_id'],target['owner_id'])
        if ident in new_ids or target['sandbox_id'] in old_ids:raise ValueError('replacement owner is duplicated or belongs to a fenced sandbox')
        new_ids.add(ident)
        proofs.append(verify_replacement_owner(item['reference'],owner_journal=owner,expected_primary=expected_primary,
            expected_target=target,worker_index=rank,nonce=nonce,get_sandbox=get_sandbox,observe_owner=observe_owner,
            failed_sandbox_id=owner['workers'][str(rank)]['sandbox_id'],now=now))
    owner_reference(journal_ref['path'],journal_ref['sha256'],expected_primary)
    result={'schema':'natlang.diloco-full-restore-fence/1','run_id':owner['run_id'],'primary':expected_primary,
        'primary_fenced':True,'all_workers_quiesced':True,'quiesced_workers':sorted(expected_workers),
        'owner_journal':journal_ref,'completed_owners':observations,'replacement_owners':proofs,'nonce':nonce}
    result['proof_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result
