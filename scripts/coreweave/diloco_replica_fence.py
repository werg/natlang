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
    program=('import fcntl,json,os,sys\nfrom pathlib import Path\n'
             'sys.path.insert(0,sys.argv[5]+"/training/neuralese")\nfrom natlang_neuralese.common.jsonio import write_json_durable\n'+inspect.getsource(function)+
             '\nprint(json.dumps('+action+'(sys.argv[1],expected_owner=sys.argv[2],expected_boot=sys.argv[3],nonce=sys.argv[4])))\n')
    results=[]
    for worker in workers:
        if set(worker)!={'sandbox_id','directory','owner_id','boot_id','runtime'}:raise ValueError('invalid worker owner binding')
        cwctl.load_env_file();cwctl.require_key();sandbox=cwctl.read_sandbox(worker['sandbox_id'],patience=60)
        result=sandbox.exec(['python3','-c',program,worker['directory'],worker['owner_id'],worker['boot_id'],nonce,worker['runtime']],timeout_seconds=30).result(timeout=60)
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


def verify_owned_restart_fence(proof, *, expected_primary, expected_workers, nonce, observe_primary, probe_worker, now=time.time):
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
        challenge=secrets.token_hex(24);start=now();result=probe_worker(identity,challenge);end=now()
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
