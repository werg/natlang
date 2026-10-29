"""Durable role-start fence shared with the persistent sandbox owner."""
import fcntl
import json
import os
import secrets
from pathlib import Path
from natlang_neuralese.common.jsonio import write_json_durable


def fence(directory, *, expected_owner, expected_boot, nonce, preserve_role=None):
    directory=Path(directory).resolve(strict=True)
    if not isinstance(nonce,str) or len(nonce)<32: raise ValueError('invalid supervisor generation nonce')
    with (directory/'promotion-authority.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads((directory/'state.json').read_text())
        if state.get('owner_id')!=expected_owner or state.get('boot_id')!=expected_boot:
            raise ValueError('role owner identity changed')
        if state.get('pending_start') or (directory/'request.json').exists() or (directory/'request.claimed.json').exists():
            raise ValueError('role request remains in flight')
        for name,role in state.get('roles',{}).items():
            if not role.get('alive'):continue
            events=[event for event in state.get('events',[]) if event.get('event')=='start' and event.get('role')==name and event.get('pid')==role.get('pid')]
            if (not isinstance(preserve_role,dict) or name not in ('actor','coordinator') or
                    set(preserve_role)!={'role','pid','start_ticks','argv_sha256'} or preserve_role.get('role')!=name or
                    preserve_role.get('pid')!=role.get('pid') or preserve_role.get('start_ticks')!=role.get('start_ticks') or
                    len(events)!=1 or events[0].get('argv_sha256')!=preserve_role.get('argv_sha256')):
                raise ValueError('owned roles must be quiesced or exactly preserved')
        preserved=directory/'promotion-preserved-role.json'
        declaration={'schema':'natlang.diloco-promotion-preserved-role/1','nonce':nonce,'role':preserve_role}
        if preserved.exists() and json.loads(preserved.read_text())!=declaration:raise ValueError('preserved role binding differs')
        if not preserved.exists():write_json_durable(preserved,declaration)
        path=directory/'promotion-fence.json'
        value={'schema':'natlang.diloco-role-promotion-fence/1','owner_id':expected_owner,'boot_id':expected_boot,'nonce':nonce}
        if path.exists():
            if json.loads(path.read_text())!=value:raise ValueError('different promotion already fenced this owner')
        else:write_json_durable(path,value)
        return value


def release(directory, *, expected_owner, expected_boot, nonce):
    directory=Path(directory).resolve(strict=True)
    with (directory/'promotion-authority.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        import re
        if not isinstance(nonce,str) or not re.fullmatch('[A-Za-z0-9_-]{32,128}',nonce):raise ValueError('invalid promotion nonce')
        path=directory/'promotion-fence.json';released=directory/('promotion-release-'+nonce+'.json')
        expected={'schema':'natlang.diloco-role-promotion-fence/1','owner_id':expected_owner,'boot_id':expected_boot,'nonce':nonce}
        state=json.loads((directory/'state.json').read_text())
        if state.get('owner_id')!=expected_owner or state.get('boot_id')!=expected_boot:raise ValueError('role owner identity changed')
        if not path.exists():
            if released.is_file() and json.loads(released.read_text())==expected:return
            raise ValueError('promotion fence is absent without matching release receipt')
        if json.loads(path.read_text())!=expected:raise ValueError('promotion fence identity differs')
        if released.exists() and json.loads(released.read_text())!=expected:raise ValueError('promotion release receipt differs')
        if not released.exists():write_json_durable(released,expected)
        authorization=directory/'promotion-coordinator.json'
        if authorization.exists():os.replace(authorization,directory/('promotion-coordinator-'+nonce+'.json'))
        (directory/'promotion-seed-server.json').unlink(missing_ok=True)
        (directory/'promotion-preserved-role.json').unlink(missing_ok=True)
        path.unlink();fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def authorize_coordinator(directory, *, expected_owner, expected_boot, nonce, request):
    """Authorize exactly one bound coordinator request while workers remain fenced."""
    import hashlib
    directory=Path(directory).resolve(strict=True)
    if (not isinstance(request,dict) or request.get('owner_id')!=expected_owner or
            request.get('role')!='coordinator' or request.get('action')!='start' or request.get('promotion_nonce')!=nonce):
        raise ValueError('invalid coordinator promotion request')
    digest=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    with (directory/'promotion-authority.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        fence=json.loads((directory/'promotion-fence.json').read_text())
        if fence!={'schema':'natlang.diloco-role-promotion-fence/1','owner_id':expected_owner,'boot_id':expected_boot,'nonce':nonce}:
            raise ValueError('promotion fence differs')
        state=json.loads((directory/'state.json').read_text())
        if state.get('owner_id')!=expected_owner or state.get('boot_id')!=expected_boot or state.get('generation')!=request.get('expected_generation'):
            raise ValueError('coordinator owner generation changed')
        value={'schema':'natlang.diloco-coordinator-promotion-authorization/1','nonce':nonce,'request_sha256':digest}
        path=directory/'promotion-coordinator.json'
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('another coordinator request is already authorized')
        if not path.exists():write_json_durable(path,value)
        return value


def authorize_seed_server(directory, *, expected_owner, expected_boot, nonce, request):
    """Authorize exactly a hash-bound seed server or replica peer under a fence."""
    import hashlib,re
    directory=Path(directory).resolve(strict=True);argv=request.get('argv',[]) if isinstance(request,dict) else []
    if (request.get('owner_id')!=expected_owner or request.get('role') not in ('actor','coordinator') or request.get('action')!='start'
            or request.get('promotion_nonce')!=nonce or len(argv)!=8 or argv[1:3]!=['-m','natlang_neuralese.train.diloco.replica_cli'] or argv[3] not in ('seed-serve','peer')
            or argv[4]!='--config' or argv[6]!='--config-sha256' or not re.fullmatch('[0-9a-f]{64}',argv[7])):
        raise ValueError('invalid read-only seed server request')
    digest=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    with (directory/'promotion-authority.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=json.loads((directory/'promotion-fence.json').read_text());state=json.loads((directory/'state.json').read_text())
        if marker!={'schema':'natlang.diloco-role-promotion-fence/1','owner_id':expected_owner,'boot_id':expected_boot,'nonce':nonce} or state.get('owner_id')!=expected_owner or state.get('boot_id')!=expected_boot or state.get('generation')!=request.get('expected_generation'):
            raise ValueError('seed owner fence or generation differs')
        value={'schema':'natlang.diloco-seed-promotion-authorization/1','nonce':nonce,'request_sha256':digest}
        path=directory/'promotion-seed-server.json'
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('another seed request is authorized')
        if not path.exists():write_json_durable(path,value)
        return value


def record_preserved_role(directory, *, expected_owner, expected_boot, nonce, preserve_role):
    """Record the exact already-authorized peer role before a later re-fence."""
    directory=Path(directory).resolve(strict=True)
    with (directory/'promotion-authority.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=json.loads((directory/'promotion-fence.json').read_text());state=json.loads((directory/'state.json').read_text())
        if marker!={'schema':'natlang.diloco-role-promotion-fence/1','owner_id':expected_owner,'boot_id':expected_boot,'nonce':nonce} or state.get('owner_id')!=expected_owner or state.get('boot_id')!=expected_boot:
            raise ValueError('preserved role fence differs')
        if not isinstance(preserve_role,dict) or set(preserve_role)!={'role','pid','start_ticks','argv_sha256'} or preserve_role['role'] not in ('actor','coordinator'):
            raise ValueError('invalid preserved role')
        current=state.get('roles',{}).get(preserve_role['role'],{})
        events=[event for event in state.get('events',[]) if event.get('event')=='start' and event.get('role')==preserve_role['role'] and event.get('pid')==preserve_role['pid']]
        if (not current.get('alive') or current.get('pid')!=preserve_role['pid'] or current.get('start_ticks')!=preserve_role['start_ticks'] or
                len(events)!=1 or events[0].get('argv_sha256')!=preserve_role['argv_sha256']):raise ValueError('preserved role is not the exact live child')
        authorization=json.loads((directory/'promotion-seed-server.json').read_text())
        publisher=json.loads((directory/('publisher-'+events[0]['request_id']+'.json')).read_text())
        if authorization.get('nonce')!=nonce or authorization.get('request_sha256')!=publisher.get('request_sha256'):
            raise ValueError('preserved role was not authorized by this fence')
        path=directory/'promotion-preserved-role.json';old=json.loads(path.read_text())
        if old!={'schema':'natlang.diloco-promotion-preserved-role/1','nonce':nonce,'role':None} and old.get('role')!=preserve_role:
            raise ValueError('another role is already preserved')
        value={'schema':'natlang.diloco-promotion-preserved-role/1','nonce':nonce,'role':preserve_role};write_json_durable(path,value)
        return value
