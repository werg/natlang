"""Durable role-start fence shared with the persistent sandbox owner."""
import fcntl
import json
import os
import secrets
from pathlib import Path
from natlang_neuralese.common.jsonio import write_json_durable


def fence(directory, *, expected_owner, expected_boot, nonce):
    directory=Path(directory).resolve(strict=True)
    if not isinstance(nonce,str) or len(nonce)<32: raise ValueError('invalid supervisor generation nonce')
    with (directory/'promotion-authority.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads((directory/'state.json').read_text())
        if state.get('owner_id')!=expected_owner or state.get('boot_id')!=expected_boot:
            raise ValueError('role owner identity changed')
        if state.get('pending_start') or (directory/'request.json').exists() or (directory/'request.claimed.json').exists():
            raise ValueError('role request remains in flight')
        if any(role.get('alive') for role in state.get('roles',{}).values()):
            raise ValueError('owned roles must be quiesced first')
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
        path=directory/'promotion-fence.json';value=json.loads(path.read_text())
        if value!={'schema':'natlang.diloco-role-promotion-fence/1','owner_id':expected_owner,'boot_id':expected_boot,'nonce':nonce}:
            raise ValueError('promotion fence identity differs')
        state=json.loads((directory/'state.json').read_text())
        if state.get('owner_id')!=expected_owner or state.get('boot_id')!=expected_boot:raise ValueError('role owner identity changed')
        (directory/'promotion-coordinator.json').unlink(missing_ok=True)
        (directory/'promotion-seed-server.json').unlink(missing_ok=True)
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
