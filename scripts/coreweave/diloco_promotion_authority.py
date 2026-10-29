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
        path.unlink();fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
