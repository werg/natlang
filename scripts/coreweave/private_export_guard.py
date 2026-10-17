"""Serialize exact-content exports and reuse only verified, still-present objects."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def _identity(value):
    if (not isinstance(value, dict) or not re.fullmatch('[0-9a-f]{64}', str(value.get('sha256', '')))
            or type(value.get('bytes')) is not int or value['bytes'] <= 0):
        raise ValueError('private export needs an exact nonempty SHA-256/size binding')
    return {key:value[key] for key in ('sha256', 'bytes')}


def validate_receipt(receipt, identity):
    """A PUT success alone is insufficient; require the independent complete GET."""
    identity = _identity(identity)
    if (not isinstance(receipt, dict) or receipt.get('private') is not True
            or _identity(receipt.get('identity')) != identity
            or not isinstance(receipt.get('independent_verification'), dict)
            or any(receipt['independent_verification'].get(key) != value for key,value in identity.items())
            or not isinstance(receipt.get('run'), str) or not receipt['run']
            or not isinstance(receipt.get('name'), str) or not receipt['name']
            or not isinstance(receipt.get('source'), str) or not receipt['source']):
        raise ValueError('private export receipt lacks matching full independent verification')
    return receipt


class ExportLease:
    def __init__(self, path, scope, identity, receipt):
        self.path, self.scope, self.identity, self.receipt = path, scope, identity, receipt

    def record(self, receipt):
        validate_receipt(receipt, self.identity)
        value = {'schema':'natlang.private-export-cache/1', 'scope':self.scope,
                 'identity':self.identity, 'receipt':receipt}
        pending = None
        try:
            fd,pending = tempfile.mkstemp(prefix='.'+self.path.name, dir=self.path.parent)
            with os.fdopen(fd,'w') as stream:
                json.dump(value,stream,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            os.replace(pending,self.path);pending = None
            fd = os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        finally:
            if pending is not None:Path(pending).unlink(missing_ok=True)
        self.receipt = receipt


@contextmanager
def export_lease(directory, *, scope, identity, object_exists):
    """Hold through the existing export's complete PUT+independent GET lifecycle.

    object_exists must perform an authenticated provider check. Return False only
    for confirmed absence; provider/auth/network errors must propagate.
    """
    identity = _identity(identity)
    if not isinstance(scope,str) or not scope:raise ValueError('private export needs its provider/owner scope')
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=True)
    key = hashlib.sha256(json.dumps({'scope':scope,'identity':identity},sort_keys=True).encode()).hexdigest()
    path = directory/(key+'.json')
    with (directory/(key+'.lock')).open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            receipt = None
            if path.exists():
                cached = json.loads(path.read_text())
                if (cached.get('schema') != 'natlang.private-export-cache/1'
                        or cached.get('scope') != scope or cached.get('identity') != identity):
                    raise ValueError('private export cache identity changed')
                receipt = validate_receipt(cached.get('receipt'),identity)
                exists = object_exists(receipt)
                if type(exists) is not bool:raise ValueError('private object existence check must return a boolean')
                if not exists:receipt = None
            yield ExportLease(path,scope,identity,receipt)
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def receipt_for_source(receipt, *, source, identity):
    """Bind already-hashed equivalent source bytes without inventing a new upload."""
    import copy
    validate_receipt(receipt,identity)
    if not isinstance(source,str) or not source:raise ValueError('equivalent export source path is required')
    result = copy.deepcopy(receipt)
    result['source'] = source
    result['reused_verified_export'] = {
        'upload_source':receipt.get('reused_verified_export',{}).get('upload_source',receipt['source']),
        'verified_at':receipt.get('verified_at'),
        'identity':_identity(identity),
        'new_upload':False,
        'independent_readback_repeated':False,
    }
    return result
