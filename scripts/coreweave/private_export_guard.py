"""Serialize exact-content exports and reuse only verified, still-present objects."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile


def commit_uploaded_run_file(api, run_path, name):
    """Use the SDK's upload notification; callers still verify the full GET.

    Match Run.upload_file without uploading again or changing a finished run's
    state. Older servers retain their native object-store notification path.
    Service errors propagate: notification failure is not verified custody.
    """
    from wandb.proto import wandb_api_pb2 as apb
    from wandb.proto import wandb_internal_pb2 as pb

    entity, project, run_id = run_path.split('/')
    service = api.run(run_path)._service_api
    if not service.feature_enabled(pb.MARK_RUN_FILES_UPLOADED):
        return False
    service.send_api_request(apb.ApiRequest(
        mark_run_files_uploaded_request=apb.MarkRunFilesUploadedRequest(
            entity=entity, project=project, run_id=run_id, files=[name])))
    return True


class CacheBoundedReader(io.FileIO):
    """Read unchanged file bytes while advising away consumed clean cache pages.

    FileIO preserves fileno/tell/seek and length discovery for HTTP clients.
    Seeked-over ranges are never treated as consumed. Advice is an I/O hint;
    callers still hash every byte and independently verify the private object.
    """
    def __init__(self, path, *, release_bytes=16*1024**2):
        if type(release_bytes) is not int or release_bytes <= 0:
            raise ValueError('cache release interval must be a positive byte count')
        super().__init__(path, 'r')
        self.release_bytes = release_bytes
        self._released = self._next = 0

    def _before_read(self):
        position = self.tell()
        if position != self._next:
            self._released = position

    def _after_read(self, eof):
        position = self.tell()
        count = position-self._released
        if count and (count >= self.release_bytes or eof):
            os.posix_fadvise(self.fileno(), self._released, count, os.POSIX_FADV_DONTNEED)
            self._released = position
        self._next = position

    def read(self, size=-1):
        self._before_read()
        value = super().read(size)
        self._after_read(not value)
        return value

    def readinto(self, buffer):
        self._before_read()
        count = super().readinto(buffer)
        self._after_read(not count)
        return count


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
            or receipt['independent_verification'].get('status') == 'awaiting_target_readback'
            or receipt.get('status') == 'awaiting_target_readback'
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


@contextmanager
def sandbox_upload_lease(directory, sandbox_id, *, verified_legacy_reader=None):
    """Serialize PUTs while allowing completed-object readbacks to overlap.

    Old exporters held their original lock throughout GET verification. During
    migration a caller may prove that exact lock holder is already only reading
    its completed object. All new uploaders still share the new upload lock.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(sandbox_id.encode()).hexdigest()
    legacy = (directory / ('sandbox-upload-' + key + '.lock')).open('a+b')
    modern = (directory / ('sandbox-put-' + key + '.lock')).open('a+b')
    legacy_held = False
    try:
        try:
            fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            legacy_held = True
        except BlockingIOError:
            if verified_legacy_reader is None or not verified_legacy_reader(legacy):
                fcntl.flock(legacy, fcntl.LOCK_EX)
                legacy_held = True
        fcntl.flock(modern, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(modern, fcntl.LOCK_UN)
        if legacy_held:
            fcntl.flock(legacy, fcntl.LOCK_UN)
        modern.close()
        legacy.close()
