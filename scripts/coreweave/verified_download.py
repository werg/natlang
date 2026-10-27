"""Verify a complete private-object stream, optionally retaining the same bytes.

Failure to retain an optional local copy does not weaken remote verification.
Only a complete, hash-verified temporary file can become a recovery candidate.
Existing recovery files are never replaced by this operation.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile


def verify_download_blocks(blocks, identity, *, destination=None, reserve_bytes=0,
                           sync_bytes=32 << 20, on_progress=None):
    if (type(identity.get('bytes')) is not int or identity['bytes'] < 0
            or not isinstance(identity.get('sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', identity['sha256']) is None):
        raise ValueError('download requires an exact SHA-256 and nonnegative byte count')
    if type(reserve_bytes) is not int or reserve_bytes < 0 or type(sync_bytes) is not int or sync_bytes < 1:
        raise ValueError('download reserve and synchronization window must be valid byte counts')
    pending = None
    output = None
    local = None
    destination = Path(destination) if destination is not None else None

    def abandon(error):
        nonlocal output, pending, local
        local = {'committed': False, 'error': str(error)}
        if output is not None:
            try:
                output.close()
            except OSError:
                pass
            output = None
        if pending is not None:
            try:
                pending.unlink(missing_ok=True)
            except OSError as cleanup_error:
                local.update(unverified_pending=str(pending), cleanup_error=str(cleanup_error))
            pending = None

    try:
        if destination is not None:
            try:
                if destination.exists() or destination.is_symlink():
                    local = {'committed': False, 'reason': 'existing candidate requires its normal local verification'}
                else:
                    free = os.statvfs(destination.parent)
                    if free.f_bavail * free.f_frsize < reserve_bytes + identity['bytes']:
                        raise OSError('insufficient space for the complete copy and recovery reserve')
                    output = tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.owner-verified-', delete=False)
                    pending = Path(output.name)
            except OSError as error:
                abandon(error)
        digest = hashlib.sha256()
        size = window = reported = 0
        for block in blocks:
            digest.update(block)
            size += len(block)
            if size > identity['bytes']:
                raise ValueError('private object exceeds its bound byte count')
            if on_progress is not None and size - reported >= sync_bytes:
                on_progress(size)
                reported = size
            if output is not None:
                try:
                    output.write(block)
                    window += len(block)
                    if window >= sync_bytes:
                        output.flush()
                        os.fdatasync(output.fileno())
                        os.posix_fadvise(output.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        free = os.statvfs(destination.parent)
                        if free.f_bavail * free.f_frsize < reserve_bytes:
                            raise OSError('local recovery free-space reserve reached')
                        window = 0
                except OSError as error:
                    abandon(error)
        if size != identity['bytes'] or digest.hexdigest() != identity['sha256']:
            raise ValueError('independent complete-object SHA/size mismatch')
        if on_progress is not None and reported != size:
            on_progress(size)
        if output is not None:
            try:
                output.flush()
                os.fsync(output.fileno())
                output.close()
                output = None
                # This new, verified inode has no live source readers. Link it
                # exclusively so a concurrent recovery publisher wins safely.
                try:
                    os.link(pending, destination)
                except FileExistsError:
                    local = {'committed': False, 'reason': 'concurrent candidate requires normal local verification'}
                else:
                    local = {'committed': True, 'path': str(destination), **identity}
                pending.unlink()
                pending = None
                fd = os.open(destination.parent, os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError as error:
                abandon(error)
        return {'bytes': size, 'sha256': digest.hexdigest(),
                **({'local_copy': local} if destination is not None else {})}
    finally:
        if output is not None:
            try:
                output.close()
            except OSError:
                pass
        if pending is not None:
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                pass
