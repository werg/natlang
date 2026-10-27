"""Reuse immutable archive files only after normal decode and store admission."""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import stat


def signature(path):
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f'native archive is not a regular file: {path}')
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def read_archive(path):
    path = Path(path).resolve(strict=True)
    before = signature(path)
    body = path.read_bytes()
    if signature(path) != before:
        raise ValueError(f'native archive changed during verification: {path}')
    return body, (path, before, hashlib.sha256(body).hexdigest())


def remember(engine, ident, reference):
    if len(reference) == 2:
        # Newly serialized files must be bound to the admitted values, rather
        # than blessing whatever bytes happen to be at that path afterwards.
        path, expected = reference
        if ident.startswith('nzs1_'):
            from ..serve.store import encode_native_state
            record = engine.store.get_native_state(ident)
            body = encode_native_state(record.block)
        else:
            from ..model.native_artifact import encode_native_tensor_artifact
            record = engine.store.get_native_tensor_artifact(ident)
            body = encode_native_tensor_artifact(record.artifact)
        reference = (path, expected, hashlib.sha256(body).hexdigest())
    if not hasattr(engine, '_verified_native_archive_files'):
        engine._verified_native_archive_files = {}
    engine._verified_native_archive_files[ident] = reference


def link_verified(engine, ident, target):
    """Link exact admitted bytes when their original immutable inode is unchanged.

    Content-addressed archives are published once and never edited in place.
    A replaced, removed or changed source uses the admitted store values instead.
    Cross-filesystem outputs retain ordinary serialization.
    """
    references = getattr(engine, '_verified_native_archive_files', {})
    reference = references.get(ident)
    if reference is None:
        return False
    source, expected, digest = reference
    try:
        if signature(source) != expected:
            references.pop(ident, None)
            return False
        os.link(source, target)
    except ValueError:
        references.pop(ident, None)
        return False
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.EXDEV, errno.EPERM, errno.EACCES,
                          errno.EMLINK, errno.EOPNOTSUPP, errno.EEXIST}:
            return False
        raise
    actual = signature(target)
    observed = hashlib.sha256()
    try:
        with Path(target).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                observed.update(chunk)
        stable = signature(target) == actual
    except OSError:
        Path(target).unlink(missing_ok=True)
        references.pop(ident, None)
        raise
    if actual[:4] != expected[:4] or not stable or observed.hexdigest() != digest:
        Path(target).unlink()
        references.pop(ident, None)
        return False
    # Creating the link changes inode ctime, not its content. Future links check
    # that updated identity rather than treating our own link as a source edit.
    references[ident] = (source, actual, digest)
    return True
