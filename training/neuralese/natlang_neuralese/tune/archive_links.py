"""Reuse immutable archive files only after normal decode and store admission."""
from __future__ import annotations

import errno
import os
from pathlib import Path
import stat


def signature(path):
    value = Path(path).stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f'native archive is not a regular file: {path}')
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def read_archive(path):
    path = Path(path).resolve(strict=True)
    before = signature(path)
    body = path.read_bytes()
    if signature(path) != before:
        raise ValueError(f'native archive changed during verification: {path}')
    return body, (path, before)


def remember(engine, ident, reference):
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
    source, expected = reference
    try:
        if signature(source) != expected:
            references.pop(ident, None)
            return False
        os.link(source, target)
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.EXDEV, errno.EPERM, errno.EACCES,
                          errno.EMLINK, errno.EOPNOTSUPP, errno.EEXIST}:
            return False
        raise
    actual = signature(target)
    if actual[:4] != expected[:4]:
        Path(target).unlink()
        references.pop(ident, None)
        return False
    # Creating the link changes inode ctime, not its content. Future links check
    # that updated identity rather than treating our own link as a source edit.
    references[ident] = (source, actual)
    return True
