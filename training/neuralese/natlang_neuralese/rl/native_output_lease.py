"""Exclusive native output ownership, reentrant only in the owning thread.

Admission, actor-store setup and nested training share one flock descriptor.
Independent callers cannot overwrite a source before the learner lock rejects
it. Process identity is included so a fork cannot inherit reentrant permission.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import threading

_guard=threading.Lock()
_owners={}


@contextmanager
def coefficient_output_lease(out):
    directory=Path(out).resolve();directory.mkdir(parents=True,exist_ok=True)
    key=(os.getpid(),str(directory));thread=threading.get_ident()
    with _guard:
        entry=_owners.get(key)
        if entry is not None:
            if entry['thread']!=thread:
                raise BlockingIOError('native coefficient output is owned by another caller')
            entry['depth']+=1
        else:
            stream=(directory/'learner.lock').open('a+')
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:
                stream.close()
                raise
            entry={'thread':thread,'depth':1,'stream':stream};_owners[key]=entry
    try:
        yield directory
    finally:
        with _guard:
            entry['depth']-=1
            if not entry['depth']:
                del _owners[key]
                entry['stream'].close()
