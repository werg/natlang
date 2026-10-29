"""Bound negative-reference scans to a revalidated JSON namespace snapshot."""
from __future__ import annotations
import os
from pathlib import Path


def _snapshot_once(roots):
    """Capture directory and JSON identities, refusing inaccessible roots or races.

    Directory identities detect newly created/removed names; JSON identities
    detect content changes. No symlink directory traversal is performed.
    """
    result={}
    def identity(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    def walk_error(error):raise error
    for root in map(Path,roots):
        if not root.is_dir():raise ValueError('reference root unavailable: '+str(root))
        for directory,dirs,files in os.walk(root,followlinks=False,onerror=walk_error):
            path=Path(directory);before=identity(path.stat());result[str(path)]=before
            for name in files:
                if name.endswith('.json'):
                    p=path/name;result[str(p)]=identity(p.stat())
            if identity(path.stat())!=before:raise RuntimeError('reference directory changed during snapshot')
    return result


def snapshot(roots, *, attempts=3):
    for attempt in range(attempts):
        try:return _snapshot_once(roots)
        except (RuntimeError,FileNotFoundError):
            if attempt+1==attempts:raise
    raise RuntimeError('reference namespace did not stabilize')


def revalidate(roots, expected):
    if snapshot(roots)!=expected:raise RuntimeError('reference namespace changed since scan')


def refresh_references(roots, previous, references, digests, scan, *, attempts=3):
    """Refresh only JSON names whose five-field identity changed or appeared.

    Deleted names are removed from the reference map. Rescan changed files and
    retry if the namespace changes again; no stale negative is accepted.
    """
    refs={key:set(value) for key,value in references.items()}
    for _ in range(attempts):
        current=snapshot(roots)
        changed={p for p in set(previous)|set(current) if p.endswith('.json') and previous.get(p)!=current.get(p)}
        for paths in refs.values():paths.difference_update(changed)
        existing=[p for p in changed if p in current]
        if existing:
            fresh=scan(digests,existing)
            for digest in digests:refs[digest].update(fresh[digest])
        after=snapshot(roots)
        if current==after:return after,refs
        previous=current
    raise RuntimeError('reference namespace did not stabilize within bounded retries')
