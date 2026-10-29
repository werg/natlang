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


def _checked_changed(paths, digests, *, attempts=3):
    results={d:set() for d in digests}
    def sig(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    for path in paths:
        for attempt in range(attempts):
            try:
                with open(path,'rb') as handle:
                    before=sig(os.fstat(handle.fileno()));tail=b'';found=set()
                    while block:=handle.read(1<<20):
                        data=tail+block
                        for digest in digests:
                            if digest.encode('ascii') in data:found.add(digest)
                        tail=data[-63:]
                    if sig(os.fstat(handle.fileno()))!=before or sig(os.stat(path))!=before:raise RuntimeError('changed during checked read')
                for digest in found:results[digest].add(path)
                break
            except FileNotFoundError:
                if not os.path.lexists(path):break
                if attempt+1==attempts:raise
            except RuntimeError:
                if attempt+1==attempts:raise
    return results


def refresh_point_in_time(roots, previous, references, digests):
    """One namespace snapshot and checked delta, for a bounded commit fence.

    Each changed file is checked by descriptor and path identity. Concurrent
    writers outside the shared lease lock mean this is not a global snapshot.
    """
    current=snapshot(roots);refs={d:set(references[d]) for d in digests}
    changed={p for p in set(previous)|set(current) if p.endswith('.json') and previous.get(p)!=current.get(p)}
    for values in refs.values():values.difference_update(changed)
    fresh=_checked_changed([p for p in changed if p in current],digests)
    for digest in digests:refs[digest].update(fresh[digest])
    return current,refs


def refresh_references(roots, previous, references, digests, scan, *, attempts=3, allow_busy=False):
    """Refresh only JSON names whose five-field identity changed or appeared.

    Deleted names are removed from the reference map. Rescan changed files and
    retry if the namespace changes again; no stale negative is accepted.
    """
    refs={key:set(value) for key,value in references.items()}
    for _ in range(attempts):
        current=snapshot(roots)
        changed={p for p in set(previous)|set(current) if p.endswith('.json') and previous.get(p)!=current.get(p)}
        prior={key:set(value) for key,value in refs.items()}
        for paths in refs.values():paths.difference_update(changed)
        existing=[p for p in changed if p in current]
        if existing:
            try:fresh=scan(digests,existing)
            except (RuntimeError,FileNotFoundError):
                refs=prior
                continue
            for digest in digests:refs[digest].update(fresh[digest])
        after=snapshot(roots)
        # Directory identities guard each walk, but unrelated non-JSON writes
        # need not invalidate a completed scan of the JSON namespace.
        def json_files(items):return {p:v for p,v in items.items() if p.endswith('.json')}
        if json_files(current)==json_files(after):return after,refs
        previous=current
    if not allow_busy:raise RuntimeError('reference namespace did not stabilize within bounded retries')
    # Managed consumers participate in the retention lock. Other writers do
    # not, so this final refresh is explicitly a point-in-time observation,
    # not an assertion of a globally atomic filesystem namespace.
    current=snapshot(roots)
    changed={p for p in set(previous)|set(current) if p.endswith('.json') and previous.get(p)!=current.get(p)}
    for paths in refs.values():paths.difference_update(changed)
    fresh=_checked_changed([p for p in changed if p in current],digests)
    for digest in digests:refs[digest].update(fresh[digest])
    return current,refs
