#!/usr/bin/env python3
"""Bound digest-cache history without walking source files or changing their bytes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
import time


def _path_kind(key):
    try:
        if key.startswith('content|'):
            path,*numbers=key[8:].rsplit('|',5)
            if len(numbers)!=5:return None
            tuple(map(int,numbers));return path,1
        path,size,mtime=key.rsplit('|',2);int(size);int(mtime);return path,0
    except (ValueError,AttributeError):return None


def _build_index(source, connection, limit):
    scanned=discarded=offset=dropped=0
    while offset<limit:
        raw=source.readline();offset+=len(raw)
        if not raw:break
        scanned+=1
        if raw.endswith(b'\n'):
            try:row=json.loads(raw)
            except (ValueError,UnicodeError):row={}
            if not isinstance(row,dict):row={}
            key=row.get('key');digest=row.get('sha256')
            parsed=_path_kind(key) if isinstance(key,str) else None
            if parsed and isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest):
                path,kind=parsed
                if path.startswith('/tmp/') or path.startswith('/var/tmp/'):
                    discarded+=1
                else:
                    # Keep the first digest for each exact key, including when
                    # signatures recur. Keep only the last observed signature
                    # per source pathname/kind in the compacted lookup log.
                    connection.execute('INSERT OR IGNORE INTO digests VALUES (?,?)',(key,digest))
                    connection.execute('INSERT INTO latest VALUES (?,?,?) ON CONFLICT(path,kind) DO UPDATE SET cache_key=excluded.cache_key',(path,kind,key))
        if offset-dropped>=8*1024**2 and hasattr(os,'posix_fadvise'):
            os.posix_fadvise(source.fileno(),dropped,offset-dropped,os.POSIX_FADV_DONTNEED);dropped=offset
    return scanned,discarded


def _compact(path):
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode):raise ValueError('hash cache must be a regular file')
    temporary=index=None;connection=None
    try:
        descriptor,name=tempfile.mkstemp(prefix='.compact-index-',dir=path.parent);os.close(descriptor);index=Path(name)
        connection=sqlite3.connect(index)
        for sql in ('PRAGMA journal_mode=OFF','PRAGMA synchronous=OFF','PRAGMA cache_size=-8192','PRAGMA temp_store=FILE'):
            connection.execute(sql)
        connection.execute('CREATE TABLE digests (cache_key TEXT PRIMARY KEY, digest TEXT)')
        connection.execute('CREATE TABLE latest (path TEXT,kind INTEGER,cache_key TEXT,PRIMARY KEY(path,kind))')
        with path.open('rb') as source:
            scanned,discarded=_build_index(source,connection,before.st_size)
        connection.commit();kept=0
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as target:
            temporary=Path(target.name)
            for key,digest in connection.execute('SELECT latest.cache_key,digest FROM latest JOIN digests USING(cache_key)'):
                target.write((json.dumps({'key':key,'sha256':digest})+'\n').encode());kept+=1
            target.flush();os.fsync(target.fileno());os.chmod(temporary,stat.S_IMODE(before.st_mode))
        latest=path.lstat()
        if (latest.st_dev,latest.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('hash cache was replaced while compacting')
        # Cache writers need not cooperate: an append missed at this boundary
        # causes a cache miss and a fresh source hash, never unverified reuse.
        os.replace(temporary,path);temporary=None
        after=path.stat();state={'device':after.st_dev,'inode':after.st_ino,'bytes':after.st_size,'unix':time.time()}
        state_path=path.with_name(path.name+'.compact-state.json')
        state_path.write_text(json.dumps(state)+'\n')
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return {'compacted':True,'before_bytes':before.st_size,'after_bytes':after.st_size,
                'scanned_entries':scanned,'retained_entries':kept,'discarded_disposable_entries':discarded,
                'observed_concurrent_append_bytes':max(0,latest.st_size-before.st_size),
                'scope':'derived digest cache only; retain first digest for each surviving exact key and last observed signature per path/kind; source signatures are checked on lookup; misses rehash source bytes'}
    finally:
        if connection is not None:connection.close()
        for item in (temporary,index):
            if item is not None:item.unlink(missing_ok=True)


def compact(path, *, threshold=128*1024**2):
    path=Path(path)
    if not path.exists() or path.stat().st_size<threshold:return {'compacted':False}
    with path.with_name(path.name+'.compact.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'compacted':False,'reason':'another compactor is active'}
        if threshold>0:
            try:
                prior=json.loads(path.with_name(path.name+'.compact-state.json').read_text());current=path.stat()
                if ((current.st_dev,current.st_ino)==(prior['device'],prior['inode'])
                        and current.st_size-prior['bytes']<64*1024**2 and time.time()-prior['unix']<86400):
                    return {'compacted':False,'reason':'recently compacted; little new cache history'}
            except (OSError,ValueError,KeyError,TypeError):pass
        return _compact(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path',type=Path,default=Path.home()/'.cache/natlang/sha256-cache.jsonl')
    parser.add_argument('--threshold-mib',type=int,default=128)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args();result=compact(args.path,threshold=args.threshold_mib*1024**2)
    if args.receipt:args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
