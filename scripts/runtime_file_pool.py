"""Deduplicate sealed runtime inputs without symlinks to mutable checkouts.

Snapshots retain independent paths and can be exported with ordinary tar. Their
files are immutable inputs: edit main and create a new snapshot instead of
changing a sealed inode. Only executable/read permission bits are retained.
"""
import contextlib
import fcntl
import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import uuid
from pathlib import Path


def _signature(path):
    value=Path(path).stat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode)


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _commit(pool):
    with (pool/'.commit.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def _sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _writer_identity():
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    pid=os.getpid()
    start=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    return boot,str(pid),start


def _writer_live(boot,pid,start):
    try:
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=boot:return False
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]==start
    except FileNotFoundError:return False
    except (OSError,ValueError,IndexError):return True


def prune_unreferenced(pool):
    """Retire only blobs with no snapshot links; scans stay outside commit locks."""
    pool=Path(pool)
    removed=0
    for candidate in pool.glob('.copy.*'):
        parts=candidate.name.split('.')
        if len(parts)!=6 or _writer_live(*parts[2:5]):continue
        with _commit(pool):
            if _writer_live(*parts[2:5]):continue
            try:size=candidate.stat().st_size;candidate.unlink();removed+=size
            except FileNotFoundError:pass
    for candidate in pool.glob('[0-9a-f][0-9a-f]/*'):
        try:before=candidate.stat()
        except FileNotFoundError:continue
        if before.st_nlink!=1:continue
        with _commit(pool):
            try:current=candidate.stat()
            except FileNotFoundError:continue
            if current.st_ino==before.st_ino and current.st_nlink==1:
                candidate.unlink();removed+=current.st_size
    return removed


def seal_file(path,pool,*,durable=True):
    path,pool=Path(path),Path(pool)
    if path.is_symlink() or not path.is_file():raise ValueError('pool needs a regular snapshot file')
    pool.mkdir(parents=True,exist_ok=True,mode=0o700)
    initial=_signature(path)
    digest=_hash(path)
    if _signature(path)!=initial:raise ValueError('runtime input changed during hashing')
    mode=stat.S_IMODE(initial[-1]) & ~0o222
    bucket=pool/digest[:2];bucket.mkdir(exist_ok=True)
    blob=bucket/(digest+'.'+format(mode,'o'))
    temporary=None
    created=False
    try:
        while True:
            # Checking a large existing blob is deliberately outside the lock.
            try:
                existing=blob.stat()
                if (existing.st_size!=initial[2] or stat.S_IMODE(existing.st_mode)!=mode
                        or _hash(blob)!=digest):
                    raise ValueError('immutable runtime pool blob is corrupt')
            except FileNotFoundError:existing=None
            if existing is None and temporary is None:
                fd,name=tempfile.mkstemp(prefix='.copy.'+'.'.join(_writer_identity())+'.',dir=pool)
                temporary=Path(name)
                with os.fdopen(fd,'wb') as output,path.open('rb') as source:
                    shutil.copyfileobj(source,output,length=1024*1024)
                    output.flush();os.fchmod(output.fileno(),mode)
                    if durable:os.fsync(output.fileno())
                if _signature(path)!=initial or _hash(temporary)!=digest:
                    raise ValueError('runtime input changed during pooling')
            with _commit(pool):
                if _signature(path)!=initial:raise ValueError('runtime input changed before sealing')
                try:current=blob.stat()
                except FileNotFoundError:current=None
                if current is None:
                    if temporary is None:continue  # concurrent pruning; recopy outside lock
                    os.link(temporary,blob)
                    if durable:_sync_directory(bucket)
                    created=True
                elif existing is None or (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns) != (
                        existing.st_dev,existing.st_ino,existing.st_size,existing.st_mtime_ns):
                    continue  # concurrent publication; authenticate winner outside lock
                if path.stat().st_ino!=blob.stat().st_ino:
                    alias=path.with_name('.'+path.name+'.pool-'+uuid.uuid4().hex)
                    try:
                        os.link(blob,alias);os.replace(alias,path)
                        if durable:_sync_directory(path.parent)
                    finally:alias.unlink(missing_ok=True)
                return {'bytes':initial[2],'new_blob':created,'sha256':digest}
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def seal_runtime_tree(root,pool=None):
    root=Path(root).resolve(strict=True)
    pool=Path(pool) if pool is not None else root.parent/'.native-runtime-file-pool'
    pool=pool.resolve()
    if pool==root or pool.is_relative_to(root):raise ValueError('runtime pool must be outside snapshot')
    pool.mkdir(parents=True,exist_ok=True,mode=0o700)
    reclaimed=prune_unreferenced(pool)
    totals={'schema':'natlang.immutable-runtime-file-pool/1','pool':str(pool),'files':0,
            'logical_bytes':0,'new_blob_bytes':0,'reused_bytes':0,'pruned_unreferenced_bytes':reclaimed,
            'contract':'read-only shared file inodes; independent snapshot paths; tar exports remain self-contained'}
    # Do not follow directory or file symlinks. Their exact existing meanings
    # remain bound by the normal runtime/dependency manifests.
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=sorted(name for name in dirs if not (Path(directory)/name).is_symlink())
        for name in sorted(files):
            path=Path(directory)/name
            if path.is_symlink():continue
            # The snapshot is not published until the complete filesystem flush
            # below. Per-file flushes serialize tens of thousands of journal
            # commits, even though intermediate trees are not recovery states.
            result=seal_file(path,pool,durable=False)
            totals['files']+=1;totals['logical_bytes']+=result['bytes']
            totals['new_blob_bytes' if result['new_blob'] else 'reused_bytes']+=result['bytes']
    subprocess.run(['sync','-f',str(root)],check=True)
    return totals
