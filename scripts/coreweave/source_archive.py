#!/usr/bin/env python3
"""Exact TAR+zstd transport for an owner-selected immutable source directory.

The existing mined-source archive protocol is retained. No extraction, source
deletion, quality decision or training admission occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import tempfile


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _files(source, *, dereference_file_symlinks=False):
    paths = sorted(source.rglob('*'))
    for path in paths:
        if path.is_symlink() and (not dereference_file_symlinks or not path.is_file()):
            raise ValueError('source archive refuses symlinks unless explicit file dereference is enabled')
    return [path for path in paths if path.is_file()]


def _write_json(path, value):
    pending = path.with_name(path.name + '.pending')
    with pending.open('w') as stream:
        json.dump(value, stream, indent=1)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def pack(source, spool, name, *, reserve_bytes=20 * 2**30, dereference_file_symlinks=False):
    source, spool = Path(source), Path(spool)
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('archive name must be one ordinary path component')
    if not source.is_dir() or source.is_symlink():
        raise ValueError('source must be a real directory')
    spool.mkdir(parents=True, exist_ok=True)
    if spool.resolve().is_relative_to(source.resolve()):
        raise ValueError('spool must be outside the archived source')
    target = spool / (name + '.tar.zst')
    pending = target.with_suffix('.pending')
    if target.exists() or pending.exists():
        raise FileExistsError(target if target.exists() else pending)
    files = _files(source, dereference_file_symlinks=dereference_file_symlinks)
    if not files:
        raise ValueError('source contains no files')
    if source.joinpath('transfer-manifest.json').exists():
        raise ValueError('source uses the reserved transfer-manifest.json name')
    bindings = []

    class VerifiedReader:
        def __init__(self, stream):
            self.stream, self.digest, self.size = stream, hashlib.sha256(), 0

        def read(self, count):
            if shutil.disk_usage(spool).free < reserve_bytes:
                raise OSError('archive spool reached its declared reserve')
            data = self.stream.read(count)
            self.digest.update(data)
            self.size += len(data)
            return data

    with pending.open('xb') as output:
        process = subprocess.Popen(['zstd', '-q', '-T1', '-3', '--long=27', '-c'],
                                   stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=process.stdin, mode='w|', copybufsize=8 << 20,
                              dereference=True) as archive:
                for path in files:
                    link = (os.readlink(path), path.lstat().st_ino, str(path.resolve(strict=True))) if path.is_symlink() else None
                    before = path.stat()
                    info = archive.gettarinfo(str(path), arcname=name + '/' + str(path.relative_to(source)))
                    if not info.isfile():
                        raise ValueError('archive member is not a regular file')
                    with path.open('rb') as stream:
                        reader = VerifiedReader(stream)
                        archive.addfile(info, reader)
                    after = path.stat()
                    after_link = (os.readlink(path), path.lstat().st_ino, str(path.resolve(strict=True))) if path.is_symlink() else None
                    if ((before.st_size, before.st_mtime_ns, before.st_ino) !=
                            (after.st_size, after.st_mtime_ns, after.st_ino) or reader.size != before.st_size
                            or link != after_link):
                        raise ValueError('source changed during archive: ' + str(path))
                    bindings.append({'path': str(path.relative_to(source)), 'bytes': reader.size,
                                     'sha256': reader.digest.hexdigest(),
                                     **({'source_symlink': link[0], 'resolved_source': link[2]} if link else {})})
                if _files(source, dereference_file_symlinks=dereference_file_symlinks) != files:
                    raise ValueError('source file set changed during archive')
                manifest = {'schema': 'natlang.mined-source-transfer/1', 'source': str(source),
                            'name': name, 'training_admission': False, 'files': bindings,
                            'format': 'tar+zstd; scoped private transfer; exact bytes; '
                                      'no source deletion before independent durable verification'}
                payload = json.dumps(manifest, sort_keys=True, indent=1).encode()
                info = tarfile.TarInfo(name + '/transfer-manifest.json')
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('zstd archive compression failed')
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            process.terminate()
            process.wait()
            pending.unlink(missing_ok=True)
            raise
    pending.replace(target)
    directory = os.open(spool, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    receipt = {'schema': 'natlang.mined-source-archive-receipt/1', 'source': str(source), 'name': name,
               'archive': str(target), 'archive_bytes': target.stat().st_size,
               'archive_sha256': digest_file(target), 'source_bytes': sum(row['bytes'] for row in bindings),
               'source_files': len(bindings), 'manifest': manifest, 'private_durability': 'pending',
               'training_admission': False, 'at': time.time()}
    _write_json(spool / (name + '.receipt.json'), receipt)
    return receipt


def verify(spool, name):
    spool = Path(spool)
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('archive name must be one ordinary path component')
    receipt = json.loads((spool / (name + '.receipt.json')).read_text())
    archive = Path(receipt['archive'])
    if archive.stat().st_size != receipt['archive_bytes'] or digest_file(archive) != receipt['archive_sha256']:
        raise ValueError('archive bytes differ from the sealed receipt')
    expected = {name + '/' + row['path']: row for row in receipt['manifest']['files']}
    if len(expected) != len(receipt['manifest']['files']):
        raise ValueError('duplicate source member in receipt')
    seen = set()
    process = subprocess.Popen(['zstd', '-dcq', '--long=27', '--', str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as tar:
            for member in tar:
                if not member.isfile() or member.name in seen:
                    raise ValueError('nonregular or duplicate archive member')
                seen.add(member.name)
                stream = tar.extractfile(member)
                if member.name == name + '/transfer-manifest.json':
                    if json.load(stream) != receipt['manifest']:
                        raise ValueError('embedded manifest differs from receipt')
                    continue
                row = expected[member.name]
                digest, size = hashlib.sha256(), 0
                while block := stream.read(8 << 20):
                    digest.update(block)
                    size += len(block)
                if size != member.size or size != row['bytes'] or digest.hexdigest() != row['sha256']:
                    raise ValueError('decoded member differs from original bytes: ' + member.name)
        if seen != set(expected) | {name + '/transfer-manifest.json'} or process.wait() != 0:
            raise ValueError('archive member set or decoder exit differs')
    except BaseException:
        process.terminate()
        process.wait()
        raise
    proof = {'status': 'passed', 'name': name, 'archive_sha256': receipt['archive_sha256'],
             'files': len(expected), 'source_bytes': receipt['source_bytes'], 'private_durability': 'pending',
             'training_admission': False,
             'control': 'Independent decoded TAR member hashes match every original byte binding; '
                        'no extraction or source deletion'}
    _write_json(spool / (name + '.roundtrip-proof.json'), proof)
    return proof


def restore(spool, name, target):
    """Restore an already sealed archive atomically per member, with exact bytes."""
    spool,target=Path(spool),Path(target)
    proof=verify(spool,name)
    target.mkdir(parents=True,exist_ok=True);target=target.resolve(strict=True)
    receipt=json.loads((spool/(name+'.receipt.json')).read_text())
    expected={name+'/'+row['path']:row for row in receipt['manifest']['files']}
    for row in expected.values():
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('archive restore member escaped its target')
    process=subprocess.Popen(['zstd','-dcq','--long=27','--',str(receipt['archive'])],stdout=subprocess.PIPE)
    restored=0;seen=set()
    try:
        with tarfile.open(fileobj=process.stdout,mode='r|') as tar:
            for member in tar:
                if member.name in seen:raise ValueError('duplicate archive restore member')
                seen.add(member.name)
                stream=tar.extractfile(member)
                if member.name==name+'/transfer-manifest.json':
                    if json.load(stream)!=receipt['manifest']:raise ValueError('embedded restore manifest changed')
                    continue
                row=expected[member.name]
                if not member.isfile():raise ValueError('archive restore refuses nonregular members')
                path=target/row['path']
                if not path.parent.resolve().is_relative_to(target):
                    raise ValueError('archive restore target escaped through a link')
                path.parent.mkdir(parents=True,exist_ok=True)
                if not path.parent.resolve().is_relative_to(target) or path.is_symlink():
                    raise ValueError('archive restore target escaped through a link')
                digest,size=hashlib.sha256(),0
                with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.cw-restore-',delete=False) as out:
                    pending=Path(out.name)
                    try:
                        while block:=stream.read(8<<20):out.write(block);digest.update(block);size+=len(block)
                        out.flush();os.fsync(out.fileno())
                        if size!=row['bytes'] or size!=member.size or digest.hexdigest()!=row['sha256']:
                            raise ValueError('restored member differs from declared bytes')
                        os.replace(pending,path)
                        directory=os.open(path.parent,os.O_DIRECTORY)
                        try:os.fsync(directory)
                        finally:os.close(directory)
                    finally:pending.unlink(missing_ok=True)
                restored+=1
        if process.wait()!=0 or restored!=len(expected):raise ValueError('archive restore decoder/member count differs')
    except BaseException:
        process.terminate();process.wait();raise
    return {**proof,'target':str(target),'restored_files':restored,
        'control':'Decoded archive and every target SHA/size verified; each member fsynced and atomically replaced; no training admission'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('pack', 'verify', 'restore'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--spool', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--target',type=Path)
    parser.add_argument('--reserve-gb', type=float, default=20)
    parser.add_argument('--dereference-file-symlinks', action='store_true',
                        help='archive immutable linked file bytes as regular members, recording target provenance')
    args = parser.parse_args(argv)
    if args.operation == 'pack':
        if args.source is None or args.reserve_gb < 0:
            parser.error('pack requires --source and a nonnegative reserve')
        result = pack(args.source, args.spool, args.name, reserve_bytes=int(args.reserve_gb * 2**30),
                      dereference_file_symlinks=args.dereference_file_symlinks)
    elif args.operation=='restore':
        if args.target is None:parser.error('restore requires --target')
        result=restore(args.spool,args.name,args.target)
    else:
        result = verify(args.spool, args.name)
    print(json.dumps({key: value for key, value in result.items() if key != 'manifest'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
