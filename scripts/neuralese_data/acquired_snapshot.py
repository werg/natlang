"""Explicit immutable file subsets of an incompletely downloaded pinned source."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from .common import Reject


def _stat(path):
    value=path.stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


@lru_cache(maxsize=32768)
def _git_blob_cached(path,identity):
    digest=hashlib.sha1(('blob '+str(identity[2])+'\0').encode())
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    if _stat(Path(path))!=identity:
        raise Reject('source changed during pinned Git blob verification')
    return digest.hexdigest()


def git_blob_hex(path):
    path=Path(path).resolve(strict=True)
    return _git_blob_cached(str(path),_stat(path))


def snapshot(ref,root,catalogue,selected,output):
    """Freeze explicitly selected authenticated files; retain every pending path."""
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.common.jsonio import write_json_durable
    from . import acquired_agents
    root=Path(root).resolve();catalogue=Path(catalogue).resolve();output=Path(output).resolve()
    sources=[source for source in json.loads(catalogue.read_text())['sources'] if source['ref']==ref]
    if len(sources)!=1:raise Reject('snapshot requires unique pinned catalogue source')
    source=sources[0];known={entry['path']:entry for entry in source['files']}
    if len(known)!=len(source['files']) or len(set(selected))!=len(selected) or not set(selected)<=known.keys():
        raise Reject('snapshot selection differs from exact unique source catalogue')
    files=[]
    for name in sorted(selected):
        path=(root/name).resolve(strict=True)
        if not path.is_relative_to(root) or path.stat().st_size!=known[name]['bytes']:
            raise Reject('snapshot selected source differs from pinned path/size')
        files.append({'path':name,'bytes':path.stat().st_size,'sha256':sha256_file_hex_cached(path)})
    value={'schema':'natlang.acquired-file-snapshot/1','status':'acquired-file-snapshot',
        'repo':source['repo'],'revision':source['revision'],'card':source.get('card',{}),
        'source_complete':False,'training_admission':False,
        'catalogue':{'path':os.path.relpath(catalogue,output.parent),'bytes':catalogue.stat().st_size,
                     'sha256':sha256_file_hex_cached(catalogue)},
        'files':files,'pending_files':sorted(known.keys()-set(selected))}
    acquired_agents.validate_acquisition(ref,value,output,root=root)
    if output.exists():
        if json.loads(output.read_text())!=value:raise Reject('immutable snapshot output already binds different files')
    else:write_json_durable(output,value)
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',required=True);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--catalogue',required=True,type=Path);parser.add_argument('--file',action='append',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv);value=snapshot(args.ref,args.root,args.catalogue,args.file,args.output)
    print(json.dumps({'output':str(args.output),'files':len(value['files']),
        'pending_files':len(value['pending_files']),'source_complete':False,'training_admission':False}))


if __name__=='__main__':main()
