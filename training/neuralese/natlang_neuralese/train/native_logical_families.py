"""Exact source-bound, explicit logical objective families."""
import hashlib,json,re
from pathlib import Path

def _verified_bytes(reference):
    path=Path(reference['path']).resolve(strict=True)
    before=path.stat();raw=path.read_bytes();after=path.stat()
    stat=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if stat(before)!=stat(after):raise ValueError('logical family map changed while reading')
    if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('logical family map SHA mismatch')
    return path,raw

def checked(reference):
    return _verified_bytes(reference)[0]

def load_map(reference, source):
    _,raw=_verified_bytes(reference)
    doc=json.loads(raw)
    if doc.get('schema')!='natlang.logical-family-map/1' or doc.get('source_sha256')!=source.file_sha256:
        raise ValueError('logical family map source differs')
    rows=doc.get('rows')
    if not isinstance(rows,list) or len(rows)!=len(source.entries):raise ValueError('logical family map row count differs')
    result=[]
    for row,entry in zip(rows,source.entries,strict=True):
        if (not isinstance(row,dict) or set(row)!={'id','binding','family'}
                or row['id']!=entry['id'] or row['binding']!=entry['binding']
                or not isinstance(row['family'],str) or not re.fullmatch('[a-z][a-z0-9_-]*',row['family'])):
            raise ValueError('logical family row identity or family differs')
        result.append(row['family'])
    return result
