"""Exact source-bound, explicit logical objective families."""
import hashlib,json,re
from pathlib import Path

def checked(reference):
    path=Path(reference['path']).resolve(strict=True)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=reference['sha256']:raise ValueError('logical family map SHA mismatch')
    return path

def load_map(reference, source):
    path=checked(reference);before=path.stat();doc=json.loads(path.read_text())
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
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('logical family map changed while reading')
    return result
