#!/usr/bin/env python3
"""Send an exact actor transport, automatically reusing the peer's previous view.

The peer cache is a performance hint, never model identity or training credit.
Missing bases use a full transfer; an available base uses the smaller of an
authenticated XOR delta and the complete view. Receipts commit after readback.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shlex,subprocess,tempfile,time
from pathlib import Path
try:
    from scripts.coreweave.artifact_delta import build,digest_file
except ModuleNotFoundError:
    from artifact_delta import build,digest_file


def remote(host,code,*args):
    command=shlex.join(['python3','-I','-',*map(str,args)])
    result=subprocess.run(['ssh','-o','ConnectTimeout=10',host,command],input=code,text=True,capture_output=True,timeout=1800)
    if result.returncode:raise RuntimeError('peer operation failed: '+result.stderr[-3000:])
    return json.loads(result.stdout)


def copy(host,source,target,*,compress=False):
    subprocess.run(['scp',*(['-C'] if compress else []),'--',str(source),host+':'+str(target)],check=True,timeout=1800)


def send(transport,expected_sha256,host,remote_root):
    if host.startswith('-'):raise ValueError('invalid SSH host')
    transport=Path(transport).resolve(strict=True)
    if digest_file(transport)!=expected_sha256:raise ValueError('transport receipt differs from sender binding')
    value=json.loads(transport.read_text());view=value['view'];source=value['source']['sha256']
    local_view=transport.parent/view['path'];view_receipt=transport.parent/(value['view_receipt']['sha256']+'.json')
    if (not Path(remote_root).is_absolute() or '/' in view['path'] or len(source)!=64
            or local_view.stat().st_size!=view['bytes'] or digest_file(local_view)!=view['sha256']
            or digest_file(view_receipt)!=value['view_receipt']['sha256']):
        raise ValueError('actor transport dependencies differ from their exact bindings')
    stage=Path(remote_root)/source;target=stage/view['path']
    refs=[(transport,expected_sha256),(view_receipt,value['view_receipt']['sha256'])]
    metadata=[{'path':str(stage/local.name),'sha256':sha} for local,sha in refs]
    probe=remote(host,'''import json,sys,hashlib
from pathlib import Path
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
root,target=map(Path,sys.argv[1:3]);root.mkdir(parents=True,exist_ok=True);target.parent.mkdir(parents=True,exist_ok=True)
state=root/'.natlang-actor-transfer.json';hint=json.loads(state.read_text()) if state.exists() else None
actual=None
if target.exists():actual=digest(target)
metadata={r['path']:digest(r['path']) for r in json.loads(sys.argv[3]) if Path(r['path']).is_file()}
print(json.dumps({'target_sha256':actual,'base':hint,'base_present':bool(hint and Path(hint['receiver_view']).is_file()),'metadata':metadata}))
''',remote_root,target,json.dumps(metadata))
    if probe['target_sha256'] not in (None,view['sha256']):raise ValueError('peer target exists with different bytes; preserve and investigate')
    mode,transferred='already-verified',0
    if probe['target_sha256'] is None:
        hint=probe['base'];delta=None
        if probe['base_present'] and Path(hint['sender_view']).is_file():
            cache=transport.parent/'.transfer-deltas'/(hint['sha256']+'-'+view['sha256'])
            receipt=cache/'receipt.json'
            if not receipt.exists():build(hint['sender_view'],local_view,cache)
            delta=json.loads(receipt.read_text())
            if (delta['base']['sha256']!=hint['sha256'] or delta['target']!={'sha256':view['sha256'],'bytes':view['bytes']}
                    or digest_file(cache/'delta.zlib')!=delta['delta']['sha256']):
                raise ValueError('cached delta differs from current peer or target binding')
        if delta and delta['delta']['bytes']<view['bytes']*.85:
            mode='xor-zlib';remote_cache=stage/'.delta'
            remote(host,"import pathlib,sys,json;pathlib.Path(sys.argv[1]).mkdir(exist_ok=True);print('{}')",remote_cache)
            for name in ('delta.zlib','receipt.json'):copy(host,cache/name,remote_cache/name)
            implementation=Path(__file__).with_name('artifact_delta.py');copy(host,implementation,remote_cache/'artifact_delta.py')
            # Pin the actual decoder before executing it on the peer.
            remote(host,'''import hashlib,json,runpy,sys
from pathlib import Path
tool,sha,base,manifest,out,digest,size=sys.argv[1:]
assert hashlib.sha256(Path(tool).read_bytes()).hexdigest()==sha,'decoder source changed'
module=runpy.run_path(tool);result=module['apply'](base,manifest,out,expected_sha256=digest,expected_bytes=int(size),min_free_bytes=1073741824)
print(json.dumps(result))
''',remote_cache/'artifact_delta.py',digest_file(implementation),hint['receiver_view'],remote_cache/'receipt.json',target,view['sha256'],view['bytes'])
            transferred=delta['delta']['bytes']
        else:
            mode='full';pending=target.with_name('.'+target.name+'.transfer')
            copy(host,local_view,pending,compress=True)
            remote(host,'''import hashlib,json,os,sys
from pathlib import Path
p,t=map(Path,sys.argv[1:3]);expected=sys.argv[3]
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==expected,'full transfer hash differs'
with p.open('rb') as f:os.fsync(f.fileno())
os.link(p,t);p.unlink();print('{}')
''',pending,target,view['sha256'])
            transferred=view['bytes']
    for local,sha in refs:
        if probe['metadata'].get(str(stage/local.name))!=sha:copy(host,local,stage/local.name)
    result=remote(host,'''import hashlib,json,os,sys,tempfile
from pathlib import Path
root,target=map(Path,sys.argv[1:3]);sha,sender,metadata,source=sys.argv[3:]
for ref in json.loads(metadata):
 assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256'],'metadata transfer differs'
h=hashlib.sha256()
with target.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==sha,'actor view readback differs'
state={'schema':'natlang.peer-actor-view-cache/1','sha256':sha,'sender_view':sender,'receiver_view':str(target),'source_checkpoint_sha256':source}
fd,pending=tempfile.mkstemp(prefix='.actor-cache-',dir=root)
with os.fdopen(fd,'w') as f:json.dump(state,f);f.flush();os.fsync(f.fileno())
os.replace(pending,root/'.natlang-actor-transfer.json');fd=os.open(root,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(state))
''',remote_root,target,view['sha256'],local_view,json.dumps(metadata),source)
    return {'schema':'natlang.actor-transfer-readback/1','mode':mode,'payload_bytes':transferred,
        'full_view_bytes':view['bytes'],'transport_sha256':expected_sha256,'peer':host,'verified':result,
        'scope':'exact inference transport; no training resume, update credit or capability qualification'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('transport','expected-sha256','host','remote-root'):p.add_argument('--'+name,required=True)
    args=vars(p.parse_args());result=send(**args)
    directory=Path(args['transport']).resolve().parent/'transfer-readbacks';directory.mkdir(exist_ok=True)
    result['implementation']={'sender_sha256':digest_file(__file__),
        'decoder_sha256':digest_file(Path(__file__).with_name('artifact_delta.py'))}
    receipt=directory/(str(time.time_ns())+'.json');fd,pending=tempfile.mkstemp(prefix='.readback-',dir=directory)
    try:
        with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(pending,receipt);Path(pending).unlink();fd=os.open(directory,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    finally:Path(pending).unlink(missing_ok=True)
    print(json.dumps({**result,'receipt':{'path':str(receipt),'sha256':digest_file(receipt)}}),flush=True)


if __name__=='__main__':main()
