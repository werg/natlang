"""Fetch a selected ready-packet closure by exact hash from the source sandbox."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from scripts.coreweave.teacher_tranche_admission import file_binding, read_bound

SANDBOX='78530f71-08fc-4f57-ace9-7fbad64aeeb8'
READY_PATH=Path('/home/werg/data/dgxops/teacher-computed-result202-future-sequence-v1-20261020/ready-packet-sequence.json')
SPOOL=Path('/home/werg/data/natlang-internal-spool/teacher-qwen36-continuation-20261023')
SCAN_SHA='80dc44d7a2d39dac55b075f3f57d03ba9f83b0edd3726fd00e45f161f86f0482'
SCAN_CODE='fb21932f0e670c0e8574ddaeb51364e3e84302d1f20e1419fc8e621f81d7b6d2'

def fetch(batch:int,target:str)->dict:
    ready_ref=file_binding(READY_PATH)
    if ready_ref.get('sha256')!='da767d1c8791bb043a7f9420b43200cb436556ac7441b9b0765d8f2b7f641bda': raise ValueError('ready sequence changed')
    ready=read_bound(ready_ref)
    scan_path=SPOOL/'assets/ready-packet/candidate-scan.json';scan_ref=file_binding(scan_path)
    if scan_ref.get('sha256')!=SCAN_SHA: raise ValueError('full candidate scan changed')
    scan=read_bound(scan_ref)
    if scan.get('remote_scan_code_sha256')!=SCAN_CODE or scan.get('original_ready_packet')!=ready_ref or scan.get('candidate_count')!=len(ready['entries']): raise ValueError('candidate scan is not exact source sequence scan')
    rows=(scan.get('scan') or {}).get('candidates') or []
    ords=[i for i,x in enumerate(rows) if x.get('cases_binding',{}).get('sha256')==target]
    if len(ords)!=1: raise ValueError(f'exact source case hash has {len(ords)} ready packet matches')
    ordinal=ords[0];entry=ready['entries'][ordinal];candidate=rows[ordinal]
    source_root=candidate['candidate_remote_path'].rsplit('/',1)[0]
    output=SPOOL/'assets/ready-packet/remote-recovered'/f'packet-{ordinal:05d}'
    output.mkdir(parents=True,exist_ok=True)
    cached=output/'remote-assets.json'
    if cached.exists():
        prior=read_bound(file_binding(cached))
        if prior.get('source_cases_sha256')==target and prior.get('batch')==batch and prior.get('source_root')==source_root:
            for item in prior['files']:
                if file_binding(item['local_ref']['path'])!=item['local_ref']: raise ValueError('cached remote packet asset changed')
            return {'cached':True,'batch':batch,'ordinal':ordinal,'remote_assets':file_binding(cached)}
    sys.path.insert(0,'/home/werg/natlang/scripts/coreweave')
    import cwctl
    cwctl.load_env_file()
    sandbox=cwctl.read_sandbox(SANDBOX,patience=180)
    def get(ref,extension):
        remote=source_root+'/'+ref['sha256']+'.'+extension
        value=sandbox.read_file(remote).result(timeout=180)
        raw=value if isinstance(value,bytes) else bytes(value)
        digest=hashlib.sha256(raw).hexdigest()
        if (len(raw),digest)!=(ref['bytes'],ref['sha256']): raise ValueError(f'exact remote asset mismatch: {remote}')
        return remote,raw
    files={}
    for role,ref in [('candidate_packet',entry['candidate_packet']),('qualified_packet',entry['qualified_packet'])]:
        remote,raw=get(ref,'json'); files[role]=(remote,ref,raw)
    packet=json.loads(files['qualified_packet'][2])
    refs={'qualification':packet['qualification'],'global_protection_join':packet['references']['global_protection_join'],
          'selection':packet['references']['selection'],'selected_cases':packet['references']['selected_cases']}
    if packet['cases']!=refs['selected_cases']: raise ValueError('qualified packet cases differ from closure ref')
    for role,ref in refs.items():
        remote,raw=get(ref,'jsonl' if role=='selected_cases' else 'json'); files[role]=(remote,ref,raw)
    records=[]
    for role,(remote,ref,raw) in files.items():
        path=output/(role+('.jsonl' if role=='selected_cases' else '.json'))
        if path.exists() and path.read_bytes()!=raw: raise ValueError('local exact asset collision')
        if not path.exists(): path.write_bytes(raw)
        records.append({'role':role,'source_sandbox_id':SANDBOX,'source_path':remote,'original_ref':ref,'local_ref':file_binding(path)})
    result={'schema':'natlang.teacher-ready-packet-exact-remote-assets/1','batch':batch,'ordinal':ordinal,
        'source_cases_sha256':target,'source_sandbox_id':SANDBOX,'source_root':source_root,
        'files':records,'training_admission':False}
    tmp=cached.with_suffix('.tmp');tmp.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');tmp.replace(cached)
    return {'cached':False,'batch':batch,'ordinal':ordinal,'remote_assets':file_binding(cached)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--source-cases-sha256',required=True);a=p.parse_args()
    print(json.dumps(fetch(a.batch,a.source_cases_sha256),sort_keys=True))
if __name__=='__main__':main()
