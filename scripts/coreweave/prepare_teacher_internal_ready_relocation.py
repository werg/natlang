"""Build a strict per-source-case path relocation from the exact 202-packet scan."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.common.paths import resolve
from scripts.coreweave.teacher_tranche_admission import file_binding, read_bound

SCAN_SHA = '80dc44d7a2d39dac55b075f3f57d03ba9f83b0edd3726fd00e45f161f86f0482'
SCAN_CODE_SHA = 'fb21932f0e670c0e8574ddaeb51364e3e84302d1f20e1419fc8e621f81d7b6d2'
SANDBOX = '78530f71-08fc-4f57-ace9-7fbad64aeeb8'
READY_PATH = resolve('data_nvme','dgxops','teacher-computed-result202-future-sequence-v1-20261020','ready-packet-sequence.json')
SPOOL = resolve('data_nvme','natlang-internal-spool','teacher-qwen36-continuation-20261023')

def run(batch: int, target: str) -> dict:
    ready_ref = file_binding(READY_PATH)
    if ready_ref != {'path': str(READY_PATH), 'bytes': 162695,
                     'sha256': 'da767d1c8791bb043a7f9420b43200cb436556ac7441b9b0765d8f2b7f641bda'}:
        raise ValueError('original ready packet binding changed')
    ready = read_bound(ready_ref)
    scan_path = SPOOL/'assets/ready-packet/candidate-scan.json'
    scan_ref = file_binding(scan_path)
    if scan_ref['sha256'] != SCAN_SHA:
        raise ValueError('exact all-candidate remote scan binding changed')
    base_scan = read_bound(scan_ref)
    if (base_scan.get('remote_scan_code_sha256') != SCAN_CODE_SHA
            or base_scan.get('source_generation_sandbox_id') != SANDBOX
            or base_scan.get('original_ready_packet') != ready_ref
            or base_scan.get('candidate_count') != len(ready['entries'])
            or base_scan.get('training_admission') is not False):
        raise ValueError('candidate scan does not bind the full original sequence')
    base_candidates = (base_scan.get('scan') or {}).get('candidates') or []
    if len(base_candidates) != len(ready['entries']):
        raise ValueError('remote candidate scan is incomplete')
    ordinals = [i for i,c in enumerate(base_candidates)
                if c.get('cases_binding',{}).get('sha256') == target]
    if len(ordinals) != 1:
        raise ValueError(f'source cases must bind exactly one selected packet; found {ordinals}')
    ordinal=ordinals[0]
    entry=ready['entries'][ordinal]
    selected=base_candidates[ordinal]
    candidate_root=selected['candidate_remote_path'].rsplit('/',1)[0]
    candidates=[]
    for i,candidate in enumerate(base_candidates):
        if (candidate.get('ordinal') != i or candidate.get('candidate_ref') != ready['entries'][i].get('candidate_packet')
                or (candidate.get('candidate_asset') or {}).get('sha256') != candidate.get('candidate_ref',{}).get('sha256')
                or (candidate.get('candidate_asset') or {}).get('bytes') != candidate.get('candidate_ref',{}).get('bytes')
                or Path(candidate.get('candidate_remote_path','')).name != candidate.get('candidate_ref',{}).get('sha256','')+'.json'):
            raise ValueError(f'candidate {i} differs from the exact original ready-sequence and remote scan')
        candidates.append({**candidate,'matches_source_cases':candidate.get('cases_binding',{}).get('sha256') == target})
    evidence={'schema':'natlang.teacher-ready-packet-candidate-scan-evidence/1',
       'original_ready_packet':ready_ref,'source_generation_sandbox_id':SANDBOX,
       'source_root':'/work/generation/strong-teacher-20261015/source-sequences',
       'source_cases_sha256':target,'candidate_count':len(ready['entries']),
       'matching_ordinals':[ordinal],'remote_scan_code_sha256':SCAN_CODE_SHA,
       'source_candidate_scan':scan_ref,
       'scan':{'schema':'natlang.teacher-ready-packet-remote-scan/1','sandbox_id':SANDBOX,
          'source_root':'/work/generation/strong-teacher-20261015/source-sequences',
          'target_cases_sha256':target,'candidate_count':len(candidates),'candidates':candidates},
       'training_admission':False}
    batch_dir=SPOOL/'assets/ready-packet/remote-recovered'/f'packet-{ordinal:05d}'
    remote_assets=read_bound(file_binding(batch_dir/'remote-assets.json'))
    if (remote_assets.get('source_cases_sha256') != target or remote_assets.get('ordinal') != ordinal
            or remote_assets.get('batch') != batch or remote_assets.get('source_sandbox_id') != SANDBOX
            or remote_assets.get('source_root') != candidate_root or remote_assets.get('training_admission') is not False):
        raise ValueError('recovered remote assets do not bind the exact selected packet and source cases')
    files={x['role']:x for x in remote_assets['files']}
    if set(files)!={'candidate_packet','qualified_packet','qualification','global_protection_join','selection','selected_cases'}:
        raise ValueError('recovered selected packet closure is incomplete')
    for role,item in files.items():
        if item.get('source_sandbox_id') != SANDBOX or item.get('source_path') != candidate_root+'/'+item['original_ref']['sha256']+('.jsonl' if role=='selected_cases' else '.json'):
            raise ValueError(f'{role} exact source asset path is not hash-addressed under selected root')
        if file_binding(item['local_ref']['path']) != item['local_ref']:
            raise ValueError(f'{role} local recovered asset differs from remote exact binding')
    if files['candidate_packet']['original_ref'] != entry['candidate_packet'] or files['qualified_packet']['original_ref'] != entry['qualified_packet']:
        raise ValueError('recovered candidate or prepared packet differs from original ready sequence')
    packet=read_bound(files['qualified_packet']['local_ref'])
    if packet.get('cases') != files['selected_cases']['original_ref'] or packet.get('qualification') != files['qualification']['original_ref']:
        raise ValueError('qualified packet does not bind relocated cases and qualification')
    expected_refs={'global_protection_join':packet['references']['global_protection_join'],
                   'selection':packet['references']['selection'],'selected_cases':packet['references']['selected_cases']}
    if any(expected_refs[role] != files[role]['original_ref'] for role in expected_refs):
        raise ValueError('qualified packet does not bind exact selected closure')
    outdir=SPOOL/'assets/ready-packet'/f'packet-{ordinal:05d}'
    outdir.mkdir(parents=True,exist_ok=True)
    evidence_path=outdir/'candidate-scan-evidence.json';write_json_durable(evidence_path,evidence)
    relocation={'schema':'natlang.teacher-ready-packet-internal-relocation/1','original_ready_packet':ready_ref,
       'source_sandbox_id':SANDBOX,'source_root':candidate_root,'source_cases_sha256':target,
       'candidate_scan':file_binding(evidence_path),'selected_ordinal':ordinal,
       'files':[{'role':r,'source_sandbox_id':i['source_sandbox_id'],'source_path':i['source_path'],
                 'original_ref':i['original_ref'],'local_ref':i['local_ref']}
                for r,i in sorted(files.items())],
       'remote_assets':file_binding(batch_dir/'remote-assets.json'), 'training_admission':False}
    relocation_path=outdir/'relocation.json';write_json_durable(relocation_path,relocation)
    return {'batch':batch,'source_cases_sha256':target,'ordinal':ordinal,'candidate_scan':file_binding(evidence_path),
            'relocation':file_binding(relocation_path),'training_admission':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--source-cases-sha256',required=True);a=p.parse_args()
    print(json.dumps(run(a.batch,a.source_cases_sha256),sort_keys=True))
if __name__=='__main__':main()
