#!/usr/bin/env python3
"""Build exact non-duplicate syntax-only child views from a completed pending census."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data import acquired_session_syntax_repair as repair

CODE=Path(__file__).read_bytes();REPAIR_CODE=Path(repair.__file__).read_bytes()

def checked(ref):
    path=Path(ref['path']).resolve(strict=True)
    if sha256_file_hex(path)!=ref['sha256']:raise ValueError('bound repair input changed: '+str(path))
    return path


def build(census_ref, existing_exports, output, export_root):
    census_path=checked(census_ref);census=json.loads(census_path.read_text())
    if census.get('schema')!='natlang.acquired-render-pending-census/1' or census.get('training_admission') is not False:
        raise ValueError('requires completed original render-pending census')
    existing={};existing_refs=[]
    for item in existing_exports:
        export_path=Path(item).resolve(strict=True);export=json.loads(export_path.read_text())
        source=(export_path.parent/'full-trajectories.jsonl.zst').resolve(strict=True)
        verified=repair.verify_derived_repair_export(export_path,export,source)
        manifest_ref=export['contract']['source_syntax_repair']['repair_manifest'];manifest=json.loads(checked(manifest_ref).read_text())
        source_path=Path(manifest['parent_source']['path']).resolve(strict=True)
        ids={row['session_id'] for row in manifest['rows']}
        if len(ids)!=len(manifest['rows']):raise ValueError('existing repair view repeats a source session')
        existing.setdefault(str(source_path),set()).update(ids)
        existing_refs.append({'path':str(export_path),'sha256':sha256_file_hex(export_path),'selected_rows':verified['selected_rows']})
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    export_root=Path(export_root);export_root.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(export_root).free < 50*1024**3:
        raise ValueError('repair outputs must preserve the external 50 GiB floor')
    total=Counter();views=[];export_paths=[];profiles={}
    for index,entry in enumerate(census['receipts']):
        receipt_path=checked(entry);receipt=json.loads(receipt_path.read_text())
        converter_ref=receipt['converter_receipt'];converter=json.loads(checked(converter_ref).read_text())
        profile=converter.get('explicit_history_reasoning_policy')
        if converter.get('source')!=receipt['source'] or profile not in (None,'keep'):
            raise ValueError('original source history profile is ambiguous or changed')
        candidate_path=receipt_path.parent/'candidates.json';audit_path=receipt_path.parent/'pending-audit.json'
        if sha256_file_hex(candidate_path)!=receipt['candidate_sha256'] or sha256_file_hex(audit_path)!=receipt['audit_sha256']:
            raise ValueError('candidate or pending audit changed')
        candidate=json.loads(candidate_path.read_text());audit=json.loads(audit_path.read_text())
        if candidate.get('source')!=entry['source'] or audit.get('source')!=entry['source']:
            raise ValueError('repair input changed its exact original source')
        old=existing.get(candidate['source']['path'],set());rows=candidate['rows'];audit_rows=audit['rows']
        if {r['id'] for r in rows}!={r['id'] for r in audit_rows}:raise ValueError('pending/candidate IDs differ')
        keep={r['id'] for r in rows}-old;total['existing_excluded']+=len({r['id'] for r in rows}&old)
        if not keep:continue
        subset=output/receipt_path.parent.name;subset.mkdir()
        cp=subset/'candidates.json';ap=subset/'pending-audit.json'
        write_json_durable(cp,{**candidate,'rows':[r for r in rows if r['id'] in keep]})
        write_json_durable(ap,{**audit,'rows':[r for r in audit_rows if r['id'] in keep]})
        if shutil.disk_usage(export_root).free < 51*1024**3:
            raise ValueError('repair batch lacks the 50 GiB floor and one GiB headroom')
        parent=Path(candidate['source']['path']).parent/'export.json'
        dest=export_root/Path(candidate['source']['path']).parent.name
        result=repair.build_repair_view(parent_export_path=parent,candidate_audit_path=cp,pending_audit_path=ap,output_dir=dest)
        total['candidate_rows']+=len(keep);total['selected_sessions']+=result['selected_sessions']
        total['repair_calls']+=result['repair_calls'];total['unresolved_sessions']+=result['unresolved_sessions']
        export=dest/'export.json';export_paths.append(str(export))
        if profile is not None:profiles[str(export)]=profile
        views.append({'path':str(export),'sha256':sha256_file_hex(export),
            'original_converter_receipt':converter_ref,'history_reasoning_policy':profile,
            'candidate_rows':len(keep),'selected_sessions':result['selected_sessions'],'unresolved_sessions':result['unresolved_sessions']})
        print(json.dumps({'kind':'acquired_syntax_repair_progress','sources':index+1,'of':len(census['receipts']),
                          'selected_sessions':total['selected_sessions']}),flush=True)
        if Path(__file__).read_bytes()!=CODE or Path(repair.__file__).read_bytes()!=REPAIR_CODE:
            raise ValueError('repair code changed during batch')
    if total['candidate_rows']+total['existing_excluded']!=census['counts']['fully_bounded_candidates']:
        raise ValueError('repair batch does not account for all exact bounded candidates')
    if sha256_file_hex(census_path)!=census_ref['sha256']:raise ValueError('pending census changed')
    write_json_durable(output/'exports.json',{'exports':export_paths,'history_reasoning_policy':profiles})
    result={'schema':'natlang.acquired-syntax-repair-batch/1','training_admission':False,
        'census':census_ref,'existing_views':existing_refs,'counts':dict(total),'views':views,
        'exports':{'path':str(output/'exports.json'),'sha256':sha256_file_hex(output/'exports.json')},
        'builder_sha256':hashlib.sha256(CODE).hexdigest(),'repair_builder_sha256':hashlib.sha256(REPAIR_CODE).hexdigest(),
        'scope':'Original source text syntax-only; no tool execution, behavior result or training admission.'}
    write_json_durable(output/'receipt.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--census',required=True);p.add_argument('--census-sha256',required=True)
    p.add_argument('--existing-export',action='append',default=[]);p.add_argument('--out',required=True);p.add_argument('--export-root',required=True)
    a=p.parse_args();print(json.dumps(build({'path':a.census,'sha256':a.census_sha256},a.existing_export,a.out,a.export_root)['counts'],sort_keys=True))
if __name__=='__main__':main()
