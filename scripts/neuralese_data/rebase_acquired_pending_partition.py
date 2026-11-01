#!/usr/bin/env python3
"""Rebind a pending partition to an exact graphified predecessor transition."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
from natlang_neuralese.common.jsonio import open_jsonl_text,write_json_durable
from scripts.neuralese_data.partition_acquired_context_pending import bind,_checked,_read
from natlang_neuralese.train.native_source_correction import source_correction_dependencies,_proof_file

SCHEMA='natlang.acquired-pending-predecessor-rebase/1'

def verify_rebase(screen, predecessor):
    proof=_read(screen.get('predecessor_rebase'));original=_read(proof['original_screen'])
    previous=_read(proof['previous_predecessor']);current=_read(proof['current_predecessor'])
    if (proof.get('schema')!=SCHEMA or _checked(proof['current_predecessor'])!=Path(predecessor).resolve(strict=True)
        or original.get('pending_partition')!=proof.get('pending_partition')
        or current.get('source_correction')!=proof.get('source_correction')):
        raise ValueError('pending predecessor rebase does not bind its exact sources')
    expected=copy.deepcopy(original);expected['predecessor_rebase']=screen['predecessor_rebase']
    if expected!=screen:raise ValueError('pending predecessor rebase changed source screen custody')
    partition=_read(proof['pending_partition'])
    if partition.get('predecessor')!=proof['previous_predecessor']:
        raise ValueError('pending predecessor rebase names another original predecessor')
    correction_path=_checked(proof['source_correction']);source_correction_dependencies(proof['source_correction'])
    correction=json.loads(correction_path.read_text())
    if ((correction.get('prior_source') or {}).get('sha256')!=(previous.get('source') or {}).get('sha256')
        or (correction.get('current_source') or {}).get('sha256')!=(current.get('source') or {}).get('sha256')
        or not correction.get('static_native_context_correction')):
        raise ValueError('pending predecessor rebase lacks exact static source correction')
    old=_read(previous['family_map']);new=_read(current['family_map'])
    if (old.get('source_sha256')!=previous['source']['sha256'] or new.get('source_sha256')!=current['source']['sha256']):
        raise ValueError('pending predecessor family maps belong to other sources')
    before={r['id']:r for r in old.get('rows',[])};after={r['id']:r for r in new.get('rows',[])}
    if (len(before)!=len(old.get('rows',[])) or len(after)!=len(new.get('rows',[])) or set(before)!=set(after)):
        raise ValueError('pending predecessor rebase changed the row inventory')
    with open_jsonl_text(_proof_file(correction_path,correction['row_bindings_path'])) as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    changes={r['id']:r for r in rows}
    if len(changes)!=len(rows) or any(r.get('disposition')!='corrected' for r in rows):raise ValueError('rebase correction ledger is invalid')
    changed=set()
    for ident,old_row in before.items():
        new_row=after[ident]
        if old_row.get('family')!=new_row.get('family'):raise ValueError('pending predecessor rebase changed family membership')
        if old_row.get('binding')!=new_row.get('binding'):
            row=changes.get(ident)
            if not row or row.get('old_binding')!=old_row['binding'] or row.get('new_binding')!=new_row['binding']:
                raise ValueError('pending predecessor rebase changed an unproved row binding')
            changed.add(ident)
    if changed!=set(changes) or sorted(changed)!=proof.get('corrected_ids'):
        raise ValueError('pending predecessor rebase correction inventory differs')
    from scripts.neuralese_data.build_acquired_streaming_successor import receipt_protection_digest
    if (receipt_protection_digest(previous)!=receipt_protection_digest(current)
        or proof.get('protected_worlds_sha256')!=receipt_protection_digest(current)):
        raise ValueError('pending predecessor rebase changed the protected-world union')
    return original,proof['previous_predecessor'],old['rows']

def rebase(screen_path,census_path,new_predecessor,output):
    screen_path=Path(screen_path).resolve(strict=True);census_path=Path(census_path).resolve(strict=True)
    new_predecessor=Path(new_predecessor).resolve(strict=True);screen=json.loads(screen_path.read_text());census=json.loads(census_path.read_text())
    partition=_read(screen.get('pending_partition'));previous_ref=partition['predecessor'];current=json.loads(new_predecessor.read_text())
    if census.get('screen',{}).get('sha256')!=bind(screen_path)['sha256'] or census.get('pending_partition')!=screen['pending_partition']:
        raise ValueError('pending census does not bind its original screen')
    correction=_read(current['source_correction']);correction_path=_checked(current['source_correction'])
    with open_jsonl_text(_proof_file(correction_path,correction['row_bindings_path'])) as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    output=Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():raise ValueError('fresh rebase directory required')
    output.mkdir();proof={'schema':SCHEMA,'original_screen':bind(screen_path),'pending_partition':screen['pending_partition'],
        'previous_predecessor':previous_ref,'current_predecessor':bind(new_predecessor),'source_correction':current['source_correction'],
        'corrected_ids':sorted(r['id'] for r in rows),'protected_worlds_sha256':partition['protected_worlds_sha256'],
        'scope':'same protected source partition, with independently verified static predecessor corrections; no admission or exposure'}
    proof_path=output/'predecessor-rebase.json';write_json_durable(proof_path,proof)
    updated=copy.deepcopy(screen);updated['predecessor_rebase']=bind(proof_path)
    verify_rebase(updated,new_predecessor)
    screen_out=output/'screen.json';write_json_durable(screen_out,updated)
    new_census={**census,'screen':bind(screen_out),'predecessor_rebase':bind(proof_path)}
    write_json_durable(output/'census.json',new_census)
    return proof

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--screen',type=Path,required=True);p.add_argument('--census',type=Path,required=True)
    p.add_argument('--predecessor',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(rebase(a.screen,a.census,a.predecessor,a.output),sort_keys=True))
if __name__=='__main__':main()
