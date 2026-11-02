#!/usr/bin/env python3
"""Compare original source-occurrence IDs without re-reading large graph payloads."""
import argparse
import json
from pathlib import Path
from natlang_neuralese.common.hashing import sha256_file_hex, canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text,write_json_durable


def checked(ref):
    p=Path(ref['path']).resolve(strict=True)
    if sha256_file_hex(p)!=ref['sha256'] or ('bytes' in ref and p.stat().st_size!=ref['bytes']):
        raise ValueError('source occurrence inventory binding changed')
    return p


def source_id(ident):
    if not isinstance(ident,str) or not ident:raise ValueError('source graph has no ID')
    return ident.split(':source-occurrence:',1)[0]


def audit(screen_ref,predecessor_ref,reserved_refs,out):
    screen=json.loads(checked(screen_ref).read_text());predecessor=json.loads(checked(predecessor_ref).read_text())
    if (screen.get('schema')!='natlang.new-acquired-source-screen/1' or screen.get('training_admission') is not False
            or screen.get('prepared_source_sha256')!=predecessor['source']['sha256']):
        raise ValueError('screen and exact prepared predecessor differ')
    ids=json.loads(checked(screen['active_ids']).read_text());families=json.loads(checked(predecessor['family_map']).read_text())
    if (len(ids)!=screen['active']['rows'] or len(set(ids))!=len(ids)
            or families.get('source_sha256')!=predecessor['source']['sha256']
            or len(families.get('rows',[]))!=predecessor['rows']):
        raise ValueError('source ID inventory is incomplete')
    prefixes=[source_id(x) for x in ids];prior={source_id(x['id']) for x in families['rows']}
    prior_overlap=sorted(set(prefixes)&prior);within=len(prefixes)-len(set(prefixes));reserved_overlaps={}
    for ref in reserved_refs:
        prior_screen=json.loads(checked(ref).read_text());active=prior_screen['active']
        if prior_screen.get('active_ids'):
            other=json.loads(checked(prior_screen['active_ids']).read_text())
        else:
            other=[]
            with open_jsonl_text(checked(active)) as stream:
                for line in stream:
                    if line.strip():other.append(json.loads(line)['id'])
        if len(other)!=active['rows']:raise ValueError('reserved source inventory is incomplete')
        reserved_overlaps[ref['sha256']]=sorted(set(prefixes)&{source_id(x) for x in other})
    result={'schema':'natlang.acquired-source-occurrence-overlap/1','training_admission':False,
        'screen':screen_ref,'predecessor':predecessor_ref,'reserved_screens':reserved_refs,
        'active_rows':len(ids),'predecessor_rows':len(families['rows']),
        'source_occurrence_suffix_missing':sum(':source-occurrence:' not in x for x in ids),
        'duplicate_source_occurrences_within_active':within,'predecessor_source_occurrence_overlaps':prior_overlap,
        'reserved_source_occurrence_overlaps':reserved_overlaps,'active_source_occurrence_set_sha256':canonical_json_sha256_hex(sorted(set(prefixes))),
        'scope':'Exact source ID prefix before source-occurrence suffix; no semantic or cross-dataset content-deduplication claim.'}
    out=Path(out)
    if out.exists():raise ValueError('overlap proof output must be fresh')
    write_json_durable(out,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('screen','screen-sha256','predecessor','predecessor-sha256','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--reserved-screen',action='append',default=[],help='JSON object with path and sha256')
    a=p.parse_args();v=audit({'path':a.screen,'sha256':a.screen_sha256},{'path':a.predecessor,'sha256':a.predecessor_sha256},
        [json.loads(x) for x in a.reserved_screen],a.out)
    print(json.dumps({k:v[k] for k in ('active_rows','source_occurrence_suffix_missing','duplicate_source_occurrences_within_active','predecessor_source_occurrence_overlaps','reserved_source_occurrence_overlaps')},sort_keys=True))
if __name__=='__main__':main()
