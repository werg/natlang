#!/usr/bin/env python3
"""Bind screened, disjoint acquired sources into an unadmitted manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds

CODE=Path(__file__).read_bytes()


def checked(ref):
    p=Path(ref['path']).resolve(strict=True);before=p.stat();sha=sha256_file_hex(p);after=p.stat()
    sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if (sha!=ref['sha256'] or ('bytes' in ref and before.st_size!=ref['bytes']) or sig(before)!=sig(after)):
        raise ValueError('combined source evidence changed')
    return p


def build(predecessor_ref,screen_refs,overlap_refs,protection_path,out,*,pending_refs=()):
    if len(screen_refs)!=len(overlap_refs) or not screen_refs:raise ValueError('each source needs its own overlap proof')
    predecessor=json.loads(checked(predecessor_ref).read_text());protection=ProtectedWorlds.load()
    quality=current_protection(Path(protection_path).resolve(strict=True))
    pending_ids=set()
    for ref in pending_refs:
        pending=json.loads(checked(ref).read_text());schema=pending.get('schema')
        if pending.get('training_admission') is not False:raise ValueError('pending evidence claims training admission')
        if schema=='natlang.acquired-unresolved-render-obligations/1':
            ledger=checked(pending['ledger']);rows=[]
            with ledger.open() as stream:
                for line in stream:
                    if line.strip():rows.append(json.loads(line)['original_omission']['id'])
            if len(rows)!=pending['rows']:raise ValueError('pending ledger count differs')
        elif schema=='natlang.acquired-repair-still-pending/1':
            original=pending['original_omission']['id'];derived=pending['derived_omission']['id']
            if original!=derived:raise ValueError('post-repair pending identity changed')
            rows=[original]
        else:raise ValueError('unknown pending render obligation evidence')
        if len(set(rows))!=len(rows) or pending_ids.intersection(rows):raise ValueError('pending render obligations overlap')
        pending_ids.update(rows)
    carried_context=None
    if predecessor.get('source_screen'):
        prior_screen=json.loads(checked(predecessor['source_screen']).read_text());partition_ref=prior_screen.get('pending_partition')
        if partition_ref:
            partition=json.loads(checked(partition_ref).read_text());rows=partition.get('pending_rows') or []
            if len({row['id'] for row in rows})!=len(rows):raise ValueError('prior context obligations repeat an ID')
            carried_context={'partition':partition_ref,'rows':len(rows)}
    hold_paths=sorted((Path(__file__).resolve().parents[2]/'training/decision-holds').glob('*.jsonl'))
    _,hold_refs=load_decision_holds(hold_paths)
    screens=[];proofs=[];seen_refs=set();total=0;tokenizer=None;source_leaves=[]
    for screen_ref,overlap_ref in zip(screen_refs,overlap_refs,strict=True):
        screen=json.loads(checked(screen_ref).read_text());proof=json.loads(checked(overlap_ref).read_text())
        if (screen.get('schema')!='natlang.new-acquired-source-screen/1' or screen.get('training_admission') is not False
                or screen.get('prepared_source_sha256')!=predecessor['source']['sha256']
                or screen.get('protected_worlds_sha256')!=protection.digest or screen.get('quality_protection')!=quality['proof']
                or screen.get('decision_holds')!=hold_refs):raise ValueError('source screen differs from the current custody union')
        if (proof.get('schema')!='natlang.acquired-source-occurrence-overlap/1' or proof.get('training_admission') is not False
                or proof.get('screen')!=screen_ref or proof.get('predecessor')!=predecessor_ref
                or proof.get('active_rows')!=screen['active']['rows'] or proof.get('source_occurrence_suffix_missing')!=0
                or proof.get('duplicate_source_occurrences_within_active')!=0
                or proof.get('predecessor_source_occurrence_overlaps') or any(proof.get('reserved_source_occurrence_overlaps',{}).values())):
            raise ValueError('source occurrence overlap is missing or not empty')
        prior_refs={x['sha256'] for x in proof.get('reserved_screens',[])}
        if prior_refs!=seen_refs:raise ValueError('overlap proof does not cover all preceding staged sources')
        active=screen['active'];counts=screen.get('counts') or {}
        builder=screen.get('builder') or {};builder_path=Path(builder.get('path',''))
        if not builder_path.is_file() or sha256_file_hex(builder_path)!=builder.get('sha256'):
            builder_path=Path(screen_ref['path']).parent/('screen-builder-'+str(builder.get('sha256'))+'.py')
        checked({'path':str(builder_path),'sha256':builder.get('sha256')})
        if screen.get('active_ids'):checked(screen['active_ids'])
        decisions=json.loads(checked(screen['decisions']).read_text())
        inventory=screen.get('output_inventory') or []
        if (active.get('rows',0)<=0 or screen.get('pending',{}).get('rows')!=0 or decisions
                or counts!={'active':active['rows'],'raw':active['rows']}
                or sum(item.get('rows',0) for item in inventory)!=active['rows']):
            raise ValueError('direct source reuse requires every original row to have passed screening')
        for item in inventory:
            checked(item)
            source_leaves.append({'kind':'native-graphs',**{k:item[k] for k in ('path','sha256','bytes')},'training_priority':1})
        fp=screen['tokenizer_fingerprint']
        if tokenizer is None:tokenizer=fp
        if fp!=tokenizer:raise ValueError('combined sources use different tokenizers')
        screens.append(screen);proofs.append(proof);seen_refs.add(screen_ref['sha256']);total+=active['rows']
    if ProtectedWorlds.load().digest!=protection.digest or current_protection(Path(protection_path))['proof']!=quality['proof'] or load_decision_holds(hold_paths)[1]!=hold_refs:
        raise ValueError('current protection changed while binding combined source')
    if len({(item['path'],item['sha256']) for item in source_leaves})!=len(source_leaves):
        raise ValueError('combined source repeats an original file')
    if Path(__file__).read_bytes()!=CODE:raise ValueError('combined-source builder changed during execution')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    builder=out/('builder-'+hashlib.sha256(CODE).hexdigest()+'.py');builder.write_bytes(CODE);builder.chmod(0o444)
    manifest=out/'train-sources.json'
    write_json_durable(manifest,{'schema':'natlang.native-graph-sources/1','sources':source_leaves})
    receipt={'schema':'natlang.combined-acquired-screened-delta/1','training_admission':False,
        'predecessor':predecessor_ref,'source':{'path':str(manifest),'sha256':sha256_file_hex(manifest)},
        'screens':screen_refs,'occurrence_proofs':overlap_refs,'builder':{'path':str(builder),'sha256':sha256_file_hex(builder)},
        'rows':total,'source_leaves':len(source_leaves),'tokenizer_fingerprint':tokenizer,
        'protected_worlds_sha256':protection.digest,'quality_protection':quality['proof'],'decision_holds':hold_refs,
        'pending_render_obligations':{'evidence':list(pending_refs),'rows':len(pending_ids)},
        'carried_over_context_obligations':carried_context,
        'source_reuse':'Each original immutable external file is reused in screened order because every row passed; no graph payload copy.',
        'limitations':['No semantic or cross-dataset content deduplication, native source indexing, effective-loss census, GPU capacity qualification, behavioral evaluation or training admission.']}
    write_json_durable(out/'receipt.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predecessor',required=True,help='JSON path/SHA reference')
    p.add_argument('--screen',action='append',required=True,help='JSON path/SHA reference in source order')
    p.add_argument('--overlap',action='append',required=True,help='JSON path/SHA reference in matching order')
    p.add_argument('--protection',required=True);p.add_argument('--out',required=True)
    p.add_argument('--pending',action='append',default=[],help='JSON path/SHA reference for retained pending obligations')
    a=p.parse_args();r=build(json.loads(a.predecessor),[json.loads(x) for x in a.screen],[json.loads(x) for x in a.overlap],a.protection,a.out,pending_refs=[json.loads(x) for x in a.pending])
    print(json.dumps({'rows':r['rows'],'source':r['source'],'training_admission':False},sort_keys=True))
if __name__=='__main__':main()
