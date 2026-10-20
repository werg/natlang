#!/usr/bin/env python3
"""Convert and register newly completed, source-bound Search/OWL shard pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from natlang_neuralese.common.hashing import sha256_file_hex as sha
from natlang_neuralese.common.jsonio import iter_jsonl_rows,write_json_durable
from natlang_neuralese.data.native_sources import NativeGraphSource
from scripts.neuralese_data import render_native_turn_corpus as renderer
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds
from scripts.neuralese_data.register_corpora import register
from scripts.neuralese_data.verify_completed_search_parent import verify as verify_parent, verified_artifacts

ROOT=Path(__file__).resolve().parents[2]
PARENT='external-expansion-search-owl-native-20261015-v1'
PREFIX='external-expansion-search-owl-native-graphs-20261016-v3-shard'
MODELS={
    'lfm25':'lfm2:f40d7a5bf651a03618ae8c06bda24a180e22fa79de964e55862c2c4984249da6',
    'mellum21':'mellum:bc700045baee1c7ea1869675c38efedd59d9db09975ea0b38cff8db152d6148f'}


def eligible(done,expected_renderer):
    return (done.get('renderer_code_sha256')==expected_renderer
        and isinstance(done.get('aggregate_outputs'),dict) and bool(done['aggregate_outputs'])
        and isinstance(done.get('trajectory_bindings'),dict) and bool(done['trajectory_bindings']))


def output_directory(root,view,alias):
    return root/('program-header' if view=='program-header' else '')/alias


def convert_one(source,destination,tokenizer,model):
    digest=sha(source)
    receipt_path=destination/'receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['source']['sha256']!=digest or receipt.get('model_id')!=model:
            raise ValueError('completed native conversion source/model changed')
        for item in receipt['outputs']:
            if sha(Path(item['path']))!=item['sha256']:raise ValueError('completed native conversion output changed')
        return receipt
    subprocess.run([sys.executable,str(ROOT/'scripts/neuralese_data/build_native_graph_text_corpus.py'),
        '--source',str(source.resolve(strict=True)),'--source-sha256',digest,
        '--base-tokenizer',tokenizer,'--model-id',model,'--output',str(destination)],check=True)
    return json.loads(receipt_path.read_text())


def admit(shard,*,launch,launch_path,state,work,tokenizers,input_binding,protection,protection_path,parent_composition=None):
    out=work.parent/f'shard{shard}-native-graphs'
    done={alias:json.loads((work/'done'/alias/f'{shard}.json').read_text()) for alias in MODELS}
    renderer_sha=sha(Path(renderer.__file__))
    parent_verification=None
    if parent_composition is not None:
        parent_verification=verify_parent(shard,launch_path=launch_path,composition_path=parent_composition,
            output=work/'admission-parent-verifications'/f'{shard}.json')
    elif not all(eligible(row,renderer_sha) for row in done.values()):return None
    selected=work/'selected'/f'{shard}.jsonl.zst'
    held=set(state['test_groups'])|set(state['reserved_groups'])|renderer.held_groups_from([Path(launch['held_text'])])
    for alias,row in done.items():
        job=(str(selected),str(work),alias,state['blocked'],PARENT,held,set(),False,set(),
            tuple(launch['renderings']),True,set(),None,input_binding,None)
        if parent_verification is None:
            renderer._verify_trajectory_receipts(job,row,{renderer_sha,renderer.LEGACY_SEARCH_RENDERER_SHA256})
        elif parent_verification['parents'][alias]['marker']['sha256']!=sha(work/'done'/alias/f'{shard}.json'):
            raise ValueError('historical Search marker changed before conversion')
        if any(row.get(key)!=0 for key in ('decontaminated','decision_excluded','target_excluded')):
            raise ValueError('Search shard has exclusions needing an explicit admission decision')
    if len({row['source_sha256'] for row in done.values()})!=1:
        raise ValueError('student shards have different selected source identity')
    hold_ids,hold_bindings=load_decision_holds(sorted((ROOT/'training/decision-holds').glob('*.jsonl')))
    selected_ids=set()
    for _,row in iter_jsonl_rows(selected):
        if row.get('split')!='train':raise ValueError('Search selected shard is not train-only')
        selected_ids.add(row['id'])
    if selected_ids&hold_ids:raise ValueError('Search selected source intersects current decision holds')
    if any(len(selected_ids)!=row['records'] for row in done.values()):
        raise ValueError('Search selected source count differs from renderer receipts')
    out.mkdir(parents=True,exist_ok=True)
    include=[];views={};fingerprints={};ids={alias:set() for alias in MODELS}
    protected=protection['held']|protection['quality']|protection['pending']
    for alias,model in MODELS.items():
        counts={}
        for view in launch['renderings']:
            source=work/view/alias/f'{shard}.jsonl.zst'
            destination=output_directory(out,view,alias)
            receipt=convert_one(source,destination,tokenizers[alias],model)
            if receipt['source_tokenizer_fingerprint']!=input_binding['tokenizers'][alias]['fingerprint']:
                raise ValueError('native conversion differs from the actual source tokenizer')
            if receipt['source']['rows']!=done[alias][view]:raise ValueError('native/source render counts differ')
            graph=NativeGraphSource(destination/'train/native-graphs.jsonl.zst',
                receipt['expanded_tokenizer_fingerprint'],required_split='train',role_ids=tuple(receipt['role_token_ids']))
            if len(graph)!=done[alias][view] or graph.groups&protected:
                raise ValueError('native Search count or current protected closure differs')
            if ids[alias]&set(graph.bindings):raise ValueError('native Search graph IDs overlap between views')
            ids[alias].update(graph.bindings)
            fingerprints.setdefault(alias,receipt['expanded_tokenizer_fingerprint'])
            if fingerprints[alias]!=receipt['expanded_tokenizer_fingerprint']:raise ValueError('student view tokenizer mismatch')
            if any(item['rows'] for item in receipt['outputs'] if '/test/' in item['path']):
                raise ValueError('unexpected held rows in train-only Search source')
            counts[view]=len(graph)
            source_copy=out/'source-text'/view/alias/source.name
            source_copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,source_copy)
            include.append(str(source_copy.relative_to(out)))
            include.extend(str(p.relative_to(out)) for p in [destination/'receipt.json',
                destination/'train/native-graphs.jsonl.zst',destination/'test/native-graphs.jsonl.zst'])
        views[alias]={'model_id':model,'views':counts}
        copied=out/'renderer-receipts'/f'{alias}-{shard}.json';copied.parent.mkdir(exist_ok=True)
        shutil.copy2(work/'done'/alias/f'{shard}.json',copied);include.append(str(copied.relative_to(out)))
    snapshots={'selection-state.json':work/'selection-state.json','launch-binding.json':launch_path,
        f'selected-shard-{shard}.jsonl.zst':selected}
    for name,path in snapshots.items():shutil.copy2(path,out/name);include.append(name)
    if parent_verification is not None:
        verified_artifacts(parent_verification,shard=shard,launch_path=launch_path,composition_path=parent_composition)
        write_json_durable(out/'historical-parent-verification.json',parent_verification)
        include.append('historical-parent-verification.json')
    proof={'schema':'natlang.search-owl-native-graph-shard-admission/2','status':'passed','shard':shard,
        'source_corpus':PARENT,'selected_source_records':len(selected_ids),'source_selection_complete':True,
        'full_source_conversion_complete':False,'earned_graph_visits':0,'views':views,
        'current_protection':protection['proof'],'current_decision_holds':hold_bindings,
        'current_protected_intersections':0,'current_decision_hold_intersections':0,
        'renderer_sha256':renderer_sha,'renderer_full_trajectory_verifier':True,
        'converter_sha256':sha(ROOT/'scripts/neuralese_data/build_native_graph_text_corpus.py'),
        'selection_state':{'path':'selection-state.json','sha256':sha(out/'selection-state.json')},
        'launch_binding':{'path':'launch-binding.json','sha256':sha(out/'launch-binding.json')},
        'selected_shard':{'path':f'selected-shard-{shard}.jsonl.zst','bytes':selected.stat().st_size,'sha256':sha(selected)},
        'transformation':'Existing role-aware ordinary native text conversion; no inferred communication edges'}
    if parent_verification is not None:
        proof['historical_parent_verification']={'path':'historical-parent-verification.json',
            'sha256':sha(out/'historical-parent-verification.json')}
        proof['historical_renderer_code_sha256s']={alias:item['renderer'] for alias,item in parent_verification['parents'].items()}
    write_json_durable(out/'admission-proof.json',proof);include.append('admission-proof.json')
    entry={'id':PREFIX+shard,'owner':'dgx','path':str(out.relative_to(ROOT)),'include':include,
        'source_class':'static-role-aware-native-graph-conversion-of-approved-search-owl-shard',
        'training_admission':True,'training_admission_reason':'Current renderer full trajectory/aggregate verification, exact native counts and current connected protection/decision holds passed for both student views.',
        'derived_from':[PARENT],'license':'Inherited MIT SearchSwarm and Apache-2.0 OWL provenance; owner accepts all licenses.',
        'native_graph_views':views,'admission_proof':'admission-proof.json',
        'notes':'Incremental completed shard only; remaining source shards and actual training exposure are separate obligations.'}
    register([entry],f'Admit source-verified Search native shard {shard} for both students')
    return {'id':entry['id'],'views':views,'proof_sha256':sha(out/'admission-proof.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-binding',type=Path,required=True)
    parser.add_argument('--protection',type=Path,required=True)
    parser.add_argument('--qualified-parent-composition',type=Path)
    parser.add_argument('--shard',action='append',default=[],help='Only these exact completed shard identities')
    parser.add_argument('--watch-seconds',type=float,default=0)
    args=parser.parse_args()
    launch=json.loads(args.launch_binding.read_text());work=ROOT/launch['out_dir']/'work'
    if sha(Path(launch['corpus_manifest']))!=launch['corpus_manifest_sha256']:raise ValueError('source manifest changed')
    state=json.loads((work/'selection-state.json').read_text())
    if state['next_source']!=len(launch['source_turn_files']):raise ValueError('source selection is incomplete')
    tokenizers=dict(item.split('=',1) for item in launch['tokenizers'])
    manifest=json.loads(Path(launch['corpus_manifest']).read_text())
    if any(Path(row['path']).name in ('native-pieces.jsonl','pieces.jsonl') for row in manifest['files']):
        raise ValueError('Search source now declares pieces; bind them through the shared renderer before admission')
    # This admitted parent has no piece sidecars. The exact renderer job
    # verifier below independently requires the same empty-piece input hash.
    # Rehashing every original source merely to rediscover this absence would
    # stall the rolling frontier on tens of GB of unrelated source reads.
    pieces={}
    renderer._init(tokenizers,pieces);input_binding=renderer._render_input_binding(tokenizers,pieces)
    while True:
        protection=current_protection(args.protection)
        registered={row['id'] for row in json.loads((ROOT/'training/neuralese_corpora.json').read_text())['corpora']}
        shards=set.intersection(*[{p.stem for p in (work/'done'/alias).glob('*.json')} for alias in MODELS])
        if args.shard:
            if not set(args.shard)<=shards:raise ValueError('requested Search shard is not completed for both students')
            shards=set(args.shard)
        for shard in sorted(shards):
            if PREFIX+shard in registered:continue
            result=admit(shard,launch=launch,launch_path=args.launch_binding,state=state,work=work,
                tokenizers=tokenizers,input_binding=input_binding,protection=protection,protection_path=args.protection,parent_composition=args.qualified_parent_composition)
            if result:print(json.dumps(result),flush=True)
        if not args.watch_seconds:break
        time.sleep(args.watch_seconds)


if __name__=='__main__':main()
