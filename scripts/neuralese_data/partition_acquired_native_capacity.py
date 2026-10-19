"""Preserve all audited ordinary graphs while exposing a declared geometry envelope."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex, sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_jsonl_text, write_json_durable
from scripts.neuralese_data.build_acquired_full_session_graphs import (
    sessions, acquire_output_lease, _record_uncommitted_cleanup)
from scripts.neuralese_data.prepare_acquired_session_admission import bound


def ordinary_geometry(graph, proof):
    """One ordinary invocation has exactly one physical/KV slot per own token.

    This restricted invariant avoids recompiling every already parsed graph.
    Imports, external state, cache-prefix exports and multi-node graphs require
    the actual general planner and are rejected here rather than guessed.
    """
    nodes=graph.get('nodes') or []
    if (len(nodes)!=1 or graph.get('external_states') or nodes[0].get('imports')
            or nodes[0].get('cache_prefix') or nodes[0].get('producing') or nodes[0].get('logical_producing')):
        raise ValueError('capacity partition requires already audited ordinary one-node graphs')
    width=len(nodes[0]['token_ids'])
    if width!=proof['tokens'] or canonical_json_sha256_hex(graph)!=proof['graph_sha256']:
        raise ValueError('capacity partition graph differs from the exact all-row admission proof')
    return {'forward_tokens':width,'kv_slots':width}


def inside(geometry,limits):
    return all(geometry[key]<=limits[key] for key in ('forward_tokens','kv_slots'))


def partition_preferences(audit_path, limits, out):
    """Partition complete source-qualified pairs using actual compiled KV slots.

    The original audit owns eligibility. This mechanism preserves every pair,
    its exact objective and held split; it grants no execution qualification.
    """
    import torch
    from natlang_neuralese.data.native_graph_schema import parse_graph
    from natlang_neuralese.train.native_preference import NativePreferenceObjective

    audit_path = Path(audit_path).resolve(strict=True)
    out = Path(out).resolve()
    audit_ref = {'path': str(audit_path), 'sha256': sha256_file_hex(audit_path)}
    audit = json.loads(audit_path.read_text())
    if (audit.get('schema') != 'natlang.acquired-offline-preference-admission/1'
            or audit.get('status') != 'passed'
            or audit.get('all_pair_token_render_verified') is not True):
        raise ValueError('preference partition requires complete original-alternative admission')
    if (not audit.get('models') or set(limits) != set(audit['models'])
            or any(set(v) != {'forward_tokens', 'kv_slots'}
                   or any(type(n) is not int or n < 1 for n in v.values()) for v in limits.values())):
        raise ValueError('declare positive physical/KV limits for every preference model')
    binding = {'audit': audit_ref, 'limits': limits, 'verifier_sha256': sha256_file_hex(__file__)}
    with acquire_output_lease(out):
        target = out / 'receipt.json'
        if target.exists():
            receipt = json.loads(target.read_text())
            if receipt['binding'] != binding:
                raise ValueError('completed preference partition has different source/envelope bindings')
            for ref in receipt['files']:
                bound(out / ref['path'], ref)
            for ref in receipt['source_bindings']:
                bound(ref['path'], ref)
            return receipt
        pending, files = out / 'pending', out / 'files'
        for directory in (pending, files):
            if directory.exists():
                _record_uncommitted_cleanup(out / 'shards', directory)
                shutil.rmtree(directory)
        pending.mkdir()
        sources, counts, manifests, objectives, obligations = [], {}, {}, {}, []
        with (pending / 'coverage.jsonl').open('w', encoding='utf-8') as coverage:
            for model, views in audit['models'].items():
                aggregate_ref = audit['training_entrypoints'][model]['objective_spec']
                aggregate_path = bound(aggregate_ref['path'], aggregate_ref)
                sources.append(aggregate_ref)
                aggregate = NativePreferenceObjective(aggregate_path)
                seen, all_objectives, held_outputs = set(), {}, []
                ids = {(split, part): [] for split in ('train', 'test') for part in ('fit', 'remaining-long')}
                counts[model], manifests[model], objectives[model] = {}, {}, {}
                with ExitStack() as stack:
                    writers = {}
                    for split, part in ids:
                        path = pending / model / split / (part + '.jsonl.zst')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        writers[split, part] = stack.enter_context(write_jsonl_text(path))
                    for view in views:
                        if view.get('source_render_verified') is not True:
                            raise ValueError('preference view lacks original-alternative render verification')
                        objective_ref = view['objective']
                        objective_path = bound(objective_ref['path'], objective_ref)
                        sources.append(objective_ref)
                        objective = NativePreferenceObjective(objective_path)
                        if {k: objective.spec[k] for k in ('schema', 'kind', 'beta', 'margin')} != {
                                k: aggregate.spec[k] for k in ('schema', 'kind', 'beta', 'margin')}:
                            raise ValueError('preference view changed aggregate objective parameters')
                        if set(all_objectives) & set(objective.rows):
                            raise ValueError('preference views repeat objective pair identities')
                        all_objectives.update(objective.rows)
                        view_seen = set()
                        for ref in view['outputs']:
                            path = bound(ref['path'], ref)
                            sources.append(ref)
                            if ref['split'] == 'test' and ref['rows']:
                                held_outputs.append(ref)
                            rows = 0
                            with open_jsonl_text(path) as stream:
                                for line in stream:
                                    if not line.strip():
                                        continue
                                    if not line.endswith('\n'):
                                        raise ValueError('preference source row lost its original LF boundary')
                                    row = json.loads(line)
                                    identifier = row['id']
                                    proof = view['proofs'].get(identifier)
                                    if (identifier in seen or proof is None
                                            or proof['graph_sha256'] != canonical_json_sha256_hex(row)
                                            or proof['split'] != row['split'] or ref['split'] != row['split']
                                            or proof['source_groups'] != row['source_groups']
                                            or proof['source_record_sha256'] != row['conversion']['source_record_sha256']
                                            or aggregate.rows.get(identifier) != objective.rows.get(identifier)):
                                        raise ValueError('preference graph differs from its exact source/split/objective proof')
                                    example = parse_graph(row, tokenizer_fingerprint=row['tokenizer_fingerprint'])
                                    with torch.no_grad():
                                        objective.forward_inputs({'examples': [example]})
                                    geometry = {'forward_tokens': example.plan.token_count,
                                                'kv_slots': len(example.plan.slot_sources)}
                                    part = 'fit' if inside(geometry, limits[model]) else 'remaining-long'
                                    key = row['split'], part
                                    index = len(ids[key])
                                    ids[key].append(identifier)
                                    writers[key].write(line)
                                    seen.add(identifier); view_seen.add(identifier); rows += 1
                                    coverage.write(json.dumps({'model': model, 'id': identifier,
                                        'split': row['split'], 'partition': part, 'partition_row_index': index,
                                        'geometry': geometry, 'source': ref,
                                        'source_graph_sha256': proof['graph_sha256'],
                                        'source_row_bytes_sha256': sha256_hex(line.encode('utf-8'))}, sort_keys=True) + '\n')
                                    if row['split'] == 'train' and part == 'remaining-long':
                                        obligations.append({'model': model, 'id': identifier,
                                            'graph_sha256': proof['graph_sha256'], 'geometry': geometry,
                                            'required_visits': 1})
                            if rows != ref['rows']:
                                raise ValueError('preference partition source row count differs from admission')
                        if view_seen != set(objective.rows) or view_seen != set(view['proofs']):
                            raise ValueError('preference partition lost a source-proven pair')
                if seen != set(aggregate.rows) or all_objectives != aggregate.rows:
                    raise ValueError('preference partition lost an aggregate objective pair')
                held = set(ids['test', 'fit']) | set(ids['test', 'remaining-long'])
                for (split, part), identifiers in ids.items():
                    key = split + ':' + part
                    counts[model][key] = {'rows': len(identifiers)}
                    if identifiers:
                        path = pending / model / split / (part + '.jsonl.zst')
                        manifest = path.with_name(part + '-sources.json')
                        write_json_durable(manifest, {'schema': 'natlang.native-graph-sources/1',
                            'sources': [{'kind': 'native-graphs', 'path': path.name,
                                         'sha256': sha256_file_hex(path), 'weight': 1., 'epochs': 1}]})
                        manifests[model][key] = str(Path('files') / manifest.relative_to(pending))
                # Existing held manifests also contain ordinary held CE. Keep
                # them intact and include EVERY held pair in either objective.
                held_ref = audit['training_entrypoints'][model]['held_native_graphs']
                held_path = bound(held_ref['path'], held_ref); sources.append(held_ref)
                held_manifest = json.loads(held_path.read_text())
                if held_manifest.get('schema') != 'natlang.native-graph-sources/1':
                    raise ValueError('preference held entrypoint needs its original source manifest')
                declared_held = {}
                for source in held_manifest.get('sources', []):
                    path = (held_path.parent / source['path']).resolve(strict=True)
                    if (path in declared_held or source.get('kind') != 'native-graphs'
                            or type(source.get('weight')) not in (int, float) or source['weight'] != 1.
                            or type(source.get('epochs')) is not int or source['epochs'] != 1):
                        raise ValueError('preference held source identity/weight/epochs changed')
                    bound(path, source)
                    declared_held[path] = source
                    sources.append({'path': str(path), 'sha256': source['sha256'], 'bytes': path.stat().st_size})
                if any(declared_held.get(Path(ref['path']).resolve(), {}).get('sha256') != ref['sha256']
                       for ref in held_outputs):
                    raise ValueError('preference held manifest omitted an admitted held pair source')
                manifests[model]['held_native_graphs'] = held_ref
                for part in ('fit', 'remaining-long'):
                    selected = set(ids['train', part]) | held
                    path = pending / model / (part + '-objective.json')
                    write_json_durable(path, {**aggregate.spec, 'rows': {
                        identifier: value for identifier, value in aggregate.rows.items() if identifier in selected}})
                    objectives[model][part] = str(Path('files') / path.relative_to(pending))
        write_json_durable(pending / 'pending-sources.json', {
            'schema': 'natlang.native-pending-source-declaration/1', 'full_cohort': audit_ref,
            'binding': binding, 'rows': obligations,
            'policy': 'Complete source-pair obligations; earned visits remain native continuation progress.'})
        for path in pending.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        pending.replace(files)
        fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        inventory = [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': sha256_file_hex(p)}
                     for p in sorted(files.rglob('*')) if p.is_file()]
        bound(audit_path, audit_ref)
        for ref in sources:
            bound(ref['path'], ref)
        receipt = {'schema': 'natlang.acquired-preference-capacity-partition/1', 'binding': binding,
            'source_bindings': sources, 'counts': counts, 'manifests': manifests, 'objectives': objectives,
            'coverage': 'files/coverage.jsonl', 'pending_source_declaration': 'files/pending-sources.json',
            'files': inventory, 'exact_original_row_bytes': True, 'source_eligibility_unchanged': True,
            'training_admission': False, 'execution_qualified': False, 'earned_pair_visits': 0,
            'full_recipe_completed': False}
        write_json_durable(target, receipt)
        return receipt


def partition(audit_path,samples_path,limits,out):
    audit_path,samples_path,out=Path(audit_path).resolve(strict=True),Path(samples_path).resolve(strict=True),Path(out).resolve()
    audit_binding={'path':str(audit_path),'sha256':sha256_file_hex(audit_path)}
    samples_binding={'path':str(samples_path),'sha256':sha256_file_hex(samples_path)}
    audit=json.loads(audit_path.read_text());sample=json.loads(samples_path.read_text())
    sample_audit=sample.get('cohort_audit',{})
    static = audit.get('schema') == 'natlang.static-repository-admission-audit/1'
    if static:
        from scripts.neuralese_data.prepare_static_repository_admission import verify_audit
        if verify_audit(audit_path) != audit or sample.get('all_rows') is not True or sample.get('static_source_replay') is not True:
            raise ValueError('static partition requires authenticated complete source fidelity replay')
    if (audit.get('schema') not in {'natlang.acquired-session-admission-audit/1', 'natlang.static-repository-admission-audit/1'} or audit.get('status')!='passed'
            or sample.get('passed') is not True or sample_audit.get('sha256')!=audit_binding['sha256']
            or not sample_audit.get('path') or Path(sample_audit['path']).resolve()!=audit_path):
        raise ValueError('capacity partition needs its complete source audit and bound source render samples')
    if (set(limits)!=set(audit['by_model']) or any(set(values)!={'forward_tokens','kv_slots'}
            or any(type(v) is not int or v<1 for v in values.values()) for values in limits.values())):
        raise ValueError('declare positive physical/KV limits for every audited model; limits do not qualify execution')
    proof_path=bound(audit_path.parent/'row-proof.jsonl',{'sha256':audit['row_proof_sha256']})
    proofs={}
    for proof in sessions(proof_path):
        key=(proof['model'],proof['id'])
        if key in proofs:raise ValueError('cohort row proof repeats a graph identity')
        proofs[key]=proof
    binding={'audit':audit_binding,'source_render_samples':samples_binding,'limits':limits,
        'verifier_sha256':sha256_file_hex(__file__)}
    with acquire_output_lease(out):
        receipt_path=out/'receipt.json'
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if receipt['binding']!=binding:raise ValueError('completed capacity partition has different source/envelope bindings')
            for item in receipt['files']:bound(out/item['path'],item)
            for item in receipt['source_bindings']:bound(item['path'],item)
            return receipt
        pending=out/'pending';files=out/'files'
        for directory in (pending,files):
            if directory.exists():
                # Reuse the converter's exact owned-uncommitted-data inventory
                # before deleting a partial, unreceipted partition shard.
                _record_uncommitted_cleanup(out/'shards',directory)
                shutil.rmtree(directory)
        pending.mkdir()
        counts=Counter();tokens=Counter();seen=set();remaining=[];source_bindings=[]
        outputs={}
        with ExitStack() as stack:
            writers={}
            for model in limits:
                for split in ('train','test'):
                    for part in ('fit','remaining-long'):
                        path=pending/model/split/(part+'.jsonl.zst');path.parent.mkdir(parents=True,exist_ok=True)
                        writers[model,split,part]=stack.enter_context(write_jsonl_text(path))
                        outputs[model,split,part]=path
            coverage=stack.enter_context((pending/'coverage.jsonl').open('w',encoding='utf-8',newline='\n'))
            for conversion in audit['conversions']:
                model=conversion['model']
                for output in conversion['outputs']:
                    path=bound(output['path'],output);source_bindings.append(output)
                    with open_jsonl_text(path) as stream:
                        for row_index,line in enumerate(stream):
                            if not line.strip():continue
                            if not line.endswith('\n'):raise ValueError('preserve exact source rows only with original LF boundaries')
                            graph=json.loads(line);key=(model,graph['id']);proof=proofs.get(key)
                            if proof is None or key in seen:raise ValueError('partition source has an unknown or duplicated audited graph')
                            if graph['split']!=proof['split'] or output['split']!=proof['split']:
                                raise ValueError('partition source changed its audited held split')
                            geometry=ordinary_geometry(graph,proof)
                            part='fit' if inside(geometry,limits[model]) else 'remaining-long'
                            label=(model,graph['split'],part);partition_index=counts[label]
                            writers[label].write(line);counts[label]+=1;tokens[label]+=proof['tokens'];seen.add(key)
                            coverage.write(json.dumps({'model':model,'id':graph['id'],'split':graph['split'],
                                'source':{'path':str(path),'sha256':output['sha256'],'row_index':row_index},
                                'source_graph_sha256':proof['graph_sha256'],'source_row_bytes_sha256':sha256_hex(line.encode('utf-8')),
                                'partition':part,'partition_row_index':partition_index,'geometry':geometry},sort_keys=True)+'\n')
                            if part=='remaining-long' and graph['split']=='train':
                                remaining.append({'model':model,'id':graph['id'],'graph_sha256':proof['graph_sha256'],
                                    'required_visits':1,'geometry':geometry})
        if seen!=set(proofs):raise ValueError('capacity partition lost audited graphs')
        summaries={};manifests={}
        for model in limits:
            summaries[model]={}
            for split in ('train','test'):
                for part in ('fit','remaining-long'):
                    label=(model,split,part);path=outputs[label]
                    summaries[model][split+':'+part]={'rows':counts[label],'tokens':tokens[label]}
                    if counts[label]:
                        manifest=path.parent/(part+'-sources.json')
                        write_json_durable(manifest,{'schema':'natlang.native-graph-sources/1',
                            'sources':[{'kind':'native-graphs','path':path.name,'sha256':sha256_file_hex(path),'weight':1.,'epochs':1}]})
                        manifests[':'.join(label)]=str(Path('files')/manifest.relative_to(pending))
        write_json_durable(pending/'pending-sources.json',{
            'schema':'natlang.native-pending-source-declaration/1','full_cohort':audit_binding,
            'binding':binding,'rows':remaining,
            'policy':'source obligations only; earned exposure belongs to existing native_continuation progress'})
        for path in pending.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:os.fsync(stream.fileno())
        pending.replace(files)
        directory_fd=os.open(out,os.O_DIRECTORY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
        inventory=[{'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':sha256_file_hex(path)}
            for path in sorted(files.rglob('*')) if path.is_file()]
        if sha256_file_hex(audit_path)!=audit_binding['sha256'] or sha256_file_hex(samples_path)!=samples_binding['sha256']:
            raise ValueError('source evidence changed during capacity partition')
        if static:
            verify_audit(audit_path)
        for item in source_bindings:bound(item['path'],item)
        receipt={'schema':'natlang.acquired-native-capacity-partition/1','binding':binding,
            'source_bindings':source_bindings,
            'counts':summaries,'graphs':len(seen),'coverage':'files/coverage.jsonl','files':inventory,
            'manifests':manifests,'pending_source_declaration':'files/pending-sources.json',
            'source_eligibility_unchanged':True,'exact_original_row_bytes':True,
            'execution_qualified':False,'production_append_requires_actual_capacity_certificate':True,
            'full_recipe_completed':False}
        write_json_durable(receipt_path,receipt)
        return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--source-render-samples',type=Path)
    parser.add_argument('--preferences',action='store_true',help='partition complete source-audited offline pair graphs')
    parser.add_argument('--limits',type=Path,required=True,help='model keyed forward_tokens/kv_slots; parameters, not certification')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.preferences:
        if args.source_render_samples:parser.error('preference admission already binds original-alternative render verification')
        receipt=partition_preferences(args.audit,json.loads(args.limits.read_text()),args.out)
    else:
        if not args.source_render_samples:parser.error('--source-render-samples is required for ordinary session graphs')
        receipt=partition(args.audit,args.source_render_samples,json.loads(args.limits.read_text()),args.out)
    print(json.dumps({'path':str(args.out/'receipt.json'),'sha256':sha256_file_hex(args.out/'receipt.json'),
        'counts':receipt['counts'],'execution_qualified':False,'full_recipe_completed':False},sort_keys=True))


if __name__=='__main__':main()
