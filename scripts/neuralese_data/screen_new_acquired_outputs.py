#!/usr/bin/env python3
"""Stage only newly completed acquired outputs after exact current custody checks."""
import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path

from natlang_neuralese.common.custody import ProtectedWorlds, world_identities
from natlang_neuralese.common.hashing import sha256_file_hex, canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from natlang_neuralese.data.native_graph_schema import indexed_graph_metadata
from natlang_neuralese.data.native_sources import _row_bindings
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds

MODULE_SOURCE_BYTES=Path(__file__).read_bytes()


def bind(path):
    path=Path(path).resolve(strict=True);before=path.stat();sha=sha256_file_hex(path);after=path.stat()
    ident=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    if ident(before)!=ident(after):raise ValueError('input changed while hashing: '+str(path))
    return {'path':str(path),'sha256':sha,'bytes':before.st_size}


def write_original(writer, line, graph):
    """Retain the exact validated JSONL bytes without re-encoding giant token arrays."""
    writer.encoder.stdin.write(line.encode('utf-8') if line.endswith('\n') else (line+'\n').encode('utf-8'))
    writer.rows += 1


def inventory(summary, old_coverages):
    old={(o['path'],o['sha256']) for c in old_coverages for r in c['receipts'] for o in r.get('outputs',[])}
    result=[o for source in summary['by_source'].values() for o in source['outputs']
            if '/train/' in o['path'] and (o['path'],o['sha256']) not in old]
    identities=[(o['path'],o['sha256']) for o in result]
    if len(set(identities))!=len(identities):raise ValueError('conversion summary repeats an output')
    return result


def case(graph):
    conv=graph.get('conversion') or {};acquired=conv.get('acquired_session') or {}
    ids=list(conv.get('source_record_ids') or [])+list(acquired.get('original_target_ids') or [])
    ids += [v for v in (conv.get('source_row_id'),acquired.get('original_source_row')) if isinstance(v,str) and v]
    return {'id':graph['id'],'split':'train','source_ids':ids,'source_groups':graph['source_groups'],
            'split_groups':graph['source_groups']}


def screen(summary_path, coverage_path, predecessor_path, protection_path, out, *, reserved_screens=()):
    summary_path,coverage_path,predecessor_path,protection_path=map(lambda p:Path(p).resolve(strict=True),
        (summary_path,coverage_path,predecessor_path,protection_path))
    inputs={k:bind(v) for k,v in [('summary',summary_path),('coverage',coverage_path),('predecessor',predecessor_path),('source_protection',protection_path)]}
    summary=json.loads(summary_path.read_text());coverage=json.loads(coverage_path.read_text());predecessor=json.loads(predecessor_path.read_text())
    if summary.get('schema')!='natlang.acquired-session-student-conversion/1' or summary.get('incomplete'):
        raise ValueError('source inventory is not a complete conversion')
    previous=[coverage]
    for field in ('parent_inventory','excluded_inventory'):
        ref=coverage[field];actual=bind(ref['path'])
        if actual['sha256']!=ref['sha256']:raise ValueError('prior coverage inventory changed')
        inputs[field]=actual;previous.append(json.loads(Path(actual['path']).read_text()))
    candidates=inventory(summary,previous)
    family_ref=predecessor['family_map'];family_path=Path(family_ref['path']).resolve(strict=True)
    if bind(family_path)['sha256']!=family_ref['sha256']:raise ValueError('prepared family map changed')
    family=json.loads(family_path.read_text());prior={r['id']:r['binding'] for r in family['rows']}
    if (len(prior)!=len(family['rows']) or len(prior)!=predecessor['rows']
            or family.get('source_sha256')!=predecessor['source']['sha256']):
        raise ValueError('prepared source ID inventory is incomplete')
    prior_pending=set()
    if predecessor.get('source_screen'):
        source_screen_ref=predecessor['source_screen'];screen_path=Path(source_screen_ref['path']).resolve(strict=True)
        if bind(screen_path)['sha256']!=source_screen_ref['sha256']:raise ValueError('prior source screen changed')
        prior_screen=json.loads(screen_path.read_text());partition_ref=prior_screen.get('pending_partition')
        if partition_ref:
            partition_path=Path(partition_ref['path']).resolve(strict=True)
            if bind(partition_path)['sha256']!=partition_ref['sha256']:raise ValueError('prior pending partition changed')
            partition=json.loads(partition_path.read_text())
            rows=partition.get('pending_rows') or []
            prior_pending={r['id'] for r in rows}
            if len(prior_pending)!=len(rows):raise ValueError('prior pending partition repeats an ID')
            inputs['prior_pending_partition']=bind(partition_path)
        inputs['prior_source_screen']=bind(screen_path)
    protection=ProtectedWorlds.load();quality=current_protection(protection_path)
    hold_paths=sorted((Path(__file__).resolve().parents[2]/'training/decision-holds').glob('*.jsonl'))
    held_ids,held_refs=load_decision_holds(hold_paths)
    held_sources={x.split(':decision:',1)[0] for x in held_ids}|held_ids
    reserved_ids=set()
    for index,reserved_path in enumerate(reserved_screens):
        screen_ref=bind(reserved_path);record=json.loads(Path(screen_ref['path']).read_text())
        if (record.get('schema')!='natlang.new-acquired-source-screen/1' or record.get('training_admission') is not False
                or record.get('prepared_source_sha256')!=predecessor['source']['sha256']
                or record.get('protected_worlds_sha256')!=protection.digest
                or record.get('quality_protection')!=quality['proof'] or record.get('decision_holds')!=held_refs):
            raise ValueError('reserved source screen differs from current predecessor or protections')
        prior_builder=record.get('builder') or {}
        builder_path=Path(prior_builder.get('path',''))
        if not builder_path.is_file() or sha256_file_hex(builder_path)!=prior_builder.get('sha256'):
            builder_path=Path(screen_ref['path']).parent/('screen-builder-'+str(prior_builder.get('sha256'))+'.py')
        actual_builder=bind(builder_path)
        if actual_builder['sha256']!=prior_builder.get('sha256'):
            raise ValueError('reserved source screen lacks its exact producer code')
        inputs['reserved_builder_'+str(index)]=actual_builder
        active_ref=record['active'];actual=bind(active_ref['path'])
        if actual!={k:active_ref[k] for k in ('path','sha256','bytes')}:
            raise ValueError('reserved screened source bytes changed')
        ids_ref=record.get('active_ids')
        if ids_ref is not None:
            ids_binding=bind(ids_ref['path'])
            if ids_binding!=ids_ref:raise ValueError('reserved source ID index changed')
            ids=json.loads(Path(ids_binding['path']).read_text())
            if not isinstance(ids,list):raise ValueError('reserved source ID index is not a list')
            inputs['reserved_ids_'+str(index)]=ids_binding
        else:
            ids=[]
            with open_jsonl_text(actual['path']) as stream:
                for line in stream:
                    if line.strip():
                        graph=json.loads(line)
                        if graph.get('split')!='train':raise ValueError('reserved source has a non-train graph')
                        ids.append(graph.get('id'))
        if (len(ids)!=active_ref['rows'] or any(not isinstance(ident,str) for ident in ids)
                or len(set(ids))!=len(ids) or set(ids)&reserved_ids or bind(actual['path'])!=actual):
            raise ValueError('reserved source repeats IDs or changed while reading')
        reserved_ids.update(ids)
        inputs['reserved_screen_'+str(index)]=screen_ref
        inputs['reserved_source_'+str(index)]=actual
    output=Path(out).resolve();output.mkdir(parents=True,exist_ok=False)
    active=GraphWriter(output/'active.jsonl.zst');pending=GraphWriter(output/'over-context-pending.jsonl.zst')
    counts=Counter();seen=set();active_ids=[];decisions=[];receipts=[];tokenizer=None
    try:
        for index,item in enumerate(candidates):
            path=Path(item['path']).resolve(strict=True);actual=bind(path)
            if actual!={k:item[k] for k in ('path','sha256','bytes')}:raise ValueError('output bytes differ from conversion inventory')
            rp=path.parent.parent/'receipt.json';rb=bind(rp);receipt=json.loads(rp.read_text())
            if (receipt.get('schema')!='natlang.acquired-full-session-graphs/1' or receipt.get('training_admission') is not False
                    or item not in receipt.get('outputs',[])):
                raise ValueError('output lacks its actual unadmitted conversion receipt')
            receipts.append(rb);observed=0
            with open_jsonl_text(path) as stream:
                for line in stream:
                    if not line.strip():continue
                    graph=json.loads(line);observed+=1;counts['raw']+=1;ident=graph.get('id')
                    fp=graph.get('tokenizer_fingerprint')
                    if tokenizer is None:tokenizer=fp
                    if (not isinstance(ident,str) or not ident or ident in seen or graph.get('split')!='train'
                            or fp!=tokenizer or fp!=receipt['expanded_tokenizer_fingerprint']):
                        raise ValueError('new output repeats IDs or changes split/tokenizer')
                    seen.add(ident);groups=graph.get('source_groups')
                    if not isinstance(groups,list) or not groups or len(set(groups))!=len(groups):raise ValueError('invalid source groups')
                    c=case(graph);related=world_identities(c,evidence_items=False)&protection.identities
                    related=sorted((k,v) for k,v in related if k in ('id','group','binding'))
                    quality_groups=sorted(set(groups)&(quality['held']|quality['quality']|quality['pending']))
                    held=sorted(set(c['source_ids']+[ident])&held_sources)
                    reason=None;row_binding=None
                    if related:reason='current-protected-world'
                    elif quality_groups:reason='current-source-quality-or-held-group'
                    elif held:reason='current-decision-hold'
                    elif ident in prior_pending:reason='already-pending-source-obligation'
                    elif ident in reserved_ids:reason='already-reserved-source-obligation'
                    elif ident in prior:
                        meta=indexed_graph_metadata(graph,tokenizer_fingerprint=tokenizer)
                        row_binding=_row_bindings([meta])[ident]
                        reason='already-present-exact' if prior[ident]==row_binding else 'already-present-different-binding-review'
                    elif any(len(node.get('token_ids') or [])>262144 for node in graph.get('nodes') or []):
                        reason='over-context-pending'
                    if reason:
                        counts[reason]+=1;decisions.append({'id':ident,'binding':row_binding,'reason':reason,
                            **({'matches':related} if related else {}),**({'groups':quality_groups} if quality_groups else {}),
                            **({'held_ids':held} if held else {})})
                        if reason=='over-context-pending':write_original(pending,line,graph)
                    else:write_original(active,line,graph);active_ids.append(ident);counts['active']+=1
            if observed!=item['rows'] or bind(path)!=actual or bind(rp)!=rb:raise ValueError('conversion output changed while screening')
            print(json.dumps({'kind':'acquired_screen_progress','files':index+1,'of':len(candidates),'rows':counts['raw']}),flush=True)
        active_ref=active.finish();pending_ref=pending.finish()
    except BaseException:
        active.abort();pending.abort();raise
    if ProtectedWorlds.load().digest!=protection.digest or current_protection(protection_path)['proof']!=quality['proof']:
        raise ValueError('current protection changed during screening')
    if any(bind(ref['path'])!=ref for ref in inputs.values()) or load_decision_holds(hold_paths)[1]!=held_refs:
        raise ValueError('source screen controls changed')
    decisions_path=output/'decisions.json';write_json_durable(decisions_path,decisions)
    active_ids_path=output/'active-ids.json';write_json_durable(active_ids_path,active_ids)
    if Path(__file__).read_bytes()!=MODULE_SOURCE_BYTES:
        raise ValueError('screen code changed while the process was running')
    builder=output/('screen-builder-'+hashlib.sha256(MODULE_SOURCE_BYTES).hexdigest()+'.py')
    builder.write_bytes(MODULE_SOURCE_BYTES);builder.chmod(0o444)
    proof={'schema':'natlang.new-acquired-source-screen/1','training_admission':False,'inputs':inputs,
        'prepared_source_sha256':predecessor['source']['sha256'],'protected_worlds_sha256':protection.digest,
        'quality_protection':quality['proof'],'decision_holds':held_refs,'receipts':receipts,
        'output_inventory':candidates,'counts':dict(counts),'active':active_ref,'pending':pending_ref,
        'decisions':bind(decisions_path),'active_ids':bind(active_ids_path),'tokenizer_fingerprint':tokenizer,'builder':bind(builder),
        'limitations':['Current protection is checked against identities actually present in each converted graph.',
          'Active rows require canonical native indexing, semantic quality checks and fresh admission before training; no exposure is claimed.',
          'Unchanged context-window obligations remain pending; no graph is truncated.']}
    write_json_durable(output/'receipt.json',proof);return proof


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('summary','coverage','predecessor','protection','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--reserved-screen',action='append',default=[])
    args=p.parse_args();print(json.dumps(screen(args.summary,args.coverage,args.predecessor,args.protection,args.out,reserved_screens=args.reserved_screen)['counts'],sort_keys=True))

if __name__=='__main__':main()
