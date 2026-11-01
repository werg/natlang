#!/usr/bin/env python3
"""Retain over-position ordinary graphs as explicit immutable pending obligations."""
from __future__ import annotations
import argparse, copy, json, os, shutil
from collections import Counter
from pathlib import Path
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from natlang_neuralese.common.custody import ProtectedWorlds, world_identities
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter

DEFAULT_MAX_TOKENS=262144

def _checked(ref):
    if not isinstance(ref,dict) or not isinstance(ref.get('path'),str) or not isinstance(ref.get('sha256'),str):
        raise ValueError('pending partition needs an exact binding')
    path=Path(ref['path']).resolve(strict=True);before=path.stat();digest=sha256_file_hex(path);after=path.stat()
    if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
        or digest!=ref['sha256'] or 'bytes' in ref and ref['bytes']!=before.st_size):
        raise ValueError('pending partition input differs from its binding')
    return path

def _read(ref): return json.loads(_checked(ref).read_text())

def _ordinary(row):
    from natlang_neuralese.data.native_graph_schema import indexed_graph_metadata
    example=indexed_graph_metadata(row,tokenizer_fingerprint=row.get('tokenizer_fingerprint'))
    nodes=row.get('nodes') or []
    if (len(nodes)!=1 or len(example.plan.invocations)!=1 or row.get('external_states') or row.get('decisions')
        or row.get('source_pairing') or row.get('source_chain') or any(nodes[0].get(k) for k in ('imports','producing','logical_producing'))
        or set(nodes[0])-{'name','token_ids','target_weights'}
        or (row.get('conversion') or {}).get('role_interpretation') not in ('ordinary-text','whole-trajectory-chat-roles')):
        raise ValueError('position partition requires a single ordinary graph')

def _protected_case(row):
    conversion=row.get('conversion') or {};acquired=conversion.get('acquired_session') or {}
    ids=list(conversion.get('source_record_ids') or [])+list(acquired.get('original_target_ids') or [])
    ids.extend(x for x in (conversion.get('source_row_id'),acquired.get('original_source_row')) if isinstance(x,str) and x)
    return {'id':row['id'],'split':row['split'],'source_ids':ids,'source_groups':row['source_groups'],'split_groups':row['source_groups']}

SCHEMA='natlang.acquired-position-pending-partition/1'

def bind(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha256_file_hex(path)}

def verify_partition(screen, predecessor):
    """Reconstruct the two partitions from the authenticated original source."""
    proof=_read(screen.get('pending_partition'))
    if proof.get('schema')!=SCHEMA or proof.get('predecessor')!=bind(predecessor):
        raise ValueError('position pending proof has different predecessor')
    candidate=_read(proof['candidate']);original=_read(proof['original_screen']);limit=proof.get('limit')
    if (candidate.get('schema')!='natlang.acquired-streaming-source-successor/1' or candidate.get('training_admission') is not False
        or candidate.get('source_screen')!=proof['original_screen'] or candidate.get('predecessor')!=proof['predecessor']
        or candidate.get('effective_loss_census')!=proof['original_census'] or candidate.get('row_signals')!=proof['original_row_signals']
        or candidate.get('appended_graphs',{}).get('sha256')!=original.get('output',{}).get('sha256')):
        raise ValueError('position pending proof does not bind the exact candidate')
    from natlang_neuralese.rl.native_direct_sft_recovery import _authenticated_refresh_publication
    published=_authenticated_refresh_publication(proof['parent_publication'])['publication']
    if (published.get('rolling_endpoint_custody_proof') or {}).get('successor')!=proof['parent_snapshot']:
        raise ValueError('position pending parent snapshot differs from authenticated publication')
    saved=_read(proof['parent_snapshot']);frames=saved.get('sft_frames')
    if (saved.get('schema')!='natlang.rolling-sft-parent-custody-snapshot/1' or saved.get('optimizer_present') is not True
        or saved.get('rng_present') is not True or (saved.get('checkpoint') or {}).get('sha256')!=(published.get('checkpoint') or {}).get('sha256')
        or not isinstance(frames,dict) or not frames):raise ValueError('position pending parent is not an exact full SFT endpoint')
    if (type(limit) is not int or not 2<=limit<=DEFAULT_MAX_TOKENS or proof.get('input_graph')!=original.get('output')
        or proof.get('active_graph')!=screen.get('output') or proof.get('protected_worlds_sha256')!=screen.get('protected_worlds',{}).get('identity_set_sha256')):
        raise ValueError('position pending proof source or protection differs')
    expected=copy.deepcopy(original);expected['output']=proof['active_graph'];expected['pending_partition']=screen['pending_partition']
    expected['accounting']['eligible_train_graphs']=proof['active_graph']['rows'];expected['accounting']['position_pending_graphs']=proof['pending_graph']['rows']
    if expected!=screen:raise ValueError('position pending screen changed unrelated custody')
    source=_checked(proof['input_graph']);active=_checked(proof['active_graph']);pending=_checked(proof['pending_graph'])
    records=proof.get('pending_rows')
    if not isinstance(records,list) or len(records)!=proof['pending_graph']['rows'] or len({x.get('id') for x in records})!=len(records):
        raise ValueError('position pending inventory is invalid')
    by_id={x['id']:x for x in records};seen=set();active_ids=[];pending_ids=[]
    with open_jsonl_text(source) as src, open_jsonl_text(active) as kept, open_jsonl_text(pending) as deferred:
        kept=iter(kept);deferred=iter(deferred)
        for line in src:
            if not line.strip():continue
            row=json.loads(line);ident=row.get('id')
            if ident in seen:raise ValueError('original screen repeats graph identity')
            seen.add(ident);long=any(len(n.get('token_ids') or [])>limit for n in row.get('nodes') or [])
            if any(ident in (f.get('source_row_bindings') or {}) or ident in (f.get('pending_source_rows') or {})
                or (f.get('exposure_counts') or {}).get(ident,0)!=0 for f in frames.values()):
                raise ValueError('position pending source was present in a saved SFT frame')
            stream=deferred if long else kept
            result=next((x for x in stream if x.strip()),None)
            if result is None or json.loads(result)!=row:raise ValueError('position partition changed or omitted original graph')
            if long:
                _ordinary(row);entry=by_id.get(ident)
                if not entry or entry.get('graph_sha256')!=canonical_json_sha256_hex(row) or entry.get('forward_tokens')!=sum(len(n['token_ids']) for n in row['nodes']):
                    raise ValueError('position pending row evidence differs')
                pending_ids.append(ident)
            else:active_ids.append(ident)
        if next((x for x in kept if x.strip()),None) is not None or next((x for x in deferred if x.strip()),None) is not None:
            raise ValueError('position partition contains extra graph')
    if (len(seen)!=original['output']['rows'] or len(active_ids)!=proof['active_graph']['rows'] or len(pending_ids)!=len(records)
        or canonical_json_sha256_hex(sorted(seen))!=original['output']['graph_ids_sha256']
        or canonical_json_sha256_hex(sorted(active_ids))!=proof['active_graph']['graph_ids_sha256']
        or canonical_json_sha256_hex(sorted(pending_ids))!=proof['pending_graph']['graph_ids_sha256']):
        raise ValueError('position partition identity inventory differs')
    return original,len(pending_ids)

def verify_partition_census(screen, census):
    proof=_read(screen['pending_partition']);original=_read(proof['original_census'])
    expected_policy=original.get('ce_policy');expected_objective=original.get('objective_sha256')
    if screen.get('predecessor_rebase'):
        from scripts.neuralese_data.rebase_acquired_pending_partition import verify_rebase
        from natlang_neuralese.train.own_output_supervision import policy_objective
        rebase=_read(screen['predecessor_rebase'])
        verify_rebase(screen,rebase['current_predecessor']['path'])
        predecessor=_read(rebase['current_predecessor']);expected_policy=predecessor['ce_policy']
        expected_objective=canonical_json_sha256_hex(policy_objective(_read(expected_policy)))
    if (original.get('schema')!='natlang.acquired-effective-learning-census/1' or original.get('training_admission') is not False
        or original.get('screen',{}).get('sha256')!=proof['original_screen']['sha256']
        or original.get('row_signals')!=proof['original_row_signals'] or census.get('pending_partition')!=screen['pending_partition']
        or census.get('ce_policy')!=expected_policy or census.get('objective_sha256')!=expected_objective
        or census.get('builder')!=original.get('builder')):
        raise ValueError('position pending census changed original objective custody')
    pending={x['id']:x for x in proof['pending_rows']};seen=set();count=0
    with open_jsonl_text(_checked(proof['original_row_signals'])) as prior, open_jsonl_text(_checked(census['row_signals'])) as active:
        active=iter(active)
        for line in prior:
            if not line.strip():continue
            item=json.loads(line);ident=item['id'];count+=1
            if ident in pending:
                entry=pending[ident];seen.add(ident)
                if any(entry[k]!=item[k] for k in ('graph_sha256','forward_tokens','effective_positive_tokens','category_mass')):
                    raise ValueError('position pending objective evidence differs')
            else:
                actual=next((x for x in active if x.strip()),None)
                if actual is None or json.loads(actual)!=item:raise ValueError('position pending active census changed')
        if next((x for x in active if x.strip()),None) is not None:raise ValueError('position pending census contains extra rows')
    if seen!=set(pending) or count!=original.get('rows') or count-len(seen)!=census.get('rows'):
        raise ValueError('position pending census inventory differs')

def partition(candidate_ref, parent_publication, output, *, limit=DEFAULT_MAX_TOKENS):
    if type(limit) is not int or not 2<=limit<=DEFAULT_MAX_TOKENS: raise ValueError('invalid position limit')
    from natlang_neuralese.rl.native_direct_sft_recovery import _authenticated_refresh_publication
    publication=_authenticated_refresh_publication(parent_publication)['publication']
    snap_ref=(publication.get('rolling_endpoint_custody_proof') or {}).get('successor')
    saved=_read(snap_ref);frames=saved.get('sft_frames')
    if (saved.get('schema')!='natlang.rolling-sft-parent-custody-snapshot/1' or saved.get('optimizer_present') is not True
        or saved.get('rng_present') is not True or (saved.get('checkpoint') or {}).get('sha256')!=(publication.get('checkpoint') or {}).get('sha256')
        or not isinstance(frames,dict) or not frames): raise ValueError('missing published full SFT endpoint')
    candidate=_read(candidate_ref)
    if candidate.get('schema')!='natlang.acquired-streaming-source-successor/1' or candidate.get('training_admission') is not False:
        raise ValueError('partition requires unadmitted candidate')
    screen=_read(candidate['source_screen']);census=_read(candidate['effective_loss_census'])
    if (screen.get('output',{}).get('sha256')!=candidate['appended_graphs']['sha256'] or
        census.get('screen',{}).get('sha256')!=candidate['source_screen']['sha256'] or
        census.get('row_signals',{}).get('sha256')!=candidate['row_signals']['sha256'] or
        census.get('rows')!=candidate['new_rows']): raise ValueError('candidate source custody differs')
    source=_checked(candidate['appended_graphs']);signals=_checked(candidate['row_signals'])
    previous=_read(candidate['predecessor']);family=_read(previous['family_map'])
    if family.get('source_sha256')!=previous['source']['sha256'] or len(family.get('rows',[]))!=candidate['predecessor_rows']:
        raise ValueError('predecessor family inventory differs')
    old={x['id'] for x in family['rows']}
    if len(old)!=candidate['predecessor_rows']: raise ValueError('duplicate predecessor identities')
    protected=ProtectedWorlds.load(); output=Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir() or output.parent.stat().st_dev!=source.stat().st_dev:
        raise ValueError('fresh output on source filesystem required')
    if shutil.disk_usage(output.parent).free<40*1024**3+3*source.stat().st_size: raise ValueError('disk floor')
    stage=output.with_name('.'+output.name+'.staging-'+str(os.getpid()));stage.mkdir()
    active=GraphWriter(stage/'active.jsonl.zst');pending=GraphWriter(stage/'pending.jsonl.zst')
    seen=set();pending_rows=[];counts=Counter();mass=Counter();stores=Counter();unknown=Counter();active_ids=[]
    try:
        with open_jsonl_text(source) as rows, open_jsonl_text(signals) as signal_rows, (stage/'row-signals.jsonl').open('x') as out:
            signal_iter=(json.loads(x) for x in signal_rows if x.strip())
            for line in rows:
                if not line.strip(): continue
                row=json.loads(line);item=next(signal_iter,None);ident=row.get('id')
                if (not isinstance(item,dict) or item.get('id')!=ident or item.get('graph_sha256')!=canonical_json_sha256_hex(row)
                    or item.get('forward_tokens')!=sum(len(n.get('token_ids') or []) for n in row.get('nodes') or [])
                    or item.get('raw_positive_tokens')!=sum(w>0 for n in row.get('nodes') or [] for w in n.get('target_weights') or [])):
                    raise ValueError('source differs from measured row census')
                if not isinstance(ident,str) or not ident or ident in seen or ident in old or row.get('split')!='train': raise ValueError('duplicate or invalid source row')
                if any(ident in (f.get('source_row_bindings') or {}) or ident in (f.get('pending_source_rows') or {}) or (f.get('exposure_counts') or {}).get(ident,0)!=0 for f in frames.values()):
                    raise ValueError('candidate row already belongs to saved SFT frame')
                if row.get('tokenizer_fingerprint')!=screen['output']['tokenizer_fingerprint'] or world_identities(_protected_case(row),evidence_items=False)&protected.identities:
                    raise ValueError('tokenizer or protected-world custody differs')
                seen.add(ident)
                if any(len(n.get('token_ids') or [])>limit for n in row['nodes']):
                    _ordinary(row)
                    pending.write(row);pending_rows.append({'id':ident,'graph_sha256':item['graph_sha256'],'forward_tokens':item['forward_tokens'],
                        'effective_positive_tokens':item['effective_positive_tokens'],'category_mass':item['category_mass'],
                        'reason':'ordinary-context-exceeds-position-limit; original evidence retained; graphification and target qualification pending'})
                else:
                    active.write(row);active_ids.append(ident);out.write(json.dumps(item,sort_keys=True)+'\n')
                    for key in ('forward_tokens','raw_positive_tokens','effective_positive_tokens'):counts[key]+=item[key]
                    counts['source_label_diagnosis_rows']+=item.get('source_label_diagnosis') is True
                    counts['oracle_verified_rows']+=item.get('oracle_verified') is True
                    counts['zero_effective_rows']+=item.get('eligible') is not True
                    mass.update(item['category_mass']);stores[item['source_store'] or '<unknown>']+=1
                    if not item['source_store']:unknown['source_store']+=1
            if next(signal_iter,None) is not None: raise ValueError('extra census rows')
            out.flush();os.fsync(out.fileno())
        if len(seen)!=candidate['new_rows'] or canonical_json_sha256_hex(sorted(seen))!=screen['output']['graph_ids_sha256']:
            raise ValueError('candidate row inventory differs')
        ar=active.finish();pr=pending.finish()
        if not pending_rows or not active_ids: raise ValueError('partition requires nonempty active and pending sets')
        for ref in (candidate_ref,candidate['appended_graphs'],candidate['source_screen'],candidate['row_signals'],candidate['effective_loss_census'],candidate['predecessor'],previous['family_map']):_checked(ref)
        if ProtectedWorlds.load().digest!=protected.digest: raise ValueError('protection changed during partition')
        ar['path']=str(output/'active.jsonl.zst');pr['path']=str(output/'pending.jsonl.zst')
        for ref,ids in ((ar,active_ids),(pr,[x['id'] for x in pending_rows])):
            ref['tokenizer_fingerprint']=screen['output']['tokenizer_fingerprint'];ref['graph_ids_sha256']=canonical_json_sha256_hex(sorted(ids))
        proof={'schema':SCHEMA,'candidate':candidate_ref,'original_screen':candidate['source_screen'],'original_census':candidate['effective_loss_census'],
            'original_row_signals':candidate['row_signals'],'parent_publication':parent_publication,'parent_snapshot':snap_ref,
            'predecessor':candidate['predecessor'],'protected_worlds_sha256':protected.digest,'limit':limit,
            'input_graph':screen['output'],'active_graph':ar,'pending_graph':pr,'pending_rows':pending_rows,
            'scope':'only unearned ordinary rows over the position bound are pending; original evidence remains immutable; no behavior qualification'}
        write_json_durable(stage/'pending-partition.json',proof)
        proofref={'path':str(output/'pending-partition.json'),'bytes':(stage/'pending-partition.json').stat().st_size,'sha256':sha256_file_hex(stage/'pending-partition.json')}
        new_screen=copy.deepcopy(screen);new_screen['output']=ar;new_screen['pending_partition']=proofref
        new_screen['accounting']['eligible_train_graphs']=len(active_ids);new_screen['accounting']['position_pending_graphs']=len(pending_rows)
        write_json_durable(stage/'screen.json',new_screen)
        sr={'path':str(output/'screen.json'),'bytes':(stage/'screen.json').stat().st_size,'sha256':sha256_file_hex(stage/'screen.json')}
        sig={'path':str(output/'row-signals.jsonl'),'bytes':(stage/'row-signals.jsonl').stat().st_size,'sha256':sha256_file_hex(stage/'row-signals.jsonl')}
        new_census={**census,'screen':sr,'row_signals':sig,'rows':len(active_ids),'counts':dict(counts),'category_mass':dict(mass),'source_stores':dict(stores),'unknown':dict(unknown),
                    'pending_partition':proofref,'status':'partitioned from authenticated completed census; source admission remains separate'}
        write_json_durable(stage/'census.json',new_census)
        os.replace(stage,output);fd=os.open(output.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return proof
    except BaseException:
        active.abort();pending.abort()
        if stage.exists():shutil.rmtree(stage)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate-receipt',type=Path,required=True);p.add_argument('--candidate-sha256',required=True)
    p.add_argument('--parent-publication',type=Path,required=True);p.add_argument('--parent-publication-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-context-tokens',type=int,default=DEFAULT_MAX_TOKENS);a=p.parse_args()
    print(json.dumps(partition({'path':str(a.candidate_receipt.resolve()),'sha256':a.candidate_sha256},
      {'path':str(a.parent_publication.resolve()),'sha256':a.parent_publication_sha256},a.output,limit=a.max_context_tokens),sort_keys=True))
if __name__=='__main__':main()
