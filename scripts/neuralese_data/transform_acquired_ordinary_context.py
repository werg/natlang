#!/usr/bin/env python3
"""Correct a screened, unearned acquired append while retaining graph identities.

The input candidate and its predecessor are exact receipt-bound artifacts. Every
row is rechecked against the current complete protected-world union and the
predecessor ID inventory. Long ordinary rows become a static derived native
graph; existing native communication graphs are never fragmented.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from natlang_neuralese.common.custody import ProtectedWorlds, world_identities
from natlang_neuralese.data.ordinary_context_windows import window_ordinary_row,DEFAULT_MAX_TOKENS,DEFAULT_OVERLAP
from natlang_neuralese.train.own_output_supervision import policy_objective
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter


def _checked(ref):
    if not isinstance(ref,dict) or not isinstance(ref.get('path'),str) or not isinstance(ref.get('sha256'),str):
        raise ValueError('acquired correction needs an exact input binding')
    path=Path(ref['path']).resolve(strict=True);before=path.stat()
    digest=sha256_file_hex(path);after=path.stat()
    if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
            or digest!=ref['sha256'] or 'bytes' in ref and ref['bytes']!=before.st_size):
        raise ValueError('acquired correction input differs from its binding')
    return path


def _read(ref):return json.loads(_checked(ref).read_text())


def _protected_case(row):
    conversion=row.get('conversion') or {};acquired=conversion.get('acquired_session') or {}
    ids=list(conversion.get('source_record_ids') or [])+list(acquired.get('original_target_ids') or [])
    ids.extend(x for x in (conversion.get('source_row_id'),acquired.get('original_source_row')) if isinstance(x,str) and x)
    return {'id':row['id'],'split':row['split'],'source_ids':ids,'source_groups':row['source_groups'],
            'split_groups':row['source_groups']}

def verify_context_correction(screen, predecessor, census=None):
    """Replay every changed row and objective from the exact unearned parent source."""
    proof=_read(screen.get('static_context_correction'));candidate=_read(proof['candidate'])
    original=_read(proof['screen']);old_census=_read(proof['prior_objective_census'])
    if (proof.get('schema')!='natlang.acquired-ordinary-context-correction/1'
        or proof.get('predecessor')!=candidate.get('predecessor') or _checked(proof['predecessor'])!=Path(predecessor).resolve(strict=True)
        or candidate.get('schema')!='natlang.acquired-streaming-source-successor/1' or candidate.get('training_admission') is not False
        or candidate.get('appended_graphs')!=proof.get('input') or candidate.get('source_screen')!=proof.get('screen')
        or candidate.get('effective_loss_census')!=proof.get('prior_objective_census')
        or candidate.get('row_signals')!=proof.get('prior_row_signals') or candidate.get('ce_policy')!=proof.get('objective')
        or proof.get('output')!=screen.get('output') or proof.get('input',{}).get('sha256')!=original.get('output',{}).get('sha256')):
        raise ValueError('static context correction does not bind exact original source')
    expected=copy.deepcopy(original);expected['output']=proof['output'];expected['static_context_correction']=screen['static_context_correction']
    if expected!=screen:raise ValueError('static context correction changed unrelated screen custody')
    from natlang_neuralese.rl.native_direct_sft_recovery import _authenticated_refresh_publication
    publication=_authenticated_refresh_publication(proof['parent_publication'])['publication']
    if ((publication.get('rolling_endpoint_custody_proof') or {}).get('successor')!=proof.get('parent_sft_snapshot')
        or publication.get('checkpoint')!=proof.get('parent_checkpoint')):raise ValueError('static context correction parent publication differs')
    saved=_read(proof['parent_sft_snapshot']);frames=saved.get('sft_frames')
    if (saved.get('schema')!='natlang.rolling-sft-parent-custody-snapshot/1' or saved.get('optimizer_present') is not True
        or saved.get('rng_present') is not True or (saved.get('checkpoint') or {}).get('sha256')!=(publication.get('checkpoint') or {}).get('sha256')
        or not isinstance(frames,dict) or not frames):raise ValueError('static context correction lacks full SFT endpoint')
    objective=policy_objective(_read(proof['objective']))
    prior_objective=policy_objective(_read(old_census['ce_policy']))
    if ({k:v for k,v in objective.items() if k!='family_balance'}!={k:v for k,v in prior_objective.items() if k!='family_balance'}
        or old_census.get('screen',{}).get('sha256')!=proof['screen']['sha256']
        or old_census.get('row_signals')!=proof['prior_row_signals']):raise ValueError('static context original objective differs')
    if census is not None and (census.get('static_context_correction')!=screen['static_context_correction']
        or census.get('ce_policy')!=old_census.get('ce_policy') or census.get('objective_sha256')!=old_census.get('objective_sha256')
        or census.get('builder')!=old_census.get('builder')):raise ValueError('static context census objective custody differs')
    evidence={p['parent_id']:p for p in proof.get('proofs',[])};seen=set();count=0
    if len(evidence)!=proof.get('corrected_rows'):raise ValueError('static context correction proof repeats rows')
    from contextlib import ExitStack
    from scripts.neuralese_data.audit_acquired_learning_signal import row_signal
    with ExitStack() as stack:
        parents=stack.enter_context(open_jsonl_text(_checked(proof['input'])))
        children=iter(stack.enter_context(open_jsonl_text(_checked(proof['output']))))
        prior=iter(stack.enter_context(open_jsonl_text(_checked(proof['prior_row_signals']))))
        active=iter(stack.enter_context(open_jsonl_text(_checked(census['row_signals'])))) if census is not None else None
        for line in parents:
            if not line.strip():continue
            parent=json.loads(line);child_line=next((x for x in children if x.strip()),None);signal_line=next((x for x in prior if x.strip()),None)
            if child_line is None or signal_line is None:raise ValueError('static correction omitted an input row')
            child=json.loads(child_line);old_signal=json.loads(signal_line);ident=parent['id'];count+=1
            if ident in seen or old_signal.get('id')!=ident or old_signal.get('graph_sha256')!=canonical_json_sha256_hex(parent):
                raise ValueError('static correction original row census differs')
            seen.add(ident)
            if any(ident in (f.get('source_row_bindings') or {}) or ident in (f.get('pending_source_rows') or {})
                or (f.get('exposure_counts') or {}).get(ident,0)!=0 for f in frames.values()):raise ValueError('static correction source was already in an SFT frame')
            if ident in evidence:
                from natlang_neuralese.data.static_native_context import verify_graphification
                from natlang_neuralese.data.ordinary_context_windows import verify_ordinary_windows
                if proof.get('representation')=='static-native':verify_graphification(parent,child,objective)
                elif proof.get('representation')=='ordinary-windows':verify_ordinary_windows(parent,child,objective)
                else:raise ValueError('unrecognized static context representation')
                item=evidence[ident]
                if item.get('parent_row_sha256')!=canonical_json_sha256_hex(parent) or item.get('graph_sha256',item.get('windowed_row_sha256'))!=canonical_json_sha256_hex(child):
                    raise ValueError('static context correction graph proof differs')
                expected_signal=row_signal(child,prior_objective)
            else:
                if child!=parent:raise ValueError('unlisted static correction changed a row')
                expected_signal=old_signal
            if active is not None:
                signal=next((x for x in active if x.strip()),None)
                if signal is None or json.loads(signal)!=expected_signal:raise ValueError('static correction census row differs')
        if any(next((x for x in stream if x.strip()),None) is not None for stream in (children,prior) if stream is not None):
            raise ValueError('static correction added rows')
        if active is not None and next((x for x in active if x.strip()),None) is not None:raise ValueError('static correction added census rows')
    if count!=candidate['new_rows'] or count!=proof['output']['rows'] or not set(evidence)<=seen:
        raise ValueError('static correction row inventory differs')
    return original,evidence


def transform(candidate_ref, output, *, parent_publication, max_tokens=DEFAULT_MAX_TOKENS, overlap=DEFAULT_OVERLAP,
              representation='static-native'):
    if representation not in ('static-native','ordinary-windows'):raise ValueError('unknown context representation')
    from natlang_neuralese.rl.native_direct_sft_recovery import _authenticated_refresh_publication
    authenticated=_authenticated_refresh_publication(parent_publication)
    publication=authenticated['publication']
    snapshot_ref=(publication.get('rolling_endpoint_custody_proof') or {}).get('successor')
    if not snapshot_ref:raise ValueError('unearned correction needs an authenticated saved SFT endpoint snapshot')
    saved=_read(snapshot_ref);frames=saved.get('sft_frames')
    if (saved.get('schema')!='natlang.rolling-sft-parent-custody-snapshot/1'
            or saved.get('optimizer_present') is not True or saved.get('rng_present') is not True
            or (saved.get('checkpoint') or {}).get('sha256')!=(publication.get('checkpoint') or {}).get('sha256')
            or not isinstance(frames,dict) or not frames):
        raise ValueError('unearned correction lacks exact published checkpoint custody')
    candidate_path=_checked(candidate_ref);candidate=json.loads(candidate_path.read_text())
    if candidate.get('schema')!='natlang.acquired-streaming-source-successor/1' or candidate.get('training_admission') is not False:
        raise ValueError('context correction requires an explicitly unadmitted acquired candidate')
    input_ref=candidate['appended_graphs'];source=_checked(input_ref)
    screen=_read(candidate['source_screen']);screen_output=screen.get('output') or {}
    if (screen_output.get('sha256')!=input_ref['sha256'] or Path(screen_output.get('path','')).resolve()!=source
            or screen_output.get('rows')!=candidate['new_rows']):
        raise ValueError('screened input differs from the exact candidate append')
    predecessor=_read(candidate['predecessor']);family_map=_read(predecessor['family_map'])
    if (family_map.get('schema')!='natlang.logical-family-map/1'
            or family_map.get('source_sha256')!=(predecessor.get('source') or {}).get('sha256')
            or len(family_map.get('rows') or [])!=candidate['predecessor_rows']):
        raise ValueError('predecessor duplicate inventory is not bound to the exact source')
    prior_ids={row['id'] for row in family_map['rows']}
    if len(prior_ids)!=candidate['predecessor_rows']:raise ValueError('predecessor identity inventory contains duplicates')
    policy=_read(candidate['ce_policy']);objective=policy_objective(policy)
    if objective.get('name')!='own-outputs/1':raise ValueError('context correction requires its exact own-output objective')
    census=_read(candidate['effective_loss_census'])
    from scripts.neuralese_data import audit_acquired_learning_signal as audited_census
    if _checked(census.get('builder'))!=Path(audited_census.__file__).resolve(strict=True):
        raise ValueError('candidate census does not bind the actual audited learning-signal program')
    census_objective=policy_objective(_read(census['ce_policy']))
    if (census.get('schema')!='natlang.acquired-effective-learning-census/1'
            or census.get('rows')!=candidate['new_rows'] or census.get('training_admission') is not False
            or census.get('screen',{}).get('sha256')!=candidate['source_screen']['sha256']
            or census.get('row_signals',{}).get('sha256')!=candidate['row_signals']['sha256']
            or census.get('objective_sha256')!=canonical_json_sha256_hex(census_objective)
            or {k:v for k,v in census_objective.items() if k!='family_balance'} !=
               {k:v for k,v in objective.items() if k!='family_balance'}):
        raise ValueError('candidate objective census differs from the exact screened source and weighting')
    signal_path=_checked(candidate['row_signals'])
    protected=ProtectedWorlds.load();protection_digest=protected.digest
    output=Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError('correction needs a fresh output in an existing directory')
    if output.parent.stat().st_dev!=source.stat().st_dev:
        raise ValueError('bulk correction output must remain on the acquired external data filesystem')
    if shutil.disk_usage(output.parent).free < 40*1024**3+3*source.stat().st_size:
        raise ValueError('context correction would breach the 40 GiB filesystem floor')
    stage=output.with_name('.'+output.name+'.staging-'+str(os.getpid()))
    stage.mkdir();writer=GraphWriter(stage/'train/native-graphs.jsonl.zst')
    seen=set();proofs=[];before_rows=0;counts=Counter();mass=Counter();stores=Counter();unknown=Counter()
    try:
        with open_jsonl_text(source) as rows, open_jsonl_text(signal_path) as signals, (stage/'row-signals.jsonl').open('x') as signal_out:
            signal_iter=(json.loads(line) for line in signals if line.strip())
            for line in rows:
                if not line.strip():continue
                row=json.loads(line);before_rows+=1
                signal=next(signal_iter,None)
                if (not isinstance(signal,dict) or signal.get('id')!=row.get('id')
                        or signal.get('graph_sha256')!=canonical_json_sha256_hex(row)
                        or signal.get('forward_tokens')!=sum(len(node.get('token_ids') or ()) for node in row.get('nodes') or [])
                        or signal.get('raw_positive_tokens')!=sum(w>0 for node in row.get('nodes') or [] for w in node.get('target_weights') or [])):
                    raise ValueError('candidate row differs from its measured objective census')
                ident=row.get('id')
                if not isinstance(ident,str) or not ident or ident in seen or ident in prior_ids:
                    raise ValueError('candidate contains duplicate or predecessor graph identity')
                if any(ident in (frame.get('source_row_bindings') or {})
                       or ident in (frame.get('pending_source_rows') or {})
                       or (frame.get('exposure_counts') or {}).get(ident,0)!=0 for frame in frames.values()):
                    raise ValueError('candidate row already belongs to a saved SFT frame; use typed source correction')
                if row.get('split')!='train':raise ValueError('candidate correction only accepts train rows')
                if row.get('tokenizer_fingerprint')!=screen_output.get('tokenizer_fingerprint'):
                    raise ValueError('candidate tokenizer differs from original screen')
                if world_identities(_protected_case(row),evidence_items=False)&protected.identities:
                    raise ValueError('candidate intersects the current complete protected-world union')
                seen.add(ident)
                if any(len(node.get('token_ids') or ())>max_tokens for node in row.get('nodes') or []):
                    if representation=='static-native':
                        from natlang_neuralese.data.static_native_context import graphify
                        corrected,proof=graphify(row,objective,max_tokens=max_tokens,consumer_context=max(2,overlap),
                                                state_tokens=min(16,max_tokens-1))
                    else:
                        corrected,proof=window_ordinary_row(row,objective,max_tokens=max_tokens,overlap=overlap)
                    from natlang_neuralese.train.own_output_supervision import CATEGORIES
                    if (proof['effective_positive_targets']!=signal.get('effective_positive_tokens')
                            or any(abs(proof['category_mass'].get(str(i),0.)-signal.get('category_mass',{}).get(name,0.)) >
                                   1e-8*max(1.,abs(signal.get('category_mass',{}).get(name,0.)))
                                   for i,name in enumerate(CATEGORIES))):
                        raise ValueError('windowed row changed the measured effective target census')
                    signal=audited_census.row_signal(corrected,census_objective)
                    proofs.append(proof);writer.write(corrected)
                else:writer.write(row)
                signal_out.write(json.dumps(signal,sort_keys=True)+'\n')
                for key in ('forward_tokens','raw_positive_tokens','effective_positive_tokens'):counts[key]+=signal[key]
                counts['zero_effective_rows']+=signal.get('eligible') is not True
                counts['source_label_diagnosis_rows']+=signal.get('source_label_diagnosis') is True
                counts['oracle_verified_rows']+=signal.get('oracle_verified') is True
                stores[signal['source_store'] or '<unknown>']+=1
                if not signal['source_store']:unknown['source_store']+=1
                mass.update(signal['category_mass'])
            if next(signal_iter,None) is not None:raise ValueError('objective census contains extra rows')
            signal_out.flush();os.fsync(signal_out.fileno())
        if before_rows!=candidate['new_rows'] or canonical_json_sha256_hex(sorted(seen))!=screen_output['graph_ids_sha256']:
            raise ValueError('candidate rows differ from the original screened census')
        result=writer.finish()
        if ProtectedWorlds.load().digest!=protection_digest:raise ValueError('protected-world union changed during correction')
        _checked(candidate_ref);_checked(input_ref);_checked(candidate['ce_policy']);_checked(candidate['source_screen'])
        _checked(candidate['predecessor']);_checked(predecessor['family_map'])
        _checked(candidate['effective_loss_census']);_checked(candidate['row_signals']);_checked(census['ce_policy'])
        result['path']=str(output/'train/native-graphs.jsonl.zst')
        result['tokenizer_fingerprint']=screen_output['tokenizer_fingerprint']
        result['graph_ids_sha256']=screen_output['graph_ids_sha256']
        receipt={'schema':'natlang.acquired-ordinary-context-correction/1',
            'candidate':candidate_ref,'input':input_ref,'screen':candidate['source_screen'],
            'parent_publication':parent_publication,'parent_checkpoint':publication['checkpoint'],
            'parent_sft_snapshot':snapshot_ref,'candidate_rows_absent_from_saved_sft_frames':True,
            'predecessor':candidate['predecessor'],'predecessor_family_map':predecessor['family_map'],
            'objective':candidate['ce_policy'],'prior_objective_census':candidate['effective_loss_census'],
            'prior_row_signals':candidate['row_signals'],'protected_worlds_digest':protection_digest,
            'input_rows':before_rows,'output':result,'corrected_rows':len(proofs),'proofs':proofs,
            'max_tokens':max_tokens,'overlap':overlap,'representation':representation,'row_ids_preserved':True,
            'scope':'static derived training representation; original evidence and effective supervision checked per graph; no observed teacher execution, capability, exposure or admission claim'}
        write_json_durable(stage/'receipt.json',receipt)
        receipt_ref={'path':str(output/'receipt.json'),'bytes':(stage/'receipt.json').stat().st_size,
                     'sha256':sha256_file_hex(stage/'receipt.json')}
        revised_screen=copy.deepcopy(screen);revised_screen['output']=result;revised_screen['static_context_correction']=receipt_ref
        write_json_durable(stage/'screen.json',revised_screen)
        screen_ref={'path':str(output/'screen.json'),'bytes':(stage/'screen.json').stat().st_size,
                    'sha256':sha256_file_hex(stage/'screen.json')}
        signal_ref={'path':str(output/'row-signals.jsonl'),'bytes':(stage/'row-signals.jsonl').stat().st_size,
                    'sha256':sha256_file_hex(stage/'row-signals.jsonl')}
        revised_census={**census,'screen':screen_ref,'row_signals':signal_ref,'rows':before_rows,
            'counts':dict(counts),'category_mass':dict(mass),'source_stores':dict(stores),'unknown':dict(unknown),
            'static_context_correction':receipt_ref,'status':'derived from original exact source and objective; admission remains separate'}
        write_json_durable(stage/'census.json',revised_census)
        os.replace(stage,output)
        fd=os.open(output.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return receipt
    except BaseException:
        writer.abort()
        # Only the newly-created incomplete staging directory is disposable.
        if stage.exists():shutil.rmtree(stage)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-receipt',type=Path,required=True)
    parser.add_argument('--candidate-sha256',required=True)
    parser.add_argument('--parent-publication',type=Path,required=True)
    parser.add_argument('--parent-publication-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-context-tokens',type=int,default=DEFAULT_MAX_TOKENS)
    parser.add_argument('--context-overlap',type=int,default=DEFAULT_OVERLAP)
    parser.add_argument('--representation',choices=('static-native','ordinary-windows'),default='static-native')
    args=parser.parse_args(argv)
    receipt=transform({'path':str(args.candidate_receipt.resolve()),'sha256':args.candidate_sha256},args.output,
                      parent_publication={'path':str(args.parent_publication.resolve()),'sha256':args.parent_publication_sha256},
                      max_tokens=args.max_context_tokens,overlap=args.context_overlap,representation=args.representation)
    print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__=='__main__':main()
