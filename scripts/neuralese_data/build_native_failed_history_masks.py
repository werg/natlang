#!/usr/bin/env python3
"""Prepare exact failed-history CE corrections for explicitly selected TRAIN leaves.

No source IDs, raw rows, indexes or completed exposure are replaced. The small
manifest and tokenizer-only snapshot are consumed through the existing CE policy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable,write_bytes_durable
from natlang_neuralese.data.native_sources import _jsonl
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
from natlang_neuralese.train.native_failed_history import (
    SCHEMA,binding,mask_law,mask_record,validate_source_evidence,derive_context_mask,load_manifest)


def capture_evidence(graph,raw,line_sha256,*,tokenizer=None):
    """Original collector custody and the shared no-embedded-boundary verifier."""
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import reviewed_teacher_context_guard
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
    teacher=graph['conversion']['teacher_source']
    if raw.get('id')!=teacher['source_row_id']or line_sha256!=teacher['source_row_sha256']:
        raise ValueError('failed-history original capture row identity mismatch')
    index=teacher['teacher_turn_index']
    if type(index)is not int or index<0:raise ValueError('failed-history capture turn index invalid')
    turn=raw['trajectory'][index];assistant=turn['assistant']
    target={'role':'assistant','content':assistant.get('content',''),
            'tool_calls':turn['model_response']['raw_calls']}
    if assistant.get('reasoning'):target['reasoning_content']=assistant['reasoning']
    target_binding=graph.get('target_render_binding')
    if target_binding is None:
        guard=reviewed_teacher_context_guard(turn,turn['context'],target,turn['tools_offered'])
    else:
        from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import captured_text_emulation_context
        from natlang_neuralese.data.native_graph_conversion import unambiguous_chat_input_guard
        resolved,tools,_,omission=captured_text_emulation_context(turn,turn['context'],turn['tools_offered'])
        if omission is not None:raise ValueError('failed-history original resolved prefix is unavailable')
        guard={'schema':'natlang.failed-history-prefix-input-guard/1',
            'policy':'unambiguous-rendered-context-with-exact-captured-target',
            'prefix':unambiguous_chat_input_guard(resolved,None,tools),
            'target_render_binding_sha256':digest(target_binding)}
    evidence={'kind':'verified-teacher-capture','teacher_source_sha256':digest(teacher),
              'structured_input_guard':guard}
    if target_binding is not None:
        from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import render_teacher_packet,teacher_action_graph_id
        from natlang_neuralese.serve.chat import bind_history_reasoning
        profiles=[]
        for profile in (None,'keep','last_turn_only'):
            for mask in ('preserve','mask-runtime-failures'):
                modern='text:'+teacher_action_graph_id(teacher['source_row_id'],teacher['source_row_sha256'],index,profile,mask)
                legacy=f"text:{teacher['source_row_id']}:action:{index:04d}"
                if profile is not None:legacy+=':history-reasoning:'+profile
                if mask!='preserve':legacy+=':assistant-history:'+mask
                if graph['id']in(modern,legacy):profiles.append(profile)
        if tokenizer is None or len(profiles)!=1:
            raise ValueError('failed-history target render needs exact tokenizer and source-bound history profile')
        profile=profiles[0];bind_history_reasoning(tokenizer,policy=profile,allow_transition=True)
        packet,omission=render_teacher_packet(tokenizer,turn['context'],target,turn['tools_offered'],turn)
        if omission is not None:raise ValueError('failed-history original captured target render is unavailable')
        (text,ids,boundary),_=packet
        actual={'token_ids_sha256':digest(ids),'source_text_sha256':sha256_hex(text.encode()),
                'source_suffix_start':boundary}
        if actual!=target_binding:raise ValueError('failed-history original captured rendering differs from raw graph')
        evidence['captured_target_render']={'schema':'natlang.failed-history-captured-target-render/1',
            'binding':actual,'teacher_source_sha256':digest(teacher),'history_reasoning_profile':profile}
    return evidence


def native_record_evidence(candidates,receipts,tokenizer,overrides):
    """Authenticate existing native-view decisions through their original renderer."""
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
    from natlang_neuralese.serve.chat import bind_history_reasoning
    from natlang_neuralese.data.text_corpus import authenticated_crisp_context_messages,target_with_teacher_reasoning,native_gold_packet
    from natlang_neuralese.data.trajectory_text import crisp_messages,handover_notes
    from natlang_neuralese.data.native_graph_conversion import unambiguous_chat_input_guard
    requested=[item for item in candidates if item.get('custody')=='native']
    for receipt_path in receipts:
        ref=binding(receipt_path);receipt=json.loads(Path(receipt_path).read_text())
        corpus=receipt.get('corpus_id');sources=receipt.get('sources')
        if (receipt.get('schema')!='natlang.native-turn-render/1'or not isinstance(corpus,str)or not corpus
                or not isinstance(sources,dict)or not sources):
            raise ValueError('failed-history native render receipt is missing exact original sources')
        template=getattr(tokenizer,'chat_template',None)
        if not isinstance(template,str):raise ValueError('failed-history native renderer needs its exact template')
        options=[item for item in(receipt.get('tokenizers')or{}).values()
            if(item.get('files')or{}).get('chat_template.jinja')==sha256_hex(template.encode())]
        if len(options)!=1 or(options[0].get('history_reasoning')or{}).get('policy')not in('keep','last_turn_only'):
            raise ValueError('failed-history native receipt does not bind this tokenizer template/history policy')
        policy=options[0]['history_reasoning']['policy'];bind_history_reasoning(tokenizer,policy=policy,allow_transition=True)
        wanted={}
        for candidate in requested:
            graph=candidate['graph'];ids=graph['conversion'].get('source_record_ids')
            if not isinstance(ids,list)or len(ids)!=1:raise ValueError('native failed-history decision needs one exact source record')
            modes=[mode for mode in('full','program-header')if graph['id']==f'text:native-view-qualified:{mode}:{corpus}:{ids[0]}']
            if modes:wanted.setdefault(ids[0],[]).append((candidate,modes[0]))
        if not wanted:continue
        seen=set()
        for original,declared in sources.items():
            path=overrides.get(declared.get('sha256'),Path(original))
            if not path.is_absolute():path=Path(__file__).resolve().parents[2]/path
            before=binding(path)
            if(before['sha256'],before['bytes'])!=(declared.get('sha256'),declared.get('bytes')):
                raise ValueError('failed-history original native JSON file differs from renderer receipt')
            with path.open('rb')as stream:
                for line in stream:
                    record=json.loads(line);identifier=record.get('id')
                    if identifier not in wanted:continue
                    if identifier in seen:raise ValueError('failed-history native record identity is ambiguous')
                    seen.add(identifier)
                    if(record.get('version')!='natlang.teacher_training_turn.native/1'or record.get('split')!='train'
                            or(record.get('training_admission')or{}).get('approved')is not True):
                        raise ValueError('failed-history native source is not an approved TRAIN decision')
                    messages,context,augmentations=authenticated_crisp_context_messages(record,{},{})
                    if context or augmentations:raise ValueError('failed-history native rendering needs its complete context custody')
                    target=crisp_messages([target_with_teacher_reasoning(record)],{},handover_notes(record))[0]
                    guard=unambiguous_chat_input_guard(messages,target,record.get('tools')or[],capture=record)
                    for candidate,mode in wanted[identifier]:
                        graph=candidate['graph'];expected=graph['native_render_binding']
                        if sorted(set(record.get('source_groups')or[]))!=graph['source_groups']:
                            raise ValueError('failed-history native source groups differ from original graph')
                        text,ids,boundary=native_gold_packet(tokenizer,messages,target,record.get('tools')or[],prompt_rendering=mode)
                        actual={'token_ids_sha256':digest(ids),'source_text_sha256':sha256_hex(text.encode()),'source_suffix_start':boundary}
                        if actual!=expected:raise ValueError('failed-history native gold rendering differs from the original graph')
                        evidence={'kind':'verified-native-record-render','structured_input_guard':guard,
                            'native_render_receipt':ref,'native_record':{'source':before,'id':identifier,
                                'line_sha256':sha256_hex(line.rstrip(b'\r\n')),'record_sha256':digest(record),
                                'split':'train','source_groups':graph['source_groups'],'training_admission':record['training_admission']},
                            'native_render':{'corpus_id':corpus,'prompt_rendering':mode,'history_reasoning_policy':policy,'binding':actual}}
                        validate_source_evidence(graph,evidence,target_render_binding=expected)
                        if candidate['record']['source_evidence']is not None:raise ValueError('ambiguous native render authority')
                        candidate['record']['source_evidence']=evidence
            if binding(path)!=before:raise ValueError('original native JSON changed during verification')
        if seen!=set(wanted):raise ValueError('failed-history original approved native source records are missing')
        if binding(receipt_path)!=ref:raise ValueError('native render receipt changed during preparation')
    if any(item['record']['source_evidence']is None for item in requested):
        raise ValueError('failed-history native row lacks an exact original renderer receipt')


def build(sources,*,tokenizer,out,capture_overrides=(),native_render_receipts=(),work=None,prior_derivation_modules=()):
    """Bounded by explicit selected leaves; one pass per original capture file."""
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
    out=Path(out)
    if out.exists():raise ValueError('failed-history output must be a new immutable directory')
    overrides={}
    for path in capture_overrides:
        item=binding(path)
        if item['sha256']in overrides:raise ValueError('ambiguous capture relocation')
        overrides[item['sha256']]=Path(path)
    fingerprint=tokenizer_fingerprint(tokenizer)
    work=Path(work)if work is not None else None
    if work is not None:work.mkdir(parents=True,exist_ok=True)
    from scripts.neuralese_data.source_policy_identity import scoped_python_policy
    from natlang_neuralese.train import native_failed_history as mask_helper
    mask_module=mask_helper.__file__
    callable_law=scoped_python_policy(mask_module,('derive_context_mask','mask_record'),allow_imports=True)['policy_sha256']
    derivation={'version':2,'mask_law_sha256':mask_law(),'callable_law_sha256':callable_law,
        'tokenizer_fingerprint':fingerprint,'native_custody_enabled':bool(native_render_receipts)}
    prior_derivations=[]
    for path in prior_derivation_modules:
        original=binding(path)
        if scoped_python_policy(path,('derive_context_mask','mask_record'),allow_imports=True)['policy_sha256']!=callable_law:
            raise ValueError('prior candidate derivation changes the actual mask callable law')
        prior_derivations.append({'version':1,'mask_law_sha256':derivation['mask_law_sha256'],
            'mask_module_sha256':original['sha256'],'tokenizer_fingerprint':fingerprint,
            'native_custody_enabled':bool(native_render_receipts)})
    candidates=[];source_bindings=[];seen=set();counts={'rows':0,'affected_rows':0,'zeroed_tokens':0}
    for path in sources:
        path=Path(path).resolve(strict=True);source=binding(path)
        if source['sha256']in seen:raise ValueError('duplicate failed-history source leaf')
        seen.add(source['sha256']);source_bindings.append({**source,'split':'train'})
        key=digest({'source_sha256':source['sha256'],**derivation})
        cache=work/(key+'.json')if work is not None else None
        selected_derivation=derivation;selected_key=key
        if cache is not None and not cache.exists():
            for old_derivation in prior_derivations:
                old_key=digest({'source_sha256':source['sha256'],**old_derivation})
                old_cache=work/(old_key+'.json')
                if old_cache.exists():
                    cache=old_cache;selected_derivation=old_derivation;selected_key=old_key;break
        if cache is not None and cache.exists():
            saved=json.loads(cache.read_text());payload=saved.get('payload')or{}
            if (saved.get('schema')!='natlang.failed-history-candidate-work/1'
                    or saved.get('admitted')is not False or saved.get('payload_sha256')!=digest(payload)
                    or payload.get('key')!=selected_key or payload.get('derivation')!=selected_derivation
                    or payload.get('source_sha256')!=source['sha256']or payload.get('source_bytes')!=source['bytes']):
                raise ValueError('failed-history NON-ADMITTED candidate work binding mismatch')
            restored=payload.get('candidates')
            if (not isinstance(restored,list)or type(payload.get('rows'))is not int
                    or payload['rows']<len(restored)or any(
                        item['record']['source_sha256']!=source['sha256']
                        or item['record']['tokenizer_fingerprint']!=fingerprint
                        or item['graph']['id']!=item['record']['id']for item in restored)):
                raise ValueError('failed-history candidate work row identity mismatch')
            candidates.extend(restored);counts['rows']+=payload['rows']
            continue
        first_candidate=len(candidates);first_row=counts['rows']
        for graph in _jsonl(path):
            counts['rows']+=1
            if graph.get('schema')!='natlang.neuralese-native-graph/1' or graph.get('split')!='train':
                raise ValueError('failed-history producer requires explicitly selected native TRAIN graph leaves')
            conversion=graph.get('conversion')or{}
            if conversion.get('role_interpretation')not in ('ordinary-text','whole-trajectory-chat-roles'):
                continue
            if graph.get('tokenizer_fingerprint')!=fingerprint:
                raise ValueError('failed-history decoder differs from actual source row tokenizer')
            spans,failure_evidence,target_binding=derive_context_mask(graph,tokenizer)
            if not spans:continue
            if conversion.get('role_interpretation')=='ordinary-text':
                teacher=conversion.get('teacher_source')or{}
                native=not teacher and bool(native_render_receipts)
                if not native and not all(teacher.get(key)for key in ('source_artifact','source_artifact_sha256',
                        'source_row_id','source_row_sha256')):
                    raise ValueError('affected ordinary graph lacks original verified teacher capture custody')
                if native and target_binding is not None:
                    raise ValueError('failed-history native target needs unambiguous original rendered roles')
                evidence=None
            else:
                evidence={'kind':'declared-whole-trajectory',
                    'supervision_policy_sha256':digest(conversion.get('supervision_policy')or{}),
                    'source_text_sha256':conversion.get('source_text_sha256')}
                validate_source_evidence(graph,evidence)
            record=mask_record(graph,source_sha256=source['sha256'],source_evidence=evidence,
                zero_spans=spans,failure_evidence=failure_evidence)
            # Never retain full token/weight arrays across the streamed corpus.
            native=conversion.get('role_interpretation')=='ordinary-text'and not conversion.get('teacher_source')
            candidates.append({'graph':{'id':graph['id'],'conversion':conversion,'source_groups':graph['source_groups'],
                **({'target_render_binding':target_binding}if target_binding is not None else{}),
                **({'native_render_binding':{'token_ids_sha256':digest(graph['nodes'][0]['token_ids']),
                    'source_text_sha256':conversion['source_text_sha256'],'source_suffix_start':conversion['source_suffix_start']}}if native else{})},
                'record':record,'custody':'native'if native else'teacher'})
        if binding(path)!=source:raise ValueError('failed-history TRAIN source changed during preparation')
        if cache is not None:
            payload={'key':key,'derivation':derivation,'source_sha256':source['sha256'],
                'source_bytes':source['bytes'],'rows':counts['rows']-first_row,
                'candidates':candidates[first_candidate:]}
            write_json_durable(cache,{'schema':'natlang.failed-history-candidate-work/1',
                'admitted':False,'payload_sha256':digest(payload),'payload':payload})
    # Resolve requested capture turns while streaming each raw file once. Neither
    # full graph arrays nor entire captured trajectories accumulate in memory.
    requested={}
    for candidate in candidates:
        if candidate['record']['source_evidence'] is not None or candidate['custody']=='native':continue
        teacher=candidate['graph']['conversion']['teacher_source'];sha=teacher['source_artifact_sha256']
        entry=requested.setdefault(sha,{'path':overrides.get(sha,Path(teacher['source_artifact'])),'rows':{}})
        entry['rows'].setdefault(teacher['source_row_id'],[]).append(candidate)
    custody_errors=[]
    for sha,request in requested.items():
        path=request['path'];seen_rows=set()
        try:
            before=binding(path)
            if before['sha256']!=sha:raise ValueError('original capture artifact hash mismatch')
            with path.open('rb')as stream:
                for line in stream:
                    raw=json.loads(line)
                    if raw.get('id')not in request['rows']:continue
                    if raw['id']in seen_rows:raise ValueError('ambiguous original capture row')
                    seen_rows.add(raw['id']);line_sha=sha256_hex(line.rstrip(b'\r\n'))
                    for candidate in request['rows'][raw['id']]:
                        try:
                            evidence=capture_evidence(candidate['graph'],raw,line_sha,tokenizer=tokenizer)
                            validate_source_evidence(candidate['graph'],evidence,
                                target_render_binding=candidate['graph'].get('target_render_binding'))
                            candidate['record']['source_evidence']=evidence
                        except (ValueError,KeyError,IndexError,TypeError)as error:
                            custody_errors.append({'source_sha256':candidate['record']['source_sha256'],
                                'raw_row_sha256':candidate['record']['raw_row_sha256'],
                                'id':candidate['graph']['id'],'capture_sha256':sha,
                                'capture_row_id':raw['id'],'error_type':type(error).__name__,'error':str(error)})
            if seen_rows!=set(request['rows'])or binding(path)!=before:
                raise ValueError('missing or changed original capture rows')
        except (OSError,ValueError,KeyError,TypeError)as error:
            custody_errors.append({'capture_sha256':sha,'path':str(path),
                'missing_row_ids':sorted(set(request['rows'])-seen_rows),
                'error_type':type(error).__name__,'error':str(error)})
    if work is not None:
        write_json_durable(work/'custody-report.json',{'schema':'natlang.failed-history-custody-census/1',
            'admitted':False,'derivation':derivation,'sources':source_bindings,
            'candidate_rows':len(candidates),'requested_captures':len(requested),'errors':custody_errors})
    if custody_errors:
        raise ValueError(f'failed-history original capture custody incomplete: {len(custody_errors)} errors; no manifest admitted')
    native_record_evidence(candidates,native_render_receipts,tokenizer,overrides)
    records=[candidate['record']for candidate in candidates]
    counts['zeroed_tokens']=sum(stop-start for record in records for start,stop in record['zero_spans'])
    if not records:raise ValueError('selected leaves contain no positively supervised runtime-failed assistant history')
    out.mkdir(parents=True);decoder=out/'tokenizer';tokenizer.save_pretrained(decoder)
    from transformers import AutoTokenizer
    restored=AutoTokenizer.from_pretrained(decoder,local_files_only=True,trust_remote_code=False)
    if tokenizer_fingerprint(tokenizer)!=fingerprint or tokenizer_fingerprint(restored)!=fingerprint:raise ValueError('saved decoder changes tokenizer identity')
    files=[]
    for path in sorted(decoder.iterdir()):
        if not path.is_file():raise ValueError('tokenizer snapshot must contain explicit regular files')
        write_bytes_durable(path,path.read_bytes())
        files.append({**binding(path),'path':str(path.relative_to(out))})
    counts['affected_rows']=len(records)
    value={'schema':SCHEMA,'mask_law_sha256':mask_law(),'sources':source_bindings,
        'tokenizer':{'directory':'tokenizer','fingerprint':fingerprint,'files':files},
        'rows':records,'counts':counts,
        'scope':'Explicit selected TRAIN leaves; same raw source IDs/row bindings/exposure; no corpus admission.'}
    path=out/'manifest.json';write_json_durable(path,value);load_manifest(binding(path))
    return binding(path),counts


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,action='append',required=True,
        help='exact admitted TRAIN native graph leaf, not a broad directory census')
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--capture',type=Path,action='append',default=[],
        help='optional locally relocated original capture, matched by exact original SHA')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--work',type=Path,help='durable NON-ADMITTED candidate preparation cache and custody census')
    parser.add_argument('--prior-derivation-module',type=Path,action='append',default=[],
        help='exact preceding mask module for a checked equal-callable-law candidate-work migration')
    parser.add_argument('--native-render-receipt',type=Path,action='append',default=[],
        help='existing native-turn renderer receipt binding original approved native records and source view')
    args=parser.parse_args(argv)
    from transformers import AutoTokenizer
    from natlang_neuralese.model.shared_state import provision_role_tokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    provision_role_tokenizer(tokenizer)
    reference,counts=build(args.source,tokenizer=tokenizer,out=args.out,capture_overrides=args.capture,
        native_render_receipts=args.native_render_receipt,work=args.work,prior_derivation_modules=args.prior_derivation_module)
    print(json.dumps({'failed_history_mask':reference,**counts},sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
