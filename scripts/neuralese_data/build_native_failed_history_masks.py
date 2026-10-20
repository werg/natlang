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


def build(sources,*,tokenizer,out,capture_overrides=()):
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
    candidates=[];source_bindings=[];seen=set();counts={'rows':0,'affected_rows':0,'zeroed_tokens':0}
    for path in sources:
        path=Path(path).resolve(strict=True);source=binding(path)
        if source['sha256']in seen:raise ValueError('duplicate failed-history source leaf')
        seen.add(source['sha256']);source_bindings.append({**source,'split':'train'})
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
                if not all(teacher.get(key)for key in ('source_artifact','source_artifact_sha256',
                        'source_row_id','source_row_sha256')):
                    raise ValueError('affected ordinary graph lacks original verified teacher capture custody')
                evidence=None
            else:
                evidence={'kind':'declared-whole-trajectory',
                    'supervision_policy_sha256':digest(conversion.get('supervision_policy')or{}),
                    'source_text_sha256':conversion.get('source_text_sha256')}
                validate_source_evidence(graph,evidence)
            record=mask_record(graph,source_sha256=source['sha256'],source_evidence=evidence,
                zero_spans=spans,failure_evidence=failure_evidence)
            # Never retain full token/weight arrays across the streamed corpus.
            candidates.append({'graph':{'id':graph['id'],'conversion':conversion,
                **({'target_render_binding':target_binding}if target_binding is not None else{})},'record':record})
        if binding(path)!=source:raise ValueError('failed-history TRAIN source changed during preparation')
    # Resolve requested capture turns while streaming each raw file once. Neither
    # full graph arrays nor entire captured trajectories accumulate in memory.
    requested={}
    for candidate in candidates:
        if candidate['record']['source_evidence'] is not None:continue
        teacher=candidate['graph']['conversion']['teacher_source'];sha=teacher['source_artifact_sha256']
        entry=requested.setdefault(sha,{'path':overrides.get(sha,Path(teacher['source_artifact'])),'rows':{}})
        entry['rows'].setdefault(teacher['source_row_id'],[]).append(candidate)
    for sha,request in requested.items():
        path=request['path'];before=binding(path)
        if before['sha256']!=sha:raise ValueError('original capture artifact hash mismatch')
        seen_rows=set()
        with path.open('rb')as stream:
            for line in stream:
                raw=json.loads(line)
                if raw.get('id')in request['rows']:
                    if raw['id']in seen_rows:raise ValueError('ambiguous original capture row')
                    seen_rows.add(raw['id']);line_sha=sha256_hex(line.rstrip(b'\r\n'))
                    for candidate in request['rows'][raw['id']]:
                        evidence=capture_evidence(candidate['graph'],raw,line_sha,tokenizer=tokenizer)
                        validate_source_evidence(candidate['graph'],evidence,
                            target_render_binding=candidate['graph'].get('target_render_binding'))
                        candidate['record']['source_evidence']=evidence
        if seen_rows!=set(request['rows'])or binding(path)!=before:
            raise ValueError('missing or changed original capture rows')
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
    args=parser.parse_args(argv)
    from transformers import AutoTokenizer
    from natlang_neuralese.model.shared_state import provision_role_tokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    provision_role_tokenizer(tokenizer)
    reference,counts=build(args.source,tokenizer=tokenizer,out=args.out,capture_overrides=args.capture)
    print(json.dumps({'failed_history_mask':reference,**counts},sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
