"""Compile authenticated provider episodes to training graphs without model capture.

This path produces token/source topology only. It never constructs a native state,
claims a student capture, or supplies sampled-policy probabilities.
"""
from __future__ import annotations
import base64, hashlib, json
from .provider_native_graph import _replace_refs
from .provider_teacher_source import load_verified_luna_teacher_turn, _jsonl_by_id, _read_bound
from .native_graph_schema import parse_graph


def _sha(data): return hashlib.sha256(data).hexdigest()
def _canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()

def compile_verified_episode(audit_path, record_ids, renderer, *, tokenizer_fingerprint):
    """Use the actual chat renderer and audited turns; no model, KV, or sampled policy.

    The renderer supplies the production tokenizer, roles, prompt policy, and the
    same forced-target formatting implementation as the executing engine.
    """
    audit=json.loads(__import__('pathlib').Path(audit_path).read_bytes())
    ref=audit['python_corpus_conversion']; data=_read_bound(ref['path'],ref['sha256'])
    loaded=[]
    for record_id in record_ids:
        raw=_jsonl_by_id(data,record_id); opening=(raw.get('native_source_conversion') or {}).get('source_messages')
        if not isinstance(opening,list):raise ValueError('audited turn lacks exact source opening')
        item=load_verified_luna_teacher_turn(audit_path,record_id,replay_messages=opening,
            replay_tools=raw.get('tools') or [],allow_context_only=True)
        loaded.append((item,opening,raw.get('tools') or []))
    row,example=_compile_loaded_episode(loaded,renderer,tokenizer_fingerprint=tokenizer_fingerprint)
    row['static_training_provenance']['audit_path']=str(__import__('pathlib').Path(audit_path).resolve())
    return row,example


def _compile_loaded_episode(loaded,renderer,*,tokenizer_fingerprint):
    from .text_corpus import tokenizer_fingerprint as actual_tokenizer_fingerprint
    if actual_tokenizer_fingerprint(renderer.tokenizer)!=tokenizer_fingerprint:raise ValueError('static compiler tokenizer fingerprint differs')
    if not loaded:raise ValueError('empty provider episode')
    receipts=[item['source_receipt'] for item,_,_ in loaded]; first=receipts[0]
    if first.get('split')!='train' or any(item.get('audit_sha256')!=loaded[0][0].get('audit_sha256') for item,_,_ in loaded):raise ValueError('provider episode audit or split differs')
    common=('source_row_id','source_row_sha256','source_file_sha256','trace_sha256','plan_sha256','runtime_receipt_sha256','source_groups','split','actor')
    if any(any(r.get(k)!=first.get(k) for k in common) for r in receipts):raise ValueError('provider episode custody differs')
    ix=[r.get('source_turn_index') for r in receipts]
    if any(type(x) is not int for x in ix) or ix!=list(range(ix[0],ix[0]+len(ix))):raise ValueError('provider episode is not contiguous')
    bodies={}
    for item,_,_ in loaded:
        for body in item['typed_text_bodies']:
            ident=body['block_id']; stable=(body['body'],body['body_sha256'],body['type'])
            if _sha(body['body'].encode())!=body['body_sha256'] or ident in bodies and bodies[ident]!=stable:raise ValueError('provider text body identity differs')
            bodies[ident]=stable
    # These strings are renderer-local placeholders only, never actual state IDs.
    placeholders={key:'nzs1_'+base64.b32encode(hashlib.sha256(key.encode()).digest()).decode().lower().rstrip('=') for key in bodies}
    actions=[]; target_spans=[]
    for (item,opening,tools),receipt in zip(loaded,receipts):
        bindings={x['block_id']:{'native_block_id':placeholders[x['block_id']],'source_occurrence':'static:'+x['block_id']} for x in item['typed_text_bodies']}
        counts={x['block_id']:x['context_occurrences'] for x in item['typed_text_bodies']}
        rewritten=_replace_refs(opening,bindings,counts)
        rendered=renderer.render_request(rewritten,tools)
        own=[]; imported=[]
        reverse={v:k for k,v in placeholders.items()}
        for part in rendered.segments:
            if isinstance(part,str): own.extend(renderer._template_tokens(part,rendered.escape_nonce))
            else:
                if type(part) is not int or part<0 or part>=len(rendered.blocks):raise ValueError('invalid rendered block')
                key=reverse[rendered.blocks[part]]; own.append(renderer.role_tokens.consume_open); imported.append((len(own),key)); own.append(renderer.role_tokens.consume_close)
        from .provider_static_renderer import render_forced_message_plan
        target=render_forced_message_plan(renderer,item['record']['target'],tools,opening=rewritten,spans=item['native_text_spans'])
        terminal=next((i for i,token in enumerate(target) if token in renderer.stop_ids),None)
        if terminal is not None:target=target[:terminal]
        spans=[]; cursor=0
        for x in item['native_text_spans']:
            bodyids=renderer.tokenizer(x['text'],add_special_tokens=False,split_special_tokens=True)['input_ids']
            needle=[renderer.role_tokens.produce_open,*bodyids,renderer.role_tokens.produce_close]
            hits=[i for i in range(cursor,len(target)-len(needle)+1) if target[i:i+len(needle)]==needle]
            if len(hits)!=1:raise ValueError('producer target span is not uniquely token aligned')
            pos=hits[0]; cursor=pos+len(needle); spans.append((x,len(own)+pos+1,len(own)+pos+1+len(bodyids)))
        actions.append({'own':own,'target':target,'imports':imported});target_spans.append(spans)
    inline={}; by_action={}
    for i,((item,_,_),receipt) in enumerate(zip(loaded,receipts)):
        for span,lo,hi in target_spans[i]:
            matches=[w for r in receipts for w in r.get('typed_text_writes',[]) if w.get('writer_call_id')==next((x.get('source_turn_id') for x in receipt.get('target_text_spans',[]) if x.get('marker')==span.get('marker')),None) and w.get('body_sha256')==_sha(span['text'].encode())]
            stable={(w['block_id'],w['writer_node']) for w in matches}
            if not stable:continue
            if len(stable)!=1:raise ValueError('inline writer identity ambiguous')
            key,name=next(iter(stable)); detail=(i,name,lo,hi)
            if key in inline and inline[key]!=detail:raise ValueError('inline writer identity conflicts')
            inline[key]=detail;by_action.setdefault(i,[]).append((name,lo,hi))
    nodes=[]; writer={}
    for key,(body,digest,value_type) in bodies.items():
        if key in inline:continue
        from ..serve.chat import escape_specials
        ids=renderer._template_tokens(escape_specials(body,renderer.specials,'static'),'static')
        if not ids:raise ValueError('empty encoder payload')
        name='teacher-text-encoder:'+_sha((first['source_row_id']+key).encode())[:24];writer[key]=(name,1,1+len(ids))
        nodes.append({'name':name,'token_ids':[renderer.role_tokens.produce_open,*ids,renderer.role_tokens.produce_close],'target_weights':[0.0]*(len(ids)+2),'producing':[[1,1+len(ids)]]})
    for key,(_,name,lo,hi) in inline.items():writer[key]=(name,lo,hi)
    for i,(action,receipt) in enumerate(zip(actions,receipts)):
        details=by_action.get(i,[]); names={x[0] for x in details}
        if len(names)>1:raise ValueError('action has multiple writer names')
        name=next(iter(names)) if names else 'luna-action:'+_sha((first['source_row_id']+':'+str(receipt['source_turn_index'])).encode())[:24]
        own,target=action['own'],action['target']; supervise=receipt.get('target_supervision')
        if type(supervise) is not bool or not target:raise ValueError('target supervision is malformed')
        nodes.append({'name':name,'token_ids':own+target,'target_weights':[0.0]*len(own)+[float(supervise)]*len(target),'imports':[{'at':at,'source':writer[key][0],'start':writer[key][1],'stop':writer[key][2]} for at,key in action['imports']],**({'producing':[[lo,hi] for _,lo,hi in details]} if details else {})})
    row={'schema':'natlang.neuralese-native-graph/1','id':'luna-native-episode:'+first['source_row_id'],'split':'train','source_groups':first['source_groups'],'tokenizer_fingerprint':tokenizer_fingerprint,'nodes':nodes,'teacher_source_receipts':receipts,'teacher_text_encoder_sources':[{'teacher_block_id':key,'body_sha256':digest,'type':value_type,'source_kind':'inline-supervised-target' if key in inline else 'authenticated-host-text-encoder','producer':writer[key][0],'producer_span':[writer[key][1],writer[key][2]]} for key,(body,digest,value_type) in bodies.items()],'static_training_provenance':{'schema':'natlang.provider-static-native-graph/1','audit_sha256':loaded[0][0]['audit_sha256'],'scope':'Authenticated teacher tokens and graph topology; no model capture, KV, sampled policy, or probabilities.'}}
    return row,parse_graph(row,tokenizer_fingerprint=tokenizer_fingerprint)
