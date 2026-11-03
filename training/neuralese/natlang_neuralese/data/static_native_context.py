"""Derive a training graph from an ordinary row without inventing teacher calls.

Producer chunks cover the original prefix. Their exported native states are
imported by a target-bearing consumer. The graph remains one source row; its
new edges describe a static training construction, not observed executions.

Given a tokenizer, each chunk exports a seeded summary span appended to the chunk (a deterministic mechanical
summary from data/operator_seeds.py: the spans selected by the consumer's question, exact identifiers, and the
structure outline), supervised as producer output with a bounded mass and recorded provenance (schema /2). Without
one the chunk exports its last ``state_tokens`` tokens, an arbitrary fragment (schema /1): it exists to replay and
verify the earlier rows that were built that way, not as a basis for new conversions.
"""
from __future__ import annotations
import copy, hashlib, math
from ..common.hashing import canonical_json_sha256_hex
from .ordinary_context_windows import DEFAULT_MAX_TOKENS, _ordinary, weighting_contract

SCHEMA='natlang.static-native-context-graph/1'
SEED_SCHEMA='natlang.static-native-context-graph/2'
# Per-chunk cap on the seed span (tokens) and the share of the parent's effective loss mass the seeds may add.
DEFAULT_SEED_TOKENS=256
DEFAULT_SEED_SHARE=.25
SEED_METHOD='natlang_neuralese.data.operator_seeds.chunk_summary_seed'

def _ids_sha256(ids):
    return hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()


def graphify(row, objective, *, max_tokens=DEFAULT_MAX_TOKENS, consumer_context=1024, state_tokens=16,
             tokenizer=None, question=None, seed_tokens=DEFAULT_SEED_TOKENS, seed_share=DEFAULT_SEED_SHARE):
    """Graphify an over-position ordinary row. With ``tokenizer`` every chunk exports a seeded summary of itself
    (at most ``seed_tokens`` tokens, ``state_tokens`` the least admitted); ``question`` overrides the consumer
    context text that selects the question-relevant spans. Without a tokenizer: the legacy tail-token export."""
    if tokenizer is not None:
        return _graphify_seeded(row,objective,max_tokens=max_tokens,consumer_context=consumer_context,
            state_tokens=state_tokens,tokenizer=tokenizer,question=question,seed_tokens=seed_tokens,seed_share=seed_share)
    if (type(max_tokens) is not int or not 4<=max_tokens<=DEFAULT_MAX_TOKENS or type(consumer_context) is not int
        or consumer_context<2 or type(state_tokens) is not int or state_tokens<1 or state_tokens>=max_tokens):
        raise ValueError('invalid static native context geometry')
    source=_ordinary(row);ids=source['token_ids'];raw=source['target_weights']
    if len(ids)<=max_tokens:raise ValueError('static context graph requires an over-position ordinary row')
    from ..train.own_output_supervision import graph_weights, _raw_target_span_signature
    weights,categories=graph_weights(row,objective);positive=[i for i,w in enumerate(weights) if w>0]
    if not positive:raise ValueError('static native context graph requires effective target loss')
    first=positive[0];start=max(0,first-consumer_context)
    if start<1 or first-start<2:raise ValueError('effective target distribution cannot use a target-bearing consumer')
    chunks=[(i,min(i+max_tokens,start)) for i in range(0,start,max_tokens)]
    imports=sum(min(state_tokens,b-a) for a,b in chunks)
    if len(ids)-start+imports>max_tokens:
        raise ValueError('target-bearing consumer and imported states exceed the position limit')
    signature,_=_raw_target_span_signature(ids,raw)
    suffix=(row.get('conversion') or {}).get('source_suffix_start')
    kind='ordinary' if (row.get('conversion') or {}).get('role_interpretation')=='ordinary-text' and suffix is not None else None
    contract=weighting_contract(objective,row_kind=kind,node_name=source['name'],target_sha256=signature)
    meta={'schema':SCHEMA,'parent_row_sha256':canonical_json_sha256_hex(row),'parent_node_name':source['name'],
      'parent_tokens':len(ids),'parent_target_sha256':signature,'parent_row_kind':kind,'weighting_sha256':contract,
      'max_tokens':max_tokens,'consumer_context':consumer_context,'state_tokens':state_tokens,
      'producer_spans':[list(x) for x in chunks],'consumer_start':start,
      'effective_weights':weights,'categories':categories,
      'raw_mask_scope':'source parent retains original raw mask; producer raw weights are zero because effective loss is zero',
      'graph_origin':'static derived training graph; edges are not observed teacher invocations'}
    digest=canonical_json_sha256_hex(meta);prefix=source['name']+':static-context:'+digest+':'
    nodes=[]
    for i,(a,b) in enumerate(chunks):
        n=b-a;take=min(state_tokens,n)
        nodes.append({'name':prefix+'producer:'+str(i),'token_ids':list(ids[a:b]),'target_weights':[0.]*n,
                      'producing':[[n-take,n]]})
    consumer_raw=list(raw[start:]);consumer_raw[0]=0.;consumer_raw[1]=0.
    nodes.append({'name':prefix+'consumer','token_ids':list(ids[start:]),'target_weights':consumer_raw,
                  'imports':[{'at':1,'source':node['name'],'start':len(node['token_ids'])-min(state_tokens,len(node['token_ids'])),
                              'stop':len(node['token_ids'])} for node in nodes]})
    result=copy.deepcopy(row);result['nodes']=nodes;result['conversion']['static_native_context']=meta
    verify_graphification(row,result,objective)
    return result,{'schema':'natlang.static-native-context-proof/1','parent_id':row['id'],
        'parent_row_sha256':meta['parent_row_sha256'],'graph_sha256':canonical_json_sha256_hex(result),
        'original_tokens':len(ids),'producer_count':len(chunks),'imported_state_tokens':imports,
        'effective_positive_targets':len(positive),'effective_mass':math.fsum(weights),
        'category_mass':{str(k):math.fsum(w for w,c in zip(weights,categories) if c==k) for k in sorted(set(categories))},
        'scope':'all original tokens retained; causal state imports and source-parent raw mask provenance; no capability claim'}


def _chunks(start, size):
    return [(i,min(i+size,start)) for i in range(0,start,size)]


def _graphify_seeded(row, objective, *, max_tokens, consumer_context, state_tokens, tokenizer, question, seed_tokens, seed_share):
    from .operator_seeds import chunk_summary_seed
    if (type(max_tokens) is not int or not 4<=max_tokens<=DEFAULT_MAX_TOKENS or type(consumer_context) is not int
        or consumer_context<2 or type(state_tokens) is not int or state_tokens<1 or type(seed_tokens) is not int
        or not state_tokens<=seed_tokens<=max_tokens//2 or not isinstance(seed_share,(int,float)) or not 0<seed_share<=1):
        raise ValueError('invalid static native context geometry')
    source=_ordinary(row);ids=source['token_ids'];raw=source['target_weights']
    if len(ids)<=max_tokens:raise ValueError('static context graph requires an over-position ordinary row')
    from ..train.own_output_supervision import graph_weights, _raw_target_span_signature, CATEGORIES
    weights,categories=graph_weights(row,objective);positive=[i for i,w in enumerate(weights) if w>0]
    if not positive:raise ValueError('static native context graph requires effective target loss')
    first=positive[0];start=max(0,first-consumer_context)
    if start<1 or first-start<2:raise ValueError('effective target distribution cannot use a target-bearing consumer')
    # A chunk plus its seed stays inside the position limit: chunks are cut `seed_tokens` short.
    chunks=_chunks(start,max_tokens-seed_tokens)
    room=(max_tokens-(len(ids)-start))//len(chunks)
    cap=min(seed_tokens,room)
    if cap<state_tokens:raise ValueError('target-bearing consumer and imported states exceed the position limit')
    # The consumer's own pre-target context is the question the exported states must serve, unless one is supplied.
    asked=question if question is not None else tokenizer.decode(ids[start:first],skip_special_tokens=True)
    seeds=[];seed_ids=[]
    for a,b in chunks:
        summary=chunk_summary_seed(tokenizer.decode(ids[a:b],skip_special_tokens=True),asked)
        encoded=list(tokenizer.encode(summary['text'],add_special_tokens=False))[:cap]
        if not encoded or any(type(t) is not int or t<0 for t in encoded):
            raise ValueError('chunk summary seed has no tokens')
        seed_ids.append(encoded)
        seeds.append({'tokens':len(encoded),'text_sha256':hashlib.sha256(summary['text'].encode()).hexdigest(),
                      'token_ids_sha256':_ids_sha256(encoded),'provenance':summary['provenance'],'parts':summary['parts']})
    total=sum(len(x) for x in seed_ids)
    mass=math.fsum(weights);seed_weight=min(mass/len(positive),seed_share*mass/total)
    signature,_=_raw_target_span_signature(ids,raw)
    suffix=(row.get('conversion') or {}).get('source_suffix_start')
    kind='ordinary' if (row.get('conversion') or {}).get('role_interpretation')=='ordinary-text' and suffix is not None else None
    contract=weighting_contract(objective,row_kind=kind,node_name=source['name'],target_sha256=signature)
    meta={'schema':SEED_SCHEMA,'parent_row_sha256':canonical_json_sha256_hex(row),'parent_node_name':source['name'],
      'parent_tokens':len(ids),'parent_target_sha256':signature,'parent_row_kind':kind,'weighting_sha256':contract,
      'max_tokens':max_tokens,'consumer_context':consumer_context,'seed_tokens':cap,'seed_share':seed_share,
      'seed_weight':seed_weight,'seed_method':SEED_METHOD,'seeds':seeds,
      'producer_spans':[list(x) for x in chunks],'consumer_start':start,
      'effective_weights':weights,'categories':categories,
      'raw_mask_scope':'source parent retains original raw mask; chunk tokens have zero producer weight; each chunk exports a seeded summary span with bounded effective weight',
      'graph_origin':'static derived training graph; edges are not observed teacher invocations'}
    digest=canonical_json_sha256_hex(meta);prefix=source['name']+':static-context:'+digest+':'
    nodes=[]
    for i,((a,b),seed) in enumerate(zip(chunks,seed_ids)):
        n=b-a;k=len(seed)
        nodes.append({'name':prefix+'producer:'+str(i),'token_ids':list(ids[a:b])+seed,
                      'target_weights':[0.]*n+[1.]*k,'producing':[[n,n+k]]})
    consumer_raw=list(raw[start:]);consumer_raw[0]=0.;consumer_raw[1]=0.
    nodes.append({'name':prefix+'consumer','token_ids':list(ids[start:]),'target_weights':consumer_raw,
                  'imports':[{'at':1,'source':node['name'],'start':node['producing'][0][0],'stop':node['producing'][0][1]}
                             for node in nodes]})
    result=copy.deepcopy(row);result['nodes']=nodes;result['conversion']['static_native_context']=meta
    verify_graphification(row,result,objective)
    return result,{'schema':'natlang.static-native-context-proof/2','parent_id':row['id'],
        'parent_row_sha256':meta['parent_row_sha256'],'graph_sha256':canonical_json_sha256_hex(result),
        'original_tokens':len(ids),'producer_count':len(chunks),'imported_state_tokens':total,
        'seed_provenance':dict.fromkeys(sorted({x['provenance'] for x in seeds}),len(seeds)),
        'seed_weight':seed_weight,'seed_mass':seed_weight*total,
        'effective_positive_targets':len(positive),'effective_mass':mass,
        'category_mass':{str(k):math.fsum(w for w,c in zip(weights,categories) if c==k) for k in sorted(set(categories))},
        'scope':'all original tokens retained; each chunk exports a mechanical summary seed under bounded producer CE; causal state imports and source-parent raw mask provenance; no capability claim'}


def _seeded_supervision(row,objective,meta):
    from ..train.own_output_supervision import CATEGORIES
    nodes=row.get('nodes') or [];limit=meta.get('max_tokens');size=meta.get('parent_tokens');start=meta.get('consumer_start')
    chunks=meta.get('producer_spans');values=meta.get('effective_weights');categories=meta.get('categories');seeds=meta.get('seeds')
    seed_weight=meta.get('seed_weight');cap=meta.get('seed_tokens')
    if (type(limit) is not int or not 4<=limit<=DEFAULT_MAX_TOKENS or type(size) is not int or size<=limit
      or type(start) is not int or not 1<=start<size or type(cap) is not int or not 1<=cap<=limit//2
      or type(seed_weight) not in (int,float) or not math.isfinite(seed_weight) or seed_weight<=0
      or not isinstance(chunks,list) or not chunks or len(nodes)!=len(chunks)+1 or not isinstance(seeds,list) or len(seeds)!=len(chunks)
      or meta.get('seed_method')!=SEED_METHOD
      or not isinstance(values,list) or len(values)!=size or not isinstance(categories,list) or len(categories)!=size
      or any(type(w) not in (int,float) or not math.isfinite(w) or w<0 for w in values)
      or any(type(c) is not int or not 0<=c<len(CATEGORIES) for c in categories)
      or any(values[:start+2]) or not any(values[start+2:]) or row.get('external_states') or row.get('decisions')
      or row.get('source_pairing') or row.get('source_chain')):raise ValueError('invalid static native context custody')
    if weighting_contract(objective,row_kind=meta.get('parent_row_kind'),node_name=meta.get('parent_node_name'),target_sha256=meta.get('parent_target_sha256'))!=meta.get('weighting_sha256'):
        raise ValueError('static native context belongs to different supervision arithmetic')
    digest=canonical_json_sha256_hex(meta);prefix=meta['parent_node_name']+':static-context:'+digest+':'
    cursor=0;expected_imports=[];flat=[];flat_categories=[];imported=0
    for i,(node,span,seed) in enumerate(zip(nodes[:-1],chunks,seeds,strict=True)):
        if not isinstance(span,list) or len(span)!=2 or any(type(x) is not int for x in span):raise ValueError('invalid source chunk span')
        a,b=span;n=b-a;k=seed.get('tokens') if isinstance(seed,dict) else None
        if type(k) is not int or not 1<=k<=cap or n+k>limit or not isinstance(seed.get('provenance'),str) or not seed['provenance']:
            raise ValueError('invalid static seed custody')
        if (a!=cursor or not 0<n or b>start or set(node)!={'name','token_ids','target_weights','producing'}
          or node['name']!=prefix+'producer:'+str(i) or len(node['token_ids'])!=n+k
          or node['target_weights']!=[0.]*n+[1.]*k or node['producing']!=[[n,n+k]]
          or _ids_sha256(node['token_ids'][n:])!=seed.get('token_ids_sha256')):raise ValueError('static source producer changed')
        expected_imports.append({'at':1,'source':node['name'],'start':n,'stop':n+k});cursor=b;imported+=k
        flat+=values[a:b]+[seed_weight]*k;flat_categories+=categories[a:b]+[1]*k
    consumer=nodes[-1];n=size-start
    if (cursor!=start or set(consumer)!={'name','token_ids','target_weights','imports'} or consumer['name']!=prefix+'consumer'
      or len(consumer['token_ids'])!=n or len(consumer['target_weights'])!=n or consumer['imports']!=expected_imports
      or n+imported>limit or consumer['target_weights'][:2]!=[0.,0.]
      or any(w>0 and raw<=0 for w,raw in zip(values[start:],consumer['target_weights']))):
        raise ValueError('static target consumer changed')
    return flat+values[start:],flat_categories+categories[start:],sum(seed['tokens'] for seed in seeds)


def static_supervision(row,objective):
    from ..train.own_output_supervision import CATEGORIES
    meta=(row.get('conversion') or {}).get('static_native_context') or {};nodes=row.get('nodes') or []
    if meta.get('schema')==SEED_SCHEMA:return _seeded_supervision(row,objective,meta)[:2]
    limit=meta.get('max_tokens');size=meta.get('parent_tokens');start=meta.get('consumer_start');state=meta.get('state_tokens')
    chunks=meta.get('producer_spans');values=meta.get('effective_weights');categories=meta.get('categories')
    if (meta.get('schema')!=SCHEMA or type(limit) is not int or not 4<=limit<=DEFAULT_MAX_TOKENS or type(size) is not int or size<=limit
      or type(start) is not int or not 1<=start<size or type(state) is not int or not 1<=state<limit
      or not isinstance(chunks,list) or len(nodes)!=len(chunks)+1 or not chunks
      or not isinstance(values,list) or len(values)!=size or not isinstance(categories,list) or len(categories)!=size
      or any(type(w) not in (int,float) or not math.isfinite(w) or w<0 for w in values)
      or any(type(c) is not int or not 0<=c<len(CATEGORIES) for c in categories)
      or any(values[:start+2]) or not any(values[start+2:]) or row.get('external_states') or row.get('decisions')
      or row.get('source_pairing') or row.get('source_chain')):raise ValueError('invalid static native context custody')
    if weighting_contract(objective,row_kind=meta.get('parent_row_kind'),node_name=meta.get('parent_node_name'),target_sha256=meta.get('parent_target_sha256'))!=meta.get('weighting_sha256'):
        raise ValueError('static native context belongs to different supervision arithmetic')
    digest=canonical_json_sha256_hex(meta);prefix=meta['parent_node_name']+':static-context:'+digest+':'
    cursor=0;expected_imports=[]
    for i,(node,span) in enumerate(zip(nodes[:-1],chunks,strict=True)):
        if not isinstance(span,list) or len(span)!=2 or any(type(x) is not int for x in span):raise ValueError('invalid source chunk span')
        a,b=span;n=b-a;take=min(state,n)
        if (a!=cursor or not 0<n<=limit or b>start or set(node)!={'name','token_ids','target_weights','producing'}
          or node['name']!=prefix+'producer:'+str(i) or len(node['token_ids'])!=n or node['target_weights']!=[0.]*n
          or node['producing']!=[[n-take,n]]):raise ValueError('static source producer changed')
        expected_imports.append({'at':1,'source':node['name'],'start':n-take,'stop':n});cursor=b
    consumer=nodes[-1];n=size-start
    if (cursor!=start or set(consumer)!={'name','token_ids','target_weights','imports'} or consumer['name']!=prefix+'consumer'
      or len(consumer['token_ids'])!=n or len(consumer['target_weights'])!=n or consumer['imports']!=expected_imports
      or n+sum(e['stop']-e['start'] for e in expected_imports)>limit or consumer['target_weights'][:2]!=[0.,0.]
      or any(w>0 and raw<=0 for w,raw in zip(values[start:],consumer['target_weights']))):
        raise ValueError('static target consumer changed')
    return values,categories

def verify_graphification(parent,child,objective):
    from ..train.own_output_supervision import graph_weights
    source=_ordinary(parent);values,categories=graph_weights(parent,objective)
    meta=child['conversion']['static_native_context'];seeded=meta.get('schema')==SEED_SCHEMA
    if seeded:
        projected,actual_categories,seed_total=_seeded_supervision(child,objective,meta)
    else:
        projected,actual_categories=static_supervision(child,objective)
    restored={**child,'nodes':parent['nodes'],'conversion':dict(child['conversion'])};restored['conversion'].pop('static_native_context')
    if restored!=parent or meta['parent_row_sha256']!=canonical_json_sha256_hex(parent):raise ValueError('static graph changed parent provenance')
    if seeded:
        # Seed tokens are exported producer output, not original tokens: strip them before comparing with the parent.
        original=[];kept=[];kept_categories=[];offset=0
        for node,(a,b) in zip(child['nodes'][:-1],meta['producer_spans']):
            n=b-a;k=len(node['token_ids'])-n
            original+=node['token_ids'][:n];kept+=projected[offset:offset+n];kept_categories+=actual_categories[offset:offset+n];offset+=n+k
        original+=child['nodes'][-1]['token_ids'];kept+=projected[offset:];kept_categories+=actual_categories[offset:]
        projected,actual_categories=kept,kept_categories
    else:
        original=[x for node in child['nodes'] for x in node['token_ids']]
    if original!=source['token_ids'] or projected!=values or actual_categories!=categories:
        raise ValueError('static graph changed original tokens or effective supervision')
    start=meta['consumer_start']
    if child['nodes'][-1]['target_weights'][2:]!=source['target_weights'][start+2:]:raise ValueError('static graph changed consumer raw mask')
    from .native_graph_schema import parse_graph
    example=parse_graph(child,tokenizer_fingerprint=parent['tokenizer_fingerprint'])
    if max(example.plan.own_positions,default=0)>=meta['max_tokens'] or max(example.plan.slot_positions,default=0)>=meta['max_tokens']:
        raise ValueError('static native graph exceeds positional bound')
    return True
