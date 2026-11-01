"""Derive a training graph from an ordinary row without inventing teacher calls.

Producer chunks cover the original prefix. Their final native states are
imported by a target-bearing consumer. The graph remains one source row; its
new edges describe a static training construction, not observed executions.
"""
from __future__ import annotations
import copy, math
from ..common.hashing import canonical_json_sha256_hex
from .ordinary_context_windows import DEFAULT_MAX_TOKENS, _ordinary, weighting_contract

SCHEMA='natlang.static-native-context-graph/1'

def graphify(row, objective, *, max_tokens=DEFAULT_MAX_TOKENS, consumer_context=1024, state_tokens=16):
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

def static_supervision(row,objective):
    from ..train.own_output_supervision import CATEGORIES
    meta=(row.get('conversion') or {}).get('static_native_context') or {};nodes=row.get('nodes') or []
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
    projected,actual_categories=static_supervision(child,objective);meta=child['conversion']['static_native_context']
    restored={**child,'nodes':parent['nodes'],'conversion':dict(child['conversion'])};restored['conversion'].pop('static_native_context')
    if restored!=parent or meta['parent_row_sha256']!=canonical_json_sha256_hex(parent):raise ValueError('static graph changed parent provenance')
    if [x for node in child['nodes'] for x in node['token_ids']]!=source['token_ids'] or projected!=values or actual_categories!=categories:
        raise ValueError('static graph changed original tokens or effective supervision')
    start=meta['consumer_start']
    if child['nodes'][-1]['target_weights'][2:]!=source['target_weights'][start+2:]:raise ValueError('static graph changed consumer raw mask')
    from .native_graph_schema import parse_graph
    example=parse_graph(child,tokenizer_fingerprint=parent['tokenizer_fingerprint'])
    if max(example.plan.own_positions,default=0)>=meta['max_tokens'] or max(example.plan.slot_positions,default=0)>=meta['max_tokens']:
        raise ValueError('static native graph exceeds positional bound')
    return True
