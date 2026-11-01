"""Bound positions of an ordinary row without manufacturing communication edges.

The row and quota remain one graph. Its independent nodes are contiguous text
windows, not invocations observed in the source. Full-row supervision is
projected before slicing so cut turns retain their actual category and mass.
"""
from __future__ import annotations
import copy
import math
from ..common.hashing import canonical_json_sha256_hex

SCHEMA='natlang.ordinary-context-windows/1'
DEFAULT_MAX_TOKENS=262144
DEFAULT_OVERLAP=1024


def weighting_contract(objective, *, row_kind, node_name, target_sha256):
    """Bind precisely the controls that change graph_weights, excluding family scale."""
    keys=('name','role_tokens','reasoning_share','reasoning_markers','mass')
    matched=[x for x in objective.get('decision_factors',[])
             if (x['row_kind'],x['node_name'],x['target_sha256'])==(row_kind,node_name,target_sha256)]
    if len(matched)>1:raise ValueError('ambiguous original target decision factor')
    return canonical_json_sha256_hex({**{key:objective.get(key) for key in keys},'matched_decision_factor':matched})


def _ordinary(row):
    from .native_graph_schema import indexed_graph_metadata
    example=indexed_graph_metadata(row,tokenizer_fingerprint=row.get('tokenizer_fingerprint'))
    nodes=row.get('nodes') or []
    if (len(nodes)!=1 or len(example.plan.invocations)!=1 or row.get('external_states')
            or row.get('decisions') or row.get('source_pairing') or row.get('source_chain')
            or any(nodes[0].get(key) for key in ('imports','producing','logical_producing'))
            or set(nodes[0])-{'name','token_ids','target_weights'}):
        raise ValueError('context windows require one ordinary node without imports, decisions or producer state')
    conversion=row.get('conversion') or {}
    if conversion.get('ordinary_context_windows') or conversion.get('static_native_context'):
        raise ValueError('an already transformed row cannot be transformed again')
    if conversion.get('role_interpretation') not in ('ordinary-text','whole-trajectory-chat-roles'):
        raise ValueError('context windows require an explicit ordinary text role contract')
    return nodes[0]


def window_ordinary_row(row, objective, *, max_tokens=DEFAULT_MAX_TOKENS, overlap=DEFAULT_OVERLAP):
    """Return the original row unchanged or one graph with independently bounded nodes."""
    if type(max_tokens) is not int or not 2<=max_tokens<=DEFAULT_MAX_TOKENS or type(overlap) is not int or not 1<=overlap<max_tokens:
        raise ValueError('window limit and overlap must be integers with 1 <= overlap < limit')
    node=_ordinary(row);ids=node['token_ids'];raw=node['target_weights'];size=len(ids)
    if size<=max_tokens:return copy.deepcopy(row),None
    from ..train.own_output_supervision import (graph_weights, _raw_target_span_signature,
        REPLY, REASONING, ASSISTANT_CONTEXT)
    effective,categories=graph_weights(row,objective)
    if len(effective)!=size or len(categories)!=size:raise ValueError('parent supervision is not token-aligned')
    targets=[i for i,value in enumerate(effective) if value>0]
    if not targets:raise ValueError('cannot make nonempty learning windows from a targetless row')
    suffix=(row.get('conversion') or {}).get('source_suffix_start')
    row_kind='ordinary' if (row.get('conversion') or {}).get('role_interpretation')=='ordinary-text' and suffix is not None else None
    signature,_=_raw_target_span_signature(ids,raw)
    contract=weighting_contract(objective,row_kind=row_kind,node_name=node['name'],target_sha256=signature)
    windows=[];cursor=0
    while cursor<size:
        start=max(0,cursor-overlap) if cursor else 0
        end=min(size,start+max_tokens)
        owned=[p for p in targets if cursor<=p<end]
        if not owned:raise ValueError('cannot cover source tokens without an empty-learning window')
        if end<size and not any(p>=end for p in targets):
            # Give the final window its own real target; never duplicate one.
            end=owned[-1]
            if end<=cursor or not any(cursor<=p<end for p in targets) or size-max(0,end-overlap)>max_tokens:
                raise ValueError('tail cannot be covered without an empty-learning window')
        if end<=cursor:raise ValueError('window progression made no progress')
        prefix=cursor-start
        weights=[0.]*prefix+list(raw[cursor:end])
        projected=[0.]*prefix+list(effective[cursor:end])
        cats=[ASSISTANT_CONTEXT if c in (REPLY,REASONING) else c for c in categories[start:cursor]]+list(categories[cursor:end])
        windows.append({'start':start,'owned_start':cursor,'end':end,
                        'token_ids':list(ids[start:end]),'target_weights':weights,
                        'effective_weights':projected,'categories':cats})
        cursor=end
    meta={'schema':SCHEMA,'parent_row_sha256':canonical_json_sha256_hex(row),'parent_node_name':node['name'],
          'parent_tokens':size,'parent_target_sha256':signature,'parent_row_kind':row_kind,
          'weighting_sha256':contract,'max_tokens':max_tokens,'overlap':overlap,
          'windows':[{k:w[k] for k in ('start','owned_start','end','effective_weights','categories')} for w in windows]}
    metadata_sha=canonical_json_sha256_hex(meta)
    result=copy.deepcopy(row)
    result['nodes']=[{'name':node['name']+':ordinary-window:'+str(i)+':'+metadata_sha,
                      'token_ids':w['token_ids'],'target_weights':w['target_weights']} for i,w in enumerate(windows)]
    result['conversion']['ordinary_context_windows']=meta
    verify_ordinary_windows(row,result,objective)
    return result, {'schema':'natlang.ordinary-context-window-proof/1','parent_id':row['id'],
        'parent_row_sha256':meta['parent_row_sha256'],'windowed_row_sha256':canonical_json_sha256_hex(result),
        'window_count':len(windows),'original_tokens':size,'window_tokens':sum(len(w['token_ids']) for w in windows),
        'raw_positive_targets':sum(value>0 for value in raw),'effective_positive_targets':len(targets),
        'raw_mass':math.fsum(raw),'effective_mass':math.fsum(effective),
        'category_mass':{str(kind):math.fsum(weight for weight,cat in zip(effective,categories) if cat==kind)
                         for kind in sorted(set(categories))},
        'scope':'one original graph and quota; real text windows with explicit overlapping context; no native communication edge'}


def window_supervision(row, objective):
    """Validate the exact projected metadata on every consumption."""
    from ..train.own_output_supervision import CATEGORIES
    meta=(row.get('conversion') or {}).get('ordinary_context_windows') or {}
    nodes=row.get('nodes') or [];windows=meta.get('windows')
    limit=meta.get('max_tokens');overlap=meta.get('overlap');length=meta.get('parent_tokens')
    if (meta.get('schema')!=SCHEMA or type(limit) is not int or limit<2 or type(overlap) is not int
            or not 1<=overlap<limit or type(length) is not int or length<=limit
            or limit>DEFAULT_MAX_TOKENS or not isinstance(windows,list) or len(windows)!=len(nodes) or len(nodes)<2
            or row.get('external_states') or row.get('decisions') or row.get('source_pairing') or row.get('source_chain')
            or not isinstance(meta.get('parent_row_sha256'),str) or len(meta['parent_row_sha256'])!=64):
        raise ValueError('invalid ordinary context-window custody')
    if weighting_contract(objective,row_kind=meta.get('parent_row_kind'),node_name=meta.get('parent_node_name'),
                          target_sha256=meta.get('parent_target_sha256'))!=meta.get('weighting_sha256'):
        raise ValueError('ordinary context windows belong to different supervision arithmetic')
    metadata_sha=canonical_json_sha256_hex(meta)
    values=[];categories=[];cursor=0;previous=None
    for index,(node,window) in enumerate(zip(nodes,windows,strict=True)):
        start=window.get('start');owned=window.get('owned_start');end=window.get('end')
        expected_name=meta['parent_node_name']+':ordinary-window:'+str(index)+':'+metadata_sha
        if (set(node)!={'name','token_ids','target_weights'} or node['name']!=expected_name
                or any(type(x) is not int for x in (start,owned,end)) or owned!=cursor
                or start!=(max(0,cursor-overlap) if cursor else 0) or not start<=owned<end<=length
                or end-start>limit or len(node['token_ids'])!=end-start or len(node['target_weights'])!=end-start):
            raise ValueError('ordinary context-window span or node binding changed')
        prefix=owned-start;weights=window.get('effective_weights');cats=window.get('categories')
        if (not isinstance(weights,list) or not isinstance(cats,list) or len(weights)!=end-start or len(cats)!=end-start
                or any(type(w) not in (int,float) or not math.isfinite(w) or w<0 for w in weights)
                or any(type(c) is not int or not 0<=c<len(CATEGORIES) for c in cats)
                or any(node['target_weights'][i]!=0 or weights[i]!=0 for i in range(prefix))
                or not any(w>0 for w in weights)
                or any(w>0 and raw<=0 for w,raw in zip(weights,node['target_weights']))):
            raise ValueError('ordinary context-window target mask or effective supervision changed')
        if previous is not None and prefix and node['token_ids'][:prefix]!=previous[-prefix:]:
            raise ValueError('ordinary context-window overlap token bytes differ')
        previous=node['token_ids'];cursor=end;values.extend(weights);categories.extend(cats)
    if cursor!=length:raise ValueError('ordinary context windows omit the source tail')
    return values,categories


def verify_ordinary_windows(parent, child, objective):
    """Independently compare all source tokens, raw masks, effective targets and roles."""
    from ..train.own_output_supervision import graph_weights, REPLY, REASONING, ASSISTANT_CONTEXT
    source=_ordinary(parent);effective,categories=graph_weights(parent,objective)
    projected,projected_categories=window_supervision(child,objective)
    meta=child['conversion']['ordinary_context_windows'];cursor=0;out=0
    restored={**child,'nodes':parent['nodes'],'conversion':dict(child['conversion'])}
    restored['conversion'].pop('ordinary_context_windows')
    if meta['parent_row_sha256']!=canonical_json_sha256_hex(parent) or restored!=parent:
        raise ValueError('ordinary context windows changed their parent identity or provenance')
    for node,window in zip(child['nodes'],meta['windows'],strict=True):
        start,owned,end=window['start'],window['owned_start'],window['end'];prefix=owned-start;n=end-start
        if (node['token_ids']!=source['token_ids'][start:end]
                or node['target_weights'][prefix:]!=source['target_weights'][owned:end]
                or projected[out+prefix:out+n]!=effective[owned:end]
                or projected_categories[out:out+prefix]!=[ASSISTANT_CONTEXT if c in (REPLY,REASONING) else c for c in categories[start:owned]]
                or projected_categories[out+prefix:out+n]!=categories[owned:end]):
            raise ValueError('ordinary context windows differ from exact source token/mask/role projection')
        cursor=end;out+=n
    if cursor!=len(source['token_ids']):raise ValueError('ordinary context source coverage is incomplete')
    return True
