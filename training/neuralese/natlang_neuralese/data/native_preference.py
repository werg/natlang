"""Joint offline preference graphs with exact native serving target suffixes."""
from __future__ import annotations

import copy

from ..common.hashing import canonical_json_sha256_hex
from .native_graph_schema import SCHEMA,parse_graph
from .text_corpus import native_gold_packet

KIND='native-offline-pairwise'
SCHEMA_PREFERENCE='natlang.offline-native-preference/1'


def pair_graph(record,tokenizer,*,tokenizer_fingerprint):
    """Render divergent assistant turns; all invocations share ONE graph forward.

    Ordinary CE weights are zero deliberately. The declared preference objective
    alone owns its immutable suffix masks; rejected actions can never become
    accidental positive SFT, including on a manually selected pair graph file.
    """
    if (record.get('schema')!=SCHEMA_PREFERENCE or record.get('required_objective')!=KIND
            or record.get('offline') is not True or record.get('native_on_policy') is not False):
        raise ValueError('offline native preference record/schema/provenance is required')
    if record.get('split') not in ('train','test') or not record.get('source_groups'):
        raise ValueError('offline pair needs exact protected split and source groups')
    alternatives=record.get('alternatives') or {}
    if set(alternatives)!= {'chosen','rejected'}:
        raise ValueError('offline pair must retain exactly chosen and rejected histories')
    messages={name:copy.deepcopy(alternatives[name]['messages']) for name in alternatives}
    common=0
    for left,right in zip(messages['chosen'],messages['rejected']):
        if left!=right:break
        common+=1
    if common!=record.get('common_message_count'):
        raise ValueError('offline pair common history differs from its immutable declaration')
    nodes=[];suffixes={};counts={}
    for name in ('chosen','rejected'):
        alternative=alternatives[name]
        if alternative.get('split')!=record['split'] or set(alternative.get('split_groups') or [])!=set(record['source_groups']):
            raise ValueError('offline pair alternative changed its protected split/groups')
        for message in messages[name]:
            if 'reasoning' in message:message['reasoning_content']=message.pop('reasoning')
        counts[name]=0
        for index,target in enumerate(messages[name]):
            if index<common or target.get('role')!='assistant':continue
            _,ids,boundary=native_gold_packet(tokenizer,messages[name][:index],target,alternative.get('tools'))
            if not 0<boundary<len(ids):raise ValueError('offline pair target suffix lacks a predecessor or output')
            node=name+':'+str(index)
            nodes.append({'name':node,'token_ids':ids,'target_weights':[0.]*len(ids)})
            suffixes[node]={'alternative':name,'source_message_index':index,'suffix_start':boundary,
                            'suffix_stop':len(ids),'token_ids_sha256':canonical_json_sha256_hex(ids)}
            counts[name]+=len(ids)-boundary
        if counts[name]==0:raise ValueError('offline pair lacks divergent assistant tokens in an alternative')
    source_record_sha256=canonical_json_sha256_hex(record)
    identity='offline-preference:'+canonical_json_sha256_hex({'source_record_sha256':source_record_sha256,
        'suffixes':suffixes,'token_counts':counts})
    row={'schema':SCHEMA,'id':identity,'split':record['split'],
         'source_groups':record['source_groups'],'tokenizer_fingerprint':tokenizer_fingerprint,
         'nodes':nodes,'required_objective':KIND,
         'conversion':{'kind':'offline-preference-serving-continuations','offline':True,
             'source_preference':record['source'],'source_record_sha256':source_record_sha256,
             'common_message_count':common,'native_on_policy':False,'ordinary_ce_targets':0}}
    example=parse_graph(row,tokenizer_fingerprint=tokenizer_fingerprint)
    from ..rl.native_replay import graph_identity
    spec={'graph_sha256':canonical_json_sha256_hex(graph_identity(example)),
          'split':record['split'],'suffixes':suffixes,'token_counts':counts,
          'source_preference':record['source'],'source_record_sha256':source_record_sha256}
    return row,spec
