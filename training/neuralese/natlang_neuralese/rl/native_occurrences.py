"""Exact captured state-use occurrence binding, independent of artifact equality."""
from __future__ import annotations
from collections import Counter


def _source(event):
    inputs=event.get('inputs')
    if not isinstance(inputs,list):raise ValueError('native occurrence read lacks exact input edges')
    selected=[edge.get('node') for edge in inputs if isinstance(edge,dict) and edge.get('block')==event.get('block')]
    if len(selected)!=1 or not isinstance(selected[0],str) or not selected[0]:
        raise ValueError('native occurrence read requires one actual writer or external origin')
    return selected[0]


def capture_occurrences(name,capture,nodes,*,semantic=False):
    """Consume every graph event and captured state once; never pick latest ID.

    Identical same-origin uses may exchange event positions without changing the
    graph; their actual prompt token/import coordinates always come from capture.
    Different origins require explicit transported source-occurrence metadata.
    """
    writes=[event for event in nodes.values() if event.get('kind')=='block_write' and event.get('turn')==name]
    reads=[event for event in nodes.values() if event.get('kind')=='block_read' and event.get('turn')==name]
    for event in writes+reads:
        if not isinstance(event.get('block'),str) or not isinstance(event.get('node'),str):
            raise ValueError('native occurrence requires exact block and graph node identities')
    captured_writes=capture.get('writes',[])
    captured_reads=[segment for segment in capture.get('prompt',[]) if segment.get('kind')=='native_state']
    if len(writes)!=len(captured_writes) or len(reads)!=len(captured_reads):
        raise ValueError('whole-episode state capture differs from occurrence-matched actual graph inventories')
    bound_writes=[]
    for index,write in enumerate(captured_writes):
        candidates=[event for event in writes if event['block']==write.get('state_id')]
        if not candidates:raise ValueError('whole-episode state capture changed an actual writer artifact')
        event=candidates[0];writes.remove(event)
        bound_writes.append({'index':index,'id':event['block'],'node':event['node']})
    bound_reads=[]
    for index,segment in enumerate(captured_reads):
        observed=segment.get('id');source_id=segment.get('source_state_id',observed)
        if source_id!=observed and (not semantic or not isinstance(segment.get('semantic_adaptation'),dict)):
            raise ValueError('native decoded/reencoded inputs require explicit differentiable lowering')
        if 'stop_gradient' in segment and segment['stop_gradient'] is not True:
            raise ValueError('native input stop_gradient must be explicitly true or absent')
        source=segment.get('source_occurrence')
        if source is not None and (not isinstance(source,str) or not source):
            raise ValueError('native input needs its actual nonempty source occurrence')
        candidates=[event for event in reads if event['block']==source_id and
                    event.get('stop_gradient',False)==segment.get('stop_gradient',False) and
                    (source is None or _source(event)==source)]
        if not candidates or any('stop_gradient' in event and event['stop_gradient'] is not True for event in candidates):
            raise ValueError('whole-episode native input gradient use differs from occurrence-matched actual graph read')
        if len({_source(event) for event in candidates})!=1:
            raise ValueError('equal native artifact IDs require explicit source-occurrence provenance')
        event=candidates[0];reads.remove(event)
        bound_reads.append({'index':index,'id':observed,'source_state_id':source_id,'node':event['node'],
            'source':_source(event),**({'stop_gradient':True} if segment.get('stop_gradient') else {})})
    return bound_writes,bound_reads


def graph_binding(name,capture,nodes,*,semantic=False):
    writes,reads=capture_occurrences(name,capture,nodes,semantic=semantic)
    write_counts=Counter(item['id'] for item in writes)
    read_counts=Counter(item['source_state_id'] for item in reads)
    binding={'graph_turn':name,'written_nodes':{item['id']:item['node'] for item in writes if write_counts[item['id']]==1},
        'read_bindings':{item['source_state_id']:item['source'] for item in reads if read_counts[item['source_state_id']]==1 and item['source']!='external'}}
    if any(count>1 for count in write_counts.values()):binding['written_occurrences']=writes
    if any(count>1 for count in read_counts.values()) or any(item['id']!=item['source_state_id'] for item in reads):
        binding['read_occurrences']=reads
    return binding
