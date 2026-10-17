"""Explicit historical graph continuation for captured JSON tool-history imports."""
from __future__ import annotations
import copy
import json
from collections import Counter
from .episode_graph import rollout_turn_names, validate_rollout_graph


def _tool_imports(messages):
    found=[]
    def visit(value):
        if isinstance(value,list):
            if value and all(isinstance(p,dict)and p.get('type')in('text','neuralese','native_state')for p in value):
                found.extend(p for p in value if p.get('type')=='native_state')
            else:
                for child in value:visit(child)
        elif isinstance(value,dict):
            for child in value.values():visit(child)
    for message in messages:
        for call in message.get('tool_calls')or[]:
            value=call.get('function',call).get('arguments')
            if isinstance(value,str):value=json.loads(value)if value.strip()else{}
            visit(value)
    return found


def repair_tool_history_reads(rollout):
    """Return a new strict-valid graph, adding only independently captured reads.

    Every missing occurrence must also exist in exact recorded JSON arguments and
    name a known writer of the same artifact. Existing graph events are immutable.
    """
    result=copy.deepcopy(rollout);events=result.get('execution_graph')or[]
    nodes={event['node']:event for event in events}
    original_order={event['node']:index for index,event in enumerate(events)}
    if len(nodes)!=len(events):raise ValueError('duplicate original graph occurrence')
    additions=[]
    key=lambda part:(part.get('id'),part.get('source_occurrence'),part.get('stop_gradient',False))
    for name,turn in zip(rollout_turn_names(result.get('turns')or[]),result.get('turns')or[]):
        captured=[p for p in(turn.get('sampling')or{}).get('prompt',[])if p.get('kind')=='native_state']
        reads=[event for event in events if event.get('kind')=='block_read'and event.get('turn')==name]
        if len(reads)==len(captured):continue
        actual=Counter(key(part)for part in _tool_imports(turn.get('messages')or[]))
        existing=Counter()
        for event in reads:
            sources=[edge.get('node')for edge in event.get('inputs',[])if edge.get('block')==event.get('block')]
            if len(sources)!=1:raise ValueError('existing read source is ambiguous')
            existing[(event.get('block'),sources[0],event.get('stop_gradient',False))]+=1
        required=Counter(key(part)for part in captured)
        if existing-required:raise ValueError('original reads exceed actual captured imports')
        missing=required-existing
        if missing-actual:raise ValueError('missing imports are not exact recorded JSON argument parts')
        for index,part in enumerate(captured):
            use=key(part)
            if not missing[use]:continue
            writer=nodes.get(part.get('source_occurrence'))
            if (not writer or writer.get('kind')!='block_write'or writer.get('block')!=part.get('id')
                    or writer.get('native_identity')!=part.get('native_identity')
                    or name not in original_order or original_order[writer['node']]>=original_order[name]):
                raise ValueError('captured import lacks its exact known native writer identity')
            if part.get('stop_gradient',False)is not False and part.get('stop_gradient')is not True:
                raise ValueError('invalid captured gradient use')
            node=name+'#reconstructed-tool-history-read-'+str(index)
            if node in nodes:raise ValueError('reconstructed occurrence collides with original graph')
            event={'kind':'block_read','node':node,'turn':name,'call_id':turn['invocation_id'],
                'block':part['id'],'format':'native-state','native_identity':copy.deepcopy(part['native_identity']),
                'inputs':[{'node':part['source_occurrence'],'block':part['id'],'port':'block'}],
                'reconstruction':'recorded-json-tool-arguments-and-actual-server-native-capture/1',
                **({'stop_gradient':True}if part.get('stop_gradient')else{})}
            events.append(event);nodes[node]=event;additions.append(node);missing[use]-=1
    result['execution_graph']=events
    validate_rollout_graph(result)
    return result,additions
