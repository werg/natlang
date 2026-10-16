"""Exact recorded occurrences lowered through the shared native graph compiler.

This is an internal mechanism, not the public GradSession dispatcher. Host
graph bindings are independently reconstructed from supplied execution events;
matching block text or an equal content ID never selects a writer occurrence.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace

from ..model.native_graph_plan import compile_graph, invocation_dict
from ..rl.native_replay import compile_replay, capture_invocation_actors


def authenticated_turns(turns, events):
    if not isinstance(turns, list) or not turns or not isinstance(events, list):
        raise ValueError('native recorded learning requires actual turns and execution graph events')
    nodes = {}
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get('node'), str):
            raise ValueError('native recorded graph event needs an exact node identity')
        name = event['node']
        if name in nodes:
            raise ValueError('native recorded graph repeats an occurrence identity')
        nodes[name] = event
    prepared, ids, linked = [], set(), set()
    for turn in turns:
        native = turn.get('native') if isinstance(turn, dict) else None
        if (not isinstance(native, dict) or native.get('schema') != 'natlang.native-recorded-turn/1'
                or not isinstance(native.get('id'), str) or not native['id'] or native['id'] in ids
                or type(native.get('retry')) is not int or native['retry'] < 0):
            raise ValueError('native recorded turn needs its unique exact receipt identity')
        ids.add(native['id'])
        binding = native.get('graph')
        if not isinstance(binding, dict) or set(binding) != {'graph_turn','written_nodes','read_bindings'}:
            raise ValueError('native recorded turn lacks its exact graph occurrence binding')
        name = binding['graph_turn']
        node = nodes.get(name)
        kind = native.get('kind')
        if (name in linked or node is None or node.get('kind') != 'model_turn'
                or kind not in {'generation','decision'}
                or (node.get('readout') == 'decision') != (kind == 'decision')
                or native.get('invocation_id') is not None and native['invocation_id'] != node.get('call_id')):
            raise ValueError('native recorded receipt changed its actual model-turn occurrence')
        linked.add(name)
        writes, reads, read_ids = {}, {}, set()
        for event in nodes.values():
            if event.get('turn') != name or event.get('kind') not in {'block_write','block_read'}:
                continue
            block = event.get('block')
            if not isinstance(block, str):
                raise ValueError('native block event needs its exact artifact identity')
            if event['kind'] == 'block_write':
                if block in writes:
                    raise ValueError('native repeated writes need distinct occurrence bindings')
                writes[block] = event['node']
            else:
                if block in read_ids:
                    raise ValueError('native repeated reads need distinct occurrence bindings')
                read_ids.add(block)
                inputs = event.get('inputs')
                if not isinstance(inputs, list) or any(not isinstance(item, dict) for item in inputs):
                    raise ValueError('native read needs exact execution-graph input edges')
                sources = [item['node'] for item in inputs if item.get('block') == block and item.get('node') != 'external']
                if len(sources) > 1:
                    raise ValueError('native read has ambiguous producer occurrence edges')
                if sources:
                    source = nodes.get(sources[0])
                    if source is not None and (source.get('kind') != 'block_write' or source.get('block') != block):
                        raise ValueError('native read edge does not reference its actual block writer')
                    reads[block] = sources[0]
        if binding['written_nodes'] != writes or binding['read_bindings'] != reads:
            raise ValueError('native recorded occurrence maps differ from actual execution graph events')
        capture = native.get('sampling' if kind == 'generation' else 'decision')
        if not isinstance(capture, dict) or kind == 'decision' and capture.get('schema') != 'natlang.native-decision-capture/1':
            raise ValueError('native recorded turn lacks its actual capture kind')
        captured_reads = [segment.get('id') for segment in capture.get('prompt',[])
                          if segment.get('kind')=='native_state']
        if len(set(captured_reads)) != len(captured_reads) or set(captured_reads) != read_ids:
            raise ValueError('native recorded imports need exact occurrence-matched state; decoded/reencoded inputs require explicit differentiable lowering')
        captured_writes = [write.get('state_id') for write in capture.get('writes',[])]
        if len(set(captured_writes)) != len(captured_writes) or set(captured_writes) != set(writes):
            raise ValueError('native recorded writes differ from actual captured producer artifacts')
        prepared.append({**copy.deepcopy(binding), 'sampling': copy.deepcopy(capture),
                         'receipt_id': native['id'], 'kind': kind})
    captured_writers={node for turn in prepared for node in turn['written_nodes'].values()}
    for turn in prepared:
        outside={block:node for block,node in turn['read_bindings'].items() if node not in captured_writers}
        turn['external_read_bindings']=outside
        turn['read_bindings']={block:node for block,node in turn['read_bindings'].items() if block not in outside}
    return prepared


@dataclass
class RecordedGraph:
    replay: object
    # Receipt ID -> actual invocation names (one generation or all decision options).
    invocations: dict
    # Receipt ID -> per-option actual conditional next-token coordinates.
    decision_actions: dict
    origin_bindings: dict

    def row(self, *, tokenizer_fingerprint, source_id):
        from .store import native_state_id
        return json.loads(json.dumps({'schema':'natlang.neuralese-native-graph/1','id':source_id,'split':'train',
            'source_groups':[source_id], 'tokenizer_fingerprint':tokenizer_fingerprint,
            'recorded_turn_bindings':self.origin_bindings,
            'nodes':[{**invocation_dict(node),'target_weights':[0.] * len(node.token_ids)}
                     for node in self.replay.plan.invocations],
            'external_states':{name:{'id':native_state_id(block), 'token_count':len(block.token_ids)}
                               for name,block in self.replay.external_blocks.items()}}))


def compile_recorded_graph(turns, events, *, identity, roles, fetch_block):
    prepared = authenticated_turns(turns, events)
    actors = capture_invocation_actors(prepared, identity)
    replay = compile_replay(prepared, identity=identity, roles=roles, fetch_block=fetch_block,
                            invocation_actors=actors, structural=True)
    by_name = {turn['graph_turn']:turn for turn in prepared}
    nodes, actor_map, invocations, local_actions = [], {}, {}, {}
    reserved = set(by_name) | set(replay.external_blocks)
    for node in replay.plan.invocations:
        turn = by_name[node.name]
        names = []
        if turn['kind'] == 'generation':
            nodes.append(node)
            names.append(node.name)
            actor_map[node.name] = actors['invocations'][node.name]
        else:
            options = turn['sampling'].get('continuations')
            if not isinstance(options,list) or not options:
                raise ValueError('native recorded decision requires actual option continuations')
            for index, ids in enumerate(options):
                if not isinstance(ids,list) or not ids or any(type(token) is not int or token < 0 for token in ids):
                    raise ValueError('native decision continuation needs nonempty exact token IDs')
                name = f'{node.name}:option:{index}'
                if name in reserved:
                    raise ValueError('native option identity collides with an actual source occurrence')
                reserved.add(name)
                nodes.append(replace(node,name=name,token_ids=(*node.token_ids,*ids)))
                names.append(name)
                actor_map[name] = copy.deepcopy(actors['invocations'][node.name])
                local_actions[name] = [{'query':len(node.token_ids)+position-1,'token_id':token,
                                       'temperature':1.,'allowed_token_ids':None,'banned_token_ids':None}
                                       for position,token in enumerate(ids)]
        invocations[turn['receipt_id']] = names
    plan = compile_graph(nodes, external_sizes={name:len(block.token_ids) for name,block in replay.external_blocks.items()})
    offsets = dict(zip((node.name for node in nodes),plan.offsets))
    decisions = {turn['receipt_id']:[[{**action,'query':offsets[name]+action['query']}
                 for action in local_actions[name]] for name in invocations[turn['receipt_id']]]
                 for turn in prepared if turn['kind']=='decision'}
    receipts = copy.deepcopy(actors)
    receipts['invocations'] = actor_map
    expanded={turn['graph_turn']:invocations[turn['receipt_id']] for turn in prepared}
    external_invocations={source:[{**copy.deepcopy(item),'invocation':name}
        for item in entries for name in expanded[item['invocation']]]
        for source,entries in (replay.external_invocations or {}).items()}
    external_receipts={source:[copy.deepcopy(item['receipt']) for item in entries if item['receipt'] is not None]
        for source,entries in external_invocations.items()
        if any(item['receipt'] is not None for item in entries)}
    origins={turn['receipt_id']:{key:copy.deepcopy(turn[key]) for key in
        ('graph_turn','written_nodes','read_bindings','external_read_bindings')} for turn in prepared}
    return RecordedGraph(replace(replay,plan=plan,invocation_actors=receipts,
        external_invocations=external_invocations,external_receipts=external_receipts),invocations,decisions,origins)
