"""Explicit differentiable codec dependencies between native masked graph segments.

Raw graphs retain the ordinary shared executor. A codec consumes complete source
state and produces target state, so its producer/codec/baseline/consumer passes
are named and counted separately. No sampled latent density is invented.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections import defaultdict

import torch

from .shared_state import Invocation, ImportSpan, compile_graph, capture_block, capture_logical_span
from .state_block import NativeStateBlock
from .native_actor_graph import invocation_actor_bank, invocation_graph_scope, suspend_invocation_graph


@dataclass(frozen=True)
class CodecDependency:
    input_name: str
    target_participant: str
    source_model: str
    target_model: str
    text: str
    source_state_id: str
    adapter_sha256: str
    source_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    source_node: str | None = None
    source_start: int = 0
    source_stop: int = 0
    source_logical: bool = False
    external_source: str | None = None

    def validate(self):
        import re
        if (any(not isinstance(value,str) or not value for value in
                (self.input_name,self.target_participant,self.source_model,self.target_model,self.source_state_id))
                or not isinstance(self.text,str) or not re.fullmatch('[0-9a-f]{64}',self.adapter_sha256)
                or (self.source_node is None)==(self.external_source is None)
                or self.source_node is not None and (not isinstance(self.source_node,str) or not self.source_node)
                or self.external_source is not None and (not isinstance(self.external_source,str) or not self.external_source)
                or type(self.source_logical) is not bool
                or any(type(value) is not int or value<0 for value in (self.source_start,self.source_stop))
                or self.source_stop<self.source_start
                or any(type(token) is not int or token<0 for token in (*self.source_token_ids,*self.target_token_ids))):
            raise ValueError('semantic policy dependency needs exact source/use, token and adapter bindings')
        return self


def _identity(block):
    return {key:getattr(block,key) for key in ('checkpoint_sha256','tokenizer_fingerprint',
                                              'embedding_weight_sha256','architecture')}


def execute_codec_graph(participants, plans, dependencies, *, channel, external_blocks=None):
    """Run every actual invocation once, preserving all cross-segment adjoints.

    Participants expose model/tokenizer/identity. The caller owns backend scopes,
    optimizer/RNG and the operation lease through backward; no weights change
    inside this mechanism. Inputs are already admitted complete episode graphs.
    """
    if not isinstance(plans,dict) or not plans or not set(plans)<=set(participants):
        raise ValueError('semantic policy graph requires its actual loaded participant plans')
    dependencies=[dependency.validate() for dependency in dependencies]
    if len({dependency.input_name for dependency in dependencies})!=len(dependencies):
        raise ValueError('semantic policy dependency input names must be unique')
    codecs={dependency.input_name:dependency for dependency in dependencies}
    nodes={};owners={}
    for owner,plan in plans.items():
        for node in plan.invocations:
            if node.name in nodes:raise ValueError('semantic policy invocation names must be globally unique')
            nodes[node.name]=node;owners[node.name]=owner
    uses=defaultdict(list)
    for name,node in nodes.items():
        for edge in node.imports:
            if edge.source in codecs:uses[edge.source].append((name,edge))
            elif edge.source in nodes and owners[edge.source]!=owners[name]:
                raise ValueError('different native models communicate through an explicit trained codec, not raw KV')
    for dependency in dependencies:
        if (not uses[dependency.input_name]
                or any(owners[name]!=dependency.target_participant or edge.start!=0
                       or edge.stop!=len(dependency.target_token_ids) or edge.logical
                       for name,edge in uses[dependency.input_name])
                or dependency.source_node is not None and dependency.source_node not in nodes
                or dependency.adapter_sha256!=channel.adapter_sha256):
            raise ValueError('semantic dependency differs from its actual consumers/source/adapter')
    # Semantic edges require a later segment; raw edges in a segment remain one
    # masked graph. Stable declared node order controls equal-level scheduling.
    levels={};active=set()
    def level(name):
        if name in levels:return levels[name]
        if name in active:raise ValueError('semantic policy dependencies must form an acyclic complete graph')
        active.add(name);value=0
        for edge in nodes[name].imports:
            if edge.source in nodes:value=max(value,level(edge.source))
            elif edge.source in codecs:
                dependency=codecs[edge.source]
                if dependency.source_node is not None:value=max(value,level(dependency.source_node)+1)
        active.remove(name);levels[name]=value;return value
    groups=defaultdict(list)
    for name in nodes:groups[(level(name),owners[name])].append(name)
    external_blocks=external_blocks or {}
    hidden={};exports={};decoded={};execution=[]
    # Retain only source spans, rather than all layer-state arrays after a segment.
    needed=defaultdict(set)
    for dependency in dependencies:
        if dependency.source_node is not None:
            needed[dependency.source_node].add((dependency.source_start,dependency.source_stop,
                                                dependency.source_logical,dependency.text))
    for name,node in nodes.items():
        for edge in node.imports:
            if edge.source in nodes and levels[edge.source]<levels[name]:
                needed[edge.source].add((edge.start,edge.stop,edge.logical,None))
    for (_,owner),names in sorted(groups.items()):
        participant=participants[owner];model=participant.model
        bank=invocation_actor_bank(model.port)
        own_identity=dict(participant.identity)
        def actor(name):
            return bank.receipt()['invocations'][name] if bank is not None else {
                'native_identity':own_identity,'adapter_binding':None}
        for name in names:
            for edge in nodes[name].imports:
                if edge.source not in codecs or edge.source in decoded:continue
                dependency=codecs[edge.source]
                source=(exports[(dependency.source_node,dependency.source_start,dependency.source_stop,
                                 dependency.source_logical,dependency.text)] if dependency.source_node is not None
                        else external_blocks[dependency.external_source])
                if (source.token_ids.tolist()!=list(dependency.source_token_ids) or source.text!=dependency.text
                        or channel.model_name(source)!=dependency.source_model):
                    raise ValueError('semantic policy source differs from its exact retained producer span')
                targets=[actor(reader) for reader,_ in uses[edge.source]]
                if any(target!=targets[0] for target in targets):
                    raise ValueError('one semantic dependency cannot silently choose between different target actors')
                target=targets[0]
                def adapt():
                    with suspend_invocation_graph(model.port):
                        return channel.adapt(source,target_name=dependency.target_model,target_model=model,
                            target_tokenizer=participant.tokenizer,target_identity=target['native_identity'],
                            target_adapter_binding=target['adapter_binding'],allow_nearest=True,
                            source_state_id=dependency.source_state_id)
                block,receipt=bank.execute(name,adapt) if bank is not None else adapt()
                if (block.token_ids.tolist()!=list(dependency.target_token_ids)
                        or receipt['mode']!='learned-semantic-state-deltas'):
                    raise ValueError('semantic policy decode changed its actual target tokenization or transport law')
                decoded[edge.source]=block
                execution.append({'kind':'semantic-codec','input':edge.source,'source_model':dependency.source_model,
                    'target_model':dependency.target_model,'adapter_sha256':dependency.adapter_sha256,
                    'posterior':'mean','source_state_id':dependency.source_state_id})
                execution.append({'kind':'semantic-target-baseline','participant':owner,
                                  'tokens':len(block.token_ids)+2})
        selected=[];blocks={};sizes={}
        for name in names:
            imports=[]
            for index,edge in enumerate(nodes[name].imports):
                if edge.source in names:
                    imports.append(edge);continue
                key=f'__segment_input:{name}:{index}'
                if edge.source in codecs:block=decoded[edge.source];start,stop=0,len(block.token_ids)
                elif edge.source in nodes:
                    block=exports[(edge.source,edge.start,edge.stop,edge.logical,None)];start,stop=0,len(block.token_ids)
                else:
                    block=external_blocks[edge.source];start,stop=edge.start,edge.stop
                blocks[key]=block;sizes[key]=len(block.token_ids)
                imports.append(ImportSpan(edge.at,key,start,stop,stop_gradient=edge.stop_gradient))
            selected.append(replace(nodes[name],imports=tuple(imports)))
        plan=compile_graph(selected,external_sizes=sizes)
        with invocation_graph_scope(model.port,names):
            result=model(plan,external_blocks=blocks,external_identity={key:_identity(block) for key,block in blocks.items()},
                         retain_states=any(needed[name] for name in names))
        execution.append({'kind':'native-graph-segment','participant':owner,'invocations':list(names),
                          'physical_tokens':plan.token_count,'logical_slots':sum(map(len,plan.sequences))})
        for node,offset in zip(plan.invocations,plan.offsets):
            hidden[node.name]=result['h_final'][:,offset:offset+len(node.token_ids)]
            identity=actor(node.name)
            for start,stop,logical,text in needed[node.name]:
                if logical:
                    values=capture_logical_span(result,node.name,start,stop)
                    block=NativeStateBlock(text=text or '',**identity['native_identity'],
                        adapter_binding=identity['adapter_binding'],**values).validate()
                else:
                    block=capture_block(result,node.name,start,stop,text=text or '',**identity['native_identity'])
                exports[(node.name,start,stop,logical,text)]=block
        result.pop('states',None)
    results={}
    for owner,plan in plans.items():
        results[owner]={'h_final':torch.cat([hidden[node.name] for node in plan.invocations],dim=1),
                        'plan':plan,'codec_execution':execution}
        bank=invocation_actor_bank(participants[owner].model.port)
        if bank is not None:results[owner]['invocation_actors']=bank.receipt()
    if set(hidden)!=set(nodes):raise ValueError('semantic policy execution lost an actual episode invocation')
    return results,execution
