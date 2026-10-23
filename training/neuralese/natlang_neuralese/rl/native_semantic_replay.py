"""Authenticate the original compact wire and actual decoded observation.

These are crisp archive/transport checks, not a policy for choosing a dialect.
The dependency executor recomputes captured producers and the real codec; this
module never promotes an observed decoded tensor to an in-episode producer.
"""
from ..model.semantic_message import SemanticStateMessage,retained_source_offsets
from ..model.semantic_vae import alignment_weights


def authenticate_semantic_input(segment,observed,*,identity,channel,fetch_block):
    if channel is None:
        raise ValueError('native semantic replay needs the exact differentiable semantic channel')
    source_id=segment.get('source_state_id')
    if not isinstance(source_id,str) or source_id==segment.get('id'):
        raise ValueError('native semantic replay needs original compact and observed decoded artifact identities')
    message=fetch_block(source_id)
    if not isinstance(message,SemanticStateMessage):
        raise ValueError('native semantic replay source must be the actual compact message')
    message.validate();observed.require_native(**identity)
    source=channel.model_name(message)
    targets=[name for name,binding in channel.bindings.items() if binding==identity]
    if len(targets)!=1:
        raise ValueError('semantic replay target must be one exactly trained participant')
    target=targets[0]
    source_offsets,policy=retained_source_offsets(channel.tokenizers[source],message.text,message.token_ids.tolist())
    tokens=channel.tokenizers[target](message.text,add_special_tokens=False,return_offsets_mapping=True)
    _,matching=alignment_weights(source_offsets,tokens['offset_mapping'],allow_nearest=True)
    matching['best_effort']=matching['best_effort'] or policy!='exact-tokenizer-offsets'
    expected={'source_model':source,'adapter_sha256':channel.adapter_sha256,'posterior':'mean',
        'source_native_identity':channel.bindings[source],'source_offset_policy':policy,
        'target_model':target,'alignment':matching,'mode':'learned-semantic-state-deltas',
        'target_native_identity':identity}
    if (message.adapter_sha256!=channel.adapter_sha256 or message.source_model!=source
            or message.offsets!=source_offsets or message.offset_policy!=policy
            or segment.get('semantic_adaptation')!=expected
            or observed.text!=message.text or observed.token_ids.tolist()!=tokens['input_ids']
            or segment.get('native_identity')!=identity):
        raise ValueError('semantic replay changed its exact adapter, tokenizer, source or actual decode receipt')
    return {'source_state_id':source_id,'source_model':source,'target_model':target,
        'text':message.text,'adapter_sha256':channel.adapter_sha256,
        'source_token_ids':tuple(message.token_ids.tolist()),'target_token_ids':tuple(observed.token_ids.tolist())}


def compile_semantic_episode(rollout,participants,*,channel,fetch_block):
    """Compile every actual participant call, then join verified codec edges.

    Per-participant parsing uses the shared native capture compiler. A foreign
    source is temporarily an external dependency of that participant's parser;
    the verified whole-episode occurrence inventory then reconnects it to its
    real source invocation before any model executes. It is never detached or
    externalized in the resulting episode dependency graph.
    """
    import copy
    from dataclasses import replace
    from .episode_graph import whole_episode_turns
    from .native_replay import compile_replay,_require_scored_training_episode
    from ..model.shared_state import RoleTokens
    if not participants:raise ValueError('semantic policy episode needs its actual loaded participants')
    identities={name:dict(participant.identity) for name,participant in participants.items()}
    for name,identity in identities.items():channel.require_target(name,identity)
    # Shared-checkpoint participants have one durable checkpoint, independently
    # bound tokenizer/embedding/architecture identities. The header is collection
    # ownership; each actual call is checked against its exact participant below.
    header=next(iter(identities.values()))
    _require_scored_training_episode(rollout,header,semantic=True)
    turns=whole_episode_turns(rollout,semantic=True)
    groups={};owners={}
    for turn in turns:
        matches=[name for name,identity in identities.items() if identity==(turn.get('sampling') or {}).get('policy')]
        if len(matches)!=1:raise ValueError('semantic episode call is not one exactly bound participant actor')
        owner=matches[0];groups.setdefault(owner,[]).append(copy.deepcopy(turn));owners[turn['graph_turn']]=owner
    replays={};writers={}
    for owner,selected in groups.items():
        local_writers={item['node'] for turn in selected for item in turn.get('written_occurrences',[])}
        local_writers.update(node for turn in selected for node in turn.get('written_nodes',{}).values())
        for turn in selected:
            inside={};outside=dict(turn.get('external_read_bindings') or {})
            for ident,origin in turn.get('read_bindings',{}).items():
                (inside if origin in local_writers else outside)[ident]=origin
            turn['read_bindings']=inside;turn['external_read_bindings']=outside
            # Repeated read receipts carry occurrence origins directly. The
            # shared parser classifies a foreign actual writer as outside its
            # local invocation set, retaining that exact origin for reconnection.
        participant=participants[owner]
        replay=compile_replay(selected,identity=identities[owner],roles=RoleTokens(*participant.model.marker_ids),
            fetch_block=fetch_block,semantic_channel=channel)
        replays[owner]=replay
        for source in replay.written_sources:
            if source['node'] in writers:raise ValueError('semantic episode repeats a verified producer occurrence')
            writers[source['node']]=source
    dependencies=[];externals={}
    for owner,replay in replays.items():
        joined=[]
        for dependency in replay.semantic_dependencies:
            source=writers.get(dependency.source_occurrence)
            if source is not None:
                if source['state_id']!=dependency.source_state_id:
                    raise ValueError('semantic episode codec edge changed the actual producer artifact')
                dependency=replace(dependency,source_node=source['invocation'],source_start=source['start'],
                    source_stop=source['stop'],external_source=None)
            joined.append(replace(dependency,target_participant=owner))
        replay.semantic_dependencies=tuple(joined);dependencies.extend(joined)
        needed={edge.source for node in replay.plan.invocations for edge in node.imports}
        needed.update(dependency.external_source for dependency in joined if dependency.external_source is not None)
        for name,block in replay.external_blocks.items():
            if name not in needed:continue
            if name in externals:
                from ..serve.store import native_state_id
                if native_state_id(externals[name])!=native_state_id(block):
                    raise ValueError('semantic episode external name aliases different actual artifacts')
            externals[name]=block
    if set(owners)!={node.name for replay in replays.values() for node in replay.plan.invocations}:
        raise ValueError('semantic policy compiler lost an actual whole-episode model call')
    return replays,tuple(dependencies),externals


def semantic_episode_loss(participants,replays,dependencies,*,channel,external_blocks,advantage,mode='grpo',
                          clip_low=.2,clip_high=.28,chunk_tokens=128,training_execution=None,behavior_channel=None,reference=False):
    """One joint policy objective; this mechanism never steps any optimizer."""
    import math
    import torch
    from .native_replay import action_logps
    from .policy_math import clipped_token_loss
    from ..model.native_codec_graph import execute_codec_graph
    if mode not in {'grpo','rft'} or not math.isfinite(advantage):
        raise ValueError('semantic policy objective needs an actual finite admitted advantage and supported mode')
    results,execution=execute_codec_graph(participants,{name:replay.plan for name,replay in replays.items()},
        dependencies,channel=channel,external_blocks=external_blocks,
        training_execution=training_execution,behavior_channel=behavior_channel,reference=reference)
    terms=[];count=0;by_model={}
    for name,replay in replays.items():
        from ..model.native_adapters import require_model_execution_actor
        require_model_execution_actor(participants[name].model,replay.identity,replay.adapter_binding)
        logps,behavior=action_logps(participants[name].model,results[name]['h_final'][0],replay.actions,
                                  chunk_tokens=chunk_tokens)
        values=-logps if mode=='rft' else clipped_token_loss(logps,behavior,advantage,
                                                          clip_low=clip_low,clip_high=clip_high)
        terms.append(values.sum());count+=len(replay.actions);by_model[name]=len(replay.actions)
    if not count:raise ValueError('semantic policy episode contains no actual sampled actions')
    return torch.stack(terms).sum()/count,{'sampled_tokens':count,'sampled_tokens_by_model':by_model,
        'execution':execution,'objective':mode,'posterior':'mean'}
