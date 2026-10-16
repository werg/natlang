"""Public objective terms lowered to one shared graph, without executing it.

This internal compiler does not advertise or dispatch the unfinished public
GradSession derivative/optimizer protocol. It uses the production renderer,
actual captured source occurrences, native actor identities and query loss.
"""
from __future__ import annotations

import copy

from ..common.hashing import canonical_json_sha256_hex
from ..model.native_graph_plan import Invocation, compile_graph, invocation_dict
from ..rl.native_replay import _input_stream
from .grad import GradSession
from .native_recorded_graph import compile_recorded_graph


def _capture(turn):
    native=turn['native']
    return native['sampling' if native['kind']=='generation' else 'decision']


def _prompt_items(engine,turn):
    expected=[]
    for kind,value in engine._prompt_plan(turn['messages'],turn.get('tools')):
        if kind=='tokens':expected.extend(('token',token,False) for token in value)
        elif kind=='native_state':
            expected.extend([('token',engine.role_tokens.consume_open,False),('state',value),
                             ('token',engine.role_tokens.consume_close,False)])
        else:raise ValueError('native recorded request cannot use legacy projected imports')
    actual=[]
    for segment in _capture(turn)['prompt']:
        if segment['kind']=='tokens':
            actual.extend(('token',token,producing) for token,producing in zip(segment['token_ids'],segment['producing']))
        else:actual.append(('state',segment['id']))
    if expected!=actual:
        raise ValueError('native recorded prompt differs from its exact rendered messages/tools/import roles')


def lower_recorded_objective(engine,turns,events,terms,*,owner=None):
    from ..model.native_adapters import resolve_adapters,effective_identity,SCHEMA
    from ..model.native_readout import snapshot_readout
    from .native_grad import _identity
    identity=_identity(engine)
    def fetch(ident):
        record=engine.store.get_native_state(ident)
        if record is None:raise ValueError('native recorded objective is missing an exact archived state')
        return record.block
    compiled=compile_recorded_graph(turns,events,identity=identity,roles=engine.role_tokens,fetch_block=fetch)
    recorded={turn['native']['id']:turn for turn in turns}
    by_name={node.name:node for node in compiled.replay.plan.invocations}
    nodes=list(compiled.replay.plan.invocations)
    declarations={}
    for turn in turns:
        _prompt_items(engine,turn)
        resolved=resolve_adapters(engine,turn.get('adapters'))
        binding=({'schema':SCHEMA,'base_native_identity':identity,
            'adapters':[{'id':ident,'scale_hex':scale.hex()} for ident,scale,*_ in resolved]} if resolved else None)
        if _capture(turn)['policy']!=effective_identity(engine,resolved) or _capture(turn).get('adapter_binding')!=binding:
            raise ValueError('native recorded actor differs from its actual immutable adapter/base binding')
        for name in compiled.invocations[turn['native']['id']]:declarations[name]=copy.deepcopy(turn.get('adapters') or [])
    pending=[]
    target_values=[]
    teacher_readout=None
    def stopped_readout():
        nonlocal teacher_readout
        if teacher_readout is None:teacher_readout=snapshot_readout(engine,owner=owner).id
        return teacher_readout
    def selected(ident,kind=None):
        turn=recorded.get(ident)
        if turn is None or kind is not None and turn['native']['kind']!=kind:
            raise ValueError('native objective must select its exact recorded occurrence and readout kind')
        return turn
    def verify_side(side,turn):
        for key in ('messages','tools','adapters','native_state_modes'):
            if side.get(key)!=turn.get(key):
                # Empty adapters and absent adapters express the same exact base actor.
                if key=='adapters' and not side.get(key) and not turn.get(key):continue
                raise ValueError('native objective changes its recorded messages/tools/actor/transport contract')
    def prompt_length(turn):
        return sum(len(segment['token_ids']) for segment in _capture(turn)['prompt'] if segment['kind']=='tokens')
    def target_branch(turn,target):
        if not isinstance(target,dict):raise ValueError('native CE/distillation requires its target assistant message')
        _,rest=GradSession(engine)._target_items(turn['messages'],turn.get('tools'),target)
        source=by_name[compiled.invocations[turn['native']['id']][0]]
        length=prompt_length(turn)
        tokens=list(source.token_ids[:length]);flags=[False]*length
        for start,stop in source.producing:
            flags[start:min(stop,length)]=[True]*max(0,min(stop,length)-start)
        for kind,value in rest:
            if kind=='tok':tokens.append(int(value));flags.append(False)
            elif kind=='block':
                block=fetch(value)
                same=block.tokenizer_fingerprint==identity['tokenizer_fingerprint']
                ids=block.token_ids.tolist() if same else engine.tokenizer(block.text,
                    add_special_tokens=False,split_special_tokens=True)['input_ids']
                tokens.extend([engine.role_tokens.produce_open,*ids,engine.role_tokens.produce_close])
                flags.extend([False,*([True]*len(ids)),False])
                target_values.append({'id':value,'tokenizer_fingerprint':block.tokenizer_fingerprint,
                    'mode':'retained-source-token-ids' if same else 'best-effort-literal-text-retokenization'})
            else:raise ValueError('native target has an unsupported renderer item')
        if len(tokens)==length:raise ValueError('native objective target contains no actual continuation tokens')
        producing=[];start=None
        for index,flag in enumerate([*flags,False]):
            if flag and start is None:start=index
            if not flag and start is not None:producing.append((start,index));start=None
        name=f'objective-branch-{len(nodes)}'
        if name in by_name:raise ValueError('native objective branch collides with a recorded occurrence')
        node=Invocation(name,tuple(tokens),imports=source.imports,producing=tuple(producing))
        nodes.append(node);by_name[name]=node;declarations[name]=copy.deepcopy(turn.get('adapters') or [])
        return [(name,local-1,token,1.,None,None) for local,token in enumerate(tokens) if local>=length]
    def observed(turn,law):
        if law not in ('model-conditional','sampled-policy'):
            raise ValueError('native likelihood must explicitly declare model-conditional or sampled-policy law')
        capture=_capture(turn)
        if law=='sampled-policy' and capture['schema']=='natlang.native-structural-capture/1':
            raise ValueError('native policy gradient requires an actual positive-temperature sampled trajectory')
        name=compiled.invocations[turn['native']['id']][0]
        length=prompt_length(turn)
        stream,_=_input_stream(capture,length)
        actual=capture.get('sampling_law',{})
        result=[]
        for index,sample in enumerate(stream):
            if sample['forced']:continue
            result.append((name,length+index-1,sample['token_id'],
                actual['temperature'] if law=='sampled-policy' else 1.,
                sample.get('allowed_token_ids') if law=='sampled-policy' else None,
                sample.get('banned_token_ids') if law=='sampled-policy' else None))
        if not result:raise ValueError('native likelihood contains no actual sampled-token actions')
        return result
    def decision_options(turn):
        result=[]
        for name,ids in zip(compiled.invocations[turn['native']['id']],_capture(turn)['continuations']):
            length=prompt_length(turn)
            result.append([(name,length+index-1,token,1.,None,None) for index,token in enumerate(ids)])
        return result
    if not isinstance(terms,list) or not terms:raise ValueError('native recorded objective requires nonempty terms')
    for term in terms:
        if not isinstance(term,dict):raise ValueError('native recorded term must be an exact objective object')
        kind=term.get('kind')
        turn=selected(term.get('native_turn_id'),'decision' if kind=='decision' else 'generation')
        verify_side(term,turn)
        weight=term.get('weight',1.)
        if kind in ('crossEntropy','selfDistill'):
            student=target_branch(turn,term.get('target'))
            if kind=='crossEntropy':pending.append({'kind':'nll','actions':student,'reduction':'mean','weight':weight})
            else:
                if term.get('target')!=turn['reply']:raise ValueError('native self-distillation changes its actual student target')
                teacher=selected(term.get('teacher_native_turn_id'),'generation')
                verify_side({'messages':term.get('teacher_messages'),'tools':teacher.get('tools'),
                    'adapters':term.get('teacher_adapters'),'native_state_modes':term.get('teacher_native_state_modes')},teacher)
                pending.append({'kind':'distill','actions':student,'teacher_actions':target_branch(teacher,term['target']),
                    'teacher_readout':stopped_readout(),'weight':weight})
        elif kind=='logLikelihood':
            if term.get('target')!=turn['reply']:raise ValueError('native likelihood changes its actual observed reply')
            pending.append({'kind':'nll','actions':observed(turn,term.get('likelihood_law')),'reduction':'sum','weight':weight})
        elif kind=='decision':
            if term.get('options')!=turn['decision']['options']:raise ValueError('native decision changed its actual option inventory')
            options=decision_options(turn)
            if term.get('teacher') is not None:
                side=term['teacher'];teacher=selected(side.get('native_turn_id'),'decision');verify_side(side,teacher)
                if teacher['decision']['options']!=term['options']:raise ValueError('native teacher decision options differ')
                pending.append({'kind':'decision-distill','options':options,'teacher_options':decision_options(teacher),
                    'rule':term.get('rule'),'teacher_readouts':[stopped_readout()]*len(options),'weight':weight})
            else:
                goal=term.get('target') or {}
                pending.append({'kind':'decision','options':options,'rule':term.get('rule'),'weight':weight,
                    **{key:goal[key] for key in ('probabilities','rewards') if key in goal}})
        else:raise ValueError('native recorded objective is not defined for this target kind; raw KV has no Gaussian prior')
    plan=compile_graph(nodes,external_sizes={name:len(block.token_ids) for name,block in compiled.replay.external_blocks.items()})
    offsets=dict(zip((node.name for node in nodes),plan.offsets))
    def actions(items):
        return [{'query':offsets[name]+query,'token_id':token,'temperature':temperature,
                 'allowed_token_ids':allowed,'banned_token_ids':banned}
                for name,query,token,temperature,allowed,banned in items]
    for term in pending:
        for key in ('actions','teacher_actions'):
            if key in term:term[key]=actions(term[key])
        for key in ('options','teacher_options'):
            if key in term:term[key]=[actions(option) for option in term[key]]
    source=canonical_json_sha256_hex({'turns':turns,'events':events,'terms':terms})
    graph=compiled.row(tokenizer_fingerprint=identity['tokenizer_fingerprint'],source_id='native-recorded:'+source)
    graph['nodes']=[{**invocation_dict(node),'token_ids':list(node.token_ids),
        'target_weights':[0.]*len(node.token_ids)} for node in nodes]
    # Normalize tuple-valued invocation fields through the existing JSON boundary.
    import json
    graph=json.loads(json.dumps(graph))
    graph['recorded_objective_sha256']=source
    graph['target_native_values']=target_values
    return {'schema':'natlang.native-gradient-request/1','native_identity':identity,'graph':graph,
        'invocation_adapters':declarations,'objective':{'kind':'native-query','schema':'natlang.native-query-objective/1',
            'graph_sha256':canonical_json_sha256_hex(graph),'terms':pending},
        'native_state_modes':{key:value for turn in turns for key,value in (turn.get('native_state_modes') or {}).items()
            if key in {binding['id'] for binding in graph['external_states'].values()}}}
