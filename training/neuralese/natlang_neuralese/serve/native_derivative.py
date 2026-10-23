"""Finite native derivative DAG execution over the shared graph and optimizer.

Internal until public GradSession translation/archive contracts are qualified.
No communication graph is recursively replayed: each objective is one shared
masked graph evaluation; explicitly declared optimizer nodes change its inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import json
import torch
from ..common.hashing import canonical_json_sha256_hex
from ..common.native_higher_order import reference_arithmetic
from ..model.native_artifact import NativeTensorArtifact
from ..model.native_derivative_contract import PROGRAM,EXECUTION,optimizer_binding,optimizer_options,native_id,validate_finite_program,program_dependencies
from ..model.native_state_derivation import state_values,replace_state_values,optimize_state
from ..model.native_adapters import _Operation
from ..train.optim import functional_named_service_step
from .native_grad import native_graph_gradient


@dataclass
class Value:
    kind:str
    tensors:dict
    ident:str|None
    names:list
    source:dict
    moments:dict|None=None
    history_id:str|None=None


def _root(engine,declaration,owner):
    if not isinstance(declaration,dict) or set(declaration)-{'kind','id','names'}:
        raise ValueError('native derivative inputs need an exact typed target')
    kind=declaration.get('kind');ident=declaration.get('id')
    if kind=='adapter':
        from ..model.native_adapters import resolve_adapters
        resolved=resolve_adapters(engine,[ident])[0]
        available=resolved[3]
        source=engine.store.get_native_tensor_artifact(ident).artifact
    elif kind=='native-state':
        record=engine.store.get_native_state(ident)
        if record is None or record.block.metadata()['schema']!='neuralese.native-state/1':
            raise ValueError('native derivative state input needs its exact raw archived state')
        from ..model.native_state_derivation import verify_state_derivation
        verify_state_derivation(engine.store,record.block)
        available=state_values(record.block);source=record.block
    elif kind=='model-parameters':
        available=dict(engine.shared_model.named_parameters())
        source=None
    else:raise ValueError('native derivative input kind is model-parameters, adapter or native-state')
    names=declaration.get('names',sorted(available) if kind!='model-parameters' else None)
    if (not isinstance(names,list) or not names or any(not isinstance(name,str) or name not in available for name in names)
            or len(set(names))!=len(names)):
        raise ValueError('native derivative inputs need unique actual named tensors')
    # Unselected coefficients/state remain part of the exact actor, with no root
    # derivative coordinate. Model roots copy only explicitly selected tensors.
    all_names=names if kind=='model-parameters' else available
    tensors={name:available[name].detach().to(engine.device).clone().requires_grad_(name in names) for name in all_names}
    if kind=='model-parameters':
        snapshot=engine.store.put_native_tensor_artifact(NativeTensorArtifact('parameter-set',engine.native_identity,
            tensors,provenance={'scope':PROGRAM,'root':True}),owner)
        if ident is not None and ident!=snapshot.id:raise ValueError('native derivative root parameter snapshot differs from the actual live checkpoint')
        ident=snapshot.id
    return Value(kind,tensors,ident,sorted(names),source)


def _rewrite_request(request,bindings,values,roots):
    body=copy.deepcopy(request)
    if not isinstance(body,dict) or body.get('schema')!='natlang.native-gradient-request/1':
        raise ValueError('native derivative objectives use the shared complete native graph request')
    if any(key in body for key in ('target','order','derived','create_graph')):
        raise ValueError('native derivative objective targets are declared by its program node')
    if not isinstance(bindings,dict) or any(name not in roots or value not in values for name,value in bindings.items()):
        raise ValueError('native derivative bindings must select available values for exact declared roots')
    selected={name:values[bindings.get(name,name)] for name in roots}
    if any(value.kind!=roots[name].kind for name,value in selected.items()):raise ValueError('native derivative binding changes input kind')
    replacements={roots[name].ident:value.ident for name,value in selected.items() if roots[name].kind in ('adapter','native-state')}
    def adapters(entries):
        if entries is None:return None
        return [replacements.get(entry,entry) if isinstance(entry,str) else {**entry,'id':replacements.get(entry['id'],entry['id'])} for entry in entries]
    if 'adapters' in body:body['adapters']=adapters(body['adapters'])
    if 'invocation_adapters' in body:body['invocation_adapters']={key:adapters(entries) for key,entries in body['invocation_adapters'].items()}
    graph=body.get('graph')
    if isinstance(graph,dict) and body.get('graph_sha256') not in (None,canonical_json_sha256_hex(graph)):
        raise ValueError('native derivative objective graph differs from its declared original digest')
    if not isinstance(graph,dict):raise ValueError('native derivative objective needs its actual graph row')
    for declaration in (graph.get('external_states') or {}).values():
        declaration['id']=replacements.get(declaration['id'],declaration['id'])
    if 'native_state_modes' in body:body['native_state_modes']={replacements.get(key,key):value for key,value in body['native_state_modes'].items()}
    if 'graph_sha256' in body:body['graph_sha256']=canonical_json_sha256_hex(graph)
    # Policy manifests bind actual immutable actors/rollouts and must be retargeted
    # through that objective's shared current-policy implementation, never by JSON
    # replacement. Refuse unsupported retargeting before evaluating wrong logits.
    if isinstance(body.get('objective'),dict) and body['objective'].get('kind')=='native-query':
        objective=body['objective']
        if objective.get('graph_sha256')!=canonical_json_sha256_hex(request['graph']):
            raise ValueError('native derivative query objective changed its original graph binding')
        objective['graph_sha256']=canonical_json_sha256_hex(graph)
    elif isinstance(body.get('objective'),dict) and any(roots[name].ident!=value.ident for name,value in selected.items()):
        raise ValueError('native derivative registered-policy updates need exact shared objective retargeting')
    return body,selected


def _evaluate(engine,request,bindings,values,roots,target_name,finish,owner,*,target_aliases=None):
    if isinstance(request,dict) and request.get('schema')=='natlang.native-recorded-objective-request/1':
        from ..model.native_derivative_contract import validate_recorded_objective_request
        from .native_recorded_objective import lower_recorded_objective_request
        validate_recorded_objective_request(request)
        if not isinstance(bindings,dict) or any(key not in roots or value not in values for key,value in bindings.items()):
            raise ValueError('recorded derivative bindings must select exact available root values')
        selected={name:values[bindings.get(name,name)] for name in roots}
        declared={item['root']:item for item in request['targets']}
        aliases=({item['root']:item['root'] for item in request['targets']} if target_aliases is None
            else dict(target_aliases))
        if set(aliases.values())-set(declared):
            raise ValueError('recorded derivative target aliases must map to declared objective roots')
        specs={}
        for alias,root_name in aliases.items():
            target=declared[root_name];value=selected[root_name]
            if target['kind']!=value.kind or target['id']!=roots[root_name].ident:
                raise ValueError('recorded objective target differs from its exact declared derivative root')
            if values.get(bindings.get(root_name,root_name)) is not value:
                raise ValueError('recorded objective target differs from its actual derivative value binding')
            names=target.get('names',value.names)
            if sorted(names)!=sorted(value.names):
                raise ValueError('recorded objective target names differ from its exact differentiable value')
            specs[alias]={'kind':value.kind,'id':value.ident,'names':list(names),
                'tensors':{name:value.tensors[name] for name in names}}
        lowered=lower_recorded_objective_request(engine,request,owner=owner)
        body,selected=_rewrite_request(lowered,bindings,values,roots)
        from ..common.native_higher_order import active_reference_arithmetic
        def observe_recorded(loss,target_groups,metadata):
            provenance=metadata['provenance']
            overlays=sorted({value.ident:{'id':value.ident,'names':sorted(value.tensors)} for value in selected.values()
                    if value.kind=='model-parameters'}.values(),key=lambda item:item['id'])
            receipt=provenance.get('invocation_actors')
            actor={'schema':'natlang.native-functional-actor/1','base_native_identity':engine.native_identity,
                'parameter_values':overlays,
                'uniform_adapters':(provenance.get('adapter_binding') or {}).get('adapters',[]),
                'invocation_adapters':({name:(value.get('adapter_binding') or {}).get('adapters',[])
                    for name,value in receipt['invocations'].items()} if receipt is not None else None),
                'arithmetic':active_reference_arithmetic().binding()}
            if overlays:
                provenance={key:value for key,value in provenance.items() if key not in
                    {'effective_native_identity','adapter_binding','invocation_actors','actor_graph_sha256'}}
                if provenance.get('external_adapter_state_contracts'):
                    provenance['external_adapter_reference_contracts']=provenance.pop('external_adapter_state_contracts')
            metadata={**metadata,'provenance':{**provenance,'functional_actor':actor,
                'functional_actor_sha256':canonical_json_sha256_hex(actor),
                'evaluation_inputs':{key:{'kind':value.kind,'value_id':value.ident,'names':value.names}
                    for key,value in selected.items()}}}
            grouped={alias:{'value':selected[aliases[alias]],'tensors':group['tensors']}
                for alias,group in target_groups.items()}
            return finish(loss,grouped,metadata)
        return native_graph_gradient(engine,body,owner,evaluation_callback=observe_recorded,
            adapter_leaves_override={value.ident:value.tensors for value in selected.values() if value.kind=='adapter'},
            native_state_overrides={value.ident:replace_state_values(value.source,value.tensors)
                for value in selected.values() if value.kind=='native-state'},
            evaluation_target_specs=specs)
    body,selected=_rewrite_request(request,bindings,values,roots)
    target=selected[target_name]
    declaration={'kind':target.kind,'names':target.names}
    if target.kind!='model-parameters':declaration['id']=target.ident
    body['target']=declaration
    leaves={value.ident:value.tensors for value in selected.values() if value.kind=='adapter'}
    states={value.ident:replace_state_values(value.source,value.tensors) for value in selected.values() if value.kind=='native-state'}
    parameters={}
    for value in selected.values():
        if value.kind!='model-parameters':continue
        for name,tensor in value.tensors.items():
            if name in parameters and parameters[name] is not tensor:raise ValueError('native derivative evaluation has conflicting functional parameter values')
            parameters[name]=tensor
    def observed(loss,selected_tensors,metadata):
        from ..common.native_higher_order import active_reference_arithmetic
        provenance=metadata['provenance']
        overlays=sorted({value.ident:{'id':value.ident,'names':sorted(value.tensors)} for value in selected.values()
                if value.kind=='model-parameters'}.values(),key=lambda item:item['id'])
        receipt=provenance.get('invocation_actors')
        actor={'schema':'natlang.native-functional-actor/1','base_native_identity':engine.native_identity,
            'parameter_values':overlays,
            'uniform_adapters':(provenance.get('adapter_binding') or {}).get('adapters',[]),
            'invocation_adapters':({name:(value.get('adapter_binding') or {}).get('adapters',[])
                for name,value in receipt['invocations'].items()} if receipt is not None else None),
            'arithmetic':active_reference_arithmetic().binding()}
        if overlays:
            # Those old hashes describe coefficient-only reference actors. A
            # parameter-set overlay requires the full executable descriptor,
            # not a fabricated checkpoint identity for another weight set.
            provenance={key:value for key,value in provenance.items() if key not in
                {'effective_native_identity','adapter_binding','invocation_actors','actor_graph_sha256'}}
            if provenance.get('external_adapter_state_contracts'):
                provenance['external_adapter_reference_contracts']=provenance.pop('external_adapter_state_contracts')
        metadata={**metadata,'provenance':{**provenance,'functional_actor':actor,
            'functional_actor_sha256':canonical_json_sha256_hex(actor),
            'evaluation_inputs':{key:{'kind':value.kind,'value_id':value.ident,'names':value.names}
                for key,value in selected.items()}}}
        return finish(loss,{name:target.tensors[name] for name in target.names},metadata)
    def execute():
        return native_graph_gradient(engine,body,owner,evaluation_callback=observed,
            adapter_leaves_override=leaves,native_state_overrides=states,evaluation_targets={name:target.tensors[name] for name in target.names})
    if parameters:
        return torch.func.functional_call(_Operation(engine.shared_model,execute),{'model.'+name:value for name,value in parameters.items()},())
    return execute()


def execute_derivative_program(engine,program,owner=None):
    """Evaluate a declared finite program without modifying the serving model."""
    validate_finite_program(program)
    from ..model.native_readout import readout_scope
    with engine.native_execution_lock,readout_scope(),torch.inference_mode(False),torch.enable_grad(),reference_arithmetic(
            engine.shared_model,query_chunk_tokens=program['arithmetic']['query_chunk_tokens']) as arithmetic:
        roots={name:_root(engine,value,owner) for name,value in program['inputs'].items()}
        coordinates=[(value.kind,value.ident,name) for value in roots.values() for name in value.names]
        if len(set(coordinates))!=len(coordinates):raise ValueError('native derivative inputs overlap the same exact coordinate')
        actor_ids=[(value.kind,value.ident) for value in roots.values() if value.kind!='model-parameters']
        if len(set(actor_ids))!=len(actor_ids):raise ValueError('native derivative actor inputs must uniquely bind each complete coefficient or state artifact')
        values=dict(roots);gradients={};outputs={};optimizer_states={};receipts=[]
        bound=copy.deepcopy(program)
        for name,value in roots.items():bound['inputs'][name]['id']=value.ident
        text=json.dumps(bound,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        import hashlib
        dependencies=program_dependencies(bound)
        declaration=engine.store.put_native_tensor_artifact(NativeTensorArtifact('derivative-program',engine.native_identity,{},
            provenance={'schema':PROGRAM,'program_json':text,'program_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'arithmetic':arithmetic.binding(),'dependencies':sorted(dependencies)}),owner)
        for node in program['nodes']:
            name=node.get('name')
            if not isinstance(name,str) or not name or name in values or name in gradients:raise ValueError('native derivative nodes require unique ordered names')
            kind=node.get('kind')
            if kind=='gradient':
                targets=(node.get('targets') if 'targets' in node else [{'name':name,'value':node.get('value')}])
                if not isinstance(targets,list) or not targets or any(item['value'] not in values for item in targets):
                    raise ValueError('native derivative gradient references an unavailable earlier value')
                if not isinstance(node['bindings'],dict) or any(key not in roots or item not in values for key,item in node['bindings'].items()):
                    raise ValueError('native derivative bindings must select available values for exact declared roots')
                root_for_alias={}
                for target in targets:
                    matches=[root for root in roots if node['bindings'].get(root,root)==target['value']]
                    if len(matches)!=1:raise ValueError('each recorded gradient alias must bind one exact original root')
                    root_for_alias[target['name']]=matches[0]
                def finish(loss,groups,metadata):
                    if metadata.get('target_groups') is None:
                        alias=targets[0]['name'];target_value=values[targets[0]['value']]
                        selected=groups
                        tensors=(torch.autograd.grad(loss,tuple(selected.values()),create_graph=True,retain_graph=True,allow_unused=True)
                            if loss.requires_grad else (None,)*len(selected))
                        live={key:gradient if gradient is not None else selected[key]*0 for key,gradient in zip(selected,tensors)}
                        record=engine.store.put_native_tensor_artifact(NativeTensorArtifact(
                            {'adapter':'adapter-gradient','native-state':'state-gradient','model-parameters':'parameter-gradient'}[target_value.kind],
                            engine.native_identity,live,target_value.ident if target_value.kind!='model-parameters' else None,
                            {**metadata['provenance'],'derivative_program_id':declaration.id,'node':alias,'gradient_node':name,
                             'arithmetic':arithmetic.binding(),'graph_sha256':metadata['graph_sha256'],'loss':float(loss.detach()),
                             'native_derivative_target':{'schema':'natlang.native-derivative-target/1','stage':'inner',
                                'variable':targets[0]['value'],'value_id':target_value.ident,'kind':target_value.kind,'names':sorted(live)},
                             **({'parameter_value_id':target_value.ident} if target_value.kind=='model-parameters' else {})}),owner)
                        return record,live
                    entries=[(alias,key,tensor,groups[alias]['value']) for alias in root_for_alias
                        for key,tensor in groups[alias]['tensors'].items()]
                    raw=(torch.autograd.grad(loss,tuple(item[2] for item in entries),create_graph=True,
                        retain_graph=True,allow_unused=True) if loss.requires_grad else (None,)*len(entries))
                    live_by_alias={alias:{} for alias in root_for_alias}
                    for (alias,key,tensor,_),gradient in zip(entries,raw):
                        live_by_alias[alias][key]=gradient if gradient is not None else tensor*0
                    records={}
                    for alias,root_name in root_for_alias.items():
                        target_value=groups[alias]['value'];live=live_by_alias[alias]
                        record=engine.store.put_native_tensor_artifact(NativeTensorArtifact(
                            {'adapter':'adapter-gradient','native-state':'state-gradient','model-parameters':'parameter-gradient'}[target_value.kind],
                            engine.native_identity,live,target_value.ident if target_value.kind!='model-parameters' else None,
                            {**metadata['provenance'],'derivative_program_id':declaration.id,'node':alias,'gradient_node':name,
                             'arithmetic':arithmetic.binding(),'graph_sha256':metadata['graph_sha256'],'loss':float(loss.detach()),
                             'target_kind':target_value.kind,'target_names':sorted(live),
                             'native_derivative_target':{'schema':'natlang.native-derivative-target/1','stage':'inner',
                                'variable':targets[[item['name'] for item in targets].index(alias)]['value'],
                                'value_id':target_value.ident,'kind':target_value.kind,'names':sorted(live)},
                             **({'parameter_value_id':target_value.ident} if target_value.kind=='model-parameters' else {})}),owner)
                        records[alias]=(record,live,target_value.ident)
                    return records
                if node['request'].get('schema')=='natlang.native-recorded-objective-request/1':
                    alias_map={target['name']:root_for_alias[target['name']] for target in targets}
                else:alias_map=None
                evaluated=_evaluate(engine,node['request'],node['bindings'],values,roots,root_for_alias[targets[0]['name']],finish,owner,
                    target_aliases=alias_map)
                if isinstance(evaluated,dict) and all(alias in evaluated for alias in root_for_alias):
                    for alias,record_live in evaluated.items():
                        gradients[alias]=record_live;outputs[alias]=record_live[0].id
                    record_ids={alias:record.id for alias,(record,_,_) in evaluated.items()}
                else:
                    alias=targets[0]['name'];record,live=evaluated
                    value=values[targets[0]['value']]
                    gradients[alias]=(record,live,value.ident);outputs[alias]=record.id;record_ids={alias:record.id}
            elif kind=='optimizer':
                if set(node)!={'kind','name','value','gradient','optimizer','optimizer_state'} or node['value'] not in values or node['gradient'] not in gradients:
                    raise ValueError('native derivative update references unavailable earlier values or gradients')
                previous=values[node['value']];record,live_gradient,gradient_parent=gradients[node['gradient']]
                if gradient_parent!=previous.ident:raise ValueError('native derivative gradient belongs to another exact earlier value')
                if record.artifact.source_artifact_id not in (previous.ident,None):raise ValueError('native derivative update gradient belongs to another value')
                options=optimizer_options(optimizer_binding(node['optimizer']))
                history=node['optimizer_state']
                if history is None:moment_tensors={};history_id=None
                elif history in values:
                    prior=values[history];moment_tensors=prior.moments;history_id=prior.history_id
                    if prior.ident!=previous.ident or moment_tensors is None:raise ValueError('native derivative update history belongs to another value')
                elif native_id(history,'nnt1_'):
                    old=engine.store.get_native_tensor_artifact(history)
                    if old is None or old.artifact.kind!='optimizer-state' or old.artifact.source_artifact_id!=previous.ident:raise ValueError('native derivative optimizer artifact belongs to another exact value')
                    moment_tensors=old.artifact.tensors;history_id=history
                else:raise ValueError('native derivative optimizer needs explicit new or exact earlier history')
                if previous.kind=='native-state':
                    result,block,moments=optimize_state(engine,record,options,history_id,owner,values=previous.tensors,
                        gradients=live_gradient,moments=moment_tensors,program_id=declaration.id)
                    value=Value(previous.kind,state_values(block),result['native_state_id'],previous.names,block,moments,result['optimizer_state_id'])
                else:
                    updated,moments=functional_named_service_step({key:previous.tensors[key] for key in live_gradient},live_gradient,moment_tensors,**options)
                    tensors={**previous.tensors,**updated}
                    parent=engine.store.get_native_tensor_artifact(previous.ident).artifact
                    artifact=NativeTensorArtifact('adapter-coefficients' if previous.kind=='adapter' else 'parameter-set',engine.native_identity,tensors,
                        parent.source_artifact_id if previous.kind=='adapter' else record.id,
                        {**{key:value for key,value in (parent.provenance or {}).items() if key!='root'},'derivative_program_id':declaration.id,'parent_value_id':previous.ident,
                         'gradient_id':record.id,'optimizer_state_parent_id':history_id,'optimizer':optimizer_binding(options),'node':name})
                    saved=engine.store.put_native_tensor_artifact(artifact,owner)
                    state=engine.store.put_native_tensor_artifact(NativeTensorArtifact('optimizer-state',engine.native_identity,moments,saved.id,
                        {'scope':PROGRAM,'optimizer':options['name'],'derivative_program_id':declaration.id,'gradient_id':record.id,
                         'optimizer_state_parent_id':history_id,'step':int(moments['optimizer.step'].item())}),owner)
                    value=Value(previous.kind,tensors,saved.id,previous.names,artifact,moments,state.id)
                values[name]=value;outputs[name]=value.ident;optimizer_states[name]=value.history_id
            else:raise ValueError('native derivative node kind must be gradient or optimizer')
            receipts.append({'name':name,'kind':kind,'artifact_id':outputs.get(name),
                **({'gradients':record_ids} if kind=='gradient' and 'record_ids' in locals() else {})})
            if kind=='gradient' and 'record_ids' in locals():del record_ids
        outer=program['outer']
        if not isinstance(outer,dict) or set(outer)!={'request','bindings'}:raise ValueError('native derivative outer objective needs its exact request and input bindings')
        def finish_outer(loss,selected,metadata):
            coordinates=[(name,key,roots[name].tensors[key]) for name in program['wrt'] for key in roots[name].names]
            raw=(torch.autograd.grad(loss,tuple(value for _,_,value in coordinates),allow_unused=True)
                if loss.requires_grad else (None,)*len(coordinates))
            grads=tuple(gradient if gradient is not None else value*0 for (_,_,value),gradient in zip(coordinates,raw))
            return loss.detach(),coordinates,grads,metadata
        loss,coordinates,grads,metadata=_evaluate(engine,outer['request'],outer['bindings'],values,roots,
            next(iter(roots)),finish_outer,owner)
        execution=engine.store.put_native_tensor_artifact(NativeTensorArtifact('derivative-program',engine.native_identity,{},
            provenance={'schema':EXECUTION,'program_id':declaration.id,'arithmetic':arithmetic.binding(),'nodes':receipts,
                'outputs':outputs,'optimizer_states':optimizer_states,'outer_graph_sha256':metadata['graph_sha256'],'loss':float(loss),
                'outer_evaluation_inputs':metadata['provenance']['evaluation_inputs'],
                'outer_functional_actor':metadata['provenance']['functional_actor'],
                'dependencies':sorted({declaration.id,*outputs.values(),*optimizer_states.values()})}),owner)
        result={}
        for name in program['wrt']:
            root=roots[name]
            tensors={key:gradient for (root_name,key,_),gradient in zip(coordinates,grads) if root_name==name}
            artifact=engine.store.put_native_tensor_artifact(NativeTensorArtifact(
                {'adapter':'adapter-gradient','native-state':'state-gradient','model-parameters':'parameter-gradient'}[root.kind],
                engine.native_identity,tensors,root.ident if root.kind!='model-parameters' else None,
                {'derivative_program_id':declaration.id,'derivative_execution_id':execution.id,'arithmetic':arithmetic.binding(),
                 **metadata['provenance'],'objective':'native-finite-derivative-program/1','loss':float(loss),
                 'target_kind':root.kind,'target_names':sorted(tensors),
                 'native_derivative_target':{'schema':'natlang.native-derivative-target/1','stage':'outer','variable':name,
                    'value_id':root.ident,'kind':root.kind,'names':sorted(tensors)},
                 **({'parameter_value_id':root.ident} if root.kind=='model-parameters' else {})}),owner)
            result[name]={'id':artifact.id,**artifact.meta()}
        return {'schema':EXECUTION,'program_id':declaration.id,'execution_id':execution.id,'loss':float(loss),
            'arithmetic':arithmetic.binding(),'gradients':result,'outputs':outputs,'optimizer_states':optimizer_states}
