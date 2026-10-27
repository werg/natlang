"""Native graph gradients returned as identity-bound named tensor artifacts.

The request is a graph, not a legacy projected-value GradSession. Graph
execution and loss are delegated to the shared native graph objective; this
module only binds exact stored imports, selects real trainable tensors, and
durably names the resulting gradients.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from ..common.jsonio import canonical_json_bytes
from ..data.native_graph import graph_loss, parse_graph
from ..model.native_artifact import NativeTensorArtifact
from .chat import RequestError


def _identity(engine):
    return {'checkpoint_sha256': engine.checkpoint_sha256,
            'tokenizer_fingerprint': engine.model_config['tokenizer_fingerprint'],
            'embedding_weight_sha256': engine.embedding_weight_sha256,
            'architecture': engine.model_config['architecture']}


def _leaf_block(block):
    """Clone each imported floating tensor as a leaf while preserving IDs."""
    layers = []
    leaves = {}
    for index, layer in enumerate(block.layers):
        copied = dict(layer)
        for name in ('raw_k', 'v', 'gated_inputs', 'mixed_qkv', 'z', 'b', 'a'):
            if name in copied:
                value = copied[name].detach().clone().requires_grad_(True)
                copied[name] = value
                leaves[f'layers.{index}.{name}'] = value
        layers.append(copied)
    # Imported embeddings/token IDs/positions are provenance and logical input
    # metadata; raw KV and hybrid sequence tensors are the consumed state.
    return replace(block, layers=tuple(layers)), leaves


def _parameter_targets(model, requested):
    available = dict(model.named_parameters())
    if not isinstance(requested, list) or not requested or any(not isinstance(name, str) for name in requested):
        raise RequestError('neuralese-native-grad-target', 'parameter gradients require explicit trainable parameter names')
    if len(set(requested)) != len(requested) or any(name not in available for name in requested):
        raise RequestError('neuralese-native-grad-target', 'requested names must uniquely select native model parameters')
    return {name: available[name] for name in requested}


def native_coefficient_gradients(engine, body, targets, owner=None):
    """Learner seam: all selected coefficient adjoints from ONE current graph.

    The HTTP single-target contract stays unchanged. This shares its exact
    objective/import/actor execution and emits existing per-actor artifacts.
    """
    if (not isinstance(targets,list) or not targets or any(not isinstance(item,dict) or
            set(item)-{'id','names'} or not isinstance(item.get('id'),str) for item in targets)
            or len({item['id'] for item in targets})!=len(targets)):
        raise ValueError('coefficient gradient targets must uniquely declare actual actor IDs and named tensors')
    return native_graph_gradient(engine,{**body,'target':{'kind':'adapter',**targets[0]}},owner,
                                 coefficient_targets=targets)


def native_recorded_gradient(engine, request, owner=None):
    """Evaluate a recorded objective's named adapter/state roots in one backward."""
    from .native_recorded_objective import lower_recorded_objective_request
    from ..model.native_derivative_contract import validate_recorded_objective_request
    validate_recorded_objective_request(request)
    body=lower_recorded_objective_request(engine,request,owner=owner)
    targets={item['root']:{key:item[key] for key in ('kind','id','names') if key in item}
        for item in request['targets']}
    return native_graph_gradient(engine,body,owner,evaluation_target_specs=targets)


def native_graph_gradient(engine, body, owner, *, coefficient_targets=None, evaluation_callback=None,
                          adapter_leaves_override=None, native_state_overrides=None, evaluation_targets=None,
                          evaluation_target_specs=None):
    if body.get('schema') != 'natlang.native-gradient-request/1':
        raise RequestError('neuralese-native-grad-schema', 'native graph gradients require schema natlang.native-gradient-request/1')
    if type(body.get('order',1)) is not int or body.get('order',1) != 1 or body.get('derived') or body.get('create_graph'):
        raise RequestError('neuralese-native-grad-order', 'native graph requests are first-order; send a second-order objective as a natlang.native-derivative-program/1 request')
    from ..model.native_adapters import resolve_adapters, run_native
    try:
        declarations=body.get('invocation_adapters')
        if 'invocation_adapters' in body and not isinstance(declarations,dict):
            raise ValueError('invocation_adapters must declare an actual invocation-to-adapter map')
        if declarations is not None:
            example=parse_graph(body.get('graph'),tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
            names=[node.name for node in example.plan.invocations]
            if (not isinstance(declarations,dict) or any(name not in names for name in declarations)
                    or any(not isinstance(entries,list) for entries in declarations.values())):
                raise ValueError('invocation_adapters must map actual invocation names to explicit adapter lists')
            source_names={name:name for name in names}
            if isinstance(body.get('objective'),dict) and body['objective'].get('kind')=='native-policy':
                from ..train.native_state_sft import pack_native_graphs
                packed,_,_,_=pack_native_graphs([example])
                source_names={before.name:after.name for before,after in zip(example.plan.invocations,packed.invocations)}
                declarations={source_names[name]:value for name,value in declarations.items()}
                names=[node.name for node in packed.invocations]
            entries=[body.get('adapters'),*declarations.values()]
        else:
            entries=[body.get('adapters')]
        leaves={}
        for bindings in entries:
            for ident,_,_,coefficients,_ in resolve_adapters(engine,bindings):
                if ident not in leaves:
                    leaves[ident]=(adapter_leaves_override[ident] if adapter_leaves_override is not None
                        and ident in adapter_leaves_override else
                        {name:value.detach().clone().requires_grad_(True) for name,value in coefficients.items()})
        adapter_target_ids=None
        target_specs=(coefficient_targets if coefficient_targets is not None else
                      ([body.get('target')] if isinstance(body.get('target'),dict)
                       and body['target'].get('kind')=='adapter' else []))
        if target_specs:
            selected_ids=set()
            for declaration in target_specs:
                ident=declaration.get('id')
                available=leaves.get(ident,{})
                requested=declaration.get('names')
                # Keep this selection identical to _native_graph_gradient_impl:
                # omitted names mean every active coefficient tensor.  Passing
                # the mapping itself through the sequence check silently made
                # the carrier scope select no auxiliary cotangent routes.
                requested=tuple(available) if requested is None else requested
                if isinstance(requested,(list,tuple)):
                    selected_ids.update(id(available[name]) for name in requested if name in available)
            adapter_target_ids=selected_ids
        if declarations is not None:
            from ..model.native_actor_graph import run_invocation_actors
            selected=(_parameter_targets(engine.shared_model,body['target'].get('names'))
                      if (body.get('target') or {}).get('kind')=='model-parameters' else {})
            # Base-weight adjoints must exist while actor weights are derived,
            # before entering any functional substitution.
            restore={parameter:parameter.requires_grad for parameter in selected.values()}
            try:
                for parameter in restore:parameter.requires_grad_(True)
                return run_invocation_actors(engine,names,declarations,
                    lambda bank:_native_graph_gradient_impl(engine,body,owner,leaves,actor_bank=bank,coefficient_targets=coefficient_targets,evaluation_callback=evaluation_callback,
                        native_state_overrides=native_state_overrides,evaluation_targets=evaluation_targets,
                        evaluation_target_specs=evaluation_target_specs),
                    default=body.get('adapters'),leaves=leaves,
                    gradient_precision=body.get('adapter_gradient_precision','storage'),source_names=source_names,
                    track_adapter_cotangents=(body.get('target',{}).get('kind')!='model-parameters'),
                    adapter_cotangent_target_ids=adapter_target_ids,
                    cotangent_parameter_ids=({id(value) for value in selected.values()}
                        if (body.get('target') or {}).get('kind')=='model-parameters' else None))
            finally:
                for parameter,enabled in restore.items():parameter.requires_grad_(enabled)
        selected=(_parameter_targets(engine.shared_model,body['target'].get('names'))
                  if (body.get('target') or {}).get('kind')=='model-parameters' else {})
        # run_native snapshots its carrier inventory before entering the callback.
        # Enable exact requested backbone leaves first so BF16 cotangent anchors
        # exist for this operation; the inner target selection restores to this
        # temporary state, then this outer boundary restores serving's frozen base.
        restore={parameter:parameter.requires_grad for parameter in selected.values()}
        try:
            for parameter in restore:
                parameter.requires_grad_(True)
            return run_native(engine, lambda: _native_graph_gradient_impl(engine, body, owner, leaves,coefficient_targets=coefficient_targets,evaluation_callback=evaluation_callback,
                native_state_overrides=native_state_overrides,evaluation_targets=evaluation_targets,
                evaluation_target_specs=evaluation_target_specs),
                              body.get('adapters'), leaves=leaves,
                              gradient_precision=body.get('adapter_gradient_precision', 'storage'),
                              track_adapter_cotangents=(body.get('target',{}).get('kind')!='model-parameters'),
                              adapter_cotangent_target_ids=adapter_target_ids,
                              cotangent_parameter_ids=({id(value) for value in selected.values()}
                                  if (body.get('target') or {}).get('kind')=='model-parameters' else None))
        finally:
            for parameter,enabled in restore.items():
                parameter.requires_grad_(enabled)
    except RequestError:
        raise
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        raise RequestError('neuralese-native-grad-invalid', str(error)) from error


def _native_graph_gradient_impl(engine, body, owner, adapter_leaves, *, actor_bank=None, coefficient_targets=None,
                                evaluation_callback=None,native_state_overrides=None,evaluation_targets=None,
                                evaluation_target_specs=None):
    """Evaluate one actual native graph and return a durable named gradient.

    `target` selects either named native model parameters or one exact imported
    NativeStateBlock. No implicit parameter inventory or lossy block flattening
    is allowed, so callers can pair the result with a declared optimizer/update.
    """
    objective_request = body.get('objective', 'graph-ce')
    if objective_request not in ('graph-ce', None) and not isinstance(objective_request, dict):
        raise RequestError('neuralese-native-grad-objective', 'objective must be graph-ce or a registered native-policy spec binding')
    row = body.get('graph')
    if not isinstance(row, dict):
        raise RequestError('neuralese-native-grad-graph', 'request must include one complete native graph row')
    restore_grad_flags = {}
    try:
        example = parse_graph(row, tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
        if example.split != 'train':
            raise ValueError('native graph parameter learning accepts train rows only')
        graph_sha256 = canonical_json_sha256_hex(row)
        if body.get('graph_sha256') not in (None, graph_sha256):
            raise ValueError('declared graph digest differs from the supplied graph row')
        expected = _identity(engine)
        from ..model.native_adapters import execution_identity, execution_binding
        effective = execution_identity(engine)
        if body.get('native_identity') not in (None, expected):
            raise ValueError('gradient request is bound to another exact live checkpoint')
        objective = None
        query_objective = None
        is_policy = isinstance(objective_request,dict) and objective_request.get('kind')=='native-policy'
        if isinstance(objective_request,dict) and objective_request.get('kind')=='native-query':
            from .native_query_objective import NativeQueryObjective
            from ..model.native_readout import bind_readout
            query_objective = NativeQueryObjective(objective_request,row,example,vocabulary=engine.shared_model.port.output_weight.shape[0],
                resolve_readout=lambda ident:bind_readout(engine,ident))
        elif isinstance(objective_request, dict):
            if objective_request.get('kind') != 'native-policy':
                raise ValueError('only the registered native-policy objective is supported')
            spec_path = objective_request.get('path')
            if not isinstance(spec_path, str):
                raise ValueError('native-policy objective needs its admitted spec path')
            if sha256_file_hex_cached(spec_path) != objective_request.get('sha256'):
                raise ValueError('native-policy objective bytes differ from the admitted SHA-256')
            from ..rl.native_replay import NativePolicyObjective
            objective = NativePolicyObjective(spec_path)
            objective.bind_actor(engine,example=example)
            objective.native_state_modes = body.get('native_state_modes') or {}
        observation_identity = objective.identity if objective is not None else effective
        modes = body.get('native_state_modes')
        if modes is not None and (not isinstance(modes, dict) or any(
                key not in {value['id'] for value in example.external_states.values()} or value != 'same-base-adapter-state'
                for key, value in modes.items())):
            raise ValueError('native_state_modes must declare actual external IDs and same-base-adapter-state')
        imported, leaf_map, state_overrides, input_identities, consumption_receipts = {}, {}, {}, {}, {}
        state_leaf_cache={}
        for name, binding in example.external_states.items():
            record = engine.store.get_native_state(binding['id'])
            if record is None:
                if is_policy:
                    continue  # The registered policy objective resolves its immutable block bundle.
                raise ValueError(f'unknown external native state {binding["id"]}')
            block = record.block
            if block.derivation is not None:
                from ..model.native_state_derivation import verify_state_derivation
                verify_state_derivation(engine.store,block)
            if len(block.token_ids) != binding['token_count']:
                raise ValueError(f'external state token count changed for {name}')
            if actor_bank is not None:
                source_identity={key:getattr(block,key) for key in expected}
                consumers=[node.name for node in example.plan.invocations if any(span.source==name for span in node.imports)]
                for consumer in consumers:
                    if (modes or {}).get(binding['id']) is not None:
                        from ..model.native_adapters import adapter_state_input_receipt
                        receipt=actor_bank.execute(actor_bank.source_names[consumer],lambda:adapter_state_input_receipt(engine,block,binding['id'],target_position_start=0))
                        consumption_receipts[canonical_json_sha256_hex([consumer,name])]={'invocation':consumer,'input_name':name,**receipt}
                    else:
                        block.require_native(**actor_bank.actors[actor_bank.source_names[consumer]][0][1])
                input_identities[name]=source_identity
            elif (modes or {}).get(binding['id']) is not None:
                from ..model.native_adapters import adapter_state_input_receipt
                consumption_receipts[name] = adapter_state_input_receipt(engine, block, binding['id'], target_position_start=0)
                input_identities[name] = consumption_receipts[name]['source_native_identity']
            else:
                block.require_native(**observation_identity)
                input_identities[name] = observation_identity
            if binding['id'] in state_leaf_cache:
                copied,leaves=state_leaf_cache[binding['id']]
            elif native_state_overrides is not None and binding['id'] in native_state_overrides:
                copied=native_state_overrides[binding['id']]
                copied.validate()
                leaves={f'layers.{index}.{key}':value for index,layer in enumerate(copied.layers)
                        for key,value in layer.items() if key!='kind'}
                state_leaf_cache[binding['id']]=(copied,leaves)
            else:
                copied, leaves = _leaf_block(block)
                state_leaf_cache[binding['id']]=(copied,leaves)
            leaf_map.update({f'external.{name}.{key}': value for key, value in leaves.items()})
            if is_policy:
                state_overrides[binding['id']] = copied
            else:
                imported[name] = copied

        target = body.get('target')
        target_groups=None
        if evaluation_target_specs is not None:
            if not isinstance(evaluation_target_specs,dict) or not evaluation_target_specs:
                raise ValueError('joint derivative targets require a named target map')
            target_groups={}
            for alias,declaration in evaluation_target_specs.items():
                if (not isinstance(alias,str) or not alias or not isinstance(declaration,dict)
                        or set(declaration)-{'kind','id','names','tensors'}):
                    raise ValueError('internal joint derivative targets need exact named typed declarations')
                kind=declaration.get('kind');ident=declaration.get('id');requested=declaration.get('names')
                supplied=declaration.get('tensors')
                if supplied is not None:
                    if (kind not in ('adapter','native-state','model-parameters') or not isinstance(supplied,dict)
                            or not supplied or (requested is not None and set(supplied)!=set(requested))):
                        raise ValueError('internal derivative target tensors differ from their exact typed declaration')
                    tensors=supplied
                elif kind=='adapter':
                    available=adapter_leaves.get(ident)
                    if available is None:raise ValueError('adapter gradient target is not an active exact coefficient artifact')
                    names=sorted(available) if requested is None else requested
                    if not isinstance(names,list) or not names or len(set(names))!=len(names) or any(name not in available for name in names):
                        raise ValueError('adapter gradient names must uniquely select active coefficient tensors')
                    tensors={name:available[name] for name in names}
                elif kind=='native-state':
                    matching=[name for name,binding in example.external_states.items() if binding['id']==ident]
                    if not matching:raise ValueError('native-state gradient target must be an imported exact graph state')
                    available={key.removeprefix(f'external.{matching[0]}.'):value for key,value in leaf_map.items()
                        if key.startswith(f'external.{matching[0]}.')}
                    names=sorted(available) if requested is None else requested
                    if not isinstance(names,list) or not names or len(set(names))!=len(names) or any(name not in available for name in names):
                        raise ValueError('native-state gradient names must select consumed named KV or hybrid-state tensors')
                    tensors={name:available[name] for name in names}
                elif kind=='model-parameters':
                    tensors=_parameter_targets(engine.shared_model,requested)
                    for parameter in tensors.values():
                        if parameter not in restore_grad_flags:restore_grad_flags[parameter]=parameter.requires_grad
                        parameter.requires_grad_(True)
                else:raise ValueError('joint native derivative targets support adapter, native-state or model-parameters')
                if not tensors:raise ValueError('joint native derivative target has no tensors')
                if ident is not None and kind in ('adapter','native-state'):
                    if any(item.get('id')==ident for item in target_groups.values()):
                        raise ValueError('one exact native artifact ID must have one joint gradient root')
                target_groups[alias]={'kind':kind,'id':ident,'tensors':tensors}
            target_tensors={(alias,name):tensor for alias,item in target_groups.items()
                for name,tensor in item['tensors'].items()}
            kind='joint'
            source_id=None
            artifact_kind='joint-gradient'
        elif evaluation_targets is not None:
            if evaluation_callback is None or not isinstance(target,dict) or target.get('kind') not in ('model-parameters','adapter','native-state'):
                raise ValueError('internal native derivative targets require an explicit objective evaluator')
            kind=target['kind']
            target_tensors=evaluation_targets
            artifact_kind={'model-parameters':'parameter-gradient','adapter':'adapter-gradient','native-state':'state-gradient'}[target['kind']]
            source_id=target.get('id') if target['kind']!='model-parameters' else None
        else:
            if not isinstance(target,dict):
                raise ValueError('target must identify model parameters or one imported native state')
            kind=target.get('kind')
            if kind == 'model-parameters':
                selected = _parameter_targets(engine.shared_model, target.get('names'))
                # Serving factories freeze the model for generation. Enable only
                # the explicitly named gradient targets during this locked call.
                restore_grad_flags = {parameter: parameter.requires_grad for parameter in selected.values()}
                for parameter in restore_grad_flags:parameter.requires_grad_(True)
                target_tensors = selected
                artifact_kind = 'parameter-gradient'
                source_id = None
            elif kind == 'adapter':
                groups=[]
                for declaration in coefficient_targets if coefficient_targets is not None else [target]:
                    selected_id=declaration.get('id')
                    available=adapter_leaves.get(selected_id)
                    if available is None:
                        raise ValueError('adapter gradient target must be active by its exact native coefficient artifact ID')
                    requested=declaration.get('names',sorted(available))
                    if (not isinstance(requested,list) or not requested or len(set(requested))!=len(requested)
                            or any(name not in available for name in requested)):
                        raise ValueError('adapter gradient names must uniquely select active coefficient tensors')
                    groups.append((selected_id,{name:available[name] for name in requested}))
                if coefficient_targets is None:
                    source_id,target_tensors=groups[0]
                else:
                    # The compound names are local gradient coordinates only;
                    # emitted artifacts retain each original parameter name.
                    source_id=None
                    target_tensors={(index,name):value for index,(_,values) in enumerate(groups) for name,value in values.items()}
                artifact_kind='adapter-gradient'
            elif kind == 'native-state':
                state_id = target.get('id')
                if not isinstance(state_id, str):
                    raise ValueError('native-state target needs its exact stored state ID')
                matching = [name for name, binding in example.external_states.items() if binding['id'] == state_id]
                if not matching:
                    raise ValueError('native-state gradient target must occur as an imported graph state')
                prefix = f'external.{matching[0]}.'
                available = {name.removeprefix(prefix): value for name, value in leaf_map.items() if name.startswith(prefix)}
                requested = target.get('names', sorted(available))
                if (not isinstance(requested, list) or not requested or len(set(requested)) != len(requested)
                        or any(name not in available for name in requested)):
                    raise ValueError('native-state gradient names must select consumed named KV or hybrid-state tensors')
                target_tensors = {name: available[name] for name in requested}
                artifact_kind = 'state-gradient'
                source_id = state_id
            else:
                raise ValueError('target.kind must be model-parameters, native-state or adapter')
        if not target_tensors:
            raise ValueError('gradient target has no tensors')
        if objective is not None:
            from ..train.native_state_sft import packed_graph_loss
            objective.external_state_overrides = state_overrides
            loss, stats = packed_graph_loss(engine.shared_model, [example], objective=objective)
        else:
            loss, stats = graph_loss(engine.shared_model, example, external_blocks=imported,
                                     external_identity=input_identities,objective=query_objective)
        def provenance_for(names):
            recorded=body.get('recorded_objective_request')
            recorded_json=canonical_json_bytes(recorded).decode() if isinstance(recorded,dict) else None
            import hashlib
            return {'objective': (query_objective.spec['schema'].removeprefix('natlang.') if query_objective is not None else
                'native-graph-token-ce/1' if objective is None else objective.spec['schema'].removeprefix('natlang.')),
            **({'recorded_objective_request_json':recorded_json,
                'recorded_objective_request_sha256':hashlib.sha256(recorded_json.encode()).hexdigest()}
                if recorded_json is not None else {}),
            **({'query_objective_sha256':canonical_json_sha256_hex(query_objective.spec),
                'query_objective_json':canonical_json_bytes(query_objective.spec).decode()}
                if query_objective is not None else {}), 'graph_id': example.id,
            'graph_sha256': graph_sha256, 'graph_json':canonical_json_bytes(row).decode(), 'target_kind': kind,
            'external_source_identities': input_identities,
            'external_adapter_state_contracts': {name: {key: value for key, value in receipt.items()
            if key not in {'target_position_start', 'tokens'}} for name, receipt in {**consumption_receipts, **(objective.evaluation_external_contracts if objective is not None else {})}.items()},
            **({'external_consumption_receipts':objective.evaluation_external_contracts}
            if objective is not None and objective.behavior_policy is not None else {}),
            'target_names': sorted(names),
            **({'invocation_actors':actor_bank.receipt(),'actor_graph_sha256':canonical_json_sha256_hex(actor_bank.receipt()),'invocation_source_names':actor_bank.source_names}
            if actor_bank is not None else {'effective_native_identity':effective}),
            'adapter_gradient_precision': body.get('adapter_gradient_precision', 'storage'),
            **({'invocation_adapter_declarations':body['invocation_adapters'],'default_adapters':body.get('adapters') or []}
            if actor_bank is not None else {'adapters':body.get('adapters') or []}),
            **({'adapter_binding': {'schema': 'natlang.native-adapter/1', 'base_native_identity': expected,
            'adapters': execution_binding(engine)}}
            if body.get('adapters') and actor_bank is None else {}),
            'loss': float(loss.detach()), 'stats': stats,
            **({'objective_spec_sha256':objective_request['sha256'],
            **({'behavior_policy':objective.behavior_policy,'behavior_policy_sha256':objective.spec['behavior_policy_sha256']}
            if objective.behavior_policy is not None else {'behavior_native_identity':objective.identity,'behavior_adapter_binding':objective.adapter_binding})}
            if objective is not None else {})}
        if evaluation_callback is not None:
            return evaluation_callback(loss,target_groups if target_groups is not None else target_tensors,{'stats':stats,'kind':kind,
                'artifact_kind':artifact_kind,'source_artifact_id':source_id,
                'target_groups':target_groups,
                'graph_sha256':graph_sha256,'graph_id':example.id,'provenance':provenance_for(target_tensors)})
        if target_groups is not None:
            flat=list(target_tensors.items())
            grads=torch.autograd.grad(loss,tuple(value for _,value in flat),allow_unused=True)
            result={}
            for ((alias,name),_),gradient in zip(flat,grads):
                if gradient is None:raise ValueError(f'objective has no gradient path to requested tensor {alias}:{name}')
                group=target_groups[alias];kind=group['kind'];ident=group['id']
                artifact_kind={'model-parameters':'parameter-gradient','adapter':'adapter-gradient','native-state':'state-gradient'}[kind]
                provenance={**provenance_for({name:group['tensors'][name]}),'target_kind':kind,'target_names':sorted(group['tensors'])}
                values=result.setdefault(alias,{})
                values[name]=gradient.detach()
            artifacts={}
            for alias,group in target_groups.items():
                kind=group['kind'];ident=group['id'];tensors=result[alias]
                artifact=NativeTensorArtifact({'model-parameters':'parameter-gradient','adapter':'adapter-gradient','native-state':'state-gradient'}[kind],
                    expected,tensors,ident if kind!='model-parameters' else None,
                    {**provenance_for(tensors),'target_kind':kind,'target_names':sorted(tensors),
                     **({'adapter_binding':{'schema':'natlang.native-adapter/1','base_native_identity':expected,
                        'adapters':execution_binding(engine)}} if kind=='adapter' and body.get('adapters') and actor_bank is None else {})}).validate()
                artifacts[alias]=engine.store.put_native_tensor_artifact(artifact,owner).meta()
            return {'loss':float(loss.detach()),'stats':stats,'gradients':artifacts,'native_identity':expected,
                'objective':query_objective.spec['schema'] if query_objective is not None else 'native-graph-token-ce/1'}
        grads = torch.autograd.grad(loss, tuple(target_tensors.values()), allow_unused=True)
        tensors = {name: grad.detach() for name, grad in zip(target_tensors, grads) if grad is not None}
        if set(tensors) != set(target_tensors):
            absent = sorted(set(target_tensors) - set(tensors))
            raise ValueError(f'objective has no gradient path to requested tensors: {absent}')
        artifacts=[]
        selections=([(selected_id,{name:tensors[(index,name)] for name in values})
                    for index,(selected_id,values) in enumerate(groups)]
                    if coefficient_targets is not None else [(source_id,tensors)])
        for selected_id,values in selections:
            artifact=NativeTensorArtifact(artifact_kind,expected,values,selected_id,
                provenance_for(values)).validate()
            artifacts.append(engine.store.put_native_tensor_artifact(artifact,owner).meta())
        return {'loss':float(loss.detach()),'stats':stats,
                **({'gradients':artifacts} if coefficient_targets is not None else {'gradient':artifacts[0]}),
                'native_identity':expected,'objective':artifact.provenance['objective'],
                **({'invocation_actors':actor_bank.receipt(),'actor_graph_sha256':canonical_json_sha256_hex(actor_bank.receipt())}
                   if actor_bank is not None else {'effective_native_identity':effective})}
    except RequestError:
        raise
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        raise RequestError('neuralese-native-grad-invalid', str(error)) from error
    finally:
        for parameter, enabled in restore_grad_flags.items():
            parameter.requires_grad_(enabled)
