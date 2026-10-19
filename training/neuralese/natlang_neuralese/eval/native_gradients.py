"""Exact complete derivative comparison with CPU reference storage."""
from __future__ import annotations
import math
import torch
from ..common.hashing import canonical_json_sha256_hex

SCHEMA='natlang.native-complete-communication-gradients/1'


def _existing_grad(tensor):
    # Reading .grad on an ordinary nonleaf emits a warning, even though it is
    # necessarily empty. Retained nonleaf grads remain part of the preservation
    # contract.
    if not tensor.is_leaf and not tensor.retains_grad:
        return None
    return tensor.grad


def gradient_metrics(actual,expected,*,atol,rtol):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
        return {'passed':False,'reason':'gradient geometry or dtype differs'}
    a,b=actual.detach().float(),expected.detach().float()
    difference=(a-b).abs();finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    an,bn=torch.linalg.vector_norm(a,dtype=torch.float64),torch.linalg.vector_norm(b,dtype=torch.float64)
    relative=float(torch.linalg.vector_norm(a-b,dtype=torch.float64)/bn.clamp_min(1e-30))
    cosine=float((a*b).sum(dtype=torch.float64)/(an*bn)) if an>0 and bn>0 else float(an==bn)
    bad=int((difference>atol+rtol*b.abs()).sum())
    return {'passed':finite and bad==0 and relative<=rtol and cosine>=1-rtol,
        'finite':finite,'max_abs':float(difference.max()) if difference.numel()else 0.,
        'shape':list(actual.shape),'dtype':str(actual.dtype),
        'relative_l2':relative,'cosine':cosine,'reference_l2':float(bn),
        'actual_l2':float(an),'mismatched_elements':bad,'elements':a.numel(),'atol':atol,'rtol':rtol}


def _backward_gradients_to_cpu(loss, names, parameters, input_tensor=None, *, on_gradient=None):
    """Run one backward and drain completed leaf gradients to CPU immediately.

    ``autograd.grad`` returns a tuple that keeps every parameter derivative on
    the accelerator until the whole backward has completed. Post-accumulate
    hooks instead transfer one completed leaf at a time and clear the temporary
    ``.grad`` before autograd reaches the next leaf. Existing parameter grad
    objects are borrowed only for the duration of backward and restored exactly.
    """
    names = tuple(names)
    parameters = tuple(parameters)
    if len(names) != len(parameters) or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError('streamed gradient targets need unique names and parameters')
    if any(not isinstance(p, torch.Tensor) or not p.requires_grad or not p.is_leaf for p in parameters):
        raise ValueError('streamed parameter gradient targets must be distinct trainable leaves')
    if input_tensor is not None and (not isinstance(input_tensor, torch.Tensor) or not input_tensor.requires_grad):
        raise ValueError('streamed explicit input target must require gradients')

    if on_gradient is not None and not callable(on_gradient):
        raise TypeError('streamed gradient observer must be callable')
    result = {name: None for name in names} if on_gradient is None else {}
    if input_tensor is not None:
        if on_gradient is None:
            result['input_embeddings'] = None
    saved = [(parameter, parameter.grad,
              None if parameter.grad is None else parameter.grad._version)
             for parameter in parameters]
    previous_input_grad = None if input_tensor is None else _existing_grad(input_tensor)
    saved_input = (previous_input_grad,
                   None if previous_input_grad is None else previous_input_grad._version)
    hooks = []
    try:
        for parameter, _, _ in saved:
            parameter.grad = None
        if input_tensor is not None:
            input_tensor.grad = None

        if input_tensor is not None:
            def capture_input(gradient):
                value = gradient.detach().to(device='cpu')
                if on_gradient is None:
                    result['input_embeddings'] = value
                else:
                    on_gradient('input_embeddings', value)
                return gradient
            hooks.append(input_tensor.register_hook(capture_input))

        for name, parameter in zip(names, parameters):
            def capture_parameter(_leaf, *, name=name, parameter=parameter):
                gradient = parameter.grad
                if gradient is None:
                    raise RuntimeError(f'autograd completed {name} without a leaf gradient')
                value = gradient.detach().to(device='cpu')
                if on_gradient is None:
                    result[name] = value
                else:
                    on_gradient(name, value)
                parameter.grad = None
            hooks.append(parameter.register_post_accumulate_grad_hook(capture_parameter))

        targets=parameters+((input_tensor,) if input_tensor is not None else ())
        torch.autograd.backward(loss, inputs=targets)
    finally:
        for hook in hooks:
            hook.remove()
        for parameter, previous, version in saved:
            parameter.grad = previous
            if parameter.grad is not previous or (previous is not None and previous._version != version):
                raise RuntimeError('streamed complete-gradient comparison changed a prior parameter .grad')
        if input_tensor is not None:
            previous, version = saved_input
            input_tensor.grad = previous
            if input_tensor.grad is not previous or (previous is not None and previous._version != version):
                raise RuntimeError('streamed complete-gradient comparison changed a prior input .grad')
    return result


def compare_complete_gradients(flat_loss,cached_loss,named,flat_inputs,cached_inputs,*,atol,rtol,cached_factory=None,gradient_observer=None,
                               flat_cotangent_scope=None):
    try:
        return _compare_complete_gradients(flat_loss,cached_loss,named,flat_inputs,cached_inputs,
            atol=atol,rtol=rtol,cached_factory=cached_factory,gradient_observer=gradient_observer,
            flat_cotangent_scope=flat_cotangent_scope)
    finally:
        # The caller entered this scope before its forward. Validation and
        # observer setup can fail before the VJP's own cleanup block begins.
        if flat_cotangent_scope is not None:
            flat_cotangent_scope.close()


def _compare_complete_gradients(flat_loss,cached_loss,named,flat_inputs,cached_inputs,*,atol,rtol,cached_factory=None,gradient_observer=None,
                                flat_cotangent_scope=None):
    """Release each CPU flat derivative as its complete cached counterpart arrives.

    A single backward drains complete leaf derivatives as they arrive. Any
    pre-existing post-accumulate hooks also run; production qualification calls
    this outside the optimizer's gated in-backward update context, so optimizer
    hooks are disarmed while finite-gradient checks remain active. Cached
    activations still need memory; the recorded bytes describe retained
    comparison tensors, not a peak claim.
    Exact CPU observer pairs and metrics avoid retaining a second full gradient
    inventory on the GPU beside the actual cached VJP and restored optimizer.
    """
    if any(type(value)not in(float,int)or not math.isfinite(value)or value<0 for value in(atol,rtol)):
        raise ValueError('complete gradient comparison needs finite nonnegative gates')
    named=list(named)
    has_inputs=flat_inputs is not None
    parameter_names=[name for name,_ in named]
    names=(["input_embeddings"] if has_inputs else [])+parameter_names
    parameters=[value for _,value in named]
    if (len(names)!=len(set(names))or len(parameters)!=len({id(p)for p in parameters})
            or any(not isinstance(name,str)or not name for name,_ in named)
            or any(not value.requires_grad for value in parameters)):
        raise ValueError('complete gradient inventory needs unique trainable names and parameters')
    if gradient_observer is not None and callable(getattr(gradient_observer,'begin_vjp_observation',None)):
        gradient_observer.begin_vjp_observation()
    if (flat_inputs is None) != (cached_inputs is None) and cached_factory is None:
        raise ValueError('complete gradient comparison must agree on explicit input presence')
    old_grads={id(p):(p.grad,None if p.grad is None else p.grad._version)for p in parameters}
    old_input_value=None if flat_inputs is None else _existing_grad(flat_inputs)
    old_input_grad=(old_input_value,None if old_input_value is None else old_input_value._version)
    cached_input_grad=None
    try:
        flat=_backward_gradients_to_cpu(flat_loss,
            (name for name,_ in named),parameters,flat_inputs if has_inputs else None)
        if flat_cotangent_scope is not None:
            flat_cotangent_scope.finish()
    finally:
        if flat_cotangent_scope is not None:
            flat_cotangent_scope.close()
    waiting=flat
    del flat
    cached_cotangent_scope=None
    if cached_factory is not None:
        produced=cached_factory()
        if len(produced)==3:
            cached_loss,cached_inputs,cached_cotangent_scope=produced
        else:
            cached_loss,cached_inputs=produced
    cached_input_value=None if cached_inputs is None else _existing_grad(cached_inputs)
    cached_input_grad=(cached_input_value,None if cached_input_value is None else cached_input_value._version)
    if (cached_inputs is not None) != has_inputs:
        if cached_cotangent_scope is not None:cached_cotangent_scope.close()
        raise ValueError('complete gradient comparison changed explicit input presence')
    retained=sum(g.numel()*g.element_size()for g in waiting.values()if g is not None)
    max_pair=0;comparisons={}
    try:
        seen=set()
        def compare_one(name, value):
            nonlocal max_pair
            if name not in waiting or name in seen:
                raise RuntimeError('cached backward emitted an unknown or duplicate gradient target: '+name)
            seen.add(name)
            previous=waiting.pop(name)
            if gradient_observer is not None:gradient_observer(name,previous,value)
            if previous is None and value is None:
                comparisons[name]={'passed':True,'unused':True}
            elif previous is None:
                comparisons[name]={'passed':False,'reason':'cached-only derivative'}
            elif value is None:
                comparisons[name]={'passed':False,'reason':'flat-only or missing cached derivative'}
            else:
                count=previous.numel()*previous.element_size()
                max_pair=max(max_pair,count+value.numel()*value.element_size())
                comparisons[name]=gradient_metrics(value,previous,atol=atol,rtol=rtol)
        _backward_gradients_to_cpu(cached_loss, parameter_names, parameters,
            cached_inputs if has_inputs else None, on_gradient=compare_one)
        if cached_cotangent_scope is not None:
            cached_cotangent_scope.finish()
        for name in tuple(waiting):
            compare_one(name, None)
    except BaseException:
        if gradient_observer is not None and callable(getattr(gradient_observer,'abort',None)):
            gradient_observer.abort()
        raise
    finally:
        waiting.clear()
        if cached_cotangent_scope is not None:
            cached_cotangent_scope.close()
    if gradient_observer is not None and callable(getattr(gradient_observer,'finish',None)):
        gradient_observer.finish()
    preserved=(all(p.grad is old_grads[id(p)][0]and(p.grad is None or p.grad._version==old_grads[id(p)][1])for p in parameters)
        and (flat_inputs is None or (_existing_grad(flat_inputs) is old_input_grad[0]
            and (old_input_grad[0] is None or old_input_grad[0]._version==old_input_grad[1])))
        and (cached_inputs is None or (_existing_grad(cached_inputs) is cached_input_grad[0]
            and (cached_input_grad[0] is None or cached_input_grad[0]._version==cached_input_grad[1]))) )
    inventory=[{'name':name,'shape':list(value.shape),'dtype':str(value.dtype)}for name,value in named]
    return {'schema':SCHEMA,'passed':preserved and all(row['passed']for row in comparisons.values()),
        'inventory':inventory,'inventory_sha256':canonical_json_sha256_hex(inventory),
        'trainable_parameter_count':len(inventory),'derivative_comparison_count':len(comparisons),
        'comparisons':comparisons,'parameter_grads_preserved':preserved,
        'retained_flat_gradient_bytes':retained,'largest_live_comparison_pair_bytes':max_pair,
        'retained_flat_gradient_device':'cpu','retained_flat_gradient_cuda_bytes':0,
        'cached_gradient_collection':'single backward; each completed leaf gradient copied to CPU, compared, then released',
        'retained_cached_gradient_map_bytes':0,
        'retained_cached_gradient_device':'none',
        'memory_scope':'the utility retains the flat CPU reference map and no cached result map; an external gradient observer may retain its own CPU snapshots, while live per-leaf backward gradients and saved activations remain additional',
        'optimizer_updates':0,'atol':atol,'rtol':rtol}


def production_inventory(model,named):
    """Select the exact declared cohort; input_embeddings is a separate comparison."""
    named=list(named)
    if (len(named)!=len({name for name,_ in named})or len(named)!=len({id(p)for _,p in named})
            or any(not isinstance(name,str)or not name for name,_ in named)):
        raise ValueError('complete gradient inventory needs unique trainable names and parameters')
    actual={id(p)for p in model.parameters()if p.requires_grad}
    if actual!={id(p)for _,p in named}:
        raise ValueError('complete qualification omitted or added a declared model trainable')
    roles={'backbone':{id(p)for layer in model.port.layers for p in layer.parameters()if p.requires_grad},
        'adapter':{id(p)for p in model.adapter.parameters()if p.requires_grad},
        'learned_input':{id(getattr(model,key))for key in('input_rows','extra_input_rows')if hasattr(model,key)},
        'learned_output':{id(getattr(model,key))for key in(('input_rows','extra_input_rows')if model.tied else('output_rows','extra_output_rows'))if hasattr(model,key)}}
    return named,{role:[name for name,p in named if id(p)in ids]for role,ids in roles.items()}


def validate_complete_proof(proof,identity,*,gradient_atol,execution_profile=None):
    if not isinstance(proof,dict)or proof.get('schema')!=SCHEMA or proof.get('passed')is not True or proof.get('native_identity')!=identity:
        raise ValueError('native readiness lacks complete communication gradient qualification')
    inventory=proof.get('inventory')
    if (not isinstance(inventory,list)or not inventory
            or any(not isinstance(row,dict)or not isinstance(row.get('name'),str)or not row['name']
                or not isinstance(row.get('shape'),list)or any(type(n)is not int or n<0 for n in row['shape'])
                or not isinstance(row.get('dtype'),str)for row in inventory)
            or len({row['name']for row in inventory})!=len(inventory)
            or proof.get('inventory_sha256')!=canonical_json_sha256_hex(inventory)
            or proof.get('parameter_grads_preserved')is not True or proof.get('optimizer_updates')!=0
            or not 0<=proof.get('atol',float('inf'))<=gradient_atol or not 0<proof.get('rtol',float('inf'))<=1e-4):
        raise ValueError('complete communication gradient inventory or gates are invalid')
    comparisons=proof.get('comparisons')or{}
    for key,count in (('trainable_parameter_count',len(inventory)),
                      ('derivative_comparison_count',len(inventory)+1)):
        if key in proof and (type(proof[key])is not int or proof[key]!=count):
            raise ValueError('complete communication gradient count differs from its exact inventory')
    if set(comparisons)!={'input_embeddings',*(row['name']for row in inventory)}or any(row.get('passed')is not True for row in comparisons.values()):
        raise ValueError('complete communication gradient coverage is incomplete')
    roles=proof.get('roles')or{}
    for role in('backbone','adapter','learned_input','learned_output'):
        if not roles.get(role)or any(name not in comparisons for name in roles[role])or not any(comparisons[name].get('reference_l2',0)>0 for name in roles[role]):
            raise ValueError('complete communication gradient role is unexercised: '+role)
    if comparisons['input_embeddings'].get('reference_l2',0)<=0 or not isinstance(proof.get('case'),dict)or proof.get('case_sha256')!=canonical_json_sha256_hex(proof['case']):
        raise ValueError('complete communication gradient input or case is unexercised')
    if proof.get('model_config_sha256')!=canonical_json_sha256_hex(proof.get('model_config')):
        raise ValueError('complete communication gradient model profile changed')
    if execution_profile is not None:
        backend=proof.get('training_backend_execution')or{};configuration=backend.get('configuration')or{}
        fp8=execution_profile.get('fp8')or{};kind=fp8.get('kind')
        if execution_profile.get('model_config_sha256')!=proof['model_config_sha256']or backend.get('passed')is not True:
            raise ValueError('complete communication gradient execution profile changed')
        if kind=='off':
            if configuration.get('enabled')is not False:raise ValueError('complete gradients used a different backend')
        else:
            from ..train.fp8_backend import BACKEND_VERSION,EXPERT_BACKEND_VERSION,expert_input_gradient_precision
            expected={'te-fp8':BACKEND_VERSION,'triton-experts':EXPERT_BACKEND_VERSION}.get(kind)
            counters=backend.get('executed_projections')or{}
            if (expected is None or configuration.get('enabled')is not True or configuration.get('backend')!=expected
                    or any(type(counters.get(key))is not int or counters[key]<=0 for key in('calls','rows'))
                    or(kind=='triton-experts'and configuration.get('input_gradient_precision')!=expert_input_gradient_precision(fp8.get('recipe')))
                    or(kind=='te-fp8'and any(configuration.get(key)!=fp8.get(key)for key in('recipe','amax_history_len')))):
                raise ValueError('complete gradients did not execute their declared FP8 backend')
    contracts={row['name']:row for row in inventory}
    for name,row in comparisons.items():
        if row.get('unused')is True:continue
        numbers=('max_abs','relative_l2','cosine','reference_l2','actual_l2','atol','rtol')
        if (any(type(row.get(key))not in(float,int)or not math.isfinite(row[key])for key in numbers)
                or row.get('finite')is not True or row.get('mismatched_elements')!=0
                or not 0<=row['atol']<=proof['atol'] or not 0<row['rtol']<=proof['rtol']
                or not 0<=row['relative_l2']<=row['rtol']or row['cosine']<1-row['rtol']
                or min(row['actual_l2'],row['reference_l2'],row['max_abs'])<0
                or(name in contracts and any(row.get(key)!=contracts[name][key]for key in('shape','dtype')))):
            raise ValueError('complete communication gradient comparison exceeds its exact gates: '+name)
