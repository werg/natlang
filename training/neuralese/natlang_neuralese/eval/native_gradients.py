"""Exact complete derivative comparison with CPU reference storage."""
from __future__ import annotations
import math
import torch
from ..common.hashing import canonical_json_sha256_hex

SCHEMA='natlang.native-complete-communication-gradients/1'


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

    autograd.grad never accumulates parameter.grad or invokes post-accumulate
    optimizer hooks. Cached partial derivatives/activations still need memory;
    the recorded bytes describe retained comparison tensors, not a peak claim.
    Exact CPU observer pairs and metrics avoid retaining a second full gradient
    inventory on the GPU beside the actual cached VJP and restored optimizer.
    """
    if any(type(value)not in(float,int)or not math.isfinite(value)or value<0 for value in(atol,rtol)):
        raise ValueError('complete gradient comparison needs finite nonnegative gates')
    named=list(named);names=['input_embeddings',*(name for name,_ in named)]
    parameters=[value for _,value in named]
    if (len(names)!=len(set(names))or len(parameters)!=len({id(p)for p in parameters})
            or any(not isinstance(name,str)or not name for name,_ in named)
            or any(not value.requires_grad for value in parameters)):
        raise ValueError('complete gradient inventory needs unique trainable names and parameters')
    if gradient_observer is not None and callable(getattr(gradient_observer,'begin_vjp_observation',None)):
        gradient_observer.begin_vjp_observation()
    old_grads={id(p):(p.grad,None if p.grad is None else p.grad._version)for p in parameters}
    flat_values=(flat_inputs,*parameters)
    try:
        flat=torch.autograd.grad(flat_loss,flat_values,allow_unused=True)
        if flat_cotangent_scope is not None:
            flat_cotangent_scope.finish()
    finally:
        if flat_cotangent_scope is not None:
            flat_cotangent_scope.close()
    waiting={name:None if value is None else value.detach().to('cpu')
             for name,value in zip(names,flat)}
    del flat
    cached_cotangent_scope=None
    if cached_factory is not None:
        produced=cached_factory()
        if len(produced)==3:
            cached_loss,cached_inputs,cached_cotangent_scope=produced
        else:
            cached_loss,cached_inputs=produced
    cached_values=(cached_inputs,*parameters)
    retained=sum(g.numel()*g.element_size()for g in waiting.values()if g is not None)
    remaining=retained;max_pair=0;comparisons={};hooks=[]
    def hook(name):
        def compare(value):
            nonlocal remaining,max_pair
            previous=waiting.pop(name)
            current=value.detach().to('cpu')
            if gradient_observer is not None:gradient_observer(name,previous,current)
            if previous is None:comparisons[name]={'passed':False,'reason':'cached-only derivative'}
            else:
                count=previous.numel()*previous.element_size();remaining-=count
                max_pair=max(max_pair,count+value.numel()*value.element_size())
                comparisons[name]=gradient_metrics(current,previous,atol=atol,rtol=rtol)
            return value
        return compare
    try:
        for name,value in zip(names,cached_values):hooks.append(value.register_hook(hook(name)))
        current=torch.autograd.grad(cached_loss,cached_values,allow_unused=True)
        if cached_cotangent_scope is not None:
            cached_cotangent_scope.finish()
        for name,value in zip(names,current):
            if name in waiting:
                previous=waiting.pop(name)
                if gradient_observer is not None:
                    gradient_observer(name,previous,None if value is None else value.detach().to('cpu'))
                comparisons[name]=({'passed':True,'unused':True}if previous is None and value is None else
                    {'passed':False,'reason':'flat-only or missing cached derivative'})
        del current
    except BaseException:
        if gradient_observer is not None and callable(getattr(gradient_observer,'abort',None)):
            gradient_observer.abort()
        raise
    finally:
        for item in hooks:item.remove()
        waiting.clear()
        if cached_cotangent_scope is not None:
            cached_cotangent_scope.close()
    if gradient_observer is not None and callable(getattr(gradient_observer,'finish',None)):
        gradient_observer.finish()
    preserved=all(p.grad is old_grads[id(p)][0]and(p.grad is None or p.grad._version==old_grads[id(p)][1])for p in parameters)
    inventory=[{'name':name,'shape':list(value.shape),'dtype':str(value.dtype)}for name,value in named]
    return {'schema':SCHEMA,'passed':preserved and all(row['passed']for row in comparisons.values()),
        'inventory':inventory,'inventory_sha256':canonical_json_sha256_hex(inventory),
        'comparisons':comparisons,'parameter_grads_preserved':preserved,
        'retained_flat_gradient_bytes':retained,'largest_live_comparison_pair_bytes':max_pair,
        'retained_flat_gradient_device':'cpu','retained_flat_gradient_cuda_bytes':0,
        'memory_scope':'CPU reference and pair tensors only; cached VJP gradients and activations additional',
        'optimizer_updates':0,'atol':atol,'rtol':rtol}


def production_inventory(model,named):
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
