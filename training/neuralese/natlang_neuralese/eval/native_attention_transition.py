"""Complete restored-model attention comparisons without optimizer updates."""
from __future__ import annotations
import contextlib
import copy
import math
import time

import torch

from ..common.hashing import canonical_json_sha256_hex
from ..model.native_precision import set_native_attention_precision
from ..model.native_graph_plan import Invocation, ImportSpan, compile_graph
from ..train.loop import capture_training_rng_state, restore_training_rng_state
from ..train.optim_restore import _children
from .native_gradients import production_inventory, compare_complete_gradients,validate_complete_proof
from .native_memory import _optimizer_versions, CapacityStateChangedError
from .native_state import qualify_existing


def transition_equivalence(report):
    """Judge arithmetic continuation separately from unchanged readiness debt.

    Cross-mode derivatives must pass the existing complete proof verifier.
    Existing cached errors may persist, but no named failure can be added or
    worsen beyond the same declared absolute/relative numerical gates.
    """
    failures=[]
    execution_setting=report.get('execution_setting','native_attention_precision')
    if execution_setting not in('native_attention_precision','ffn_chunk_tokens'):
        return {'passed':False,'failures':['unknown execution transition setting'],'readiness_qualified':False}
    def require(condition,reason):
        if not condition:failures.append(reason)
    def finite(value):return type(value)in(int,float)and math.isfinite(value)
    def finite_config(value):
        if isinstance(value,dict):return all(finite_config(v)for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite_config(v)for v in value)
        return math.isfinite(value)if type(value)in(int,float)else True
    baseline=report.get('baseline_cached_reference')or{};candidate=report.get('candidate_cached_reference')or{}
    old=baseline.get('complete_gradient_qualification')or{};new=candidate.get('complete_gradient_qualification')or{}
    atol=report.get('gradient_atol');rtol=report.get('gradient_rtol');forward_atol=report.get('forward_atol')
    require(all(finite(v)and v>0 for v in(atol,rtol,forward_atol)),'invalid declared transition gates')
    if failures:return {'passed':False,'failures':failures,'readiness_qualified':False}
    require(report.get('source_state_preserved')is True,'source state was not preserved')
    require(report.get('cross_mode_passed')is True,'complete cross-mode comparison failed')
    require(report.get('optimizer_updates')==0,'transition performed optimizer updates')
    if execution_setting=='ffn_chunk_tokens':
        counts=[report.get(key)for key in('previous_mode','candidate_mode')]
        valid_counts=all(type(value)is int and value>=0 for value in counts)
        require(valid_counts,'invalid FFN execution counts')
        nodes=(report.get('case')or{}).get('invocations')or[]
        require(valid_counts and bool(nodes)and len(nodes[0].get('token_ids',[]))>max(counts),
                'FFN reference did not cross the selected chunk boundary')
    expected_case=report.get('case_sha256')
    require(expected_case==canonical_json_sha256_hex(report.get('case')),'transition case digest differs')
    require(old.get('case_sha256')==new.get('case_sha256')==expected_case,'cached case differs')
    require(old.get('inventory')==new.get('inventory')and bool(old.get('inventory')),'cached trainable inventory differs')
    require(old.get('roles')==new.get('roles')==report.get('roles'),'cached gradient roles differ')
    unused={name for name,row in(old.get('comparisons')or{}).items()if row.get('unused')is True}
    require(unused=={name for name,row in(new.get('comparisons')or{}).items()if row.get('unused')is True},
            'cached unused derivative set changed')
    for execution in('graph','cached'):
        cross=(report.get('cross_mode')or{}).get(execution)or{}
        require(cross.get('inventory')==old.get('inventory'),'cross-mode inventory differs: '+execution)
        require(all((cross.get('backend_execution')or{}).get(mode,{}).get('passed')is True
                    for mode in('old','candidate')),'cross-mode backend did not execute: '+execution)
        require(all(finite(cross.get(key))and 0<=cross[key]<=forward_atol
                    for key in('forward_max_abs','consumer_ce_abs')),'cross-mode forward/loss failed: '+execution)
        require(unused=={name for name,row in(cross.get('comparisons')or{}).items()if row.get('unused')is True},
                'cross-mode unused derivative set changed: '+execution)
        backend=cross.get('backend_execution')or{}
        settings=[(backend.get(mode)or{}).get('configuration')for mode in('old','candidate')]
        require(settings[0]==settings[1]and all(isinstance(cfg,dict)and finite_config(cfg)for cfg in settings),
                'cross-mode backend configuration changed: '+execution)
        try:
            # The cached baseline supplies identity/case/role annotations only;
            # actual cross-mode metrics, inventory and preservation flags win.
            validate_complete_proof({**old,**cross},old.get('native_identity'),gradient_atol=atol)
        except (ValueError,TypeError,KeyError):failures.append('complete cross-mode derivatives failed: '+execution)
    for label,value in(('baseline',baseline),('candidate',candidate)):
        complete=value.get('complete_gradient_qualification')or{}
        cfg=complete.get('model_config')or{}
        expected_mode=report.get('previous_mode'if label=='baseline'else'candidate_mode')
        require(cfg.get(execution_setting,0 if execution_setting=='ffn_chunk_tokens'else'ieee')==expected_mode,'cached arithmetic identity differs: '+label)
        require(complete.get('model_config_sha256')==canonical_json_sha256_hex(cfg),'cached model profile digest differs: '+label)
        require(complete.get('inventory_sha256')==canonical_json_sha256_hex(complete.get('inventory')),
                'cached inventory digest differs: '+label)
        require(value.get('checkpoint_sha256')==report.get('parent_checkpoint_sha256')
                and(complete.get('native_identity')or{}).get('checkpoint_sha256')==report.get('parent_checkpoint_sha256'),
                'cached parent identity differs: '+label)
        require(value.get('channel_restore_exact')is True,'cached channel restore failed: '+label)
        require(complete.get('parameter_grads_preserved')is True and complete.get('optimizer_updates')==0,
                'cached derivative state was not preserved: '+label)
        require((complete.get('training_backend_execution')or{}).get('passed')is True,
                'cached backend did not execute: '+label)
        if 'training_backend_execution'in value:
            require(value['training_backend_execution'].get('passed')is True,'runtime backend did not execute: '+label)
    configs=[dict(value.get('model_config')or{})for value in(old,new)]
    for cfg in configs:cfg.pop(execution_setting,None)
    require(configs[0]==configs[1],'transition changed another model setting')
    require(baseline.get('passed')is not True or candidate.get('passed')is True,'new failed cached runtime proof')
    require(old.get('passed')is not True or new.get('passed')is True,'new failed complete cached proof')
    require(old.get('error')==new.get('error'),'new cached proof validation error')
    backend_configs=[(proof.get('training_backend_execution')or{}).get('configuration')for proof in(old,new)]
    require(backend_configs[0]==backend_configs[1]
            and all(isinstance(cfg,dict)and finite_config(cfg)for cfg in backend_configs),
            'cached backend configuration changed')
    expected={'input_embeddings',*(row['name']for row in old.get('inventory',[]))}
    before=old.get('comparisons')or{};after=new.get('comparisons')or{}
    require(set(before)==set(after)==expected,'cached named derivative coverage differs')
    for name in sorted(expected):
        a=before.get(name)or{};b=after.get(name)or{}
        if a.get('unused')is True or b.get('unused')is True:
            require(a.get('unused')is True and b.get('unused')is True
                    and a.get('passed')is True and b.get('passed')is True,
                    'cached derivative became missing/unused: '+name)
            continue
        numbers=('max_abs','relative_l2','cosine','reference_l2','actual_l2','atol','rtol')
        valid=(all(finite(row.get(key))for row in(a,b)for key in numbers)
               and all(row.get('finite')is True and 0<=row['atol']<=atol and 0<row['rtol']<=rtol
                       and min(row['max_abs'],row['reference_l2'],row['actual_l2'])>=0 for row in(a,b)))
        require(valid,'nonfinite or invalid cached derivative metrics: '+name)
        if not valid:continue
        require(all(a.get(key)==b.get(key)for key in('shape','dtype','elements')),'cached derivative geometry differs: '+name)
        require(a.get('passed')is not True or b.get('passed')is True,'new failed cached derivative: '+name)
        require(b['max_abs']<=a['max_abs']+atol+rtol*abs(a['max_abs'])
                and b['relative_l2']<=a['relative_l2']+rtol and b['cosine']>=a['cosine']-rtol,
                'worse cached derivative error: '+name)
        require(type(a.get('mismatched_elements'))is int and type(b.get('mismatched_elements'))is int
                and 0<=b['mismatched_elements']<=a['mismatched_elements'],'more failed cached coordinates: '+name)
    for key,gate in(('forward_max_abs',forward_atol),('incremental_max_abs',forward_atol),
                     ('consumer_ce_abs',forward_atol),('consumer_gradient_max_abs',atol)):
        a=baseline.get(key);b=candidate.get(key)
        require(finite(a)and finite(b)and min(a,b)>=0,'nonfinite cached runtime metric: '+key)
        if finite(a)and finite(b):
            require((a>gate or b<=gate)and b<=a+gate,'new/worse cached runtime failure: '+key)
    return {'schema':('natlang.native-ffn-chunk-transition-equivalence/1'if execution_setting=='ffn_chunk_tokens'else'natlang.native-attention-transition-equivalence/1'),'passed':not failures,
            'failures':failures,'inherited_cached_reference_debt':not report.get('cached_reference_passed'),
            'readiness_qualified':False,'optimizer_updates':0}


@contextlib.contextmanager
def _channel_copies(model, named):
    """The channel verifier round-trips copies, never source optimizer owners."""
    selected={id(p)for p in model.adapter.parameters()}
    selected.update(id(getattr(model,key))for key in('input_rows','output_rows','extra_input_rows','extra_output_rows')if hasattr(model,key))
    originals={id(p):p for p in model.parameters()if id(p)in selected}
    copies={key:torch.nn.Parameter(p.detach().clone(memory_format=torch.preserve_format),requires_grad=p.requires_grad)
            for key,p in originals.items()}
    assignments=[]
    try:
        for module in model.modules():
            for key,p in tuple(module._parameters.items()):
                if id(p)in copies:
                    assignments.append((module,key,p));module._parameters[key]=copies[id(p)]
        yield [(name,copies.get(id(p),p))for name,p in named],sum(p.numel()*p.element_size()for p in copies.values())
    finally:
        for module,key,p in assignments:module._parameters[key]=p
        if any(not torch.equal(copies[key],p)for key,p in originals.items()):
            raise CapacityStateChangedError('attention qualification numerically changed a temporary channel parameter')


def qualify_attention_transition(model, tokenizer, roles, config, *, optimizer,
        previous_mode, candidate_mode, checkpoint_sha256, forward_atol, gradient_atol,
        gradient_rtol=None, backend=None, _execution_setting='native_attention_precision'):
    """Return cross-mode and cached-reference evidence as distinct conclusions."""
    # Path consistency (cached vs reference inside each arithmetic) keeps 1e-4; the old-vs-new comparison uses the
    # transition's own bound (a deliberate tensor-core transition changes rounding by design).
    from ..model.native_precision import transition_gradient_rtol
    cross_mode_rtol = (gradient_rtol if gradient_rtol is not None else
                       transition_gradient_rtol(previous_mode, candidate_mode)
                       if _execution_setting == 'native_attention_precision' else 1e-4)
    gradient_rtol = 1e-4 if gradient_rtol is None else gradient_rtol
    named, gradient_roles = production_inventory(model, optimizer._native_named_parameters)
    source_named = named
    for _, child in _children(optimizer):
        if getattr(child, '_armed', None) is not None or (getattr(child, '_hooks', []) and not getattr(child, '_gated', False)):
            raise ValueError('attention comparison requires an unarmed, gated update boundary')
    device = model.input_rows.device
    if _execution_setting not in('native_attention_precision','ffn_chunk_tokens'):
        raise ValueError('unknown execution transition setting')
    ffn=_execution_setting=='ffn_chunk_tokens'
    original_mode=getattr(model.port,'ffn_chunk_tokens',0)if ffn else getattr(model,'native_attention_precision','ieee')
    had_ffn=hasattr(model.port,'ffn_chunk_tokens')
    if ffn and any(type(value)is not int or value<0 for value in(original_mode,previous_mode,candidate_mode)):
        raise ValueError('FFN transition requires nonnegative actual token chunk counts')
    if not ffn and original_mode != getattr(model.port,'native_attention_precision','ieee'):
        raise ValueError('model and port attention modes disagree')
    if config.get(_execution_setting,0 if ffn else'ieee')!=original_mode:
        raise ValueError('attention comparison configuration differs from live model')
    source_parameters={id(p):p for p in model.parameters()};source_parameters.update({id(p):p for _,p in named})
    versions = {key:p._version for key,p in source_parameters.items()}
    gradients = {key:(p.grad,None if p.grad is None else p.grad._version)for key,p in source_parameters.items()}
    def controls():
        return [(id(child),getattr(child,'_armed',None),getattr(child,'_gated',None),
            tuple(tuple(id(p)for p in group['params'])for group in child.param_groups))for _,child in _children(optimizer)]
    optimizer_controls=controls()
    optimizer_versions = _optimizer_versions(optimizer)
    modes = {module: module.training for module in model.modules()}
    checkpointing = model.port.checkpoint_layers
    rng = capture_training_rng_state(device)
    extra = copy.deepcopy(backend.extra_state()) if backend is not None else None
    counters = backend.collect_row_counts() if backend is not None else None
    executed = getattr(backend, 'has_executed', False)
    started = time.monotonic()

    def reset():
        restore_training_rng_state(rng, device)
        if backend is not None:
            backend.restore_extra_state(copy.deepcopy(extra))

    def select(mode):
        if ffn:model.port.ffn_chunk_tokens=mode
        else:set_native_attention_precision(model,mode)
        reset()

    def reference_report(mode):
        select(mode)
        selected_config={**config,_execution_setting:mode}
        probe={'minimum_source_tokens':max(previous_mode,candidate_mode)+1}if ffn else{}
        if backend is not None:
            from .native_backend import qualify_training_backend
            checked=qualify_training_backend(model,tokenizer,roles,selected_config,backend=backend,
                checkpoint_sha256=checkpoint_sha256,forward_atol=forward_atol,gradient_atol=gradient_atol,
                gradient_rtol=gradient_rtol,device=device,gradient_parameters=named,raise_on_failed=False,**probe)
            return checked['runtime_reference']
        return qualify_existing(model,tokenizer,roles,selected_config,
            checkpoint_sha256=checkpoint_sha256,forward_atol=forward_atol,gradient_atol=gradient_atol,
            gradient_rtol=gradient_rtol,gradient_parameters=named,before_graph=reset,**probe)

    try:
        model.eval()
        model.port.checkpoint_layers = True
        with _channel_copies(model,source_named)as (named,channel_copy_bytes):
            baseline = reference_report(previous_mode)
            case = baseline['complete_gradient_qualification']['case']
            nodes = [Invocation(**{**node, 'token_ids': tuple(node['token_ids']),
                'imports': tuple(ImportSpan(**edge) for edge in node['imports']),
                'producing': tuple(tuple(span) for span in node['producing']),
                'logical_producing': tuple(tuple(span) for span in node['logical_producing'])})
                for node in case['invocations']]
            plan = compile_graph(nodes)
            ids = plan.tensor([token for node in nodes for token in node.token_ids], device)
            positions = plan.tensor(case['positions'], device)
            labels = plan.tensor(case['labels'], device)
            comparisons = {}
            for execution in ('graph', 'cached'):
                outputs = {}
                def forward(mode, slot):
                    from ..model.native_weight_carrier import ParameterCotangentScope
                    from .native_gradients import cached_reference_cotangent_scope
                    cotangent_scope=(cached_reference_cotangent_scope(model) if execution=='cached'
                        else ParameterCotangentScope.for_model(model,mode='vjp'))
                    cotangent_scope.__enter__()
                    try:
                        select(mode)
                        embeddings = model.embed(ids[None])
                        result = model(plan, embeddings=embeddings, reference=execution == 'cached')
                        hidden = result['h_final']
                        from ..common import stable_softmax
                        loss = stable_softmax.cross_entropy(model.logits(hidden[:, positions]).float().flatten(0, 1),
                                                            labels, reduction='mean')
                        outputs[slot] = (hidden.detach().clone(), float(loss.detach()))
                        return loss, embeddings, cotangent_scope
                    except BaseException:
                        cotangent_scope.close()
                        raise
                before=backend.collect_row_counts()if backend is not None else {}
                split={}
                def candidate_forward():
                    split.update(backend.collect_row_counts()if backend is not None else {})
                    return forward(candidate_mode,'candidate')
                with backend.autocast()if backend is not None else contextlib.nullcontext():
                    old_loss,old_inputs,old_scope=forward(previous_mode,'old')
                    try:
                        comparison=compare_complete_gradients(old_loss,None,named,old_inputs,None,
                            atol=gradient_atol,rtol=cross_mode_rtol,cached_factory=candidate_forward,
                            flat_cotangent_scope=old_scope)
                    finally:
                        # Keep ownership at the caller too: a substituted or
                        # failing comparator need not reach its own cleanup.
                        old_scope.close()
                after=backend.collect_row_counts()if backend is not None else {}
                def execution_evidence(first,last):
                    delta={key:last.get(key,0)-first.get(key,0)for key in last}
                    return dict(configuration=backend.configuration()if backend is not None else{'enabled':False},
                        executed_projections=delta,passed=backend is None or(delta.get('calls',0)>0 and delta.get('rows',0)>0))
                comparison['backend_execution']=dict(old=execution_evidence(before,split),candidate=execution_evidence(split,after))
                old_hidden, old_ce = outputs['old']
                new_hidden, new_ce = outputs['candidate']
                comparison.update(forward_max_abs=float((old_hidden-new_hidden).abs().max()),
                                  consumer_ce_abs=abs(old_ce-new_ce))
                comparison['passed'] = bool(comparison['passed']
                    and comparison['forward_max_abs'] <= forward_atol
                    and comparison['consumer_ce_abs'] <= forward_atol
                    and all(value['passed']for value in comparison['backend_execution'].values()))
                comparisons[execution] = comparison
                del outputs, old_hidden, new_hidden, old_loss, old_inputs
            candidate = reference_report(candidate_mode)
            result = dict(schema=('natlang.native-ffn-chunk-transition-complete/1'if ffn else'natlang.native-attention-transition-complete/1'),
                **({'execution_setting':_execution_setting}if ffn else{}),
                parent_checkpoint_sha256=checkpoint_sha256, previous_mode=previous_mode,
                candidate_mode=candidate_mode, case=case, case_sha256=canonical_json_sha256_hex(case),
                roles=gradient_roles, cross_mode=comparisons,channel_copy_bytes=channel_copy_bytes,
                cross_mode_passed=all(row['passed'] for row in comparisons.values()),
                baseline_cached_reference=baseline, candidate_cached_reference=candidate,
                cached_reference_passed=bool(baseline['passed'] and candidate['passed']),
                optimizer_updates=0, task_quality_qualified=False, readiness_qualified=False,
                forward_atol=forward_atol, gradient_atol=gradient_atol, gradient_rtol=gradient_rtol, cross_mode_gradient_rtol=cross_mode_rtol,
                elapsed_seconds=time.monotonic()-started)
            result['passed'] = result['cross_mode_passed'] and result['cached_reference_passed']
    finally:
        if ffn:
            if had_ffn:model.port.ffn_chunk_tokens=original_mode
            elif hasattr(model.port,'ffn_chunk_tokens'):delattr(model.port,'ffn_chunk_tokens')
        else:set_native_attention_precision(model,original_mode)
        model.port.checkpoint_layers = checkpointing
        for module, training in modes.items(): module.training = training
        restore_error=None
        try:
            if backend is not None:
                backend.restore_extra_state(extra)
                if hasattr(backend,'row_counts'):
                    backend.row_counts.clear();backend.row_counts.update(counters)
                backend.has_executed=executed
        except Exception as error:restore_error=error
        try:restore_training_rng_state(rng,device)
        except Exception as error:restore_error=error
        if restore_error is not None:
            raise CapacityStateChangedError('attention comparison could not restore backend/RNG state')from restore_error
        if (any(p._version != versions[id(p)]for p in source_parameters.values())
                or any(p.grad is not gradients[id(p)][0]
                or (p.grad is not None and p.grad._version != gradients[id(p)][1]) for p in source_parameters.values())
                or controls()!=optimizer_controls
                or _optimizer_versions(optimizer) != optimizer_versions):
            raise CapacityStateChangedError('attention comparison changed parameters, gradients or optimizer history')
    result['source_state_preserved'] = True
    result['transition_equivalence']=transition_equivalence(result)
    return result


def qualify_ffn_chunk_transition(model,tokenizer,roles,config,*,optimizer,previous_chunk_tokens,
        candidate_chunk_tokens,checkpoint_sha256,forward_atol,gradient_atol,gradient_rtol=1e-4,backend=None):
    """Reuse the complete unchanged-state graph/cached execution verifier for FFN workspace."""
    actual=getattr(model.port,'ffn_chunk_tokens',0)
    return qualify_attention_transition(model,tokenizer,roles,{**config,'ffn_chunk_tokens':actual},
        optimizer=optimizer,previous_mode=previous_chunk_tokens,candidate_mode=candidate_chunk_tokens,
        checkpoint_sha256=checkpoint_sha256,forward_atol=forward_atol,gradient_atol=gradient_atol,
        gradient_rtol=gradient_rtol,backend=backend,_execution_setting='ffn_chunk_tokens')
