"""Complete restored-model attention comparisons without optimizer updates."""
from __future__ import annotations
import contextlib
import copy
import time

import torch
import torch.nn.functional as F

from ..common.hashing import canonical_json_sha256_hex
from ..model.native_precision import set_native_attention_precision
from ..model.native_graph_plan import Invocation, ImportSpan, compile_graph
from ..train.loop import capture_training_rng_state, restore_training_rng_state
from ..train.optim_restore import _children
from .native_gradients import production_inventory, compare_complete_gradients
from .native_memory import _optimizer_versions, CapacityStateChangedError
from .native_state import qualify_existing


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
        gradient_rtol=1e-4, backend=None):
    """Return cross-mode and cached-reference evidence as distinct conclusions."""
    named, gradient_roles = production_inventory(model, optimizer._native_named_parameters)
    source_named = named
    for _, child in _children(optimizer):
        if getattr(child, '_armed', None) is not None or (getattr(child, '_hooks', []) and not getattr(child, '_gated', False)):
            raise ValueError('attention comparison requires an unarmed, gated update boundary')
    device = model.input_rows.device
    original_mode = getattr(model, 'native_attention_precision', 'ieee')
    if original_mode != getattr(model.port, 'native_attention_precision', 'ieee'):
        raise ValueError('model and port attention modes disagree')
    if config.get('native_attention_precision', 'ieee') != original_mode:
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
        set_native_attention_precision(model, mode)
        reset()

    def reference_report(mode):
        select(mode)
        selected_config={**config,'native_attention_precision':mode}
        if backend is not None:
            from .native_backend import qualify_training_backend
            checked=qualify_training_backend(model,tokenizer,roles,selected_config,backend=backend,
                checkpoint_sha256=checkpoint_sha256,forward_atol=forward_atol,gradient_atol=gradient_atol,
                gradient_rtol=gradient_rtol,device=device,gradient_parameters=named,raise_on_failed=False)
            return checked['runtime_reference']
        return qualify_existing(model,tokenizer,roles,selected_config,
            checkpoint_sha256=checkpoint_sha256,forward_atol=forward_atol,gradient_atol=gradient_atol,
            gradient_rtol=gradient_rtol,gradient_parameters=named,before_graph=reset)

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
                    select(mode)
                    embeddings = model.embed(ids[None])
                    result = model(plan, embeddings=embeddings, reference=execution == 'cached')
                    hidden = result['h_final']
                    loss = F.cross_entropy(model.logits(hidden[:, positions]).float().flatten(0, 1), labels)
                    outputs[slot] = (hidden.detach().clone(), float(loss.detach()))
                    return loss, embeddings
                before=backend.collect_row_counts()if backend is not None else {}
                split={}
                def candidate_forward():
                    split.update(backend.collect_row_counts()if backend is not None else {})
                    return forward(candidate_mode,'candidate')
                with backend.autocast()if backend is not None else contextlib.nullcontext():
                    old_loss,old_inputs=forward(previous_mode,'old')
                    comparison=compare_complete_gradients(old_loss,None,named,old_inputs,None,
                        atol=gradient_atol,rtol=gradient_rtol,cached_factory=candidate_forward)
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
            result = dict(schema='natlang.native-attention-transition-complete/1',
                parent_checkpoint_sha256=checkpoint_sha256, previous_mode=previous_mode,
                candidate_mode=candidate_mode, case=case, case_sha256=canonical_json_sha256_hex(case),
                roles=gradient_roles, cross_mode=comparisons,channel_copy_bytes=channel_copy_bytes,
                cross_mode_passed=all(row['passed'] for row in comparisons.values()),
                baseline_cached_reference=baseline, candidate_cached_reference=candidate,
                cached_reference_passed=bool(baseline['passed'] and candidate['passed']),
                optimizer_updates=0, task_quality_qualified=False, readiness_qualified=False,
                forward_atol=forward_atol, gradient_atol=gradient_atol, gradient_rtol=gradient_rtol,
                elapsed_seconds=time.monotonic()-started)
            result['passed'] = result['cross_mode_passed'] and result['cached_reference_passed']
    finally:
        set_native_attention_precision(model, original_mode)
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
    return result
