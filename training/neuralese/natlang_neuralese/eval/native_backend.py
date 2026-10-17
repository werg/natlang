"""Exact live-parent training-backend check; no optimizer update or task claim."""
from __future__ import annotations

import copy
import json

from .native_state import qualify_existing
from ..train.loop import capture_training_rng_state, restore_training_rng_state


def qualify_training_backend(model, tokenizer, roles, config, *, backend,
                             checkpoint_sha256, forward_atol, gradient_atol, device,
                             gradient_parameters=None,raise_on_failed=True):
    if backend is None:
        raise ValueError('training backend qualification requires an enabled backend')
    if forward_atol is None or gradient_atol is None:
        raise ValueError('training backend qualification requires declared numerical tolerances')
    rng = capture_training_rng_state(device)
    extra = copy.deepcopy(backend.extra_state())
    counters = backend.collect_row_counts()
    executed = getattr(backend, 'has_executed', False)
    try:
        # native_modules deliberately disables FP8 and cannot qualify its execution.
        with backend.autocast():
            options={}if gradient_parameters is None else{'gradient_parameters':gradient_parameters,
                'before_graph':lambda:backend.restore_extra_state(copy.deepcopy(extra))}
            report = qualify_existing(model, tokenizer, roles, config,
                checkpoint_sha256=checkpoint_sha256, forward_atol=forward_atol,
                gradient_atol=gradient_atol,**options)
        after = backend.collect_row_counts()
        delta = {key: after.get(key, 0)-counters.get(key, 0) for key in after}
        result = {'schema': 'natlang.native-training-backend-check/1',
                  'parent_checkpoint_sha256': checkpoint_sha256,
                  'backend': backend.configuration(), 'executed_projections': delta,
                  'runtime_reference': report, 'optimizer_updates': 0,
                  'task_quality_qualified': False}
        result['passed']=bool(report['passed']and delta.get('calls',0)>0 and delta.get('rows',0)>0)
        report['training_backend_execution']={'configuration':result['backend'],'executed_projections':delta,
            'passed':delta.get('calls',0)>0 and delta.get('rows',0)>0}
        if report.get('complete_gradient_qualification')is not None:
            report['complete_gradient_qualification']['training_backend_execution']=report['training_backend_execution']
        if not result['passed']:
            report['passed']=False
            if raise_on_failed:
                raise ValueError('actual training backend failed unchanged runtime checks: '+json.dumps(result, sort_keys=True))
        return result
    finally:
        backend.restore_extra_state(extra)
        if hasattr(backend, 'row_counts'):
            backend.row_counts.clear()
            backend.row_counts.update(counters)
        backend.has_executed = executed
        restore_training_rng_state(rng, device)


def qualify_complete_runtime(model,tokenizer,roles,config,*,optimizer,backend,fp8_profile,
                             checkpoint_sha256,forward_atol,gradient_atol,device,
                             task_execution_receipt=None,backbone_optimizer_updates=0):
    """The same complete proof at ordinary and shared-training publication."""
    from ..common.hashing import canonical_json_sha256_hex
    from .native_gradients import validate_complete_proof
    named=getattr(optimizer,'_native_named_parameters',None)
    if not named:raise ValueError('complete runtime proof needs its actual production optimizer inventory')
    rng=capture_training_rng_state(device)
    try:
        args=dict(checkpoint_sha256=checkpoint_sha256,forward_atol=forward_atol,gradient_atol=gradient_atol)
        if backend is None:
            report=qualify_existing(model,tokenizer,roles,config,gradient_parameters=named,**args)
        else:
            report=qualify_training_backend(model,tokenizer,roles,config,backend=backend,
                device=device,gradient_parameters=named,raise_on_failed=False,**args)['runtime_reference']
        profile={'model_config_sha256':canonical_json_sha256_hex(config),'fp8':fp8_profile}
        report.update(qualification_profile=profile,task_execution_receipt=task_execution_receipt,
            backbone_optimizer_tested=backbone_optimizer_updates>0,
            backbone_optimizer_updates=int(backbone_optimizer_updates))
        if report['passed']:
            try:validate_complete_proof(report['complete_gradient_qualification'],report['native_identity'],
                gradient_atol=gradient_atol,execution_profile=profile)
            except ValueError as error:
                report.update(passed=False,complete_gradient_error=str(error))
        return report
    finally:restore_training_rng_state(rng,device)
