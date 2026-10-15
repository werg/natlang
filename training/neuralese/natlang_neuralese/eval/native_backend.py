"""Exact live-parent training-backend check; no optimizer update or task claim."""
from __future__ import annotations

import copy
import json

from .native_state import qualify_existing
from ..train.loop import capture_training_rng_state, restore_training_rng_state


def qualify_training_backend(model, tokenizer, roles, config, *, backend,
                             checkpoint_sha256, forward_atol, gradient_atol, device):
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
            report = qualify_existing(model, tokenizer, roles, config,
                checkpoint_sha256=checkpoint_sha256, forward_atol=forward_atol,
                gradient_atol=gradient_atol)
        after = backend.collect_row_counts()
        delta = {key: after.get(key, 0)-counters.get(key, 0) for key in after}
        result = {'schema': 'natlang.native-training-backend-check/1',
                  'parent_checkpoint_sha256': checkpoint_sha256,
                  'backend': backend.configuration(), 'executed_projections': delta,
                  'runtime_reference': report, 'optimizer_updates': 0,
                  'task_quality_qualified': False}
        if not report['passed'] or delta.get('calls', 0) <= 0 or delta.get('rows', 0) <= 0:
            raise ValueError('actual training backend failed unchanged runtime checks: '+json.dumps(result, sort_keys=True))
        return result
    finally:
        backend.restore_extra_state(extra)
        if hasattr(backend, 'row_counts'):
            backend.row_counts.clear()
            backend.row_counts.update(counters)
        backend.has_executed = executed
        restore_training_rng_state(rng, device)
