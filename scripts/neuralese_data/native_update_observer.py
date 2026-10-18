"""Read deterministic backbone samples around the actual shared optimizer update."""
from __future__ import annotations

import contextlib
import functools
import json
import math
import time
from pathlib import Path


def _coordinates(shape, count):
    size = math.prod(shape)
    if not size:
        return ()
    count = min(count, size)
    flat = [0] if count == 1 else [i * (size - 1) // (count - 1) for i in range(count)]
    result = []
    for index in flat:
        coordinate = []
        for dimension in reversed(shape):
            index, remainder = divmod(index, dimension)
            coordinate.append(remainder)
        result.append(tuple(reversed(coordinate)))
    return tuple(result)


def _sample(tensor, coordinates):
    import torch
    if tensor.layout != torch.strided:
        raise ValueError('backbone observation requires a dense sampled tensor')
    with torch.no_grad():
        if tensor.ndim == 0:
            return tensor.detach().cpu().double().reshape(1).clone()
        # Index by logical coordinates: reshape could copy an entire transposed
        # parameter. Advanced indexing allocates only the requested samples.
        indices = tuple(torch.tensor([point[axis] for point in coordinates],
                                     device=tensor.device, dtype=torch.long)
                        for axis in range(tensor.ndim))
        return tensor.detach()[indices].cpu().double()


def _before(optimizer, samples):
    named = getattr(optimizer, '_native_named_parameters', None)
    if named is None:
        raise ValueError('backbone observation requires the actual named optimizer inventory')
    result = []
    for name, parameter in named:
        if not name.startswith('backbone.') or not parameter.numel():
            continue
        coordinates = _coordinates(tuple(parameter.shape), samples)
        result.append((name, parameter, coordinates, _sample(parameter, coordinates)))
    if not result:
        raise ValueError('backbone observation found no named backbone parameters')
    return result


def _after(before):
    import torch
    result = []
    for name, parameter, coordinates, prior in before:
        current = _sample(parameter, coordinates)
        gradient = _sample(parameter.grad, coordinates) if parameter.grad is not None else None
        finite = bool(torch.isfinite(prior).all() and torch.isfinite(current).all())
        row = {'name': name, 'parameter_dtype': str(parameter.dtype),
               'shape': list(parameter.shape), 'sampled_coordinates': len(coordinates),
               'parameter_elements': parameter.numel(), 'gradient_present': gradient is not None,
               'gradient_scope': 'Remaining after the original update; an in-backward optimizer may already have cleared it',
               'finite_parameter_samples': finite}
        if finite:
            delta = current - prior
            norm = float(prior.norm())
            displacement = float(delta.norm())
            row.update(sampled_parameter_l2_before=norm,
                       sampled_update_l2=displacement,
                       sampled_update_max_abs=float(delta.abs().max()),
                       changed_sampled_coordinates=int(torch.count_nonzero(delta)),
                       sampled_relative_update_l2=displacement / norm if norm else None)
        if gradient is not None:
            row['finite_gradient_samples'] = bool(torch.isfinite(gradient).all())
            row['sampled_gradient_l2'] = float(gradient.norm()) if row['finite_gradient_samples'] else None
        result.append(row)
    return result


@contextlib.contextmanager
def observe_backbone_updates(out, *, every=250, samples=128):
    """Observe first/every-N relative calls; never grant committed exposure."""
    from natlang_neuralese.train import native_state_sft as native
    if type(every) is not int or every < 1 or type(samples) is not int or samples < 1:
        raise ValueError('backbone observation needs positive integer cadence and samples')
    out = Path(out)
    original = native.native_training_update
    state = {'attempted_updates': 0, 'completed_updates': 0, 'diagnostic_errors': [],
             'observed_updates': 0}
    enabled = True

    def failed(error):
        nonlocal enabled
        enabled = False
        state['diagnostic_errors'].append(type(error).__name__)
        print(json.dumps({'event': 'native_backbone_observation_disabled',
                          'error_class': type(error).__name__}), flush=True)

    @functools.wraps(original)
    def update(*args, **kwargs):
        state['attempted_updates'] += 1
        number = state['attempted_updates']
        before = None
        observation_started = time.monotonic()
        if enabled and (number == 1 or number % every == 0):
            try:
                optimizer = args[1] if len(args) > 1 else kwargs['optimizer']
                before = _before(optimizer, samples)
            except Exception as error:
                failed(error)
        before_seconds = time.monotonic() - observation_started
        result = original(*args, **kwargs)
        state['completed_updates'] += 1
        if before is not None:
            try:
                observation_started = time.monotonic()
                row = {'schema': 'natlang.native-backbone-update-observation/1',
                       'at': time.time(), 'relative_update': number, 'samples_per_parameter': samples,
                       'scope': 'Deterministic sampled coordinates around one actual update; not whole-tensor norms or committed checkpoint exposure',
                       'parameters': _after(before)}
                row['observation_seconds'] = before_seconds + time.monotonic() - observation_started
                out.mkdir(parents=True, exist_ok=True)
                with (out / 'native-backbone-updates.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                state['observed_updates'] += 1
            except Exception as error:
                failed(error)
        return result

    native.native_training_update = update
    try:
        yield state
    finally:
        native.native_training_update = original
        state['restored_original_function'] = True
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / 'native-backbone-observer.json').write_text(json.dumps(state, indent=2) + '\n')
        except Exception as error:
            failed(error)
