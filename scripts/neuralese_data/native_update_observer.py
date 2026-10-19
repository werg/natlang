"""Read deterministic backbone samples around the actual shared optimizer update."""
from __future__ import annotations

import contextlib
import functools
import json
import math
import time
import threading
import resource
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


def _host_resources():
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        name, _, value = line.partition(':')
        if name in ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem'):
            status[name + '_bytes'] = int(value.split()[0]) * 1024
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, _, value = line.partition(':')
        if name in ('MemAvailable', 'Shmem'):
            memory[name + '_bytes'] = int(value.split()[0]) * 1024
    for name in ('memory.current', 'memory.max'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            value = path.read_text().strip()
            memory[name] = None if value == 'max' else int(value)
    return {'process': status, 'host': memory}


def _resource_point(cuda_device=None):
    import torch
    io = {key: int(value) for key, value in
          (line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())}
    result = {**_host_resources(), 'io': io,
              'process_lifetime_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if cuda_device is not None:
        result['cuda'] = {'allocated_bytes': torch.cuda.memory_allocated(cuda_device),
                          'reserved_bytes': torch.cuda.memory_reserved(cuda_device),
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(cuda_device),
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved(cuda_device)}
    return result


class _ActualUpdateResources:
    """Bounded metadata sampling around the existing update; no model operation."""
    def __init__(self, cuda_device=None):
        import torch
        self.cuda_device = cuda_device
        self.before = _resource_point(cuda_device)
        self.peak = dict(self.before['process'])
        self.minimum_available = self.before['host']['MemAvailable_bytes']
        self.stop = threading.Event()
        self.error = None
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        def sample():
            try:
                while not self.stop.wait(.2):
                    value = _host_resources()
                    for key, count in value['process'].items():
                        self.peak[key] = max(self.peak.get(key, 0), count)
                    self.minimum_available = min(self.minimum_available, value['host']['MemAvailable_bytes'])
            except Exception as error:
                self.error = type(error).__name__
        self.thread = threading.Thread(target=sample, name='native-update-resource-observer', daemon=True)
        self.thread.start()
        self.started = time.monotonic()

    def finish(self):
        self.stop.set()
        self.thread.join()
        after = _resource_point(self.cuda_device)
        for key, count in after['process'].items():
            self.peak[key] = max(self.peak.get(key, 0), count)
        self.minimum_available = min(self.minimum_available, after['host']['MemAvailable_bytes'])
        return {'schema': 'natlang.actual-native-update-resources/1',
                'before': self.before, 'after': after,
                'io_delta': {key: after['io'][key] - count for key, count in self.before['io'].items()},
                'host_wall_seconds': time.monotonic() - self.started,
                'sampled_process_peak_bytes': self.peak,
                'sampled_minimum_guest_available_bytes': self.minimum_available,
                'host_sampling_seconds': .2, 'sampling_error': self.error,
                'scope': 'Exactly one original update. Host peaks sampled at 200ms; lifetime RSS labeled separately. CUDA allocator peaks reset at bracket entry; no synchronization or extra model computation.'}


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
        resources = None
        if enabled and (number <= 5 or number % every == 0):
            try:
                optimizer = args[1] if len(args) > 1 else kwargs['optimizer']
                devices = {parameter.device for _, parameter in optimizer._native_named_parameters if parameter.is_cuda}
                if len(devices) > 1:
                    raise ValueError('resource observation needs one actual CUDA device')
                resources = _ActualUpdateResources(next(iter(devices), None))
            except Exception as error:
                failed(error)
        update_completed = False
        try:
            result = original(*args, **kwargs)
            update_completed = True
        finally:
            if resources is not None:
                try:
                    observed_resources = resources.finish()
                    out.mkdir(parents=True, exist_ok=True)
                    with (out / 'native-update-resources.jsonl').open('a') as stream:
                        stream.write(json.dumps({'relative_update': number, 'update_completed': update_completed, **observed_resources}, allow_nan=False) + '\n')
                except Exception as error:
                    failed(error)
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
