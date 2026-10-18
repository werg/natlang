"""Observe a finite window of real native updates with the shared profiler."""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib
import json
import time
from pathlib import Path


@contextlib.contextmanager
def observe_native_phases(out):
    """Time existing calls, preserving their arguments, returned objects and errors."""
    import os
    out = Path(out)
    undo = []
    startup_undo = []
    errors = []
    sequence = 0

    def event(phase, status, **fields):
        nonlocal sequence
        sequence += 1
        row = {'schema': 'natlang.native-training-phase/1', 'sequence': sequence,
               'phase': phase, 'status': status, 'at': time.time(),
               'monotonic': time.monotonic(), **fields}
        try:
            out.mkdir(parents=True, exist_ok=True)
            with (out / 'native-startup-phases.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as error:
            errors.append(type(error).__name__)

    def timed(original, phase, after=None):
        @functools.wraps(original)
        def call(*args, **kwargs):
            started = time.monotonic()
            event(phase, 'begin')
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                event(phase, 'error', seconds=time.monotonic()-started,
                      error_type=type(error).__name__)
                raise
            event(phase, 'end', seconds=time.monotonic()-started)
            if after is not None:
                try:
                    after(result)
                except Exception as error:
                    errors.append(type(error).__name__)
                    event(phase, 'diagnostic-error', error_type=type(error).__name__)
            return result
        return call

    def patch(owner, name, phase, *, startup=True, after=None, static=False):
        original_attribute = vars(owner)[name]
        original = getattr(owner, name)
        replacement = timed(original, phase, after)
        setattr(owner, name, staticmethod(replacement) if static else replacement)
        (startup_undo if startup else undo).append(
            lambda: setattr(owner, name, original_attribute))

    def restored_optimizer(result):
        optimizer, _named = result
        # Observe this instance's already-required load, never its update method.
        old = vars(optimizer).get('load_state_dict')
        had = 'load_state_dict' in vars(optimizer)
        original = optimizer.load_state_dict
        optimizer.load_state_dict = timed(original, 'restore/optimizer-state')
        def restore():
            if had:
                optimizer.load_state_dict = old
            else:
                del optimizer.load_state_dict
        startup_undo.append(restore)

    def end_startup():
        while startup_undo:
            startup_undo.pop()()
        event('startup', 'end')

    event('control-imports', 'begin')
    try:
        from natlang_neuralese.train import native_state_sft as native
        from natlang_neuralese.model import native_model
        event('control-imports', 'end')
        patch(native.CheckpointPolicy, 'load', 'restore/checkpoint-decode', static=True)
        patch(native, 'load_native_checkpoint', 'restore/load-native-checkpoint')
        patch(native, 'load_native_model', 'restore/load-native-model')
        patch(native_model, 'load_native_model', 'restore/model-construction')
        patch(native, 'collect_sft_progress', 'restore/collect-progress')
        patch(native, 'restore_sft_progress', 'restore/restore-progress')
        patch(native.NativeGraphSource, '__init__', 'restore/source-construction')
        patch(native, 'make_native_optimizer', 'restore/optimizer-construction',
              after=restored_optimizer)
        patch(native, '_save', 'checkpoint/save', startup=False)
        yield {'diagnostic_errors': errors, 'finish_startup': end_startup}
    finally:
        while startup_undo:
            startup_undo.pop()()
        while undo:
            undo.pop()()
        event('observer', 'restored', diagnostic_errors=errors)


@contextlib.contextmanager
def profile_native_updates(out, *, first=2, last=5, profiler_factory=None, first_update_observer=None):
    import torch
    from natlang_neuralese.common import profiling
    from natlang_neuralese.train import native_state_sft as native

    if type(first) is not int or type(last) is not int or not 1 <= first <= last:
        raise ValueError('profile window requires positive relative update numbers')
    if profiling.ENABLED:
        raise ValueError('native training already has an active profiler')
    out = Path(out)
    state = {'schema': 'natlang.native-training-profile-observer/1',
             'index_scope': 'native_training_update calls in this invocation; not global checkpoint steps',
             'requested': [first, last], 'attempted_updates': 0,
             'completed_updates': 0, 'diagnostic_errors': []}
    original_update = native.native_training_update
    original_forward = native.packed_graph_loss
    original_backward = torch.autograd.backward
    profiler = None

    def restore():
        native.native_training_update = original_update
        native.packed_graph_loss = original_forward
        torch.autograd.backward = original_backward

    def diagnostic(operation):
        try:
            operation()
        except Exception as error:
            state['diagnostic_errors'].append(type(error).__name__)
            if profiler is not None:
                try:
                    profiler.close(state['completed_updates'])
                except Exception as close_error:
                    state['diagnostic_errors'].append(type(close_error).__name__)
            profiling.ENABLED = False
            restore()
            print(json.dumps({'event': 'native_profile_disabled',
                              'error_class': type(error).__name__}), flush=True)
            return False
        return True

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        with profiling.region('update/forward'):
            return original_forward(*args, **kwargs)

    @functools.wraps(original_backward)
    def backward(*args, **kwargs):
        with profiling.region('update/backward'), profiling.backward_phase():
            return original_backward(*args, **kwargs)

    @functools.wraps(original_update)
    def update(*args, **kwargs):
        state['attempted_updates'] += 1
        number = state['attempted_updates']
        if number == 1 and first_update_observer is not None:
            diagnostic(first_update_observer)
        observing = diagnostic(lambda: profiler.begin(number))
        started = time.monotonic()
        result = original_update(*args, **kwargs)
        state['completed_updates'] += 1
        seconds = time.monotonic() - started
        if observing:
            diagnostic(lambda: profiler.update_done(number, seconds))
            diagnostic(lambda: profiler.maybe_finish(number))
        if profiler.done or state['diagnostic_errors']:
            restore()
        return result

    try:
        def prepare():
            nonlocal profiler
            out.mkdir(parents=True, exist_ok=True)
            factory = profiler_factory or profiling.UpdateProfiler
            profiler = factory(out, value=f'{first}:{last}')
        if diagnostic(prepare):
            native.packed_graph_loss = forward
            torch.autograd.backward = backward
            native.native_training_update = update
        yield state
    finally:
        try:
            if profiler is not None:
                diagnostic(lambda: profiler.close(state['completed_updates']))
        finally:
            restore()
            # A closed observation never leaves model scopes enabled.
            profiling.ENABLED = False
        state['restored_original_functions'] = True
        diagnostic(lambda: (out / 'native-profile-observer.json').write_text(
            json.dumps(state, indent=2) + '\n'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-out', type=Path, required=True)
    parser.add_argument('--first-update', type=int, default=2)
    parser.add_argument('--last-update', type=int, default=5)
    parser.add_argument('--backbone-update-every', type=int, default=0,
                        help='observe sampled actual backbone displacements at this relative-update cadence')
    parser.add_argument('--backbone-update-samples', type=int, default=128)
    parser.add_argument('--backbone-update-observer', type=json.loads,
                        help='exact path/bytes/sha256 binding of the shared read-only update observer')
    parser.add_argument('--expected-observer-sha256',
                        help='supervisor-bound observer source; reject a different installed file')
    parser.add_argument('--module', default='natlang_neuralese.train.native_state_sft',
                        choices=('natlang_neuralese.train.native_state_sft',
                                 'natlang_neuralese.train.native_preference'))
    parser.add_argument('trainer_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    source = Path(__file__).resolve()
    raw = source.read_bytes()
    source_binding = {'path': str(source), 'bytes': len(raw),
                      'sha256': hashlib.sha256(raw).hexdigest()}
    if (args.expected_observer_sha256 is not None
            and args.expected_observer_sha256 != source_binding['sha256']):
        parser.error('installed profile observer differs from supervisor source binding')
    arguments = args.trainer_args
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    if not arguments:
        parser.error('pass the original trainer arguments after --')
    if args.backbone_update_every < 0 or args.backbone_update_samples < 1:
        parser.error('backbone cadence must be nonnegative and sample count positive')
    if bool(args.backbone_update_every) != bool(args.backbone_update_observer):
        parser.error('enabled backbone observation requires its exact bound helper')
    updates = contextlib.nullcontext(None)
    if args.backbone_update_observer is not None:
        reference = args.backbone_update_observer
        helper = Path(reference['path']).resolve(strict=True)
        content = helper.read_bytes()
        if (len(content) != reference['bytes']
                or hashlib.sha256(content).hexdigest() != reference['sha256']):
            parser.error('installed backbone update observer differs from its source binding')
        namespace = {'__file__': str(helper), '__name__': '_bound_native_update_observer'}
        exec(compile(content, str(helper), 'exec'), namespace)
        updates = namespace['observe_backbone_updates'](args.profile_out,
            every=args.backbone_update_every, samples=args.backbone_update_samples)
    # The finite profiler restores this outer observer after its last update;
    # periodic actual-displacement observation therefore continues afterwards.
    with observe_native_phases(args.profile_out) as phases, updates as update_state, profile_native_updates(
            args.profile_out, first=args.first_update, last=args.last_update,
            first_update_observer=phases['finish_startup']) as state:
        state['observer_source'] = source_binding
        state['phase_diagnostic_errors'] = phases['diagnostic_errors']
        state['trainer_module'] = args.module
        state['backbone_update_observer_source'] = args.backbone_update_observer
        state['backbone_update_observer'] = update_state
        return importlib.import_module(args.module).main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
