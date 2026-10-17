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
def profile_native_updates(out, *, first=2, last=5, profiler_factory=None):
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
    with profile_native_updates(args.profile_out, first=args.first_update, last=args.last_update) as state:
        state['observer_source'] = source_binding
        state['trainer_module'] = args.module
        return importlib.import_module(args.module).main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
