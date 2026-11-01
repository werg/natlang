"""Low-overhead operational diagnostics for a running native learner."""
import contextlib
import signal
import sys
import threading
import traceback


@contextlib.contextmanager
def learner_stack_signal():
    """SIGQUIT dumps Python stacks without changing checkpoint/stop signals."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, 'SIGQUIT'):
        yield
        return
    previous=signal.getsignal(signal.SIGQUIT)
    # Run with the GIL at a Python signal boundary. A native faulthandler dump
    # during concurrent CUDA serving segfaulted the live learner on 2026-11-01.
    def dump(signum, frame):
        frames = sys._current_frames()
        print('Current thread Python stacks (deferred signal boundary):', file=sys.__stderr__, flush=True)
        for identifier, current in frames.items():
            print(f'Thread {identifier}:', file=sys.__stderr__)
            traceback.print_stack(current, limit=32, file=sys.__stderr__)
        sys.__stderr__.flush()
    signal.signal(signal.SIGQUIT,dump)
    try:
        yield
    finally:
        signal.signal(signal.SIGQUIT,previous)
