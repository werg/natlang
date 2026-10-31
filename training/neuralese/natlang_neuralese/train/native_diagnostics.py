"""Low-overhead operational diagnostics for a running native learner."""
import contextlib
import faulthandler
import signal
import sys
import threading


@contextlib.contextmanager
def learner_stack_signal():
    """SIGQUIT dumps Python stacks without changing checkpoint/stop signals."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, 'SIGQUIT'):
        yield
        return
    previous=signal.getsignal(signal.SIGQUIT)
    faulthandler.register(signal.SIGQUIT,file=sys.__stderr__,all_threads=True,chain=False)
    try:
        yield
    finally:
        faulthandler.unregister(signal.SIGQUIT)
        signal.signal(signal.SIGQUIT,previous)
