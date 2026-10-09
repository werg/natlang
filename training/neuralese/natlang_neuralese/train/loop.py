"""The shared skeleton of a Neuralese training loop (ARCHITECTURE_IMPROVEMENT C1).

Every trainer repeats the same moving parts around its own forward and backward pass: stop signals, a step loop that
ends on a stop or a completion, evaluation and checkpoint cadences, RNG capture around a retried attempt, gradient
accumulation into ``.grad`` and the optimizer commit. They live here once. A trainer keeps what is its own: how a batch
is built, what the loss is, what is logged and what a checkpoint holds.

    stop = StopSignal().install()
    loop = TrainingLoop(start_step, total_steps, stop)
    evaluation, checkpoints = Cadence(eval_every), Cadence(checkpoint_every, minutes)
    for step in loop:
        ... build the batch, forward, backward (accumulate_gradients for partial graphs) ...
        clip_finite_gradients(parameters)
        commit_optimizer_step(optimizer, on_failure=drop_partial_state)
        if evaluation.due(step + 1): ...
        if checkpoints.due(step + 1, force=stop.requested or step + 1 == total_steps): save(); checkpoints.mark()
        if qualified: loop.finish('qualified')
    # loop.reason is 'complete', 'signal' or the reason given to finish()

Signals only set a flag: the loop ends at the next step boundary and the trainer checkpoints there. This module imports
no model code, so every trainer and its tests can use it.
"""
from __future__ import annotations

import random
import signal
import time

import torch


class StopSignal:
    """A stop requested by SIGTERM or SIGINT, observed at the next step boundary.

    Install it before any loading: a container's PID 1 ignores signals without a handler, so a stop requested while
    the model loads would otherwise be dropped.
    """

    def __init__(self):
        self.requested = False
        self._previous = {}

    def request(self):
        self.requested = True

    def install(self, signals=(signal.SIGTERM, signal.SIGINT)):
        for sig in signals:
            self._previous.setdefault(sig, signal.signal(sig, lambda *_: self.request()))
        return self

    def restore(self):
        """Put back the handlers that were in place before ``install`` (a trainer that returns to its caller)."""
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous = {}


class Cadence:
    """When something recurs: every N completed steps, and/or after a wall-clock interval since the last ``mark``.

    ``every_steps`` falsy disables the step trigger; ``every_minutes`` None disables the time trigger (0 is "always").
    """

    def __init__(self, every_steps=0, every_minutes=None, *, clock=time.monotonic):
        self.every_steps = int(every_steps or 0)
        self.every_minutes = every_minutes
        self.clock = clock
        self.last = clock()

    def mark(self):
        """The thing just happened: restart the wall-clock window."""
        self.last = self.clock()

    def due(self, completed_steps, *, force=False):
        if force:
            return True
        if self.every_steps and completed_steps % self.every_steps == 0:
            return True
        return self.every_minutes is not None and self.clock() - self.last >= 60 * self.every_minutes


class TrainingLoop:
    """The step loop: yields the index of each step to run, from ``start`` up to ``total``.

    It ends when the steps are done (reason ``complete``), when a stop was requested (reason ``signal``), or when the
    trainer calls ``finish(reason)``. ``stop_before_step`` chooses where a stop is observed: before the next step
    (default), or only after a step has run, so a stop requested before the first step still runs that one.
    """

    def __init__(self, start, total, stop=None, *, stop_before_step=True):
        self.start, self.total = int(start), int(total)
        self.stop = stop if stop is not None else StopSignal()
        self.stop_before_step = stop_before_step
        self.reason = None

    def finish(self, reason):
        """End the loop after the current step."""
        self.reason = reason

    def __iter__(self):
        index = self.start
        self.reason = None
        while True:
            if self.reason is not None:
                return
            if self.stop.requested and (self.stop_before_step or index > self.start):
                self.reason = 'signal'
                return
            if index >= self.total:
                self.reason = 'complete'
                return
            yield index
            index += 1


def capture_training_rng_state(device):
    """Capture every RNG stream a training update attempt consumes."""
    return {'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []}


def restore_training_rng_state(state, device):
    """Restore the RNG boundary saved in a full-state checkpoint."""
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    if str(device).startswith('cuda'):
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def accumulate_gradients(parameters, gradients):
    """Add ``torch.autograd.grad`` results into ``.grad``; a missing gradient (``None``) leaves the parameter alone.

    Taking gradients with ``autograd.grad`` instead of ``backward`` keeps an aborted attempt from leaving partial
    ``.grad`` mutations behind, and lets several graphs accumulate before one optimizer step.
    """
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None:
            if parameter.grad is None:
                parameter.grad = gradient
            else:
                parameter.grad.add_(gradient)


def commit_optimizer_step(optimizer, *, on_failure=None):
    """``optimizer.step()``; on failure run ``on_failure`` (state may be half-updated: do not checkpoint it) and re-raise."""
    try:
        optimizer.step()
    except Exception:
        if on_failure is not None:
            on_failure()
        raise
