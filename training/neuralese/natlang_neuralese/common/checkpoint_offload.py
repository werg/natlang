"""Host storage for exact nonreentrant activation-checkpoint inputs.

Wrap the port's existing checkpoint implementation, preserving its precision
contexts and RNG contract. The inner checkpoint owns activation recomputation;
PyTorch's outer hooks transfer only the tensors that checkpoint actually saves.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


class CheckpointInputOffload:
    def __init__(self, checkpoint: Callable, *, pin_memory: bool = True):
        if not callable(checkpoint):
            raise TypeError('checkpoint input offload requires an existing checkpoint callable')
        self.checkpoint = checkpoint
        self.pin_memory = pin_memory
        self.calls = 0

    def __call__(self, function, *args, **kwargs):
        if kwargs.get('use_reentrant') is not False:
            raise ValueError('checkpoint input offload requires explicit nonreentrant checkpointing')
        self.calls += 1
        with torch.autograd.graph.save_on_cpu(pin_memory=self.pin_memory):
            return self.checkpoint(function, *args, **kwargs)


def configure_checkpoint_input_offload(port, *, enabled: bool, pin_memory: bool = True):
    """Install/remove the wrapper without replacing an existing backend hook.

    Repeated setup is idempotent. Backend setup must precede this call; a caller
    can explicitly disable the mode to recover its original callable identity.
    """
    from torch.utils.checkpoint import checkpoint

    current = getattr(port, 'activation_checkpoint', checkpoint)
    if isinstance(current, CheckpointInputOffload):
        if enabled and current.pin_memory == pin_memory:
            return current
        current = current.checkpoint
    if enabled:
        wrapped = CheckpointInputOffload(current, pin_memory=pin_memory)
        port.activation_checkpoint = wrapped
        return wrapped
    port.activation_checkpoint = current
    return current
