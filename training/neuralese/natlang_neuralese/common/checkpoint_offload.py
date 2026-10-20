"""Host storage for exact nonreentrant activation-checkpoint inputs.

Wrap the port's existing checkpoint implementation, preserving its precision
contexts and RNG contract. The inner checkpoint owns activation recomputation;
PyTorch's outer hooks transfer only the tensors that checkpoint actually saves.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


class CheckpointInputOffload:
    def __init__(self, checkpoint: Callable, *, pin_memory: bool = True, directory=None):
        if not callable(checkpoint):
            raise TypeError('checkpoint input offload requires an existing checkpoint callable')
        self.checkpoint = checkpoint
        self.pin_memory = pin_memory
        self.calls = 0
        self.directory = None
        self.file_store = None
        if directory is not None:
            from .checkpoint_file_storage import CheckpointFileStore
            self.file_store = CheckpointFileStore(directory, pin_memory=pin_memory)
            self.directory = self.file_store.directory

    def __call__(self, function, *args, **kwargs):
        if kwargs.get('use_reentrant') is not False:
            raise ValueError('checkpoint input offload requires explicit nonreentrant checkpointing')
        self.calls += 1
        if self.file_store is not None:
            with torch.autograd.graph.saved_tensors_hooks(self.file_store.pack, self.file_store.unpack):
                return self.checkpoint(function, *args, **kwargs)
        with torch.autograd.graph.save_on_cpu(pin_memory=self.pin_memory):
            return self.checkpoint(function, *args, **kwargs)

    def snapshot(self):
        return {'checkpoint_calls': self.calls, **(self.file_store.snapshot()
                if self.file_store is not None else {'storage': 'cpu'})}


def configure_checkpoint_input_offload(port, *, enabled: bool, pin_memory: bool = True, directory=None):
    """Install/remove the wrapper without replacing an existing backend hook.

    Repeated setup is idempotent. Backend setup must precede this call; a caller
    can explicitly disable the mode to recover its original callable identity.
    """
    from torch.utils.checkpoint import checkpoint
    if directory is not None:
        from pathlib import Path
        directory = str(Path(directory).expanduser().resolve())
        if not enabled:
            raise ValueError('file checkpoint storage requires enabled checkpoint input offload')

    current = getattr(port, 'activation_checkpoint', checkpoint)
    if isinstance(current, CheckpointInputOffload):
        if enabled and current.pin_memory == pin_memory and current.directory == directory:
            return current
        current = current.checkpoint
    if enabled:
        wrapped = CheckpointInputOffload(current, pin_memory=pin_memory, directory=directory)
        port.activation_checkpoint = wrapped
        return wrapped
    port.activation_checkpoint = current
    return current


def checkpoint_input_storage_options(port):
    """Persist the actual installed mode, shared by semantic participants."""
    current = getattr(port, 'activation_checkpoint', None)
    result = {'checkpoint_input_offload': isinstance(current, CheckpointInputOffload)}
    if isinstance(current, CheckpointInputOffload) and current.directory is not None:
        result['checkpoint_input_offload_directory'] = current.directory
    return result


def configure_checkpoint_storage_options(port, *declarations):
    """Apply explicit settings in precedence order, inheriting the installed mode."""
    settings = checkpoint_input_storage_options(port)
    for declaration in reversed(declarations):
        for key in ('checkpoint_input_offload', 'checkpoint_input_offload_directory'):
            if key in declaration:
                settings[key] = declaration[key]
    enabled = settings['checkpoint_input_offload']
    directory = settings.get('checkpoint_input_offload_directory')
    if type(enabled) is not bool or (directory is not None and not isinstance(directory, str)):
        raise ValueError('checkpoint storage needs a boolean mode and optional directory string')
    return configure_checkpoint_input_offload(port, enabled=enabled,
        directory=(directory or None) if enabled else None)
