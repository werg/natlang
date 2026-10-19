"""Immutable physical-row provenance for graph parameter-gradient reductions.

This module is exact execution plumbing. It never evaluates a graph, changes
attention, or chooses a training policy.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from itertools import count

_ACTIVE = ContextVar('natlang_graph_gradient_partitions', default=None)
_FRAME_IDS = count()


@dataclass(frozen=True)
class GradientPartitions:
    """Contiguous invocation ownership of the current token-local operands."""
    lengths: tuple[int, ...]
    invocation_index: int | None = None
    chunk_offset: int | None = None
    invocation_length: int | None = None
    invocation_namespace: tuple[int, ...] | None = None

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in self.lengths):
            raise ValueError('gradient partition lengths must be nonnegative integers')
        if self.invocation_index is not None and (
                type(self.invocation_index) is not int or self.invocation_index < 0):
            raise ValueError('invocation index must be a nonnegative integer')
        chunk_metadata = (self.chunk_offset, self.invocation_length)
        if any(value is not None for value in chunk_metadata):
            if (self.invocation_index is None or
                    any(type(value) is not int or value < 0 for value in chunk_metadata)):
                raise ValueError('chunk provenance requires an invocation index, offset, and length')
            if len(self.lengths) != 1 or sum(self.lengths) != self.lengths[0]:
                raise ValueError('one FFN chunk must belong to exactly one physical invocation')
            if self.chunk_offset + self.lengths[0] > self.invocation_length:
                raise ValueError('FFN chunk escapes its parent invocation')
        if self.invocation_namespace is not None and (
                not isinstance(self.invocation_namespace, tuple) or not self.invocation_namespace or
                any(type(value) is not int or value < 0 for value in self.invocation_namespace)):
            raise ValueError('chunk invocation namespace must be a nonempty tuple of nonnegative integers')

    @property
    def token_count(self):
        return sum(self.lengths)

    def slices(self, *, reverse=False):
        start = 0
        result = []
        for count in self.lengths:
            if count:
                result.append(slice(start, start + count))
            start += count
        return tuple(reversed(result)) if reverse else tuple(result)



def active_gradient_partitions():
    return _ACTIVE.get()


def new_gradient_namespace():
    """Allocate one immutable forward-frame identity before checkpointing."""
    return (next(_FRAME_IDS),)


@contextmanager
def gradient_partitions(partitions):
    token = _ACTIVE.set(partitions)
    try:
        yield partitions
    finally:
        _ACTIVE.reset(token)


def capture_gradient_partitions(callback):
    """Bind immutable forward provenance across checkpoint recomputation."""
    partitions = active_gradient_partitions()
    @wraps(callback)
    def execute(*args, **kwargs):
        with gradient_partitions(partitions):
            return callback(*args, **kwargs)
    return execute


def token_slices(partitions, count, *, reverse=True):
    if partitions is None or sum(length > 0 for length in partitions.lengths) <= 1:
        # A token-local expert may select/reorder/pad rows within one known
        # invocation. Its boundary remains unique regardless of routed count.
        return (slice(0, count),)
    if partitions.token_count != count:
        raise RuntimeError('graph cotangent operands lost explicit physical-row provenance')
    return partitions.slices(reverse=reverse)


def reduce_parameter_rows(partitions, count, contribution, *, add=None):
    """Deliver reverse-invocation FP32 contributions without another model pass."""
    total = None
    for section in token_slices(partitions, count):
        value = contribution(section)
        if add is not None:
            add(value)
        elif total is None:
            total = value
        else:
            total = total + value
    return total


def token_local_chunks(h, chunk_tokens, callback):
    """Evaluate each physical FFN row once, with invocation-local routing plans.

    The layer-major graph still visits each layer once. FFN is token-local;
    keeping invocation boundaries here preserves cached expert padding and local
    chunk boundaries without replaying any attention or state-producing layer.
    """
    import torch
    partitions = active_gradient_partitions()
    if partitions is None:
        if chunk_tokens <= 0 or h.shape[1] <= chunk_tokens:
            return callback(h)
        return torch.cat([callback(part) for part in h.split(chunk_tokens, dim=1)], dim=1)
    if h.ndim != 3 or h.shape[0] != 1:
        raise RuntimeError('graph FFN chunks lost their token axis')
    sections = token_slices(partitions, h.shape[1], reverse=False)
    # Actor-specific execution and separate forward calls can reuse the same
    # local invocation index. Give this chunking call a process-local order
    # key so those distinct invocations remain distinct at a shared anchor.
    namespace = partitions.invocation_namespace or (next(_FRAME_IDS),)
    outputs = []
    for invocation_index, section in enumerate(sections):
        count = section.stop - section.start
        width = chunk_tokens if chunk_tokens > 0 else count
        for start in range(section.start, section.stop, width):
            part = h[:, start:min(start + width, section.stop)]
            with gradient_partitions(GradientPartitions(
                    (part.shape[1],),
                    invocation_index=(partitions.invocation_index if partitions.invocation_index is not None
                                      else invocation_index),
                    chunk_offset=start - section.start, invocation_length=count,
                    invocation_namespace=namespace)):
                outputs.append(callback(part))
    return torch.cat(outputs, dim=1) if outputs else h
