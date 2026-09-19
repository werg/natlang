"""Bounded host-source contract for eventful Fold inputs."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Poll:
    kind: str  # item | empty | closed | failed
    value: Any = None
    detail: str = ""

    def __post_init__(self):
        if self.kind not in ("item", "empty", "closed", "failed"):
            raise ValueError("unknown stream poll result")


class Source(Protocol):
    def poll(self) -> Poll: ...


class QueueSource:
    """A bounded producer queue; poll never treats payload text as control."""

    def __init__(self, capacity: int = 32):
        if capacity < 1:
            raise ValueError("stream capacity must be positive")
        self.capacity = capacity
        self.items = deque()
        self.terminal: Poll | None = None

    def put(self, item):
        if self.terminal is not None:
            raise ValueError("source already closed or failed")
        if len(self.items) >= self.capacity:
            raise BufferError("stream producer queue is full")
        self.items.append(item)

    def close(self):
        self.terminal = Poll("closed")

    def fail(self, detail: str):
        self.terminal = Poll("failed", detail=detail)

    def poll(self) -> Poll:
        if self.items:
            return Poll("item", self.items.popleft())
        return self.terminal or Poll("empty")


class StreamBuffer:
    """One admitted Fold item, retained until its step commits."""

    def __init__(self, source: Source):
        self.source = source
        self.current: Poll | None = None
        self.position = 0

    def peek(self) -> Poll:
        if self.current is None:
            self.current = self.source.poll()
        if self.current.kind == "empty":
            self.current = None
            return Poll("empty")
        return self.current

    def ack(self):
        if self.current is None or self.current.kind != "item":
            raise ValueError("no admitted stream item to acknowledge")
        self.current = None
        self.position += 1

    def __deepcopy__(self, memo):
        return self


class StreamDriver:
    """Drive a whole Fold invocation at each host event boundary."""

    def __init__(self, runtime, root):
        self.runtime, self.root = runtime, root
        self.terminal = None

    def drive(self):
        if self.terminal is not None:
            return self.terminal
        outcome, value = self.runtime.run_root(self.root)
        if outcome.kind != "waiting":
            self.terminal = (outcome, value)
        return outcome, value
