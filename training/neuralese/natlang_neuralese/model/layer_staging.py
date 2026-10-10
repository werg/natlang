"""Layer-lockstep backward: one update's several graphs (sequence passes, precision streams, a preserve stream) are
backpropagated layer by layer, all graphs' layer ``i`` in one engine call, so each layer's weights receive their
whole gradient in one place and can be stepped and freed there (``LionSR.step_in_backward(gated=True)``).

A plain sum of the graphs' losses backpropagated once does not achieve this: autograd runs the later-built graph
completely before the earlier one (it orders nodes by creation), so every layer's partial gradient from the later
graph waits in the engine's input buffers for the earlier graph, which is a whole gradient copy of the model.

While ``LayerStaging`` is active, the ports' ``run_layers`` (decorated with ``staged_layers``) run each layer on a
detached leaf of its input and hand on a detached leaf of its output; the tape keeps (layer, output, leaf). After the
losses' own backward (which ends at those leaves), ``LayerStaging.backward`` walks the layers from the top: one
engine call over every recorded output of layer ``i`` with its leaf's gradient, then the edges that fed layer ``i``
from outside (embeddings, a head's output). The math is the gradient of the summed losses; only the order differs.
A stream whose input depends on a deeper layer of another stream (a producer's output read at another stream's
first layer) cannot be ordered this way; that is detected and refused.
"""
from __future__ import annotations

import functools

import torch

_ACTIVE: list["LayerStaging"] = []


class LayerStaging:
    def __init__(self):
        self.outputs: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}  # layer -> [(output, leaf)]
        self.edges: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}  # layer -> [(source, leaf)]
        self.exits: dict[int, int] = {}  # id(exit leaf) -> layer
        self.processed: list[torch.Tensor] = []

    def __enter__(self):
        if _ACTIVE:
            raise RuntimeError('layer staging does not nest')
        _ACTIVE.append(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE.remove(self)
        return False

    def enter(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        if id(h) in self.exits:
            if self.exits[id(h)] >= layer:
                raise RuntimeError('layer-staged backward: a stream reads a layer at or above its own input layer, '
                                   'so the layers cannot be stepped one at a time')
            return h  # a shallower layer's leaf: this layer's engine call accumulates its gradient
        if not h.requires_grad:
            return h  # an input without a gradient path
        leaf = h.detach().requires_grad_(True)
        self.edges.setdefault(layer, []).append((h, leaf))
        return leaf

    def exit(self, layer: int, out: torch.Tensor) -> torch.Tensor:
        leaf = out.detach().requires_grad_(True)
        self.outputs.setdefault(layer, []).append((out, leaf))
        self.exits[id(leaf)] = layer
        return leaf

    def backward(self):
        """Backpropagate the recorded layers from the top, after the losses' backward reached their leaves."""
        layers = sorted(set(self.outputs) | set(self.edges), reverse=True)
        for layer in layers:
            roots = [(out, leaf.grad) for out, leaf in self.outputs.pop(layer, []) if leaf.grad is not None]
            if roots:
                torch.autograd.backward([out for out, _ in roots], [grad for _, grad in roots])
            for _, leaf in roots:
                leaf.grad = None
                self.processed.append(leaf)
            del roots
            for source, leaf in self.edges.pop(layer, []):
                if leaf.grad is not None:
                    torch.autograd.backward(source, leaf.grad)
                    leaf.grad = None
            if any(done.grad is not None for done in self.processed):
                raise RuntimeError('layer-staged backward: a stream reads a deeper layer of another stream at its '
                                   'input, so the layers cannot be stepped one at a time')
        self.processed.clear()
        self.exits.clear()


def begin() -> LayerStaging:
    """Start a tape for one update outside a ``with`` block (a trainer loop with early exits); a stale tape left by
    an aborted update is dropped. ``end()`` closes it."""
    _ACTIVE.clear()
    tape = LayerStaging()
    _ACTIVE.append(tape)
    return tape


def end() -> None:
    _ACTIVE.clear()


def active() -> LayerStaging | None:
    return _ACTIVE[0] if _ACTIVE else None


def staged_layers(run_layers):
    """Decorate a port's ``run_layers(h, layers, cache, ..., _checkpoint_layer=False)``: under ``LayerStaging`` with
    gradients enabled, run each layer on its own leaves (checkpointing, chunking and caches unchanged)."""
    @functools.wraps(run_layers)
    def wrapper(self, h, layers, cache, *args, _checkpoint_layer=False, **kwargs):
        tape = active()
        if tape is None or _checkpoint_layer or not torch.is_grad_enabled():
            return run_layers(self, h, layers, cache, *args, _checkpoint_layer=_checkpoint_layer, **kwargs)
        for index in layers:
            h, cache = run_layers(self, tape.enter(index, h), range(index, index + 1), cache, *args, **kwargs)
            h = tape.exit(index, h)
        return h, cache
    return wrapper
