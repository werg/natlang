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
A stream whose input is made from a deeper layer of another stream (a direct writer's value, made from the writer's top
layer, read at the caller's first layer: a cross-stream edge) cannot be ordered by layer alone: the caller's input
gradient exists only after the caller's layer 0, when the writer's deeper layers were already passed. Such edges are
resolved before the walk: an activation-only pre-pass (``torch.autograd.grad`` to the layer leaves, so no weight
gradient is accumulated and no latent steps) walks the same layers, sends each cross-stream edge's gradient to the
exit leaves it reads, and the main walk starts with those gradients in place, so each layer's weights still get their
whole gradient in one engine call. When the main walk reaches such an edge, its contribution to those exit leaves
(already delivered) is discarded. The pre-pass repeats while it reaches edges it had not reached before (nested
writes). Any other gradient that reaches an exit leaf after its layer was passed is refused.
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
        self.exit_leaves: dict[int, torch.Tensor] = {}  # id(exit leaf) -> leaf
        self.inputs: dict[int, list[torch.Tensor]] = {}  # layer -> the leaves its graphs start from
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
            self.inputs.setdefault(layer, []).append(h)
            return h  # a shallower layer's leaf: this layer's engine call accumulates its gradient
        if not h.requires_grad:
            return h  # an input without a gradient path
        leaf = h.detach().requires_grad_(True)
        self.edges.setdefault(layer, []).append((h, leaf))
        self.inputs.setdefault(layer, []).append(leaf)
        return leaf

    def exit(self, layer: int, out: torch.Tensor) -> torch.Tensor:
        leaf = out.detach().requires_grad_(True)
        self.outputs.setdefault(layer, []).append((out, leaf))
        self.exits[id(leaf)] = layer
        self.exit_leaves[id(leaf)] = leaf
        return leaf

    def _reads(self, source: torch.Tensor) -> list[torch.Tensor]:
        """The exit leaves an edge's source graph reads (a cross-stream edge), found by walking its autograd graph."""
        found, seen, stack = {}, set(), [source.grad_fn]
        while stack:
            node = stack.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            variable = getattr(node, "variable", None)
            if variable is not None and id(variable) in self.exits:
                found[id(variable)] = variable
            stack.extend(next_node for next_node, _ in node.next_functions)
        return list(found.values())

    def _prepass(self, cross: dict[int, list[torch.Tensor]], seeds: dict[int, torch.Tensor]):
        """Activation-only walk (no weight gradients): the gradient each cross-stream edge sends to each exit leaf it
        reads ({id(edge leaf): {id(exit leaf): gradient}}), given ``seeds`` (earlier such gradients) on top of the
        losses' gradients at the exit leaves."""
        pending = {i: leaf.grad for i, leaf in self.exit_leaves.items() if leaf.grad is not None}
        for i, g in seeds.items():
            pending[i] = g if i not in pending else pending[i] + g
        sent: dict[int, dict[int, torch.Tensor]] = {}
        for layer in sorted(set(self.outputs) | set(self.edges), reverse=True):
            roots = [(out, pending.pop(id(leaf))) for out, leaf in self.outputs.get(layer, []) if id(leaf) in pending]
            inputs = self.inputs.get(layer, [])
            if roots and inputs:
                grads = torch.autograd.grad([out for out, _ in roots], inputs, [g for _, g in roots],
                                            retain_graph=True, allow_unused=True)
                for leaf, g in zip(inputs, grads):
                    if g is not None:
                        pending[id(leaf)] = g if id(leaf) not in pending else pending[id(leaf)] + g
            for source, leaf in self.edges.get(layer, []):
                reads = cross.get(id(leaf))
                g = pending.get(id(leaf))
                if not reads or g is None:
                    continue
                grads = torch.autograd.grad(source, reads, g, retain_graph=True, allow_unused=True)
                sent[id(leaf)] = {id(r): rg for r, rg in zip(reads, grads) if rg is not None}
        return sent

    def backward(self):
        """Backpropagate the recorded layers from the top, after the losses' backward reached their leaves."""
        layers = sorted(set(self.outputs) | set(self.edges), reverse=True)
        # Cross-stream edges: an edge at layer l whose source reads exit leaves of layers >= l (a writer's value at a
        # reader's input). Their gradients go to those leaves before the walk (activation-only pre-pass).
        cross = {}
        for layer, edges in self.edges.items():
            for source, leaf in edges:
                reads = [r for r in self._reads(source) if self.exits[id(r)] >= layer]
                if reads:
                    cross[id(leaf)] = reads
        sent: dict[int, dict[int, torch.Tensor]] = {}
        if cross:
            # A newly reached nested edge can also change an edge reached in an
            # earlier round. Converge the gradients, not merely the set of edges.
            # The cross-stream graph is acyclic, so each round resolves at least
            # one more dependency; a cap exposes an unexpected cycle.
            for _ in range(len(cross) + 2):
                seeds: dict[int, torch.Tensor] = {}
                for per_edge in sent.values():
                    for i, g in per_edge.items():
                        seeds[i] = g if i not in seeds else seeds[i] + g
                reached = self._prepass(cross, seeds)
                stable = reached.keys() == sent.keys() and all(
                    reached[edge].keys() == sent[edge].keys() and all(
                        torch.allclose(g.float(), sent[edge][read].float(), rtol=1e-4, atol=1e-7)
                        for read, g in values.items())
                    for edge, values in reached.items())
                sent = reached
                if stable:
                    break
            else:
                raise RuntimeError('layer-staged backward: cross-stream pre-pass did not converge')
            for per_edge in sent.values():
                for i, g in per_edge.items():
                    leaf = self.exit_leaves[i]
                    leaf.grad = g if leaf.grad is None else leaf.grad + g
        passed: list[torch.Tensor] = []
        for layer in layers:
            outputs = self.outputs.pop(layer, [])
            roots = [(out, leaf.grad) for out, leaf in outputs if leaf.grad is not None]
            if roots:
                torch.autograd.backward([out for out, _ in roots], [grad for _, grad in roots])
            for _, leaf in roots:
                self.processed.append(leaf)
            for _, leaf in outputs:
                leaf.grad = None
                passed.append(leaf)
            del roots, outputs
            for source, leaf in self.edges.pop(layer, []):
                if leaf.grad is not None:
                    reads = cross.get(id(leaf), ())
                    before = {id(r): r.grad for r in reads}
                    torch.autograd.backward(source, leaf.grad)
                    leaf.grad = None
                    expected = sent.get(id(leaf), {})
                    for read in reads:
                        arrived = read.grad if before[id(read)] is None else read.grad - before[id(read)]
                        seed = expected.get(id(read))
                        if (arrived is None) != (seed is None) or (seed is not None and not torch.allclose(
                                arrived.float(), seed.float(), rtol=1e-3, atol=1e-6)):
                            delta = (float((arrived.float() - seed.float()).abs().max())
                                     if arrived is not None and seed is not None else None)
                            raise RuntimeError('layer-staged backward: a cross-stream edge sent a different gradient '
                                               f'than its pre-pass (layer={layer}, max_abs_delta={delta})')
                        read.grad = before[id(read)]  # already delivered by the pre-pass, before its layer
            if any(done.grad is not None for done in passed):
                raise RuntimeError('layer-staged backward: a gradient reached a layer after it was stepped (a stream '
                                   'reads a deeper layer of another stream outside a resolvable edge)')
        self.processed.clear()
        self.exits.clear()
        self.exit_leaves.clear()
        self.inputs.clear()


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


def tape_for_grad() -> LayerStaging | None:
    """The active tape when this forward builds a graph (else None): a layer loop outside ``run_layers`` (the
    isolated-sequence pass, the batched decode step) wraps each layer's input in ``enter`` and its output in ``exit``,
    so that its layers' weights get their gradient in the same per-layer engine call as every other stream's."""
    tape = active()
    return tape if tape is not None and torch.is_grad_enabled() else None


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
