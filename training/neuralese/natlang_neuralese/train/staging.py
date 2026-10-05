"""First-order function-return adjoints with one producer graph live at a time.

All parameters stay fixed until caller and producer gradients have accumulated.
Return values are temporary leaves; their adjoints are explicitly propagated by
replaying producers in reverse construction order. This is not truncated BPTT.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import gc

import torch


class GraphBudgetExceeded(RuntimeError):
    pass


@contextmanager
def graph_memory_budget(limit_bytes, measure=None):
    """Stop a joint attempt before its saved graph consumes backward headroom."""
    if not limit_bytes:
        yield
        return
    measure = measure or torch.cuda.memory_allocated
    def pack(value):
        if measure() > limit_bytes:
            raise GraphBudgetExceeded(f'joint graph exceeded {limit_bytes} bytes')
        # Do not introduce a saved-hook reference cycle through its grad_fn.
        return value.detach()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        yield


@dataclass
class WriteNode:
    compute: object
    value: torch.Tensor
    penalty_values: tuple


class StagedWrites:
    def __init__(self, observe=None, measure=None):
        self.observe, self.measure = observe, measure
        self.nodes = []
        self.replay_max_abs_error = 0.0

    def add(self, compute):
        # Use the same grad-enabled forward kernels as replay/joint execution,
        # then release this local graph. Only tiny return payloads survive.
        before = self.measure() if self.measure else 0
        value, penalties = compute()
        if self.observe and self.measure:
            self.observe(value, max(0, self.measure() - before))
        node = WriteNode(compute, value.detach().requires_grad_(True),
                         tuple(float(p.detach()) for p in penalties))
        self.nodes.append(node)
        del value, penalties
        gc.collect()
        return node

    @property
    def penalty_count(self):
        return sum(len(node.penalty_values) for node in self.nodes)

    def penalty_loss(self, weight):
        return weight * sum(sum(node.penalty_values) for node in self.nodes) / max(1, self.penalty_count)

    def backward(self, *, penalty_weight=0.0, scale=1.0):
        for node in reversed(self.nodes):
            value, penalties = node.compute()
            if len(penalties) != len(node.penalty_values):
                raise RuntimeError('staged replay changed the local objective')
            error = float((value.detach() - node.value.detach()).abs().max())
            self.replay_max_abs_error = max(self.replay_max_abs_error, error)
            # Identical parameters, control decisions, layout and kernels must
            # reproduce the primal. Never train from a changed continuation.
            torch.testing.assert_close(value.detach(), node.value.detach(), rtol=1e-5, atol=1e-6)
            outputs, adjoints = [], []
            if node.value.grad is not None and value.requires_grad:
                outputs.append(value)
                adjoints.append(node.value.grad.detach())
            for penalty in penalties:
                if penalty.requires_grad and penalty_weight:
                    outputs.append(penalty)
                    adjoints.append(torch.ones_like(penalty) * penalty_weight * scale / max(1, self.penalty_count))
            if outputs:
                torch.autograd.backward(outputs, adjoints)
            del value, penalties, outputs, adjoints
            gc.collect()

    def clear(self):
        self.nodes.clear()
        gc.collect()


def resolve_values(values):
    return {name: value.value if isinstance(value, WriteNode) else value for name, value in values.items()}
