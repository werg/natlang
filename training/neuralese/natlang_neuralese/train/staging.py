"""First-order function-return adjoints with one producer graph live at a time.

All parameters stay fixed until caller and producer gradients have accumulated.
Return values are temporary leaves; their adjoints are explicitly propagated by
replaying producers in reverse construction order. This is not truncated BPTT.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import gc

import torch

from .memory import cuda_allocated_bytes


class GraphBudgetExceeded(RuntimeError):
    pass


@contextmanager
def graph_memory_budget(limit_bytes, measure=None):
    """Stop a joint attempt before its saved graph consumes backward headroom."""
    if not limit_bytes:
        yield
        return
    measure = measure or cuda_allocated_bytes
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
    auxiliary: object = None
    core_penalty_count: int = 0
    batch: object = None


@dataclass
class WriteBatch:
    compute: object
    nodes: list


class StagedWrites:
    def __init__(self, observe=None, measure=None, observe_batch=None, collect=None):
        self.observe, self.measure = observe, measure
        self.observe_batch = observe_batch
        # Optional synchronous host instrumentation; preserve every collection.
        self.collect = collect or gc.collect
        self.nodes = []
        self.replay_max_abs_error = 0.0

    def add(self, compute, *, auxiliary=None):
        # Use the same grad-enabled forward kernels as replay/joint execution,
        # then release this local graph. Only tiny return payloads survive.
        before = self.measure() if self.measure else 0
        value, penalties = compute()
        retained = max(0, self.measure() - before) if self.measure else 0
        node = WriteNode(compute, value.detach().requires_grad_(True),
                         tuple(float(p.detach()) for p in penalties), auxiliary, len(penalties))
        self.nodes.append(node)
        del value, penalties
        self.collect()
        if auxiliary is not None:
            # The writer and gold-text graph use the same parameters/child
            # leaves but need not coexist. Preserve their original combined
            # local-objective normalization (one term for this producer).
            before = self.measure() if self.measure else 0
            loss = auxiliary()
            # Observations predict the corresponding JOINT producer tape,
            # where these two graphs do coexist. Do not train the router on
            # the smaller split peak and then underestimate joint admission.
            retained += max(0, self.measure() - before) if self.measure else 0
            node.penalty_values = (sum(node.penalty_values) + float(loss.detach()),)
            del loss
            self.collect()
        if self.observe and self.measure:
            self.observe(node.value, retained)
        return node

    def add_batch(self, compute, *, auxiliaries, observe=None):
        """Register one tensor-batched independent producer frontier.

        compute returns (value rows, penalty rows). Rows may be ragged tensors.
        All dependencies must already be nodes; no row may depend on another
        row in this batch. The caller pins membership, padding, lengths and
        checkpoint policy in this closure, including for backward replay.
        Auxiliary graphs are still released separately, retaining the exact
        existing per-producer objective normalization.
        """
        before = self.measure() if self.measure else 0
        values, penalties = compute()
        if not values or len(values) != len(penalties) or len(values) != len(auxiliaries):
            raise ValueError('staged batch rows, penalties and auxiliaries disagree')
        retained = max(0, self.measure() - before) if self.measure else 0
        group = WriteBatch(compute, [])
        for value, terms, auxiliary in zip(values, penalties, auxiliaries):
            group.nodes.append(WriteNode(compute, value.detach().requires_grad_(True),
                tuple(float(p.detach()) for p in terms), auxiliary, len(terms), group))
        self.nodes.extend(group.nodes)
        del value, terms, values, penalties
        self.collect()
        for node in group.nodes:
            if node.auxiliary is not None:
                loss = node.auxiliary()
                node.penalty_values = (sum(node.penalty_values) + float(loss.detach()),)
                del loss
                self.collect()
        # A batch has a different tape geometry. Never feed its averaged bytes
        # into the single-writer admission model and underestimate another path.
        observer = observe or self.observe_batch
        if observer and self.measure:
            # This observer calibrates only the batched writer tape. Gold
            # auxiliary tapes are released separately and never coexist with
            # that tape. Summing them would progressively disable valid batches.
            observer(tuple(n.value for n in group.nodes), retained)
        return tuple(group.nodes)

    @property
    def penalty_count(self):
        return sum(len(node.penalty_values) for node in self.nodes)

    def penalty_loss(self, weight):
        return weight * sum(sum(node.penalty_values) for node in self.nodes) / max(1, self.penalty_count)

    def backward(self, *, penalty_weight=0.0, scale=1.0):
        replayed_batches = set()
        for node in reversed(self.nodes):
            if node.batch is not None:
                if id(node.batch) in replayed_batches:
                    continue
                replayed_batches.add(id(node.batch))
                rows = node.batch.nodes
                values, penalty_rows = node.batch.compute()
                if len(values) != len(rows) or len(penalty_rows) != len(rows):
                    raise RuntimeError('staged replay changed batch membership')
            else:
                rows = (node,)
                value, penalties = node.compute()
                values, penalty_rows = (value,), (penalties,)
            outputs, adjoints = [], []
            penalty = None
            for row, value, penalties in zip(rows, values, penalty_rows):
                if len(penalties) != row.core_penalty_count:
                    raise RuntimeError('staged replay changed the local objective')
                error = float((value.detach() - row.value.detach()).abs().max())
                self.replay_max_abs_error = max(self.replay_max_abs_error, error)
                # Compare within the pinned layout, never against a different
                # singleton GEMM/attention layout with other roundoff.
                torch.testing.assert_close(value.detach(), row.value.detach(), rtol=1e-5, atol=1e-6)
                if row.value.grad is not None and value.requires_grad:
                    outputs.append(value)
                    adjoints.append(row.value.grad.detach())
                for penalty in penalties:
                    if penalty.requires_grad and penalty_weight:
                        outputs.append(penalty)
                        adjoints.append(torch.ones_like(penalty) * penalty_weight * scale / max(1, self.penalty_count))
            if outputs:
                torch.autograd.backward(outputs, adjoints)
            del value, penalties, penalty, values, penalty_rows, outputs, adjoints
            self.collect()
            for row in rows:
                if row.auxiliary is not None and penalty_weight:
                    loss = row.auxiliary()
                    if loss.requires_grad:
                        (loss * penalty_weight * scale / max(1, self.penalty_count)).backward()
                    del loss
                    self.collect()

    def clear(self):
        # Group membership is needed only through replay. Break the group/node
        # cycle so neither contexts nor graph closures await a later full GC.
        for node in self.nodes:
            node.batch = None
        self.nodes.clear()
        self.collect()


def resolve_values(values):
    return {name: value.value if isinstance(value, WriteNode) else value for name, value in values.items()}
