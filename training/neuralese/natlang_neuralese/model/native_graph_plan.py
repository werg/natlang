"""Exact positional graph structure and compiler, independent of neural kernels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch


@dataclass(frozen=True)
class ImportSpan:
    at: int
    source: str
    start: int
    stop: int
    logical: bool = False
    stop_gradient: bool = False


@dataclass(frozen=True)
class Invocation:
    name: str
    token_ids: tuple[int, ...]
    imports: tuple[ImportSpan, ...] = ()
    producing: tuple[tuple[int, int], ...] = ()
    logical_producing: tuple[tuple[int, int], ...] = ()
    # Internal exact prefix reuse, distinct from learned Neuralese production.
    # Raw complete causal state is exportable without adapting its embeddings.
    cache_prefix: bool = False


def invocation_dict(node):
    """Stable semantic serialization, preserving pre-cache graph identities."""
    value = asdict(node)
    if not node.cache_prefix:
        value.pop('cache_prefix')
    for edge in value['imports']:
        if not edge['stop_gradient']:edge.pop('stop_gradient')
    return value


@dataclass(frozen=True)
class GraphPlan:
    invocations: tuple[Invocation, ...]
    offsets: tuple[int, ...]
    sequences: tuple[tuple[int, ...], ...]
    own_positions: tuple[int, ...]
    slot_sources: tuple[int, ...]
    slot_positions: tuple[int, ...]
    node_ids: tuple[int, ...]
    slot_offsets: tuple[int, ...]
    producing: tuple[bool, ...]
    external_names: tuple[str, ...] = ()
    external_sizes: tuple[int, ...] = ()
    boundaries: tuple[tuple[int, ...], ...] = ()
    slot_stops: tuple[bool, ...] = ()

    @property
    def token_count(self):
        return len(self.own_positions)

    @property
    def attention_workspace_rows(self):
        """Q-only allocation size; never counted as real data exposure."""
        return sum((len(node.token_ids)+127)//128*128 for node in self.invocations)

    def tensor(self, values, device, dtype=torch.long):
        return torch.tensor(values, device=device, dtype=dtype)

    def history(self, row, width=None):
        node = self.node_ids[row]
        sequence = self.sequences[node]
        end = self.own_positions[row] + 1
        return sequence[max(0, end-width) if width is not None else 0:end]

    def history_stops(self,row,width=None):
        node=self.node_ids[row]
        end=self.own_positions[row]+1
        start=max(0,end-width) if width is not None else 0
        offset=self.slot_offsets[node]
        return self.slot_stops[offset+start:offset+end] if self.slot_stops else (False,)*(end-start)

    @property
    def allowed(self):
        # Diagnostic materialization only; production plans remain linear size.
        return tuple(tuple(range(self.slot_offsets[n], self.slot_offsets[n]+p+1))
                     for n, p in zip(self.node_ids, self.own_positions))

    def bounds(self, device, window=None):
        starts = self.tensor([self.slot_offsets[n] for n in self.node_ids], device)
        ends = starts + self.tensor(self.own_positions, device)
        if window is not None:
            starts = torch.maximum(starts, ends-window+1)
        return starts, ends

    def attention_mask(self, device, window=None):
        starts, ends = self.bounds(device, window)
        keys = torch.arange(len(self.slot_sources), device=device)
        return ((keys[None] >= starts[:, None]) & (keys[None] <= ends[:, None]))[None, None]


def compile_graph(invocations: tuple[Invocation, ...] | list[Invocation], *, external_sizes=None, kv_block_alignment=128) -> GraphPlan:
    invocations = tuple(invocations)
    if any(type(edge.stop_gradient) is not bool for node in invocations for edge in node.imports):
        raise ValueError('import stop_gradient must be an explicit boolean')
    stopped=any(edge.stop_gradient for node in invocations for edge in node.imports)
    sequence_stops,slot_stops=[],[]
    if not isinstance(kv_block_alignment, int) or kv_block_alignment < 1:
        raise ValueError('KV block alignment must be a positive integer')
    external_sizes = external_sizes or {}
    if any(not isinstance(size, int) or size < 0 for size in external_sizes.values()):
        raise ValueError('external state sizes must be nonnegative integers')
    external_offsets = {}
    next_external = sum(len(node.token_ids) for node in invocations)
    for name, size in external_sizes.items():
        external_offsets[name] = next_external
        next_external += size
    known, offsets, sequences, boundaries = {}, [], [], []
    own_positions, sources, positions, node_ids, slot_offsets, producing = [], [], [], [], [], []
    physical_count = 0
    for node in invocations:
        if type(node.cache_prefix) is not bool or (node.cache_prefix and
                (node.producing or node.logical_producing or len(node.imports) > 1)):
            raise ValueError('raw prefix caches require unadapted tokens and at most one complete prefix import')
        if node.name in known or node.name in external_sizes:
            raise ValueError('invocations need unique names distinct from external state names')
        offset = physical_count
        offsets.append(offset)
        count = len(node.token_ids)
        producer = [False] * count
        for start, stop in node.producing:
            if not 0 <= start <= stop <= count:
                raise ValueError('producing span is outside invocation tokens')
            producer[start:stop] = [True] * (stop-start)
        edges = {}
        for edge in node.imports:
            if not 0 <= edge.at <= count:
                raise ValueError('imports require a valid insertion boundary')
            if edge.source in external_sizes:
                if not 0 <= edge.start <= edge.stop <= external_sizes[edge.source]:
                    raise ValueError('import span is outside external state tokens')
                ext_offset = external_offsets[edge.source]
                imported = tuple(range(ext_offset+edge.start, ext_offset+edge.stop))
                imported_stops=(edge.stop_gradient,)*len(imported) if stopped else ()
            else:
                if edge.source not in known:
                    raise ValueError('imports require an earlier source or declared external state')
                source_node, source_offset, source_index = known[edge.source]
                if edge.logical:
                    lo, hi = edge.start, edge.stop
                    declared = source_node.logical_producing
                else:
                    if not 0 <= edge.start <= edge.stop <= len(source_node.token_ids):
                        raise ValueError('import span is outside source tokens')
                    lo, hi = boundaries[source_index][edge.start], boundaries[source_index][edge.stop]
                    declared = tuple((boundaries[source_index][a], boundaries[source_index][b])
                                     for a, b in source_node.producing)
                if source_node.cache_prefix:
                    if not edge.logical or edge.at != 0 or lo != 0 or hi != len(sequences[source_index]):
                        raise ValueError('raw prefix cache imports must include the complete causal prefix at position zero')
                    declared = ((0, len(sequences[source_index])),)
                if not 0 <= lo <= hi <= len(sequences[source_index]):
                    raise ValueError('import span is outside the source logical state')
                if not any(a <= lo and hi <= b for a, b in declared):
                    raise ValueError('imports must reference an explicitly producing span')
                imported = sequences[source_index][lo:hi]
                imported_stops=tuple(flag or edge.stop_gradient for flag in sequence_stops[source_index][lo:hi]) if stopped else ()
            if node.cache_prefix and (edge.source in external_sizes or not source_node.cache_prefix):
                raise ValueError('raw prefix caches can extend only another raw complete prefix cache')
            edges.setdefault(edge.at, []).append((imported,imported_stops))
        logical = []
        logical_stops=[]
        own = []
        boundary = []
        for local in range(count+1):
            boundary.append(len(logical))
            for imported,imported_stops in edges.get(local, ()):
                logical.extend(imported)
                if stopped:logical_stops.extend(imported_stops)
            if local < count:
                own.append(len(logical))
                logical.append(offset+local)
                if stopped:logical_stops.append(False)
        padding_slots = (-len(sources)) % kv_block_alignment
        sources.extend([-1] * padding_slots)
        if stopped:slot_stops.extend([False]*padding_slots)
        positions.extend([0] * padding_slots)
        slot_offset = len(sources)
        slot_offsets.append(slot_offset)
        sources.extend(logical)
        if stopped:slot_stops.extend(logical_stops)
        positions.extend(range(len(logical)))
        for pos in own:
            own_positions.append(pos)
            node_ids.append(len(sequences))
        for start, stop in node.logical_producing:
            if not 0 <= start <= stop <= len(logical):
                raise ValueError('logical producing span is outside invocation state')
            # Logical exports may mix own queries with already-produced aliases.
            # Apply the embedding transform to their own physical rows only;
            # imports retain the originating producer's existing transform.
            for local, position in enumerate(own):
                if start <= position < stop:
                    producer[local] = True
        if not logical:
            raise ValueError('an invocation must contain tokens or imported state')
        producing.extend(producer)
        sequences.append(tuple(logical))
        if stopped:sequence_stops.append(tuple(logical_stops))
        boundaries.append(tuple(boundary))
        known[node.name] = node, offset, len(sequences)-1
        physical_count += count
    if not invocations:
        raise ValueError('graph cannot be empty')
    return GraphPlan(invocations, tuple(offsets), tuple(sequences), tuple(own_positions),
                     tuple(sources), tuple(positions), tuple(node_ids), tuple(slot_offsets), tuple(producing),
                     tuple(external_sizes), tuple(external_sizes.values()), tuple(boundaries),tuple(slot_stops))


