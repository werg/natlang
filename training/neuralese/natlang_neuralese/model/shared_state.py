"""Layer-major Neuralese graphs: native state aliases, never latent recurrence.

Physical token rows execute once per layer. Logical KV slots alias physical rows
and apply RoPE at the consumer position. LFM convolution uses a bounded gather
of the same layer's gated inputs. Source private context is never imported.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ImportSpan:
    at: int
    source: str
    start: int
    stop: int
    logical: bool = False


@dataclass(frozen=True)
class Invocation:
    name: str
    token_ids: tuple[int, ...]
    imports: tuple[ImportSpan, ...] = ()
    producing: tuple[tuple[int, int], ...] = ()
    logical_producing: tuple[tuple[int, int], ...] = ()


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
                if not 0 <= lo <= hi <= len(sequences[source_index]):
                    raise ValueError('import span is outside the source logical state')
                if not any(a <= lo and hi <= b for a, b in declared):
                    raise ValueError('imports must reference an explicitly producing span')
                imported = sequences[source_index][lo:hi]
            edges.setdefault(edge.at, []).append(imported)
        logical = []
        own = []
        boundary = []
        for local in range(count+1):
            boundary.append(len(logical))
            for imported in edges.get(local, ()):
                logical.extend(imported)
            if local < count:
                own.append(len(logical))
                logical.append(offset+local)
        padding_slots = (-len(sources)) % kv_block_alignment
        sources.extend([-1] * padding_slots)
        positions.extend([0] * padding_slots)
        slot_offset = len(sources)
        slot_offsets.append(slot_offset)
        sources.extend(logical)
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
        boundaries.append(tuple(boundary))
        known[node.name] = node, offset, len(sequences)-1
        physical_count += count
    if not invocations:
        raise ValueError('graph cannot be empty')
    return GraphPlan(invocations, tuple(offsets), tuple(sequences), tuple(own_positions),
                     tuple(sources), tuple(positions), tuple(node_ids), tuple(slot_offsets), tuple(producing),
                     tuple(external_sizes), tuple(external_sizes.values()), tuple(boundaries))


class LowRankTokenEmbedding(nn.Module):
    """Identity-initialized low-rank residual within token embedding space."""
    def __init__(self, width: int, rank: int, *, device=None, dtype=None):
        super().__init__()
        if not 0 < rank <= width:
            raise ValueError('embedding rank must be positive and no greater than width')
        self.down = nn.Linear(width, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, width, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.up.weight)

    def forward(self, embeddings, producing):
        update = _linear(self.up, _linear(self.down, embeddings.to(self.down.weight.dtype)))
        return embeddings + (update * producing[..., None].to(update.dtype)).to(embeddings.dtype)


def _rotate(port, tensor, positions, layer):
    # Apply existing model-specific RoPE tables at logical consumer positions.
    dummy = tensor.new_empty(1, len(positions), port.embedding_weight.shape[1])
    cos, sin = port._rope(dummy, positions[None])
    if hasattr(port, '_apply_rope'):
        return port._apply_rope(layer, tensor, tensor, cos, sin)[0]
    from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb
    return apply_rotary_pos_emb(tensor, tensor, cos, sin)[0]


def _layer_inputs(port, layer, h, attention):
    x = layer.input_layernorm(h) if hasattr(layer, 'input_layernorm') else layer.operator_norm(h)
    if not attention:
        b, c, xx = _linear(layer.conv.in_proj, x).chunk(3, dim=-1)
        return (b * xx).float(), c
    attn = layer.self_attn
    batch, steps, _ = x.shape
    hd = attn.head_dim
    norm_q = getattr(attn, 'q_norm', None) or attn.q_layernorm
    norm_k = getattr(attn, 'k_norm', None) or attn.k_layernorm
    q = norm_q(_linear(attn.q_proj, x).view(batch, steps, -1, hd)).transpose(1, 2)
    k = norm_k(_linear(attn.k_proj, x).view(batch, steps, -1, hd)).transpose(1, 2)
    v = _linear(attn.v_proj, x).view(batch, steps, -1, hd).transpose(1, 2)
    return q, k.float(), v.float()


def _finish(port, layer, h, out, attention):
    if attention:
        projection = getattr(layer.self_attn, 'o_proj', None) or layer.self_attn.out_proj
        h = h + _linear(projection, out.transpose(1, 2).reshape(h.shape[0], h.shape[1], -1))
    else:
        h = h + _linear(layer.conv.out_proj, out)
    if hasattr(port, '_feed_forward'):
        return port._feed_forward(layer, h)
    from .lfm2_port import feed_forward_residual
    return feed_forward_residual(layer, h, getattr(port, 'ffn_chunk_tokens', 0))


def _convolve(layer, gated, c, plan, rows):
    kernel = layer.conv.conv
    width = kernel.weight.shape[-1]
    predecessor = [([-1] * max(0, width-len(plan.history(r, width))) + list(plan.history(r, width))) for r in rows]
    index = torch.tensor(predecessor, device=gated.device)
    padded = torch.cat([gated, gated.new_zeros(1, 1, gated.shape[-1])], dim=1)
    gathered = padded[:, index]  # B,Q,W,C; -1 selects the explicit zero row
    out = (gathered.float() * kernel.weight[:, 0].T.float()[None, None]).sum(2)
    if kernel.bias is not None:
        out = out + kernel.bias.float()
    return c * out.to(c.dtype)


def forward_graph(port, plan: GraphPlan, adapter: LowRankTokenEmbedding | None = None, *, embeddings=None, external_blocks=None, external_identity=None, retain_states=True):
    """Single layer-major DAG pass. No source trajectory is rerun per import."""
    device = port.embedding_weight.device
    ids = plan.tensor([t for n in plan.invocations for t in n.token_ids], device)
    h = port.embed(ids[None]) if embeddings is None else embeddings
    if h.shape[:2] != (1, plan.token_count):
        raise ValueError('graph embeddings must contain exactly its unique physical tokens')
    if adapter is not None:
        h = adapter(h, plan.tensor(plan.producing, device, torch.bool)[None])
    slots = plan.tensor(plan.slot_sources, device)
    key_positions = plan.tensor(plan.slot_positions, device)
    query_positions = plan.tensor(plan.own_positions, device)
    external = _external_layers(port, plan, external_blocks, external_identity)
    if plan.token_count == 0:
        state = tuple(tuple(torch.cat([item[i][j] for item in external], dim=2 if port.is_attention(i) else 1)
                            for j in range(2 if port.is_attention(i) else 1)) for i in range(port.num_layers))
        return {'h_final': h, 'states': state if retain_states else (), 'plan': plan}
    state = []
    masks = {}
    checkpointed = bool(getattr(port, 'checkpoint_layers', False)) and torch.is_grad_enabled()
    for i, layer in enumerate(port.layers):
        attention = port.is_attention(i)
        window = port.window(i) if attention and hasattr(port, 'window') else None
        if attention and window not in masks:
            masks[window] = graph_block_mask(plan, device, window) if h.is_cuda else plan.attention_mask(device, window)
        external_i = tuple(value for item in external for value in item[i])

        def run_layer(value, *imported, index=i, attention=attention, window=window):
            layer = port.layers[index]
            values = _layer_inputs(port, layer, value, attention)
            if attention:
                q, k, v = values
                if imported:
                    k = torch.cat([k, *imported[0::2]], dim=2)
                    v = torch.cat([v, *imported[1::2]], dim=2)
                captured = (k, v)
                q = _rotate(port, q, query_positions, index)
                kr = _rotate(port, k[:, :, slots].to(q.dtype), key_positions, index)
                out = graph_attention(q, kr, v[:, :, slots].to(q.dtype), plan, window,
                                      layer.self_attn.scaling, prepared_mask=masks[window])
            else:
                gated, c = values
                if imported:
                    gated = torch.cat([gated, *imported], dim=1)
                captured = (gated,)
                out = _convolve(layer, gated, c, plan, range(plan.token_count))
            updated = _finish(port, layer, value, out, attention)
            return (updated, *captured) if retain_states else (updated,)

        if checkpointed:
            from torch.utils.checkpoint import checkpoint
            updated = checkpoint(run_layer, h, *external_i, use_reentrant=False,
                                 preserve_rng_state=getattr(port, 'checkpoint_preserve_rng', True))
        else:
            updated = run_layer(h, *external_i)
        h = updated[0]
        if retain_states:
            state.append(updated[1:])
    return {'h_final': h, 'states': tuple(state), 'plan': plan}


def forward_reference(port, plan: GraphPlan, adapter=None, *, embeddings=None, external_blocks=None, external_identity=None):
    """Slow invocation-major native-cache oracle, for correctness checks only.

    Each completed source retains raw layer states. Each consumer rotates only
    its imported cache keys to its local positions. Per-invocation cache
    construction and conv1d are independent of the flattened graph mask/gather;
    attention uses the actual native serving kernel to preserve arithmetic.
    """
    device = port.embedding_weight.device
    if plan.token_count == 0:
        return forward_graph(port, plan, adapter, embeddings=embeddings, external_blocks=external_blocks, external_identity=external_identity)
    external = _external_layers(port, plan, external_blocks, external_identity)
    layer_cache = [[] for _ in port.layers]
    outputs = []
    for node, offset, logical in zip(plan.invocations, plan.offsets, plan.sequences):
        count = len(node.token_ids)
        if count == 0:
            continue
        def remap(rows):
            return tuple(row if row < plan.token_count else row-plan.token_count+offset+count for row in rows)
        h = (port.embed(plan.tensor(node.token_ids, device)[None]) if embeddings is None
             else embeddings[:, offset:offset+count])
        if adapter is not None:
            h = adapter(h, plan.tensor(plan.producing[offset:offset+count], device, torch.bool)[None])
        for i, layer in enumerate(port.layers):
            attention = port.is_attention(i)
            values = _layer_inputs(port, layer, h, attention)
            cache = layer_cache[i]
            if attention:
                q, k, v = values
                source_k = torch.cat([entry[0] for entry in cache] + [k] + [item[i][0] for item in external], dim=2)
                source_v = torch.cat([entry[1] for entry in cache] + [v] + [item[i][1] for item in external], dim=2)
                cache.append((k, v))
                window = port.window(i) if hasattr(port, 'window') else None
                indices = plan.tensor(remap(logical), device)
                kr = _rotate(port, source_k[:, :, indices].to(q.dtype), torch.arange(len(logical), device=device), i)
                query_positions = plan.own_positions[offset:offset+count]
                qr = _rotate(port, q, plan.tensor(query_positions, device), i)
                layout = AttentionLayout(query_positions, len(logical))
                out = graph_attention(qr, kr, source_v[:, :, indices].to(q.dtype), layout, window, layer.self_attn.scaling)

            else:
                gated, c = values
                source = torch.cat([entry[0] for entry in cache] + [gated] + [item[i][0] for item in external], dim=1)
                cache.append((gated,))
                kernel = layer.conv.conv
                width = kernel.weight.shape[-1]
                parts = []
                for local in range(count):
                    history = remap(plan.history(offset+local, width))
                    data = source[:, plan.tensor(history, device)].transpose(1, 2)
                    data = F.pad(data, (width-data.shape[-1], 0))
                    conv = F.conv1d(data.to(c.dtype), kernel.weight, kernel.bias, groups=data.shape[1])
                    parts.append(conv.transpose(1, 2) * c[:, local:local+1])
                out = torch.cat(parts, dim=1)
            h = _finish(port, layer, h, out, attention)
        outputs.append(h)
    states = tuple(tuple(torch.cat([entry[field] for entry in cache], dim=2 if len(cache[0]) == 2 else 1)
                         for field in range(len(cache[0]))) for cache in layer_cache)
    if external:
        states = tuple(tuple(torch.cat([field, *[item[i][j] for item in external]], dim=2 if len(values) == 2 else 1)
                             for j, field in enumerate(values)) for i, values in enumerate(states))
    return {'h_final': torch.cat(outputs, dim=1), 'states': states, 'plan': plan}


PRODUCE_OPEN = '<|neuralese_produce|>'
PRODUCE_CLOSE = '<|/neuralese_produce|>'
CONSUME_OPEN = '<|neuralese_consume|>'
CONSUME_CLOSE = '<|/neuralese_consume|>'
ROLE_MARKERS = (PRODUCE_OPEN, PRODUCE_CLOSE, CONSUME_OPEN, CONSUME_CLOSE)


_flex = {}


def graph_block_mask(plan, device, window=None):
    """Sparse 128-token geometry without allocating a token-by-token matrix."""
    from torch.nn.attention.flex_attention import BlockMask
    if isinstance(plan, GraphPlan):
        plan, _ = query_workspace(plan)
    starts, ends = plan.bounds(device, window)
    block = 128
    padding = (-plan.token_count) % block
    lo = F.pad(starts, (0, padding), value=len(plan.slot_sources))
    hi = F.pad(ends, (0, padding), value=-1)
    min_lo = lo.view(-1, block).amin(1)[None, None, :, None]
    max_lo = F.pad(starts, (0, padding), value=0).view(-1, block).amax(1)[None, None, :, None]
    min_hi = F.pad(ends, (0, padding), value=len(plan.slot_sources)).view(-1, block).amin(1)[None, None, :, None]
    max_hi = hi.view(-1, block).amax(1)[None, None, :, None]
    klo = torch.arange(0, len(plan.slot_sources), block, device=device)[None, None, None]
    khi = (klo+block-1).clamp(max=len(plan.slot_sources)-1)
    full = (klo >= max_lo) & (khi <= min_hi)
    partial = (khi >= min_lo) & (klo <= max_hi) & ~full

    def order(mask):
        return (mask.sum(-1).to(torch.int32),
                torch.argsort(mask.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32))

    def mask_mod(batch, head, q, kv):
        valid = q < starts.shape[0]
        safe_q = q.clamp(max=starts.shape[0]-1)
        return valid & (kv >= starts[safe_q]) & (kv <= ends[safe_q])

    counts, indices = order(partial)
    full_counts, full_indices = order(full)
    return BlockMask.from_kv_blocks(counts, indices, full_counts, full_indices,
        BLOCK_SIZE=block, mask_mod=mask_mod,
        seq_lengths=(plan.token_count, len(plan.slot_sources)))


def graph_attention(q, k, v, plan, window, scale, *, flex=None, prepared_mask=None):
    global _flex
    head_width = q.shape[-1]
    if q.is_cuda and head_width < 16:
        q, k, v = (F.pad(value, (0, 16-head_width)) for value in (q, k, v))
    if flex is None:
        flex = q.is_cuda
    if flex:
        from ..common.attention_backend import flex_kernel_options
        physical_rows = None
        if isinstance(plan, GraphPlan):
            workspace, physical_rows = query_workspace(plan)
            physical_rows = torch.tensor(physical_rows, device=q.device)
            # Pad only attention Q/output workspace. Embeddings, residuals,
            # layer norms and feed-forward never execute padding token rows.
            pieces = []
            for offset, node in zip(plan.offsets, plan.invocations):
                count = len(node.token_ids)
                if count:
                    pieces.append(F.pad(q[:, :, offset:offset+count], (0, 0, 0, (-count) % 128)))
            q = torch.cat(pieces, dim=2)
            plan = workspace
        compiled = _compiled_flex(q, k, v, scale)
        (result, _raw_lse) = compiled(q, k, v, block_mask=prepared_mask if prepared_mask is not None else graph_block_mask(plan, q.device, window),
                     scale=scale,
                     kernel_options={**flex_kernel_options(float32=q.dtype == torch.float32),
                                     **({'FLOAT32_PRECISION': "'ieee'"} if q.dtype == torch.float32 else {}),
                                     'BACKEND': 'TRITON', 'NATLANG_FIXED_DELTA': True})
        result = result[..., :head_width]
        return result if physical_rows is None else result.index_select(2, physical_rows)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=prepared_mask if prepared_mask is not None else plan.attention_mask(q.device, window),
                                         scale=scale, enable_gqa=True)[..., :head_width]


class SharedStateModel(nn.Module):
    """Four role-token rows and an embedding-space adapter, with native weights.

    Marker rows are trainable inputs AND output logits. New execution never
    calls PortHeads or NeuraleseProjection. The wrapped legacy port's two old
    marker overrides are bypassed even when their reserved rows are reused.
    """
    def __init__(self, port, marker_ids, rank=8):
        super().__init__()
        if len(marker_ids) != 4 or len(set(marker_ids)) != 4:
            raise ValueError('shared-state Neuralese requires four distinct marker IDs')
        if min(marker_ids) < 0 or max(marker_ids) >= port.embedding_weight.shape[0]:
            raise ValueError('role markers must have actual input and output vocabulary rows')
        self.port = port
        for name in ('control_rows', 'control_head_rows'):
            if hasattr(port, name):
                getattr(port, name).requires_grad_(False)
        self.marker_ids = tuple(marker_ids)
        weight = port.embedding_weight
        self.input_rows = nn.Parameter(weight[list(marker_ids)].detach().clone())
        self.tied = port.output_weight is weight
        if not self.tied:
            self.output_rows = nn.Parameter(port.output_weight[list(marker_ids)].detach().clone())
        self.adapter = LowRankTokenEmbedding(weight.shape[1], rank, device=weight.device, dtype=torch.float32)

    def embed(self, ids):
        result = F.embedding(ids, self.port.embedding_weight)
        for index, token in enumerate(self.marker_ids):
            result = torch.where((ids == token)[..., None], self.input_rows[index].to(result.dtype), result)
        return result

    def logits(self, h):
        from .readout_storage import readout_weight
        normalized = self.port.final_norm(h)
        logits = _linear_weight(normalized, readout_weight(self.port, self.port.output_weight, normalized.dtype))
        rows = self.input_rows if self.tied else self.output_rows
        logits = logits.clone()
        logits[..., list(self.marker_ids)] = _linear_weight(normalized, rows.to(normalized.dtype))
        return logits

    def forward(self, plan, *, embeddings=None, reference=False, external_blocks=None, external_identity=None, retain_states=True):
        ids = plan.tensor([t for node in plan.invocations for t in node.token_ids], self.input_rows.device)
        raw = self.embed(ids[None]) if embeddings is None else embeddings
        fn = forward_reference if reference else forward_graph
        kwargs = {} if reference else {'retain_states': retain_states}
        result = fn(self.port, plan, self.adapter, embeddings=raw, external_blocks=external_blocks, external_identity=external_identity, **kwargs)
        result['embeddings'] = raw
        result['token_ids'] = ids
        if plan.external_names:
            result['embeddings'] = torch.cat([raw, *[external_blocks[name].embeddings[None].to(raw) for name in plan.external_names]], dim=1)
            result['token_ids'] = torch.cat([ids, *[external_blocks[name].token_ids.to(ids.device) for name in plan.external_names]])
        return result


def capture_invocation_span(result, invocation, start, stop):
    """Export a logical producing span, including any imported slots inside it.

    `start`/`stop` identify own-token boundaries in the invocation. Unlike a
    physical tensor slice, a logical span includes intervening alias slots.
    """
    plan = result['plan']
    index = next((i for i, node in enumerate(plan.invocations) if node.name == invocation), None)
    if index is None:
        raise ValueError('capture invocation is not present in the graph')
    node = plan.invocations[index]
    if not 0 <= start <= stop <= len(node.token_ids):
        raise ValueError('capture span is outside invocation tokens')
    if not any(a <= start and stop <= b for a, b in node.producing):
        raise ValueError('capture requires an explicitly producing span')
    offset = plan.offsets[index]
    lo, hi = plan.boundaries[index][start], plan.boundaries[index][stop]
    return _capture_logical(result, index, lo, hi)


def capture_logical_span(result, invocation, start, stop):
    plan = result['plan']
    index = next((i for i, node in enumerate(plan.invocations) if node.name == invocation), None)
    if index is None or not any(a <= start <= stop <= b for a, b in plan.invocations[index].logical_producing):
        raise ValueError('capture requires an explicitly declared logical producing span')
    return _capture_logical(result, index, start, stop)


def _capture_logical(result, index, start, stop):
    plan = result['plan']
    sources = plan.sequences[index][start:stop]
    device = result['h_final'].device
    rows = plan.tensor(sources, device)
    layers = []
    for values in result['states']:
        if len(values) == 2:
            layers.append({'kind': 'attention', 'raw_k': values[0][0, :, rows].transpose(0, 1),
                           'v': values[1][0, :, rows].transpose(0, 1)})
        else:
            layers.append({'kind': 'finite_convolution', 'gated_inputs': values[0][0, rows]})
    return {'positions': torch.arange(start, stop, device=device), 'layers': tuple(layers),
            'token_ids': result['token_ids'][rows], 'embeddings': result['embeddings'][0, rows]}


def capture_block(result, invocation, start, stop, **metadata):
    from .state_block import NativeStateBlock
    block = NativeStateBlock(**metadata, **capture_invocation_span(result, invocation, start, stop))
    block.validate()
    return block


@dataclass(frozen=True)
class RoleTokens:
    produce_open: int
    produce_close: int
    consume_open: int
    consume_close: int

    @property
    def ids(self):
        return (self.produce_open, self.produce_close, self.consume_open, self.consume_close)


def provision_role_tokens(port, tokenizer):
    """Add four actual special tokens without stealing meaningful vocabulary rows.

    Existing tokenizer files/checkpoints remain immutable; the caller saves the
    expanded tokenizer and this model's marker contract with its new lineage.
    Both HF backbones and the native Mellum/Maple wrapper are supported.
    """
    import inspect
    modern = 'replace_extra_special_tokens' in inspect.signature(tokenizer.add_special_tokens).parameters
    key = 'extra_special_tokens' if modern else 'additional_special_tokens'
    option = 'replace_extra_special_tokens' if modern else 'replace_additional_special_tokens'
    tokenizer.add_special_tokens({key: list(ROLE_MARKERS)}, **{option: False})
    ids = tuple(tokenizer.convert_tokens_to_ids(marker) for marker in ROLE_MARKERS)
    if len(set(ids)) != 4 or any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise ValueError('tokenizer failed to provision four distinct Neuralese role markers')
    target = max(max(tokenizer.get_vocab().values())+1, port.embedding_weight.shape[0])
    old_input = port.hf.get_input_embeddings()
    old_output = port.hf.get_output_embeddings()
    tied = old_input.weight is old_output.weight
    if target > old_input.weight.shape[0]:
        # Implement the exact embedding/head resize for both the HF model and
        # native Mellum wrapper; no broad model reinitialization or row reuse.
        embedding = nn.Embedding(target, old_input.weight.shape[1], device=old_input.weight.device,
                                 dtype=old_input.weight.dtype)
        with torch.no_grad():
            embedding.weight[:old_input.weight.shape[0]].copy_(old_input.weight)
            embedding.weight[old_input.weight.shape[0]:].copy_(old_input.weight.float().mean(0))
        embedding.weight.requires_grad_(old_input.weight.requires_grad)
        port.hf.model.embed_tokens = embedding
        if tied:
            old_output.weight = embedding.weight
            if hasattr(old_output, 'out_features'):
                old_output.out_features = target
        else:
            if getattr(old_output, 'bias', None) is not None:
                raise ValueError('shared-state vocabulary expansion requires the declared bias-free LM head')
            head = nn.Linear(old_output.weight.shape[1], target, bias=False,
                             device=old_output.weight.device, dtype=old_output.weight.dtype)
            with torch.no_grad():
                head.weight[:old_output.weight.shape[0]].copy_(old_output.weight)
                head.weight[old_output.weight.shape[0]:].copy_(old_output.weight.float().mean(0))
            head.weight.requires_grad_(old_output.weight.requires_grad)
            port.hf.lm_head = head
        for configuration in (port.config, port.hf.config, getattr(port.hf.model, 'config', None)):
            if configuration is not None:
                configuration.vocab_size = target
    return RoleTokens(*ids)


def native_step(model, ids, cache=None, *, producing=False):
    """Incremental inference with the exact same native layer math as the DAG.

    Returned layer tensors cover only this chunk, retaining autograd when used
    for an exact reference. Cached keys are already rotated at consumer-local
    positions; block capture stores the raw keys separately in the result.
    """
    from .lfm2_port import PortCache, ConvState, append_kv
    port = model.port
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('native_step requires one token sequence')
    cache = cache or PortCache.empty(port.num_layers)
    if cache.pad is not None or len(set(cache.lengths)) != 1:
        raise ValueError('native state requires an unpadded cache with aligned layer positions')
    start = cache.lengths[0]
    positions = torch.arange(start, start+ids.shape[1], device=ids.device)
    raw = model.embed(ids)
    if ids.shape[1] == 0:
        # An empty span carries the exact model/layer layout but consumes no
        # positions and does not mutate the consumer's hybrid or attention state.
        layers = []
        for index, layer in enumerate(port.layers):
            if port.is_attention(index):
                attn = layer.self_attn
                shape = (0, attn.k_proj.out_features // attn.head_dim, attn.head_dim)
                layers.append({'kind': 'attention', 'raw_k': raw.new_empty(shape, dtype=torch.float32), 'v': raw.new_empty(shape, dtype=torch.float32)})
            else:
                layers.append({'kind': 'finite_convolution',
                               'gated_inputs': raw.new_empty((0, layer.conv.conv.weight.shape[0]), dtype=torch.float32)})
        return {'h_final': raw, 'cache': cache, 'token_ids': ids[0], 'embeddings': raw[0],
                'positions': positions, 'layers': tuple(layers)}
    mask = torch.full(ids.shape, producing, device=ids.device, dtype=torch.bool) if isinstance(producing, bool) else producing
    h = model.adapter(raw, mask)
    states, captures = [], []
    masks = {}
    layout = AttentionLayout(tuple(range(start, start+ids.shape[1])), start+ids.shape[1])
    for i, layer in enumerate(port.layers):
        attention = port.is_attention(i)
        values = _layer_inputs(port, layer, h, attention)
        previous = cache.states[i]
        if attention:
            q, k, v = values
            captures.append({'kind': 'attention', 'raw_k': k[0].transpose(0, 1), 'v': v[0].transpose(0, 1)})
            q, rotated_k = _rotate(port, q, positions, i), _rotate(port, k.to(q.dtype), positions, i)
            state = append_kv(previous, rotated_k, v.to(q.dtype), static=not torch.is_grad_enabled())
            window = port.window(i) if hasattr(port, 'window') else None
            if window not in masks:
                masks[window] = graph_block_mask(layout, ids.device, window) if ids.is_cuda else layout.attention_mask(ids.device, window)
            out = graph_attention(q, state.k, state.v, layout, window, layer.self_attn.scaling,
                                  prepared_mask=masks[window])

        else:
            gated, c = values
            captures.append({'kind': 'finite_convolution', 'gated_inputs': gated[0]})
            gated_channels = gated.transpose(1, 2)
            prefix = (gated_channels.new_zeros(1, gated_channels.shape[1], port.conv_window)
                      if previous is None else previous.window)
            full = torch.cat([prefix, gated_channels], dim=2)
            out = F.conv1d(full.to(c.dtype), layer.conv.conv.weight, layer.conv.conv.bias,
                           groups=gated_channels.shape[1]).transpose(1, 2) * c
            state = ConvState(full[:, :, -port.conv_window:])
        states.append(state)
        h = _finish(port, layer, h, out, attention)
    return {'h_final': h, 'cache': PortCache(tuple(states), (start+ids.shape[1],)*port.num_layers),
            'token_ids': ids[0], 'embeddings': raw[0], 'positions': positions, 'layers': tuple(captures)}


def splice_block(model, cache, block, *, checkpoint_sha256, tokenizer_fingerprint, embedding_weight_sha256):
    """Append a verified native block with consumer-local positional encoding."""
    from .lfm2_port import PortCache, ConvState, append_kv
    block.validate()
    for field, expected in (('checkpoint_sha256', checkpoint_sha256),
                            ('tokenizer_fingerprint', tokenizer_fingerprint),
                            ('embedding_weight_sha256', embedding_weight_sha256)):
        if getattr(block, field) != expected:
            raise ValueError(f'native Neuralese {field} differs; re-encode text with this model before consuming it')
    port = model.port
    if block.architecture != str(getattr(port.config, 'model_type', type(port).__name__)):
        raise ValueError('native state architecture differs from the consumer')
    cache = cache or PortCache.empty(port.num_layers)
    if cache.pad is not None or len(set(cache.lengths)) != 1 or len(block.layers) != port.num_layers:
        raise ValueError('native state layer layout or cache positions disagree')
    if block.embeddings.shape[1] != port.embedding_weight.shape[1]:
        raise ValueError('native state embedding width disagrees with the consumer')
    start, count = cache.lengths[0], len(block.token_ids)
    positions = torch.arange(start, start+count, device=port.embedding_weight.device)
    states = []
    for i, payload in enumerate(block.layers):
        previous = cache.states[i]
        if port.is_attention(i):
            if payload['kind'] != 'attention':
                raise ValueError('native state attention/convolution layout disagrees with the consumer')
            raw_k = payload['raw_k'].to(device=positions.device, dtype=port.embedding_weight.dtype).transpose(0, 1)[None]
            v = payload['v'].to(device=positions.device, dtype=port.embedding_weight.dtype).transpose(0, 1)[None]
            attn = port.layers[i].self_attn
            expected_heads = attn.k_proj.out_features // attn.head_dim
            if raw_k.shape[1:] != (expected_heads, count, attn.head_dim):
                raise ValueError('native state attention head layout disagrees with the consumer')
            state = append_kv(previous, _rotate(port, raw_k, positions, i), v, static=not torch.is_grad_enabled())
        else:
            if payload['kind'] != 'finite_convolution':
                raise ValueError('native state attention/convolution layout disagrees with the consumer')
            gated = payload['gated_inputs'].to(device=positions.device, dtype=torch.float32).T[None]
            channels = port.layers[i].conv.conv.weight.shape[0]
            if gated.shape[1] != channels:
                raise ValueError('native state convolution channels disagree with the consumer')
            prefix = (gated.new_zeros(1, channels, port.conv_window) if previous is None else previous.window)
            state = ConvState(torch.cat([prefix, gated], dim=2)[:, :, -port.conv_window:])
        states.append(state)
    return cache if count == 0 else PortCache(tuple(states), (start+count,)*port.num_layers)


def concatenate_captures(chunks):
    """Join actual producing chunks; reject gaps or mixed layer layouts."""
    if not chunks:
        raise ValueError('native capture needs an actual model-layout capture chunk')
    first_nonempty = next((chunk for chunk in chunks if len(chunk['positions'])), None)
    expected = int(first_nonempty['positions'][0]) if first_nonempty is not None else 0
    for chunk in chunks:
        positions = chunk['positions']
        if not torch.equal(positions, torch.arange(expected, expected+len(positions), device=positions.device)):
            raise ValueError('native capture chunks must form one contiguous source span')
        expected += len(positions)
        if len(chunk['layers']) != len(chunks[0]['layers']):
            raise ValueError('native capture chunks have different layer counts')
    result = {name: torch.cat([chunk[name] for chunk in chunks])
              for name in ('token_ids', 'embeddings', 'positions')}
    layers = []
    for i, first in enumerate(chunks[0]['layers']):
        if any(chunk['layers'][i]['kind'] != first['kind'] for chunk in chunks):
            raise ValueError('native capture chunks have different layer kinds')
        keys = ('raw_k', 'v') if first['kind'] == 'attention' else ('gated_inputs',)
        layers.append({'kind': first['kind'], **{key: torch.cat([chunk['layers'][i][key] for chunk in chunks]) for key in keys}})
    result['layers'] = tuple(layers)
    return result


def _external_layers(port, plan, external_blocks, external_identity):
    blocks = external_blocks or {}
    if set(blocks) != set(plan.external_names):
        raise ValueError('external state names must exactly match the compiled graph contract')
    if blocks and external_identity is None:
        raise ValueError('external native state requires the explicit target model identity')
    result = []
    kinds = tuple('attention' if port.is_attention(i) else 'finite_convolution' for i in range(port.num_layers))
    for name, count in zip(plan.external_names, plan.external_sizes):
        block = blocks[name]
        block.require_native(**external_identity, layer_kinds=kinds)
        if block.architecture != str(getattr(port.config, 'model_type', type(port).__name__)):
            raise ValueError('external native state architecture disagrees with the target decoder')
        if len(block.token_ids) != count or len(block.layers) != port.num_layers:
            raise ValueError('external state token count or decoder depth disagrees with its graph declaration')
        if block.embeddings.shape[1] != port.embedding_weight.shape[1]:
            raise ValueError('external state embedding width disagrees with the target decoder')
        if bool((block.token_ids >= port.embedding_weight.shape[0]).any()):
            raise ValueError('external state token IDs are outside the target vocabulary')
        layers = []
        for i, payload in enumerate(block.layers):
            if port.is_attention(i):
                if payload['kind'] != 'attention':
                    raise ValueError('external state layer kind disagrees with the decoder')
                attn = port.layers[i].self_attn
                heads = attn.k_proj.out_features // attn.head_dim
                if payload['raw_k'].shape[1:] != (heads, attn.head_dim):
                    raise ValueError('external state attention head layout disagrees with the target decoder')
                layers.append(tuple(payload[key].to(device=port.embedding_weight.device, dtype=torch.float32).transpose(0, 1)[None] for key in ('raw_k', 'v')))
            else:
                if payload['kind'] != 'finite_convolution':
                    raise ValueError('external state layer kind disagrees with the decoder')
                if payload['gated_inputs'].shape[1] != port.layers[i].conv.conv.weight.shape[0]:
                    raise ValueError('external state convolution channels disagree with the target decoder')
                layers.append((payload['gated_inputs'].to(device=port.embedding_weight.device, dtype=torch.float32)[None],))
        result.append(tuple(layers))
    return tuple(result)


def channel_checkpoint(model):
    """Native channel state alongside the shared trainer's full backbone state."""
    return {'format': 'neuralese.native-channel/2', 'marker_names': ROLE_MARKERS,
            'marker_ids': model.marker_ids, 'width': model.input_rows.shape[1],
            'rank': model.adapter.down.out_features,
            'adapter_dtype': str(model.adapter.down.weight.dtype),
            'shared_state_dtype': 'torch.float32',
            'kernel_contract': 'shared-state-fixed-native/2',
            'attention_query_alignment': 128,
            'attention_delta': 'fixed-fp32-row/1',
            'attention_auxiliary': 'raw-log2-lse',
            'architecture': str(getattr(model.port.config, 'model_type', type(model.port).__name__)),
            'input_rows': model.input_rows.detach().clone(),
            'output_rows': None if model.tied else model.output_rows.detach().clone(),
            'adapter': {name: value.detach().clone() for name, value in model.adapter.state_dict().items()}}


def restore_channel_checkpoint(model, state):
    """Restore an exact native channel; legacy projections are never substituted."""
    expected = channel_checkpoint(model)
    for key in ('format', 'marker_names', 'marker_ids', 'width', 'rank', 'adapter_dtype',
                'shared_state_dtype', 'kernel_contract', 'attention_query_alignment',
                'attention_delta', 'attention_auxiliary', 'architecture'):
        actual = tuple(state[key]) if key in ('marker_names', 'marker_ids') else state[key]
        if actual != expected[key]:
            raise ValueError(f'native channel checkpoint {key} differs from this model')
    if (state['output_rows'] is None) != model.tied:
        raise ValueError('native channel checkpoint tied-head contract differs')
    with torch.no_grad():
        model.input_rows.copy_(state['input_rows'])
        if not model.tied:
            model.output_rows.copy_(state['output_rows'])
        model.adapter.load_state_dict(state['adapter'], strict=True)
    return model


def _linear_weight(x, weight):
    if x.is_cuda and x.dtype in (torch.float32, torch.bfloat16):
        from ..maple.fused_moe import invariant_linear
        shape = x.shape
        return invariant_linear(x.reshape(-1, shape[-1]), weight).reshape(*shape[:-1], weight.shape[0])
    return F.linear(x, weight)


def _linear(module, x):
    # Preserve custom adapter/quantization forwards rather than bypassing them.
    if type(module) is not nn.Linear:
        return module(x)
    out = _linear_weight(x, module.weight)
    return out if module.bias is None else out + module.bias


@dataclass(frozen=True)
class AttentionLayout:
    query_positions: tuple[int, ...]
    key_count: int

    @property
    def token_count(self):
        return len(self.query_positions)

    @property
    def slot_sources(self):
        return range(self.key_count)

    def bounds(self, device, window=None):
        ends = torch.tensor(self.query_positions, device=device)
        starts = torch.zeros_like(ends)
        if window is not None:
            starts = (ends-window+1).clamp(min=0)
        return starts, ends

    def attention_mask(self, device, window=None):
        starts, ends = self.bounds(device, window)
        keys = torch.arange(self.key_count, device=device)
        return ((keys[None] >= starts[:, None]) & (keys[None] <= ends[:, None]))[None, None]


def _flex_entry(q, k, v, *, block_mask, scale, kernel_options):
    # Preserve the backend raw log2 LSE without an extra scaled public-wrapper
    # graph. The fixed per-row delta lowering below supplies the actual
    # arithmetic invariant; auxiliary-output routing alone is insufficient.
    from torch.nn.attention.flex_attention import flex_attention_hop, _identity
    for tensor in (q, k, v):
        torch._dynamo.mark_static(tensor, -3)
        torch._dynamo.mark_static(tensor, -1)
    options = dict(kernel_options)
    options.update(PRESCALE_QK=False, ROWS_GUARANTEED_SAFE=False,
                   BLOCKS_ARE_CONTIGUOUS=False, WRITE_DQ=True,
                   OUTPUT_LOGSUMEXP=True, OUTPUT_MAX=False)
    out, lse, _ = flex_attention_hop(q, k, v, _identity, block_mask.as_tuple(), scale, options)
    return out, lse


def _compiled_flex(q, k, v, scale):
    """Separate actual kernel contracts; never silently fall back to dense math.

    Dynamo's per-code-object specialization limit otherwise mixes architectures,
    precisions and cache layouts until an unfused O(T²) implementation replaces
    native attention. Query/context lengths stay dynamic within each contract.
    """
    import hashlib
    import types
    from ..common.native_attention_delta import install_native_delta_lowering
    install_native_delta_lowering()
    def layout(tensor):
        strides = tensor.stride()
        order = tuple(sorted(range(tensor.ndim), key=lambda i: strides[i]))
        # Native KV buffers have a fixed capacity larger than their current
        # logical sequence. Distinguish allocations, not every decode length.
        capacity_stride = (strides[1] if strides[2] == tensor.shape[-1]
                           and strides[1] != tensor.shape[2]*tensor.shape[-1] else 0)
        return order, capacity_stride
    key = (q.dtype, k.dtype, v.dtype, q.shape[0], q.shape[1], k.shape[1], q.shape[-1], v.shape[-1],
           scale, layout(q), layout(k), layout(v))
    if key not in _flex:
        name = 'native_flex_' + hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        # A fresh wrapper code object gives each real kernel contract its own
        # dynamic-shape compiler cache; the inner Flex implementation is inlined.
        code = _flex_entry.__code__.replace(co_name=name, co_qualname=name)
        entry = types.FunctionType(code, globals(), name)
        _flex[key] = torch.compile(entry, dynamic=True, fullgraph=True)
    return _flex[key]


@dataclass(frozen=True)
class QueryWorkspace:
    """Attention-only rows with invocation-local 128-row tile origins."""
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    key_count: int

    @property
    def token_count(self):
        return len(self.starts)

    @property
    def slot_sources(self):
        return range(self.key_count)

    def bounds(self, device, window=None):
        starts = torch.tensor(self.starts, device=device)
        ends = torch.tensor(self.ends, device=device)
        if window is not None:
            starts = torch.maximum(starts, ends-window+1)
        return starts, ends


def query_workspace(plan):
    """Match native cached query reduction origins without padding the model.

    Padding rows use zero Q, attend one finite key in their own invocation and
    have zero output adjoints after the physical-row gather. This avoids empty
    softmax rows and does not expand sparse geometry across unrelated calls.
    """
    starts, ends, physical_rows = [], [], []
    for node, offset, key_start in zip(plan.invocations, plan.offsets, plan.slot_offsets):
        count = len(node.token_ids)
        if not count:
            continue
        physical_rows.extend(range(len(starts), len(starts)+count))
        starts.extend([key_start]*count)
        node_ends = [key_start+position for position in plan.own_positions[offset:offset+count]]
        ends.extend(node_ends)
        padding = (-count) % 128
        # Use the last query's current key for dummy rows. A shared global
        # dummy key would make conservative block geometry span the whole DAG.
        starts.extend([node_ends[-1]]*padding)
        ends.extend([node_ends[-1]]*padding)
    return QueryWorkspace(tuple(starts), tuple(ends), len(plan.slot_sources)), tuple(physical_rows)
