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


@dataclass(frozen=True)
class Invocation:
    name: str
    token_ids: tuple[int, ...]
    imports: tuple[ImportSpan, ...] = ()
    producing: tuple[tuple[int, int], ...] = ()


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

    @property
    def token_count(self):
        return len(self.own_positions)

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


def compile_graph(invocations: tuple[Invocation, ...] | list[Invocation]) -> GraphPlan:
    invocations = tuple(invocations)
    known, offsets, sequences = {}, [], []
    own_positions, sources, positions, node_ids, slot_offsets, producing = [], [], [], [], [], []
    physical_count = 0
    for node in invocations:
        if node.name in known or not node.token_ids:
            raise ValueError('invocations need unique names and at least one token')
        offset = physical_count
        offsets.append(offset)
        count = len(node.token_ids)
        producer = [False] * count
        for start, stop in node.producing:
            if not 0 <= start < stop <= count:
                raise ValueError('producing span is outside invocation tokens')
            producer[start:stop] = [True] * (stop-start)
        edges = {}
        for edge in node.imports:
            if not 0 <= edge.at <= count or edge.source not in known:
                raise ValueError('imports require an earlier source and a valid insertion boundary')
            source_node, source_offset = known[edge.source]
            if not 0 <= edge.start < edge.stop <= len(source_node.token_ids):
                raise ValueError('import span is outside source tokens')
            if not any(a <= edge.start and edge.stop <= b for a, b in source_node.producing):
                raise ValueError('imports must reference an explicitly producing span')
            edges.setdefault(edge.at, []).append(tuple(range(source_offset+edge.start, source_offset+edge.stop)))
        logical = []
        own = []
        for local in range(count+1):
            for imported in edges.get(local, ()):
                logical.extend(imported)
            if local < count:
                own.append(len(logical))
                logical.append(offset+local)
        slot_offset = len(sources)
        slot_offsets.append(slot_offset)
        sources.extend(logical)
        positions.extend(range(len(logical)))
        for pos in own:
            own_positions.append(pos)
            node_ids.append(len(sequences))
        producing.extend(producer)
        sequences.append(tuple(logical))
        known[node.name] = node, offset
        physical_count += count
    if not invocations:
        raise ValueError('graph cannot be empty')
    return GraphPlan(invocations, tuple(offsets), tuple(sequences), tuple(own_positions),
                     tuple(sources), tuple(positions), tuple(node_ids), tuple(slot_offsets), tuple(producing))


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
        update = self.up(self.down(embeddings))
        return embeddings + update * producing[..., None].to(update.dtype)


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
        b, c, xx = layer.conv.in_proj(x).chunk(3, dim=-1)
        return b * xx, c
    attn = layer.self_attn
    batch, steps, _ = x.shape
    hd = attn.head_dim
    norm_q = getattr(attn, 'q_norm', None) or attn.q_layernorm
    norm_k = getattr(attn, 'k_norm', None) or attn.k_layernorm
    q = norm_q(attn.q_proj(x).view(batch, steps, -1, hd)).transpose(1, 2)
    k = norm_k(attn.k_proj(x).view(batch, steps, -1, hd)).transpose(1, 2)
    v = attn.v_proj(x).view(batch, steps, -1, hd).transpose(1, 2)
    return q, k, v


def _finish(port, layer, h, out, attention):
    if attention:
        projection = getattr(layer.self_attn, 'o_proj', None) or layer.self_attn.out_proj
        h = h + projection(out.transpose(1, 2).reshape(h.shape[0], h.shape[1], -1))
    else:
        h = h + layer.conv.out_proj(out)
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
    out = (gathered * kernel.weight[:, 0].T[None, None]).sum(2)
    if kernel.bias is not None:
        out = out + kernel.bias
    return c * out


def forward_graph(port, plan: GraphPlan, adapter: LowRankTokenEmbedding | None = None, *, embeddings=None):
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
    state = []
    for i, layer in enumerate(port.layers):
        attention = port.is_attention(i)
        values = _layer_inputs(port, layer, h, attention)
        state.append(values[1:] if attention else (values[0],))
        if attention:
            q, k, v = values
            q = _rotate(port, q, query_positions, i)
            k = _rotate(port, k[:, :, slots], key_positions, i)
            window = port.window(i) if hasattr(port, 'window') else None
            out = graph_attention(q, k, v[:, :, slots], plan, window, layer.self_attn.scaling)
        else:
            out = _convolve(layer, *values, plan, range(plan.token_count))
        h = _finish(port, layer, h, out, attention)
    return {'h_final': h, 'states': tuple(state), 'plan': plan}


def forward_reference(port, plan: GraphPlan, adapter=None, *, embeddings=None):
    """Slow invocation-major native-cache oracle, for correctness checks only.

    Each completed source retains raw layer states. Each consumer rotates only
    its imported cache keys to its local positions. Query-wise attention and
    conv1d are deliberately independent of the flattened mask/gather kernels.
    """
    device = port.embedding_weight.device
    layer_cache = [[] for _ in port.layers]
    outputs = []
    for node, offset, logical in zip(plan.invocations, plan.offsets, plan.sequences):
        count = len(node.token_ids)
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
                source_k = torch.cat([entry[0] for entry in cache] + [k], dim=2)
                source_v = torch.cat([entry[1] for entry in cache] + [v], dim=2)
                cache.append((k, v))
                parts = []
                window = port.window(i) if hasattr(port, 'window') else None
                for local in range(count):
                    position = plan.own_positions[offset+local]
                    start = 0 if window is None else max(0, position-window+1)
                    indices = plan.tensor(logical[start:position+1], device)
                    kr = _rotate(port, source_k[:, :, indices], torch.arange(start, position+1, device=device), i)
                    qr = _rotate(port, q[:, :, local:local+1], plan.tensor([position], device), i)
                    parts.append(F.scaled_dot_product_attention(qr, kr, source_v[:, :, indices],
                                 scale=layer.self_attn.scaling, enable_gqa=True))
                out = torch.cat(parts, dim=2)
            else:
                gated, c = values
                source = torch.cat([entry[0] for entry in cache] + [gated], dim=1)
                cache.append((gated,))
                kernel = layer.conv.conv
                width = kernel.weight.shape[-1]
                parts = []
                for local in range(count):
                    history = plan.history(offset+local, width)
                    data = source[:, plan.tensor(history, device)].transpose(1, 2)
                    data = F.pad(data, (width-data.shape[-1], 0))
                    conv = F.conv1d(data, kernel.weight, kernel.bias, groups=data.shape[1])
                    parts.append(conv.transpose(1, 2) * c[:, local:local+1])
                out = torch.cat(parts, dim=1)
            h = _finish(port, layer, h, out, attention)
        outputs.append(h)
    states = tuple(tuple(torch.cat([entry[field] for entry in cache], dim=2 if len(cache[0]) == 2 else 1)
                         for field in range(len(cache[0]))) for cache in layer_cache)
    return {'h_final': torch.cat(outputs, dim=1), 'states': states, 'plan': plan}


PRODUCE_OPEN = '<|neuralese_produce|>'
PRODUCE_CLOSE = '<|/neuralese_produce|>'
CONSUME_OPEN = '<|neuralese_consume|>'
CONSUME_CLOSE = '<|/neuralese_consume|>'
ROLE_MARKERS = (PRODUCE_OPEN, PRODUCE_CLOSE, CONSUME_OPEN, CONSUME_CLOSE)


def capture_span(result, start: int, stop: int):
    """Differentiable native span capture; serialization may detach explicitly.

    Raw unrotated attention keys are essential: applying a constant RoPE delta
    to already-rotated keys is not valid for every position-dependent RoPE.
    Finite convolution stores only this span's gated inputs, never private
    prefix state. A consumer continues its own logical history across imports.
    """
    plan = result['plan']
    if not 0 <= start < stop <= plan.token_count or not all(plan.producing[start:stop]):
        raise ValueError('capture requires a nonempty explicitly producing span')
    layers = []
    for values in result['states']:
        if len(values) == 2:
            layers.append({'kind': 'attention', 'raw_k': values[0][:, :, start:stop],
                           'v': values[1][:, :, start:stop]})
        else:
            layers.append({'kind': 'finite_convolution', 'gated_inputs': values[0][:, start:stop]})
    return {'positions': plan.own_positions[start:stop], 'layers': tuple(layers)}


_flex = None


def graph_block_mask(plan, device, window=None):
    """Sparse 128-token geometry without allocating a token-by-token matrix."""
    from torch.nn.attention.flex_attention import BlockMask
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
        valid = q < plan.token_count
        safe_q = q.clamp(max=plan.token_count-1)
        return valid & (kv >= starts[safe_q]) & (kv <= ends[safe_q])

    counts, indices = order(partial)
    full_counts, full_indices = order(full)
    return BlockMask.from_kv_blocks(counts, indices, full_counts, full_indices,
        BLOCK_SIZE=block, mask_mod=mask_mod,
        seq_lengths=(plan.token_count, len(plan.slot_sources)))


def graph_attention(q, k, v, plan, window, scale, *, flex=None):
    global _flex
    if flex is None:
        flex = q.is_cuda
    if flex:
        from torch.nn.attention.flex_attention import flex_attention
        from ..common.attention_backend import flex_kernel_options
        if _flex is None:
            _flex = torch.compile(flex_attention, dynamic=True)
        return _flex(q, k, v, block_mask=graph_block_mask(plan, q.device, window),
                     scale=scale, enable_gqa=True,
                     kernel_options=flex_kernel_options(float32=q.dtype == torch.float32))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=plan.attention_mask(q.device, window),
                                         scale=scale, enable_gqa=True)


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
        self.marker_ids = tuple(marker_ids)
        weight = port.embedding_weight
        self.input_rows = nn.Parameter(weight[list(marker_ids)].detach().clone())
        self.tied = port.output_weight is weight
        if not self.tied:
            self.output_rows = nn.Parameter(port.output_weight[list(marker_ids)].detach().clone())
        self.adapter = LowRankTokenEmbedding(weight.shape[1], rank, device=weight.device, dtype=weight.dtype)

    def embed(self, ids):
        result = F.embedding(ids, self.port.embedding_weight)
        for index, token in enumerate(self.marker_ids):
            result = torch.where((ids == token)[..., None], self.input_rows[index].to(result.dtype), result)
        return result

    def logits(self, h):
        from .readout_storage import readout_weight
        normalized = self.port.final_norm(h)
        logits = normalized @ readout_weight(self.port, self.port.output_weight, normalized.dtype).T
        rows = self.input_rows if self.tied else self.output_rows
        logits = logits.clone()
        logits[..., list(self.marker_ids)] = normalized @ rows.to(normalized.dtype).T
        return logits

    def forward(self, plan, *, embeddings=None, reference=False):
        ids = plan.tensor([t for node in plan.invocations for t in node.token_ids], self.input_rows.device)
        raw = self.embed(ids[None]) if embeddings is None else embeddings
        fn = forward_reference if reference else forward_graph
        result = fn(self.port, plan, self.adapter, embeddings=raw)
        result['embeddings'] = raw
        result['token_ids'] = ids
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
    if not 0 <= start < stop <= len(node.token_ids):
        raise ValueError('capture span is outside invocation tokens')
    if not any(a <= start and stop <= b for a, b in node.producing):
        raise ValueError('capture requires an explicitly producing span')
    offset = plan.offsets[index]
    lo, hi = plan.own_positions[offset+start], plan.own_positions[offset+stop-1]+1
    sources = plan.sequences[index][lo:hi]
    device = result['h_final'].device
    rows = plan.tensor(sources, device)
    layers = []
    for values in result['states']:
        if len(values) == 2:
            layers.append({'kind': 'attention', 'raw_k': values[0][0, :, rows].transpose(0, 1),
                           'v': values[1][0, :, rows].transpose(0, 1)})
        else:
            layers.append({'kind': 'finite_convolution', 'gated_inputs': values[0][0, rows]})
    return {'positions': torch.arange(lo, hi, device=device), 'layers': tuple(layers),
            'token_ids': result['token_ids'][rows], 'embeddings': result['embeddings'][0, rows]}


def capture_block(result, invocation, start, stop, **metadata):
    from .state_block import NativeStateBlock
    return NativeStateBlock(**metadata, **capture_invocation_span(result, invocation, start, stop)).validate()
