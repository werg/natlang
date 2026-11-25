"""Compiled call sites: the compiler's output is the program tuner's per-binding stack, computed in one pass.

A compiled site is exactly what program tuning fits for a binding (``tune/soft.py``, ``tune/trainer.py``), so every
engine serves it with the mechanisms it already has:

- **Memory (Mode P):** the site's soft instruction state, a ``neuralese.native-state/1`` block (attention ``raw_k`` /
  ``v`` pre-RoPE per attention layer, convolution ``gated_inputs`` per convolution layer, one row per token). It is
  spliced between consume markers at the header's real positions in the real rendered call, as ``place_states`` /
  ``soft_items_to_node`` lower it and as every engine's ``--native-state`` / ``_native_consume`` splices it. Its value
  is the text-initialized state of the header written in that call (``engine.native_init_body``) plus a residual the
  head emits from Z. The residual's output layers start at zero, so at step 0 the state is the header's own text state
  (reading it equals reading the text in context). There is no gate and no synthetic position: the training forward is
  the native graph executor's consumption of the spliced span, which ``test_tune_soft_states`` holds equal to serving.
- **Adapter:** an ``xs`` adapter (``model/native_adapters``): one r x r core per target matrix in the canonical
  weight-SVD basis, i.e. ``B C A`` with a shared ``(B, A)``. The head emits the cores ``C`` (zero at step 0). The
  release path exports it as a standard GGUF LoRA (``tune.build.export_release_nodes``) for llama.cpp and the page,
  and the native engine and SGLang serve the artifact directly.

Training runs ``serve.native_grad.native_graph_gradient`` with the compiled tensors as overrides (as the tuner's
``_graph_request`` does), so the loss is the native executor's own graph loss and gradients flow back through the
overrides into the head (FP32 parameters; the frozen base computes in its serving dtype).
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .header_encoder import HeaderEncoder, HeaderEncoderConfig

COMPILED_SITE_SCHEMA = 'natlang.compiler.compiled-site/1'
GRAPH_SCHEMA = 'natlang.neuralese-native-graph/1'
HEADER_OPEN = 'You are inside this call:'
HEADER_STOPS = ('\n\nIn eval you can use', '\n\nEval also has')
STATE_NAME = 'soft:instructions'


def site_header(messages):
    """The site header of a rendered call: the user message's ``You are inside this call: <signature>`` and its
    ``Instructions:`` paragraph, exactly as rendered (None when the call has no such header or it is not unique)."""
    for message in messages:
        content = message.get('content')
        if message.get('role') != 'user' or not isinstance(content, str) or content.count(HEADER_OPEN) != 1:
            continue
        start = content.index(HEADER_OPEN)
        stops = [content.find(stop, start) for stop in HEADER_STOPS]
        stops = [stop for stop in stops if stop > start]
        header = content[start:min(stops) if stops else len(content)].rstrip()
        return header if sum(m.get('content', '').count(header) if isinstance(m.get('content'), str) else 0
                             for m in messages) == 1 else None
    return None


def site_binding(header):
    """The site's binding: its signature line (``You are inside this call: f(x: T): R``)."""
    return header.splitlines()[0].removeprefix(HEADER_OPEN).strip()


def place_header(messages, header, state_id):
    """The call's messages with the header text replaced by the state (``tune.soft.place_states``)."""
    from ..tune.soft import place_states, soft_key
    binding = site_binding(header)
    key = soft_key(binding, 'instructions', None)
    placed_messages, placed = place_states({'binding': binding, 'messages': messages,
                                            'texts': {'instructions': header}}, {key: state_id})
    if placed != [key]:
        raise ValueError('the header does not occur exactly once in the call')
    return placed_messages


def init_header_state(engine, messages, tools, header):
    """The header's text-initialized state written in its real rendered call (the soft-state init path)."""
    placeholder = engine.native_write_text(header)['id']
    written = engine.native_init_body(place_header(messages, header, placeholder), tools, placeholder, header)
    return written['id']


def site_graph(engine, messages, tools, reply, header, state_id, block_tokens, *, graph_id, source_groups):
    """One call as a single-node native graph row importing the header state at its position; the reply's tokens are
    the targets (prompt, tool output and the state are zero-weight context)."""
    from ..serve.targets import target_items
    prompt, target = target_items(engine, place_header(messages, header, state_id), tools, reply)
    return items_graph(engine, prompt, target, state_id, block_tokens, graph_id=graph_id, source_groups=source_groups)


def items_graph(engine, prompt, target, state_id, block_tokens, *, graph_id, source_groups):
    """The graph row of rendered items (``serve.targets`` items: ``('tok', id)`` and the state's ``('block', id)``)."""
    from ..tune.soft import soft_items_to_node
    if not target:
        raise ValueError('the reply rendered no target tokens')
    ids, weights, imports, external = soft_items_to_node(engine, prompt, target, prompt_weight=0.,
                                                         state_names={state_id: STATE_NAME})
    if external[STATE_NAME]['token_count'] != block_tokens:
        raise ValueError('the imported state length differs from its block')
    return {'schema': GRAPH_SCHEMA, 'id': graph_id, 'split': 'train', 'source_groups': list(source_groups),
            'tokenizer_fingerprint': engine.model_config['tokenizer_fingerprint'],
            'nodes': [{'name': 'turn', 'token_ids': ids, 'imports': [dataclasses.asdict(edge) for edge in imports],
                       'producing': [], 'target_weights': weights}],
            'external_states': external}


def state_layout(block):
    """``[(name, rows_shape)]`` of a state's consumed tensors (``layers.<i>.<field>``), in layer order."""
    layout = []
    for index, layer in enumerate(block.layers):
        for field in ('gated_inputs', 'raw_k', 'v'):
            if field in layer:
                layout.append((f'layers.{index}.{field}', tuple(layer[field].shape[1:])))
    return layout


def default_taps(layout):
    """The header features the encoder reads: the state rows of the convolution layers nearest 25%, 60% and 85% depth
    and the last attention layer's values (LFM3's own reading of the header at several depths)."""
    conv = [name for name, _ in layout if name.endswith('.gated_inputs')]
    values = [name for name, _ in layout if name.endswith('.v')]
    if not conv:
        return [values[-1]]
    picks = list(dict.fromkeys(conv[min(len(conv) - 1, round(f * (len(conv) - 1)))] for f in (.25, .6, .85)))
    return picks + values[-1:]


def target_family(name):
    """The adapter head pooling a target: 0 attention, 1 convolution, 2 FFN up (w1, w3), 3 FFN down (w2)."""
    if '.self_attn.' in name:
        return 0
    if '.conv.' in name:
        return 1
    return 3 if '.w2.' in name else 2


@dataclass(frozen=True)
class CompiledSiteConfig:
    layout: tuple                      # ((state tensor name, row shape), ...)
    targets: tuple                     # (adapter target name, ...)
    rank: int = 16
    taps: tuple = ()
    num_slots: int = 16
    value_dim: int = 256
    key_dim: int = 128
    num_heads: int = 4
    adapter_heads: int = 4

    def to_dict(self):
        return {'layout': [[name, list(shape)] for name, shape in self.layout], 'targets': list(self.targets),
                'rank': self.rank, 'taps': list(self.taps), 'num_slots': self.num_slots, 'value_dim': self.value_dim,
                'key_dim': self.key_dim, 'num_heads': self.num_heads, 'adapter_heads': self.adapter_heads}

    @classmethod
    def from_dict(cls, data):
        return cls(layout=tuple((name, tuple(shape)) for name, shape in data['layout']), targets=tuple(data['targets']),
                   **{key: (tuple(value) if key == 'taps' else value) for key, value in data.items()
                      if key not in ('layout', 'targets')})


@dataclass
class CompiledSite:
    """One compiled site: the full state values, the xs cores and the slots they came from."""
    state_values: dict                 # layers.<i>.<field> -> [n, ...] FP32 (text init + residual)
    cores: dict                        # target name -> [rank * rank] FP32
    z_site: torch.Tensor               # [1, M, value_dim]
    k_site: torch.Tensor
    weights: torch.Tensor


class CompiledSiteHead(nn.Module):
    """``Z -> (state residual, xs cores)``; ``HeaderEncoder`` reads the header's text state for ``z_site``.

    Every parameter is FP32 (AGENTS.md precision). Output layers are zero-initialized, so an untrained head compiles a
    site to its text state and a zero adapter: the base reading the header in context."""

    def __init__(self, config: CompiledSiteConfig):
        super().__init__()
        self.config = config
        shapes = dict(config.layout)
        taps = config.taps or tuple(default_taps(config.layout))
        self.taps = tuple(taps)
        tap_dims = tuple(math.prod(shapes[name]) for name in self.taps)
        self.encoder = HeaderEncoder(HeaderEncoderConfig(tap_dims=tap_dims, num_slots=config.num_slots,
                                                         value_dim=config.value_dim, key_dim=config.key_dim,
                                                         num_heads=config.num_heads))
        d = config.value_dim
        # per-token residual readout: each header token attends over the slots Z
        self.token_query = nn.Linear(d, d)
        self.slot_key = nn.Linear(d, d)
        self.state_out = nn.ModuleDict({name.replace('.', '__'): nn.Linear(d, math.prod(shape))
                                        for name, shape in config.layout})
        for layer in self.state_out.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # xs cores: one pooling query per adapter family over Z, then a per-target core readout
        self.family_queries = nn.Parameter(torch.randn(config.adapter_heads, d) * d ** -.5)
        self.core_out = nn.ModuleDict({name.replace('.', '__'): nn.Linear(d, config.rank * config.rank)
                                       for name in config.targets})
        for layer in self.core_out.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.float()

    def encode(self, init_values):
        """``(z_site, k_site, weights, token context [1, n, d])`` from the header's text-state tensors."""
        device = self.family_queries.device
        taps = [init_values[name].to(device=device, dtype=torch.float32).flatten(1)[None] for name in self.taps]
        context = self.encoder.context(taps)
        z, k, w = self.encoder.pool(context)
        return z, k, w, context

    def forward(self, init_values, *, z_slots=None, slot_valid=None, kv=True, adapter=True):
        """Compile one site. ``z_slots`` ([1, M, d]) replaces the slots read from the header (Stage-C retrieved
        records concatenated to ``z_site``, a Tier-1.5 refined ``z_site``, or a shuffled control); ``kv`` / ``adapter``
        False leave that half at its step-0 value (the text state / a zero core) for the reader ablations."""
        z_site, k_site, weights, context = self.encode(init_values)
        z = z_site if z_slots is None else z_slots.to(context)
        mask = None if slot_valid is None else ~slot_valid.to(device=z.device, dtype=torch.bool)
        values = {}
        for name, shape in self.config.layout:
            values[name] = init_values[name].to(device=z.device, dtype=torch.float32)
        if kv:
            scores = self.token_query(context) @ self.slot_key(z).transpose(-1, -2) / math.sqrt(z.shape[-1])
            if mask is not None:
                scores = scores.masked_fill(mask[:, None, :], float('-inf'))
            mixed = (scores.softmax(-1) @ z)[0]                                       # [n, d]
            for name, shape in self.config.layout:
                delta = self.state_out[name.replace('.', '__')](mixed).reshape(mixed.shape[0], *shape)
                values[name] = values[name] + delta
        cores = {}
        if adapter:
            scores = self.family_queries @ z[0].transpose(0, 1) / math.sqrt(z.shape[-1])  # [heads, M]
            if mask is not None:
                scores = scores.masked_fill(mask[0][None], float('-inf'))
            pooled = scores.softmax(-1) @ z[0]                                         # [heads, d]
            for name in self.config.targets:
                family = target_family(name) % self.config.adapter_heads
                cores[name] = self.core_out[name.replace('.', '__')](pooled[family])
        else:
            for name in self.config.targets:
                cores[name] = torch.zeros(self.config.rank * self.config.rank, device=z.device)
        return CompiledSite(values, cores, z_site, k_site, weights)


def compiled_block(init_block, values, device=None):
    """The state block carrying the compiled values (no producer kind: every engine splices it as stored)."""
    layers = []
    for index, layer in enumerate(init_block.layers):
        copied = dict(layer)
        for field in ('gated_inputs', 'raw_k', 'v'):
            name = f'layers.{index}.{field}'
            if name in values:
                copied[field] = values[name] if device is None else values[name].to(device)
        layers.append(copied)
    return dataclasses.replace(init_block, layers=tuple(layers), producer_kind=None)


def create_compiler_adapter(engine, targets, rank, owner=None):
    """The shared xs adapter (canonical weight-SVD basis of rank ``rank`` on ``targets``) whose cores the head emits."""
    from ..model.native_adapters import create_adapter
    return create_adapter(engine, {'kind': 'xs', 'rank': int(rank), 'parameter_names': list(targets)}, owner).id


def core_tensors(engine, adapter_id, cores):
    """The adapter's coefficient tensors with the compiled cores (FP32, the artifact's own shapes)."""
    record = engine.store.get_native_tensor_artifact(adapter_id)
    out = {}
    for name, stored in record.artifact.tensors.items():
        core = cores[name]
        out[name] = core.reshape(stored.shape).to(dtype=torch.float32)
    return out


def site_loss(engine, head, site, *, adapter_id, init_block, z_slots=None, slot_valid=None, kv=True, adapter=True,
              backward=False, scale=1.0, owner=None):
    """The native graph CE of one site row under its compiled stack; ``backward`` accumulates ``scale * loss`` into
    the head. ``site``: ``{'graph', 'state_id'}``."""
    from ..serve.native_grad import native_graph_gradient
    compiled = head(init_block_values(init_block), z_slots=z_slots, slot_valid=slot_valid, kv=kv, adapter=adapter)
    state_id = site['state_id']
    device = engine.device if hasattr(engine, 'device') else None
    block = compiled_block(init_block, compiled.state_values, device)
    coefficients = core_tensors(engine, adapter_id, compiled.cores)
    body = {'schema': 'natlang.native-gradient-request/1', 'native_identity': engine.native_identity,
            'graph': site['graph'], 'invocation_adapters': {'turn': [{'id': adapter_id, 'scale': 1.0}]},
            'adapter_gradient_precision': 'fp32', 'native_state_modes': {state_id: 'same-base-adapter-state'}}
    specs = {'adapter': {'kind': 'adapter', 'id': adapter_id}, 'state': {'kind': 'native-state', 'id': state_id}}

    def evaluate(loss, groups, meta):
        if backward and loss.requires_grad:
            (loss * scale).backward()
        return float(loss.detach())

    with torch.enable_grad():
        return native_graph_gradient(engine, body, owner, evaluation_callback=evaluate,
                                     adapter_leaves_override={adapter_id: coefficients},
                                     native_state_overrides={state_id: block}, evaluation_target_specs=specs)


def init_block_values(block):
    from ..model.native_state_derivation import state_values
    return {name: value.detach() for name, value in state_values(block).items()}


@torch.no_grad()
def compile_release(engine, head, *, adapter_id, init_block, out, site_id, provenance, node=None, owner=None):
    """Write one compiled site as a program-fit release (``tune.build`` layout): ``artifacts/`` (the xs basis and the
    compiled cores), ``states/`` (the compiled header state ``.nzs``), ``result.json``. The native engine and SGLang
    bind it with ``--adapter-release``; llama.cpp and the page take ``tune.build.export_release_nodes``' GGUF LoRA and
    the state file (``--native-state``)."""
    from pathlib import Path
    from ..model.native_artifact import NativeTensorArtifact
    from ..tune.build import persist_artifacts
    out = Path(out)
    head.eval()
    initial = init_block_values(init_block)
    compiled = head(initial)
    block = compiled_block(init_block, {name: value.detach().to(device=initial[name].device, dtype=initial[name].dtype)
                                        for name, value in compiled.state_values.items()})
    state = engine.store.put_native_state(block, owner).id
    record = engine.store.get_native_tensor_artifact(adapter_id)
    cores = {name: value.detach().cpu() for name, value in core_tensors(engine, adapter_id, compiled.cores).items()}
    artifact = engine.store.put_native_tensor_artifact(NativeTensorArtifact(
        'adapter-coefficients', engine.native_identity, cores, record.artifact.source_artifact_id,
        {'adapter_spec': record.artifact.provenance['adapter_spec']}), owner).id
    persist_artifacts(engine, [artifact], out / 'artifacts', out / 'states')
    persist_artifacts(engine, [state], out / 'artifacts', out / 'states')
    result = {'schema': 'natlang.tune.build-result/1', 'compiler': COMPILED_SITE_SCHEMA,
              'artifacts': [{'artifact': artifact, 'node': node or f'lambda:{site_id}'}],
              'softStates': {STATE_NAME: {'state': state, 'site': site_id}}, 'provenance': provenance}
    (out / 'result.json').write_text(json.dumps(result, indent=1, sort_keys=True))
    return result
