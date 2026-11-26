"""Compiled call sites: the compiler's output is the program tuner's per-binding stack, computed in one pass.

A compiled site is exactly what program tuning fits for a binding (``tune/soft.py``, ``tune/trainer.py``), so every
engine serves it with the mechanisms it already has:

- **Memory (Mode P):** the site's soft instruction state, a ``neuralese.native-state/1`` block (attention ``raw_k`` /
  ``v`` pre-RoPE per attention layer, convolution ``gated_inputs`` per convolution layer, one row per token). It is
  spliced between consume markers at the header's real positions in the real rendered call, as ``place_states`` /
  ``soft_items_to_node`` lower it and as every engine's ``--native-state`` / ``_native_consume`` splices it. Its value
  is the text-initialized state of the header written in that call (``engine.native_init_body``) plus a residual the
  head emits from the unified slots ``Z = [z_site, z_kb, null]`` (site slots, evidence-gated retrieved experience
  slots, a learned null slot). The residual's output layers start at zero, so at step 0 the state is the header's own
  text state (reading it equals reading the text in context). There is no gate and no synthetic position: the training
  forward is the native graph executor's consumption of the spliced span, which ``test_tune_soft_states`` holds equal
  to serving.
- **Adapter:** an ``xs`` adapter (``model/native_adapters``): one ``r x r`` core per target matrix,
  ``C_l = tanh(diag(d_l) + U_l V_l^T)`` (AdapterReader V2; ``adapter_inner_rank=0`` emits a dense ``tanh`` core), in a
  shared ``(B, A)`` basis: the canonical weight-SVD basis, or a ``LearnedBankBasis`` (``tune/learned/bank.py``'s
  bank-basis artifact, SVD-initialized, FP32, trained jointly through live bank-basis leaves). Cores are zero at step 0.
  Scope cores add linearly in the shared basis (``compose_site_cores``: ``C = C_scope + C_body + ...``). The release
  path exports a standard GGUF LoRA (``tune.build.export_release_nodes``) for llama.cpp and the page, and the native
  engine and SGLang serve the artifact directly.

Training runs ``serve.native_grad.native_graph_gradient`` with the compiled tensors (and the live basis leaves) as
overrides (as the tuner's ``_graph_request`` does), so the loss is the native executor's own graph loss and gradients
flow back through the overrides into the head, the basis and whatever produced ``z_kb`` (FP32 parameters; the frozen
base computes in its serving dtype).
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


def split_compositional_header(header):
    """A rendered header as ``(signature line, instructions body)``: the scope and body units of compositional
    training (``C = C_scope + C_body``)."""
    lines = header.splitlines()
    sig = lines[0] if lines else header
    body = '\n'.join(lines[1:]).strip()
    return sig, body


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


def compose_site_cores(*core_dicts, scales=None):
    """Linear weight-space composition in the shared ``(B, A)`` basis: ``C_call = sum_i s_i * C_i``."""
    if not core_dicts:
        raise ValueError('compose_site_cores requires at least one core dict')
    if scales is not None and len(scales) != len(core_dicts):
        raise ValueError('scales length must match number of core dicts')
    out = {}
    keys = core_dicts[0].keys()
    if any(cores.keys() != keys for cores in core_dicts):
        raise ValueError('composed cores must cover the same targets (one shared basis)')
    for idx, cores in enumerate(core_dicts):
        s = 1.0 if scales is None else float(scales[idx])
        for name in keys:
            term = cores[name] * s
            out[name] = term if name not in out else (out[name] + term)
    return out


class LearnedBankBasis(nn.Module):
    """Trainable FP32 shared ``(B, A)`` basis: ``tune/learned/bank.py``'s bank basis (``bank-shared-basis-v1``),
    initialized from the weight SVD.

    Its live FP32 leaves ``{'left.<name>', 'right.<name>'}`` reach ``native_graph_gradient`` through
    ``adapter_leaves_override[basis_id]`` (``resolve_adapters`` slices them for the ``xs`` adapter, as the learned
    bank's trainer does with ``extra_leaves``); ``save_artifact`` writes the current values as an ordinary bank-basis
    artifact plus the ``xs`` adapter slicing it, which a release carries. The basis is never re-orthonormalized: the head
    emits cores for this basis, and a re-gauge ``B = Q R`` without re-expressing every core (``bank.transform_core``)
    would change every compiled site's update at once.
    """

    def __init__(self, shared_basis, *, basis_id: str, adapter_id: str):
        super().__init__()
        self.native_identity = dict(shared_basis.native_identity)
        self.rank = int(shared_basis.rank)
        self.targets = tuple(shared_basis.names)
        self.method = dict(shared_basis.method)
        self.basis_id = str(basis_id)
        self.adapter_id = str(adapter_id)
        self.left = nn.ParameterDict({
            name.replace('.', '__'): nn.Parameter(shared_basis.left[name].detach().float().clone())
            for name in self.targets
        })
        self.right = nn.ParameterDict({
            name.replace('.', '__'): nn.Parameter(shared_basis.right[name].detach().float().clone())
            for name in self.targets
        })

    def leaves_dict(self, device=None) -> dict[str, torch.Tensor]:
        out = {}
        for name in self.targets:
            key = name.replace('.', '__')
            left = self.left[key].float()
            right = self.right[key].float()
            if device is not None:
                left = left.to(device)
                right = right.to(device)
            out['left.' + name] = left
            out['right.' + name] = right
        return out

    @torch.no_grad()
    def to_shared_basis(self):
        from ..tune.learned.bank import SharedBasis
        left = {name: self.left[name.replace('.', '__')].detach().float().cpu().clone() for name in self.targets}
        right = {name: self.right[name.replace('.', '__')].detach().float().cpu().clone() for name in self.targets}
        return SharedBasis(
            dict(self.native_identity),
            self.rank,
            left,
            right,
            {'kind': 'compiler-learned-basis', 'sources': [self.method.get('kind', 'weight-svd')]},
        )

    @torch.no_grad()
    def save_artifact(self, engine, owner=None) -> tuple[str, str]:
        """Persist current learned basis + fresh sliced xs adapter artifact into ``engine.store``."""
        from ..model.native_adapters import create_adapter
        basis_rec = self.to_shared_basis().to_artifact(engine, owner=owner)
        adapter_rec = create_adapter(
            engine,
            {'kind': 'xs', 'rank': self.rank, 'parameter_names': list(self.targets), 'basis_id': basis_rec.id},
            owner,
        )
        return basis_rec.id, adapter_rec.id


def create_learned_compiler_basis(engine, targets, rank, owner=None) -> LearnedBankBasis:
    """Create a SVD-initialized trainable ``LearnedBankBasis`` and bind its ``bank-basis`` + ``xs`` artifacts."""
    from ..model.native_adapters import create_adapter
    from ..tune.learned.bank import weight_svd_basis
    svd = weight_svd_basis(engine, list(targets), int(rank))
    basis_rec = svd.to_artifact(engine, owner=owner)
    adapter_rec = create_adapter(
        engine,
        {'kind': 'xs', 'rank': int(rank), 'parameter_names': list(targets), 'basis_id': basis_rec.id},
        owner,
    )
    return LearnedBankBasis(svd, basis_id=basis_rec.id, adapter_id=adapter_rec.id)


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
    adapter_inner_rank: int = 4        # >0: AdapterReader V2 factorized tanh(diag + U V^T); 0: dense r*r
    null_mass: float = 0.1             # null-memory restraint mass for evidence-weighted slot gating
    kv_drop: float = 0.0               # independent KV-channel dropout during training (T2 uses 0.1)
    adapter_drop: float = 0.0          # independent adapter-channel dropout during training (T2 uses 0.1)

    def to_dict(self):
        return {
            'layout': [[name, list(shape)] for name, shape in self.layout],
            'targets': list(self.targets),
            'rank': self.rank,
            'taps': list(self.taps),
            'num_slots': self.num_slots,
            'value_dim': self.value_dim,
            'key_dim': self.key_dim,
            'num_heads': self.num_heads,
            'adapter_heads': self.adapter_heads,
            'adapter_inner_rank': self.adapter_inner_rank,
            'null_mass': self.null_mass,
            'kv_drop': self.kv_drop,
            'adapter_drop': self.adapter_drop,
        }

    @classmethod
    def from_dict(cls, data):
        # strict: a saved config naming a field this code does not have is another head, not a default
        return cls(layout=tuple((name, tuple(shape)) for name, shape in data['layout']), targets=tuple(data['targets']),
                   **{key: (tuple(value) if key == 'taps' else value) for key, value in data.items()
                      if key not in ('layout', 'targets')})


@dataclass
class CompiledSite:
    """One compiled site: the full header state values, the xs cores and the slots they came from."""
    state_values: dict                 # layers.<i>.<field> -> [n, ...] FP32 (text init + residual)
    cores: dict                        # target name -> [rank * rank] FP32
    z_site: torch.Tensor               # [1, M_site, value_dim]
    k_site: torch.Tensor
    weights: torch.Tensor


class CompiledSiteHead(nn.Module):
    """``Z = [z_site, z_kb, null] -> (header state residual, xs cores)``; ``HeaderEncoder`` reads the header's text
    state for ``z_site``.

    Every parameter is FP32 (AGENTS.md precision). The output layers start at zero, so an untrained head compiles a
    site to its text state and a zero adapter (the base reading the header in context) whatever ``z_kb`` holds.
    Retrieved slots enter with their evidence (``g = w / (null_mass + w)`` on the value, ``log w`` on the attention
    logits) next to a learned null slot at ``null_mass``, so weak evidence leaves the null and site slots dominant."""

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
        # Learned null-memory slot & slot-type embeddings (site=0, kb=1, null=2)
        self.null_slot = nn.Parameter(torch.zeros(1, 1, d))
        self.slot_type_emb = nn.Embedding(3, d)
        nn.init.normal_(self.slot_type_emb.weight, std=0.02)

        # Header state residual readout: each header token attends over evidence-weighted [z_site, z_kb, null]
        self.token_query = nn.Linear(d, d)
        self.slot_key = nn.Linear(d, d)
        self.state_out = nn.ModuleDict({name.replace('.', '__'): nn.Linear(d, math.prod(shape))
                                        for name, shape in config.layout})
        for layer in self.state_out.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # AdapterReader V2: per-family queries + optional per-target modulation -> factorized C = tanh(diag + U V^T)
        self.family_queries = nn.Parameter(torch.randn(config.adapter_heads, d) * d ** -.5)
        self.family_slot_key = nn.Linear(d, d)
        self.family_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.SiLU()) for _ in range(config.adapter_heads)
        ])
        self._inner_rank = min(int(config.adapter_inner_rank), int(config.rank)) if config.adapter_inner_rank > 0 else 0
        per_core = (config.rank * (1 + 2 * self._inner_rank)) if self._inner_rank > 0 else (config.rank * config.rank)
        self.core_out = nn.ModuleDict({name.replace('.', '__'): nn.Linear(d, per_core)
                                       for name in config.targets})
        for layer in self.core_out.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            if self._inner_rank > 0:
                # C = tanh(diag + U V^T) with U = V = 0 is a saddle: dC/dU = V and dC/dV = U are both zero, so the
                # low-rank part would never train. U starts nonzero and V at zero: C is still 0 at step 0, and V
                # (then U) receives gradient from the first step.
                with torch.no_grad():
                    layer.bias.view(config.rank, 1 + 2 * self._inner_rank)[:, 1:1 + self._inner_rank].normal_(
                        std=config.rank ** -.5)
        # Learnable core output scale (starts at 1.0; tanh keeps cores bounded while zero-init preserves step 0)
        self.core_scale = nn.Parameter(torch.ones(config.adapter_heads))
        self.float()

    def encode(self, init_values):
        """``(z_site, k_site, weights, token context [1, n, d])`` from the header's text-state tensors."""
        device = self.family_queries.device
        taps = [init_values[name].to(device=device, dtype=torch.float32).flatten(1)[None] for name in self.taps]
        context = self.encoder.context(taps)
        z, k, w = self.encoder.pool(context)
        return z, k, w, context

    def _assemble_unified_slots(
        self,
        z_site: torch.Tensor,
        w_site: torch.Tensor,
        *,
        slot_valid: torch.Tensor | None = None,
        z_kb: torch.Tensor | None = None,
        w_kb: torch.Tensor | None = None,
        kb_valid: torch.Tensor | None = None,
    ):
        """Site slots, evidence-gated KB slots and the learned null slot as ``(z_all, log_w, pad_mask)``."""
        device = z_site.device
        b, m_s, d = z_site.shape
        site_emb = self.slot_type_emb.weight[0].view(1, 1, d)
        kb_emb = self.slot_type_emb.weight[1].view(1, 1, d)
        null_emb = self.slot_type_emb.weight[2].view(1, 1, d)

        site_valid = torch.ones((b, m_s), dtype=torch.bool, device=device) if slot_valid is None else slot_valid.to(device=device, dtype=torch.bool)
        w_s = w_site.to(device=device, dtype=torch.float32).clamp_min(1e-6)
        w_s = torch.where(site_valid, w_s, torch.zeros_like(w_s))

        parts_z = [z_site + site_emb]
        parts_logw = [torch.log(w_s.clamp_min(1e-6))]
        parts_valid = [site_valid]

        if z_kb is not None and z_kb.shape[1] > 0:
            z_kb_f = z_kb.to(device=device, dtype=torch.float32)
            m_k = z_kb_f.shape[1]
            v_k = torch.ones((b, m_k), dtype=torch.bool, device=device) if kb_valid is None else kb_valid.to(device=device, dtype=torch.bool)
            if w_kb is None:
                w_k = v_k.float()
            else:
                w_k = w_kb.to(device=device, dtype=torch.float32).clamp_min(0.0) * v_k.float()
            # per-slot evidence gate g_m = w_m / (null_mass + w_m)
            null_m = float(self.config.null_mass)
            g_k = (w_k / (null_m + w_k)) * v_k.float()
            parts_z.append((z_kb_f * g_k.unsqueeze(-1)) + kb_emb)
            parts_logw.append(torch.log(w_k.clamp_min(1e-6)))
            parts_valid.append(v_k & (w_k > 0.0))

        # Append learned null-memory slot with fixed null_mass anchor so low-evidence queries stay near identity
        parts_z.append(self.null_slot.expand(b, 1, d) + null_emb)
        parts_logw.append(torch.full((b, 1), math.log(max(1e-6, float(self.config.null_mass))), device=device, dtype=torch.float32))
        parts_valid.append(torch.ones((b, 1), dtype=torch.bool, device=device))

        z_all = torch.cat(parts_z, dim=1)
        log_w = torch.cat(parts_logw, dim=1)
        valid_all = torch.cat(parts_valid, dim=1)
        return z_all, log_w, ~valid_all

    def _synthesize_core(self, raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        r = self.config.rank
        if self._inner_rank > 0:
            kc = self._inner_rank
            parts = raw.reshape(r, 1 + 2 * kc)
            diag = torch.diag(parts[:, 0])
            u = parts[:, 1:1 + kc]
            v = parts[:, 1 + kc:]
            core = torch.tanh(diag + u @ v.transpose(-1, -2)) * scale
            return core.reshape(r * r)
        return torch.tanh(raw) * scale

    def forward(
        self,
        init_values,
        *,
        z_slots=None,
        slot_valid=None,
        slot_weights=None,
        z_kb=None,
        k_kb=None,
        w_kb=None,
        kb_valid=None,
        kv: bool = True,
        adapter: bool = True,
        apply_dropout: bool = True,
    ):
        """Compile one site from its header text state ``init_values``.

        - ``z_slots`` ([1, M, d]) replaces the slots read from the header (a Tier-1.5 refined ``z_site``, an FO-MAML
          inner step, a shuffled control).
        - ``z_kb`` ([1, M_kb, d]) with ``w_kb`` / ``kb_valid``: retrieved experience slots (Live-KB writer or Stage C),
          read by both the state residual and the cores. ``k_kb`` is accepted for the retrieval record and unused here.
        - ``kv`` / ``adapter`` False leave that half at its step-0 value (the text state / a zero core): the reader
          ablations. During training each is dropped independently with ``kv_drop`` / ``adapter_drop``.
        """
        z_site, k_site, w_site, context = self.encode(init_values)
        z_s = z_site if z_slots is None else z_slots.to(context)
        w_s = w_site if slot_weights is None else slot_weights.to(context)
        if w_s.shape[:2] != z_s.shape[:2]:
            w_s = torch.ones(z_s.shape[:2], device=z_s.device, dtype=torch.float32)

        # Independent reader dropout during training (never drop both channels on the same step)
        if self.training and apply_dropout and kv and adapter:
            # drawn on the CPU generator, which a resume restores (a CUDA draw would not be resumable)
            if self.config.kv_drop > 0.0 and float(torch.rand(())) < self.config.kv_drop:
                kv = False
            elif self.config.adapter_drop > 0.0 and float(torch.rand(())) < self.config.adapter_drop:
                adapter = False

        z_all, log_w, pad_mask = self._assemble_unified_slots(
            z_s, w_s, slot_valid=slot_valid, z_kb=z_kb, w_kb=w_kb, kb_valid=kb_valid,
        )

        # header state residual (soft:instructions): each header token attends over the slots
        values = {name: init_values[name].to(device=z_s.device, dtype=torch.float32) for name, _ in self.config.layout}
        if kv:
            scores = self.token_query(context) @ self.slot_key(z_all).transpose(-1, -2) / math.sqrt(z_all.shape[-1])
            scores = scores + log_w[:, None, :]
            scores = scores.masked_fill(pad_mask[:, None, :], float('-inf'))
            mixed = (scores.softmax(-1) @ z_all)[0]                                    # [n, d]
            for name, shape in self.config.layout:
                delta = self.state_out[name.replace('.', '__')](mixed).reshape(mixed.shape[0], *shape)
                values[name] = values[name] + delta

        # AdapterReader V2 cores (factorized bounded cores per target)
        cores = {}
        if adapter:
            keyed = self.family_slot_key(z_all)[0]                                     # [M_all, d]
            a_scores = (self.family_queries @ keyed.transpose(0, 1)) / math.sqrt(z_all.shape[-1])
            a_scores = a_scores + log_w[0][None, :]
            a_scores = a_scores.masked_fill(pad_mask[0][None, :], float('-inf'))
            pooled = a_scores.softmax(-1) @ z_all[0]                                   # [heads, d]
            family_h = [self.family_mlp[h](pooled[h]) for h in range(self.config.adapter_heads)]
            for name in self.config.targets:
                family = target_family(name) % self.config.adapter_heads
                raw = self.core_out[name.replace('.', '__')](family_h[family])
                cores[name] = self._synthesize_core(raw, self.core_scale[family])
        else:
            for name in self.config.targets:
                cores[name] = torch.zeros(self.config.rank * self.config.rank, device=z_s.device)

        return CompiledSite(values, cores, z_site, k_site, w_site)


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


def create_compiler_adapter(engine, targets, rank, owner=None, *, basis_id=None):
    """The shared xs adapter on ``targets`` whose cores the head emits (uses ``basis_id`` if provided)."""
    from ..model.native_adapters import create_adapter
    options = {'kind': 'xs', 'rank': int(rank), 'parameter_names': list(targets)}
    if basis_id is not None:
        options['basis_id'] = basis_id
    return create_adapter(engine, options, owner).id


def core_tensors(engine, adapter_id, cores):
    """The adapter's coefficient tensors with the compiled cores (FP32, the artifact's own shapes)."""
    record = engine.store.get_native_tensor_artifact(adapter_id)
    out = {}
    for name, stored in record.artifact.tensors.items():
        core = cores[name]
        out[name] = core.reshape(stored.shape).to(device=stored.device, dtype=torch.float32)
    return out


def site_loss(
    engine,
    head,
    site,
    *,
    adapter_id,
    init_block,
    z_slots=None,
    slot_valid=None,
    slot_weights=None,
    z_kb=None,
    k_kb=None,
    w_kb=None,
    kb_valid=None,
    basis_module: LearnedBankBasis | None = None,
    extra_cores: dict[str, torch.Tensor] | None = None,
    kv: bool = True,
    adapter: bool = True,
    apply_dropout: bool = True,
    backward: bool = False,
    scale: float = 1.0,
    owner=None,
):
    """The native graph CE of one site row under its compiled stack; ``backward`` accumulates ``scale * loss`` into
    the head (and the basis, and whatever produced ``z_kb``). ``site``: ``{'graph', 'state_id'}``.

    ``basis_module``: a ``LearnedBankBasis`` whose live leaves the executor uses for the adapter's basis.
    ``extra_cores``: cores added to the head's in the shared basis (``C_scope``, ``C_body``, ``C_partial``)."""
    from ..serve.native_grad import native_graph_gradient
    init_vals = init_block_values(init_block)
    compiled = head(
        init_vals,
        z_slots=z_slots,
        slot_valid=slot_valid,
        slot_weights=slot_weights,
        z_kb=z_kb,
        k_kb=k_kb,
        w_kb=w_kb,
        kb_valid=kb_valid,
        kv=kv,
        adapter=adapter,
        apply_dropout=apply_dropout,
    )
    cores_eff = compiled.cores if extra_cores is None else compose_site_cores(compiled.cores, extra_cores)
    state_id = site['state_id']
    # the device of the stored state's tensors, as the tuner's state leaves are (``_leaf_block``)
    device = next(iter(init_vals.values())).device
    state_overrides = {state_id: compiled_block(init_block, compiled.state_values, device)}
    state_modes = {state_id: 'same-base-adapter-state'}
    specs = {
        'adapter': {'kind': 'adapter', 'id': adapter_id},
        'state': {'kind': 'native-state', 'id': state_id},
    }

    coefficients = core_tensors(engine, adapter_id, cores_eff)
    leaves_override = {adapter_id: coefficients}
    if basis_module is not None:
        sample_device = next(iter(coefficients.values())).device
        leaves_override[basis_module.basis_id] = basis_module.leaves_dict(device=sample_device)

    body = {
        'schema': 'natlang.native-gradient-request/1',
        'native_identity': engine.native_identity,
        'graph': site['graph'],
        'invocation_adapters': {'turn': [{'id': adapter_id, 'scale': 1.0}]},
        'adapter_gradient_precision': 'fp32',
        'native_state_modes': state_modes,
    }

    def evaluate(loss, groups, meta):
        if backward and loss.requires_grad:
            (loss * scale).backward()
        return float(loss.detach())

    with torch.enable_grad():
        return native_graph_gradient(
            engine,
            body,
            owner,
            evaluation_callback=evaluate,
            adapter_leaves_override=leaves_override,
            native_state_overrides=state_overrides,
            evaluation_target_specs=specs,
        )


def init_block_values(block):
    from ..model.native_state_derivation import state_values
    return {name: value.detach() for name, value in state_values(block).items()}


@torch.no_grad()
def compile_release(
    engine,
    head,
    *,
    adapter_id,
    init_block,
    out,
    site_id,
    provenance,
    node=None,
    owner=None,
    z_slots=None,
    z_kb=None,
    k_kb=None,
    w_kb=None,
    kb_valid=None,
    basis_module: LearnedBankBasis | None = None,
    extra_cores: dict[str, torch.Tensor] | None = None,
):
    """Write one compiled site as a program-fit release (``tune.build`` layout): ``artifacts/`` (the basis and the
    compiled cores; a learned basis is written at its current value), ``states/`` (the compiled header state
    ``.nzs``), ``result.json``. The native engine and SGLang bind it with ``--adapter-release``; llama.cpp and the page
    take ``tune.build.export_release_nodes``' GGUF LoRA and the state file (``--native-state``)."""
    from pathlib import Path
    from ..model.native_artifact import NativeTensorArtifact
    from ..tune.build import persist_artifacts
    out = Path(out)
    was_training = head.training
    head.eval()
    try:
        initial = init_block_values(init_block)
        compiled = head(
            initial,
            z_slots=z_slots,
            z_kb=z_kb,
            k_kb=k_kb,
            w_kb=w_kb,
            kb_valid=kb_valid,
            apply_dropout=False,
        )
        cores_eff = compiled.cores if extra_cores is None else compose_site_cores(compiled.cores, extra_cores)
        block = compiled_block(init_block, {name: value.detach().to(device=initial[name].device, dtype=initial[name].dtype)
                                            for name, value in compiled.state_values.items()})
        state = engine.store.put_native_state(block, owner).id
        persist_artifacts(engine, [state], out / 'artifacts', out / 'states')

        soft_states = {STATE_NAME: {'state': state, 'site': site_id}}
        effective_adapter_id = adapter_id
        if basis_module is not None:
            _, effective_adapter_id = basis_module.save_artifact(engine, owner=owner)

        record = engine.store.get_native_tensor_artifact(effective_adapter_id)
        cores = {name: value.detach().cpu() for name, value in core_tensors(engine, effective_adapter_id, cores_eff).items()}
        artifact = engine.store.put_native_tensor_artifact(NativeTensorArtifact(
            'adapter-coefficients', engine.native_identity, cores, record.artifact.source_artifact_id,
            {'adapter_spec': record.artifact.provenance['adapter_spec']}), owner).id
        persist_artifacts(engine, [artifact], out / 'artifacts', out / 'states')
        result = {
            'schema': 'natlang.tune.build-result/1',
            'compiler': COMPILED_SITE_SCHEMA,
            'artifacts': [{'artifact': artifact, 'node': node or f'lambda:{site_id}'}],
            'softStates': soft_states,
            'provenance': provenance,
            'receipt': {'base_native_identity': dict(engine.native_identity)},
        }
        (out / 'result.json').write_text(json.dumps(result, indent=1, sort_keys=True))
        return result
    finally:
        head.train(was_training)
