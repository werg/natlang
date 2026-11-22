"""Explicit native-target inventories and HF module maps per student (§5.3, §7).

Derived from the actual ports: LFM2 (`model/lfm2_port.py`, HF `Lfm2ForCausalLM`), LFM3 (`model/lfm3_port.py`, the
vendored pinned `Lfm3ForCausalLM`: LFM2's conv/feed-forward names, attention with ``o_proj`` and the per-head sigmoid
output gate ``self_attn.gate_proj``, which llama.cpp applies through `build_lora_mm` as `attn_gate`). A native target that matches no pattern is an error, never a silently
omitted tensor. The convolution kernel (``conv.conv``) is a state tensor, not a linear projection;
`native_adapters.target_parameters` already excludes it (and LFM3's 1-D q/k norms) and it has no entry here.

LFM3's per-head gate (2048 -> 16) belongs to the attention class: it is part of the attention operator, routed with
the attention matrices by the backbone optimizer, and too small for a matrix family of its own in bank groups.
`DEPLOY_QUANTIZED` records which targets a line's required deploy point quantizes (LFM3 NVFP4: feed-forward
w1/w2/w3 only, ``export/lfm3_quantized.LFM3_POINTS``): an adapter folded into those weights is re-quantized at deploy,
so its effect must be measured at that precision, not only in BF16.
"""
from __future__ import annotations

import re

from . import TuneMathError
from .export import ModuleTarget

_LAYER = r'^port\.hf\.model\.layers\.(\d+)\.'
# (pattern, HF module template(s) with {layer}/{name}, target class)
TABLES = {
    'lfm2': (
        (_LAYER + r'self_attn\.(?P<name>q_proj|k_proj|v_proj|out_proj)\.weight$', 'model.layers.{layer}.self_attn.{name}', 'attention'),
        (_LAYER + r'conv\.(?P<name>in_proj|out_proj)\.weight$', 'model.layers.{layer}.conv.{name}', 'conv-projection'),
        (_LAYER + r'feed_forward\.(?P<name>w1|w2|w3)\.weight$', 'model.layers.{layer}.feed_forward.{name}', 'mlp'),
    ),
    'lfm3': (
        (_LAYER + r'self_attn\.(?P<name>q_proj|k_proj|v_proj|o_proj|gate_proj)\.weight$', 'model.layers.{layer}.self_attn.{name}', 'attention'),
        (_LAYER + r'conv\.(?P<name>in_proj|out_proj)\.weight$', 'model.layers.{layer}.conv.{name}', 'conv-projection'),
        (_LAYER + r'feed_forward\.(?P<name>w1|w2|w3)\.weight$', 'model.layers.{layer}.feed_forward.{name}', 'mlp'),
    ),
}
DEFAULT_CLASSES = {'lfm2': ('attention', 'conv-projection', 'mlp'), 'lfm3': ('attention', 'conv-projection', 'mlp')}
# {family: {deploy point: native target name suffixes it quantizes}}; families without an entry deploy unquantized.
DEPLOY_QUANTIZED = {'lfm3': {'nvfp4': ('feed_forward.w1.weight', 'feed_forward.w2.weight', 'feed_forward.w3.weight')}}


def deploy_quantized(names, family=None):
    """``{deploy point: sorted target names it quantizes}`` for this inventory (empty when nothing is quantized)."""
    family = family or student_family(names)
    return {point: sorted(name for name in names if name.endswith(suffixes))
            for point, suffixes in DEPLOY_QUANTIZED.get(family, {}).items()}


def classify(name, family):
    for pattern, template, kind in TABLES[family]:
        match = re.match(pattern, name)
        if match:
            fields = {'layer': match.group(1), 'name': match.groupdict().get('name')}
            hf = tuple(item.format(**fields) for item in template) if isinstance(template, tuple) else template.format(**fields)
            return kind, ModuleTarget(hf)
    return None


def student_family(names):
    """Identify one actual student family; shared mappings do not establish architecture."""
    covering = [family for family in TABLES if all(classify(name, family) for name in names)]
    if len(covering) != 1:
        unmatched = {family: sorted(name for name in names if not classify(name, family))[:3] for family in TABLES}
        raise TuneMathError(f'native target inventory is not covered by exactly one student table: {unmatched}')
    return covering[0]


def module_map(names, family=None, classes=None):
    """``({native name: ModuleTarget}, {native name: class})`` for the selected target classes."""
    if family is None:
        covering = [candidate for candidate in TABLES if all(classify(name, candidate) for name in names)]
        if len(covering) > 1 and all(len({repr(classify(name, candidate)) for candidate in covering}) == 1 for name in names):
            family = covering[0]
        else:
            family = student_family(names)
    selected = set(DEFAULT_CLASSES[family] if classes is None else classes)
    unknown = selected - {kind for *_, kind in TABLES[family]}
    if unknown:
        raise TuneMathError(f'unknown target classes for {family}: {sorted(unknown)}')
    targets, kinds = {}, {}
    for name in names:
        found = classify(name, family)
        if found is None:
            raise TuneMathError(f'native target {name} has no {family} HF module mapping')
        if found[0] in selected:
            kinds[name], targets[name] = found
    return targets, kinds


# Canonical 9 projection families for global LFM3 LoRA training (lfm3opti.md §5).
# Attention gate_proj (output width 16) is excluded by default and requires explicit gate_rank <= 16 if enabled.
LFM3_LORA_FAMILIES: tuple[str, ...] = (
    'self_attn.q_proj',
    'self_attn.k_proj',
    'self_attn.v_proj',
    'self_attn.o_proj',
    'conv.in_proj',
    'conv.out_proj',
    'feed_forward.w1',
    'feed_forward.w2',
    'feed_forward.w3',
)
LFM3_ATTN_GATE_FAMILY = 'self_attn.gate_proj'
_LFM3_ALL_FAMILIES = frozenset((*LFM3_LORA_FAMILIES, LFM3_ATTN_GATE_FAMILY))


def resolve_lfm3_lora_targets(model, families=LFM3_LORA_FAMILIES, *, gate_rank: int | None = None):
    """Resolve every targeted linear projection on an HF ``Lfm3ForCausalLM`` instance.

    Fails fast if any requested family is unknown or matches zero layers on ``model``, or if ``gate_rank`` exceeds the
    gate projection's width (16 on LFM3-1.7B). Returns an ordered list of
    ``(param_name, layer_idx, layer_kind, short_name, shape)``.
    """
    requested = tuple(families)
    unknown = set(requested) - _LFM3_ALL_FAMILIES
    if unknown:
        raise TuneMathError(f'unknown LFM3 LoRA target families: {sorted(unknown)}')
    if gate_rank is not None:
        if type(gate_rank) is not int or gate_rank < 1:
            raise TuneMathError(f'gate_rank must be a positive int, got {gate_rank!r}')
        if LFM3_ATTN_GATE_FAMILY not in requested:
            requested = (*requested, LFM3_ATTN_GATE_FAMILY)
    wanted = set(requested)
    matched_families = set()
    resolved = []
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        layer_kind = 'attn' if layer.is_attention_layer else 'conv'
        candidates = (
            (('self_attn.q_proj', 'q_proj', layer.self_attn.q_proj),
             ('self_attn.k_proj', 'k_proj', layer.self_attn.k_proj),
             ('self_attn.v_proj', 'v_proj', layer.self_attn.v_proj),
             ('self_attn.o_proj', 'o_proj', layer.self_attn.o_proj),
             ('self_attn.gate_proj', 'gate_proj', layer.self_attn.gate_proj))
            if layer.is_attention_layer else
            (('conv.in_proj', 'in_proj', layer.conv.in_proj),
             ('conv.out_proj', 'out_proj', layer.conv.out_proj))
        ) + (
            ('feed_forward.w1', 'w1', layer.feed_forward.w1),
            ('feed_forward.w2', 'w2', layer.feed_forward.w2),
            ('feed_forward.w3', 'w3', layer.feed_forward.w3),
        )
        for family, short_name, module in candidates:
            if family not in wanted:
                continue
            shape = tuple(module.weight.shape)
            if family == LFM3_ATTN_GATE_FAMILY and gate_rank is not None and gate_rank > min(shape):
                raise TuneMathError(f'gate_rank {gate_rank} exceeds {family} shape {shape} at layer {layer_idx}')
            param_name = f'model.layers.{layer_idx}.{family}.weight'
            resolved.append((param_name, layer_idx, layer_kind, short_name, shape))
            matched_families.add(family)
    missing = [f for f in requested if f not in matched_families]
    if missing:
        raise TuneMathError(f'LFM3 LoRA target families matched no layers: {missing}')
    return resolved


def packed_adapter_ranks(rank: int, families=LFM3_LORA_FAMILIES, *, gate_rank: int | None = None) -> dict[str, int]:
    """Packed serving rank per SGLang fused LFM3 module (qkv_proj = 3r, gate_up_proj = 2r, others = r)."""
    wanted = set(families)
    qkv = sum(rank for name in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj') if name in wanted)
    gate_up = sum(rank for name in ('feed_forward.w1', 'feed_forward.w3') if name in wanted)
    ranks = {}
    if qkv:
        ranks['qkv_proj'] = qkv
    if 'self_attn.o_proj' in wanted:
        ranks['o_proj'] = rank
    if gate_rank is not None or LFM3_ATTN_GATE_FAMILY in wanted:
        ranks['gate_proj'] = gate_rank or rank
    if 'conv.in_proj' in wanted:
        ranks['in_proj'] = rank
    if 'conv.out_proj' in wanted:
        ranks['out_proj'] = rank
    if gate_up:
        ranks['gate_up_proj'] = gate_up
    if 'feed_forward.w2' in wanted:
        ranks['w2'] = rank
    return ranks
