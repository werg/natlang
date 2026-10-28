"""Explicit native-target inventories and HF module maps per student (§5.3, §7).

Derived from the actual ports: LFM2 (`model/lfm2_port.py`, HF `Lfm2ForCausalLM`), LFM3 (`model/lfm3_port.py`, the
vendored pinned `Lfm3ForCausalLM`: LFM2's conv/feed-forward names, attention with ``o_proj`` and the per-head sigmoid
output gate ``self_attn.gate_proj``, which llama.cpp applies through `build_lora_mm` as `attn_gate`) and Mellum (`maple/model.py`, whose `DenseExperts` fuse each expert's
``gate_proj``/``up_proj`` rows into ``gate_up``). A native target that matches no pattern is an error, never a silently
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
    'mellum': (
        (_LAYER + r'self_attn\.(?P<name>q_proj|k_proj|v_proj|o_proj)\.weight$', 'model.layers.{layer}.self_attn.{name}', 'attention'),
        (_LAYER + r'mlp\.(?P<name>gate_proj|up_proj|down_proj)\.weight$', 'model.layers.{layer}.mlp.{name}', 'mlp'),
        (_LAYER + r'mlp\.experts\.gate_up$', ('model.layers.{layer}.mlp.experts.{{expert}}.gate_proj',
                                              'model.layers.{layer}.mlp.experts.{{expert}}.up_proj'), 'experts'),
        (_LAYER + r'mlp\.experts\.down$', 'model.layers.{layer}.mlp.experts.{{expert}}.down_proj', 'experts'),
        (_LAYER + r'mlp\.gate\.weight$', 'model.layers.{layer}.mlp.gate', 'router'),
    ),
}
# Router logits are control parameters: covered by the map, trained only when a recipe explicitly selects them.
DEFAULT_CLASSES = {'lfm2': ('attention', 'conv-projection', 'mlp'), 'lfm3': ('attention', 'conv-projection', 'mlp'),
                   'mellum': ('attention', 'mlp', 'experts')}
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
    """The student whose table covers every native target; a gap is an error. A partial inventory several tables
    cover (an adapter over q_proj only fits LFM2 and LFM3) is unambiguous only when every covering table maps every
    name to the same HF module and class; otherwise it is an error."""
    covering = [family for family in TABLES if all(classify(name, family) for name in names)]
    if len(covering) > 1 and all(len({repr(classify(name, family)) for family in covering}) == 1 for name in names):
        return covering[0]
    if len(covering) != 1:
        unmatched = {family: sorted(name for name in names if not classify(name, family))[:3] for family in TABLES}
        raise TuneMathError(f'native target inventory is not covered by exactly one student table: {unmatched}')
    return covering[0]


def module_map(names, family=None, classes=None):
    """``({native name: ModuleTarget}, {native name: class})`` for the selected target classes."""
    family = family or student_family(names)
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
