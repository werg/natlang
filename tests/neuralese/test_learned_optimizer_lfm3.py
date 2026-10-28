"""LFM3-1.7B support of the tuner and the joint learned optimizer: the native target table and bank groups on the LFM3
layer inventory, the deploy-quantized targets, and the number grid's spelling in the real LFM3 tokenizer."""
import glob
import json
from pathlib import Path

import pytest

from natlang_neuralese.tune import targets as T
from natlang_neuralese.tune.learned import numbers as N
from natlang_neuralese.tune.learned.bank import target_groups

PINNED = Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
# The pinned config's layer pattern (conv / full / sliding), repeated over 24 layers.
LAYER_TYPES = ['conv', 'conv', 'full_attention', 'conv', 'sliding_attention', 'conv'] + \
    ['full_attention', 'conv', 'sliding_attention', 'conv'] * 4 + ['full_attention', 'conv']


def lfm3_inventory(layer_types):
    """Native target names of an LFM3 port (`native_adapters.target_parameters`: 2-D layer weights under port.hf)."""
    names = []
    for layer, kind in enumerate(layer_types):
        root = f'port.hf.model.layers.{layer}.'
        if kind == 'conv':
            names += [root + f'conv.{name}.weight' for name in ('in_proj', 'out_proj')]
        else:
            names += [root + f'self_attn.{name}.weight' for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj')]
        names += [root + f'feed_forward.{name}.weight' for name in ('w1', 'w2', 'w3')]
    return sorted(names)


def lfm2_inventory(layers=4):
    names = []
    for layer in range(layers):
        root = f'port.hf.model.layers.{layer}.'
        names += ([root + f'conv.{name}.weight' for name in ('in_proj', 'out_proj')] if layer % 2 == 0 else
                  [root + f'self_attn.{name}.weight' for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj')])
        names += [root + f'feed_forward.{name}.weight' for name in ('w1', 'w2', 'w3')]
    return sorted(names)


def test_pinned_config_has_the_declared_layer_pattern():
    if not (PINNED / 'config.json').exists():
        pytest.skip('pinned LFM3-1.7B snapshot is not cached here')
    config = json.loads((PINNED / 'config.json').read_text())
    assert config['model_type'] == 'lfm3' and config['layer_types'] == LAYER_TYPES and config['num_hidden_layers'] == 24


def test_lfm3_table_covers_the_inventory_with_o_proj_and_the_attention_gate():
    names = lfm3_inventory(LAYER_TYPES)
    assert T.student_family(names) == 'lfm3'
    mapped, kinds = T.module_map(names)
    assert set(mapped) == set(names)  # every default class: attention (incl. gate_proj), conv projections, feed-forward
    assert kinds['port.hf.model.layers.2.self_attn.gate_proj.weight'] == 'attention'
    assert mapped['port.hf.model.layers.2.self_attn.o_proj.weight'].hf_module == 'model.layers.2.self_attn.o_proj'
    assert mapped['port.hf.model.layers.0.conv.in_proj.weight'].hf_module == 'model.layers.0.conv.in_proj'
    assert mapped['port.hf.model.layers.23.feed_forward.w2.weight'].hf_module == 'model.layers.23.feed_forward.w2'
    assert sum(kind == 'attention' for kind in kinds.values()) == 11 * 5
    assert sum(kind == 'conv-projection' for kind in kinds.values()) == 13 * 2
    assert sum(kind == 'mlp' for kind in kinds.values()) == 24 * 3
    # Without the attention class, only conv and feed-forward targets remain.
    mapped, _ = T.module_map(names, classes=('conv-projection', 'mlp'))
    assert not any('self_attn' in name for name in mapped)


def test_lfm2_inventories_and_shared_partial_inventories_stay_unambiguous():
    assert T.student_family(lfm2_inventory()) == 'lfm2'
    # An adapter over q_proj and feed-forward only fits both LFM tables with identical modules and classes.
    partial = ['port.hf.model.layers.2.self_attn.q_proj.weight', 'port.hf.model.layers.2.feed_forward.w1.weight']
    assert T.student_family(partial) in ('lfm2', 'lfm3')
    assert T.module_map(partial)[1] == {partial[0]: 'attention', partial[1]: 'mlp'}
    with pytest.raises(Exception, match='not covered by exactly one student table'):
        T.student_family(['port.hf.model.layers.0.self_attn.q_proj.weight', 'port.hf.model.layers.0.unknown.weight'])


def test_nvfp4_deploy_point_quantizes_only_the_lfm3_feed_forward():
    names = lfm3_inventory(LAYER_TYPES)
    quantized = T.deploy_quantized(names)
    assert set(quantized) == {'nvfp4'} and len(quantized['nvfp4']) == 72
    assert all(name.split('.')[-2] in ('w1', 'w2', 'w3') for name in quantized['nvfp4'])
    assert T.deploy_quantized(lfm2_inventory()) == {}


def test_lfm3_bank_groups_are_depth_quarters_by_matrix_family():
    names = lfm3_inventory(LAYER_TYPES)
    _, kinds = T.module_map(names)
    labels, index = target_groups(names, kinds, len(LAYER_TYPES))
    assert len(labels) == 12 and {label.split(':')[1] for label in labels} == {'attention', 'convolution', 'mlp'}
    assert labels[index['port.hf.model.layers.2.self_attn.gate_proj.weight']] == 'q0:attention'
    assert labels[index['port.hf.model.layers.23.feed_forward.w3.weight']] == 'q3:mlp'


def test_real_lfm3_tokenizer_spells_the_grid_as_sign_then_digit():
    paths = glob.glob(str(PINNED / 'tokenizer.json'))
    if not paths:
        pytest.skip('pinned LFM3-1.7B tokenizer is not cached here')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(paths[0])
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    tokens = N.GridTokens.from_encoder(encode)
    assert (tokens.open, tokens.close, tokens.separator, tokens.positive, tokens.negative) == (67, 69, 20, 229, 640)
    assert tokens.digits == tuple(range(24, 34))
    for values in ([3, 0, -2], [-9, 9, 0, 1], [0]):
        assert encode(N.text(values)) == tokens.spell(values)
    # The forced template parses back exactly.
    assert N.parse(tokens, tokens.spell([4, -1, 0]), [True, True, False]) == [4, -1, 0]
