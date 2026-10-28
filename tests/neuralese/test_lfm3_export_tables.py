"""LFM3 in the llama.cpp export path: tune target tables, the converter coverage check and state-span architectures."""
import json

import pytest

from natlang_neuralese.export.native_gguf import ARCHITECTURES, convert_hf, converter_registers
from natlang_neuralese.tune import TuneMathError
from natlang_neuralese.tune.soft_export import STATE_ARCHITECTURES
from natlang_neuralese.tune.targets import classify, module_map, student_family

LFM3_NAMES = [f'port.hf.model.layers.{layer}.{name}.weight' for layer, names in (
    (0, ('conv.in_proj', 'conv.out_proj', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3')),
    (2, ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.gate_proj',
         'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3'))) for name in names]


def test_lfm3_targets_map_to_hf_modules():
    assert student_family(LFM3_NAMES) == 'lfm3'
    kind, target = classify('port.hf.model.layers.2.self_attn.gate_proj.weight', 'lfm3')
    assert kind == 'attention' and target.hf_module == 'model.layers.2.self_attn.gate_proj'
    assert classify('port.hf.model.layers.2.self_attn.out_proj.weight', 'lfm3') is None
    targets, kinds = module_map(LFM3_NAMES)
    assert len(targets) == len(LFM3_NAMES) and set(kinds.values()) == {'attention', 'conv-projection', 'mlp'}
    lfm2 = [name.replace('o_proj', 'out_proj') for name in LFM3_NAMES if 'gate_proj' not in name]
    assert student_family(lfm2) == 'lfm2'
    with pytest.raises(TuneMathError, match='exactly one student table'):
        student_family([name for name in LFM3_NAMES if 'self_attn' not in name])  # conv/mlp only: LFM2 or LFM3


def test_converter_coverage_is_checked_before_converting(tmp_path):
    converter = tmp_path / 'llama.cpp'
    (converter / 'conversion').mkdir(parents=True)
    (converter / 'convert_hf_to_gguf.py').write_text('')
    (converter / 'conversion' / 'lfm2.py').write_text('@ModelBase.register("Lfm2ForCausalLM")\nclass LFM2Model: pass\n')
    assert converter_registers(converter, 'Lfm2ForCausalLM')
    assert not converter_registers(converter, 'Lfm3ForCausalLM')
    hf_dir = tmp_path / 'hf'
    hf_dir.mkdir()
    (hf_dir / 'config.json').write_text(json.dumps({'architectures': ['Lfm3ForCausalLM'], 'model_type': 'lfm3'}))
    with pytest.raises(ValueError, match='does not register'):
        convert_hf(hf_dir, tmp_path / 'out.gguf', 'bf16', llama_cpp=converter)
    (converter / 'conversion' / 'lfm3.py').write_text('@ModelBase.register("Lfm3ForCausalLM", "Lfm3VlForConditionalGeneration")\n')
    assert converter_registers(converter, 'Lfm3ForCausalLM')


def test_lfm3_is_exportable_and_its_states_import():
    assert 'lfm3' in ARCHITECTURES and 'lfm3' in STATE_ARCHITECTURES
