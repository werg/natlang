import json

import pytest
import torch

from natlang_neuralese.tune import TuneMathError
from natlang_neuralese.tune.bases import canonical_basis
from natlang_neuralese.tune.export import ModuleTarget, export_peft, load_peft_dense
from natlang_neuralese.tune.parameterization import FreeLoRADelta, SubspaceDelta
from natlang_neuralese.tune.stack import StackMember, stack_factors

IDENTITY = {'architecture': 'test', 'checkpoint_sha256': '0'*64}


def rand(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def test_peft_round_trip_reconstructs_stack_deltas(tmp_path):
    basis = canonical_basis(IDENTITY, 'q', rand((10, 8), 1), 6)
    bases = {basis.basis_id: basis}
    targets = {'q': (10, 8), 'experts': (2, 6, 5), 'conv_in': (5, 7), 'unused': (3, 3)}
    members = [StackMember('tree:r', 'tree', 1., {'q': SubspaceDelta(basis.basis_id, 6, rand((6, 6), 2))}),
               StackMember('lambda:l', 'lambda', 0.5, {'q': SubspaceDelta(basis.basis_id, 2, rand((2, 2), 3)),
                   'experts': FreeLoRADelta(rand((2, 6, 3), 4), rand((2, 3, 5), 5), 3.),
                   'conv_in': FreeLoRADelta(rand((5, 2), 6), rand((2, 7), 7), 4.)})]
    stack = stack_factors(members, bases, targets)
    module_map = {'q': ModuleTarget('model.layers.0.self_attn.q_proj'),
                  'experts': ModuleTarget('model.layers.1.mlp.experts.{expert}.down_proj'),
                  'conv_in': ModuleTarget('model.layers.2.conv.in_proj', 'in_out')}
    receipt = export_peft(stack, module_map, tmp_path, base_identity=IDENTITY, base_model_name_or_path='test/base', lora_alpha=1.)
    assert receipt['ranks']['model.layers.0.self_attn.q_proj'] == 6 and receipt['zero_targets'] == ['unused']
    assert receipt['modes']['q'] == 'common-basis' and receipt['modes']['experts'] == 'concatenated'
    deltas = load_peft_dense(tmp_path)
    assert torch.allclose(deltas['model.layers.0.self_attn.q_proj'].float(), stack['q'].dense(targets['q']), atol=1e-5)
    for expert in range(2):
        assert torch.allclose(deltas[f'model.layers.1.mlp.experts.{expert}.down_proj'].float(),
                              stack['experts'].dense(targets['experts'])[expert], atol=1e-5)
    assert torch.allclose(deltas['model.layers.2.conv.in_proj'].float(), stack['conv_in'].dense(targets['conv_in']).T, atol=1e-5)
    config = json.loads((tmp_path / 'adapter_config.json').read_text())
    assert config['lora_alpha'] == 1. and config['rank_pattern']['model.layers.2.conv.in_proj'] == 2


def test_unmapped_or_misshaped_targets_fail_loudly(tmp_path):
    stack = stack_factors([StackMember('l', 'lambda', 1., {'w': FreeLoRADelta(rand((4, 2), 1), rand((2, 3), 2), 2.)})],
                          {}, {'w': (4, 3)})
    with pytest.raises(TuneMathError, match='no HF module mapping'):
        export_peft(stack, {}, tmp_path, base_identity=IDENTITY, base_model_name_or_path='b')
    with pytest.raises(TuneMathError, match='per expert'):
        export_peft(stack, {'w': ModuleTarget('m.{expert}.x')}, tmp_path, base_identity=IDENTITY, base_model_name_or_path='b')
