"""Exact mixed-checkpoint projection restoration for both backbone families."""
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.eval.task_cadence import restore_neuralese_projections
from natlang_neuralese.model.heads import PortHeads
from natlang_neuralese.model.neuralese_projection import NeuraleseProjection, TokenChoiceProjection


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_checkpoint_restores_writer_and_token_readout_exactly(family, tmp_path):
    if family == 'lfm':
        from test_backbone_policy import tiny_lfm
        backbone = tiny_lfm()
    else:
        from test_mellum_port import tiny_mellum
        from natlang_neuralese.maple.model import load_maple
        from natlang_neuralese.maple.maple_port import MaplePortBackbone
        from natlang_neuralese.model.lfm2_port import ControlTokens
        tiny_mellum(tmp_path)
        model = load_maple(tmp_path, device='cpu', dtype=torch.float32, ternary_attention=False)
        backbone = MaplePortBackbone(model, ControlTokens(126, 127), fast=False)
    heads = PortHeads(backbone, cutoff=max(1, backbone.num_layers - 1), profile='raw-token-v1')
    engine = SimpleNamespace(backbone=backbone, heads=heads, device='cpu')
    projection = NeuraleseProjection(backbone.config.hidden_size, hidden=16).to(torch.bfloat16)
    choice = TokenChoiceProjection(backbone.config.hidden_size, blocks=2, hidden=16, kernel=4)
    with torch.no_grad():
        projection.linear.weight.normal_(0, .02)
        for block in choice.stack.blocks:
            block.down.weight.normal_(0, .03)
    state = {'step': 42, 'neuralese_projection': projection.state_dict(),
             'token_choice_projection': choice.state_dict(),
             'identity': {'options': {'token_choice_projection': True,
                'projection_blocks': 2, 'projection_hidden': 16, 'projection_kernel': 4}}}
    binding = restore_neuralese_projections(engine, state)
    assert binding['step'] == 42 and binding['token_choice']
    for key, value in projection.state_dict().items():
        assert engine.neuralese_projection.state_dict()[key].dtype == value.dtype
        assert torch.equal(engine.neuralese_projection.state_dict()[key], value)
    for key, value in choice.state_dict().items():
        assert torch.equal(engine.token_choice_projection.state_dict()[key], value)
    assert all(not p.requires_grad for p in engine.neuralese_projection.parameters())
    bad = {**state, 'neuralese_projection': NeuraleseProjection(8).state_dict()}
    with pytest.raises(ValueError, match='geometry'):
        restore_neuralese_projections(engine, bad)
    with pytest.raises(ValueError, match='does not contain'):
        restore_neuralese_projections(engine, {**state, 'token_choice_projection': None})
