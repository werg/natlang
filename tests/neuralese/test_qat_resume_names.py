"""Mellum direct_sft resume across QAT: with the recipe's precision points installed (q4 on attention and experts,
ternary on experts), a save -> fresh-process resume keeps the optimizer schema, parameter names and shapes, and
restores the latents and optimizer state exactly; the names also equal the BF16 (pre-install) names, so a BF16
checkpoint continues into the QAT objective. Mirrors direct_sft's setup order: install, latent policy, optimizer."""
from pathlib import Path

import torch

from natlang_neuralese.maple.maple_port import MaplePortBackbone
from natlang_neuralese.maple.model import MapleConfig, MapleForCausalLM
from natlang_neuralese.model.lfm2_port import ControlTokens
from natlang_neuralese.train.backbone_policy import (backbone_trainable_state, configure_backbone_training,
                                                     plain_named_tensors)
from natlang_neuralese.train.optim import PortMuonAdamW, latent_entries, latent_options, latent_split
from natlang_neuralese.train.quantization import Quantization
from natlang_neuralese.train.recipe import resolve_recipe_data, stage_quantization

RECIPE = Path(__file__).parents[2] / 'training' / 'neuralese' / 'recipes' / 'neuralese-direct-mellum-v4.json'
LAYERS = 4


def tiny_mellum():
    torch.manual_seed(0)
    config = MapleConfig(vocab_size=64, hidden_size=64, num_hidden_layers=LAYERS, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=32, moe_intermediate_size=32, num_experts=8,
                         num_experts_per_tok=2, layer_types=['full_attention'] * LAYERS, dense_experts=True,
                         model_type='mellum', swiglu_clamp=None, partial_rotary_factor=1.0)
    model = MapleForCausalLM(config)
    for tensor in list(model.parameters()) + [b for b in model.buffers() if b.is_floating_point()]:
        torch.nn.init.normal_(tensor, std=0.2)
    model.natlang_precision = 'bf16'
    return model.eval()


def component():
    recipe = resolve_recipe_data(RECIPE)
    stage = next(s for s in recipe['stages'] if s['id'] == 'direct_sft')
    parameters = {**recipe['stage_parameter_defaults']['neuralese_direct'], **stage['parameters']}
    found = stage_quantization(recipe, stage, parameters, [])
    found['stage']['end'] = 100
    return found


def setup(install):
    model = tiny_mellum()
    backbone = MaplePortBackbone(model, controls=ControlTokens(open_id=1, close_id=2))
    quant = None
    if install:
        quant = Quantization(component())
        quant.install(backbone.hf, layer_count=backbone.num_layers)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone_named = [('backbone.' + n, q) for n, q in configure_backbone_training(backbone, 'latent')]
    latent_named, other = latent_split(backbone_named, 'muonsr')
    torch.manual_seed(1)  # direct_sft's projections (hidden matrices on Muon)
    head = [('neuralese_projection.w', torch.nn.Parameter(torch.randn(64, 64) * 0.02))]
    optimizer = PortMuonAdamW(head + other, lr=1e-3, vocab_size=1000,
                              latent=latent_entries(latent_named, 'muonsr', 1e-3, 32), latent_optimizer='muonsr',
                              latent_options=latent_options('muonsr', weight_decay=0.1))
    names = {id(q): n for n, q in head + backbone_named}
    optimizer_names = [[names[id(q)] for q in group['params']] for group in optimizer.param_groups]
    return backbone, quant, backbone_named, optimizer, optimizer_names


def step(backbone, quant, optimizer, point):
    ids = torch.randint(3, 64, (2, 12))
    with quant.context(point, 1.0):
        loss = backbone.forward_ids(ids, logits=True)['logits'].float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_mellum_qat_resume_keeps_optimizer_names_shapes_and_state():
    bf16 = setup(install=False)
    backbone, quant, named, optimizer, optimizer_names = setup(install=True)
    points = [p for p in quant.points]
    assert {'q4', 'ternary-experts'} <= set(points), points
    experts = [m for m in backbone.hf.modules() if type(m).__name__ == 'DenseExperts']
    assert experts and all(m.precision_group is not None for m in experts)
    assert any('.parametrizations.weight.original' in n for n, _ in backbone.hf.named_parameters())
    # BF16 (pre-install) and QAT setups name and shape the latents identically.
    assert [(n, tuple(q.shape)) for n, q in named] == [(n, tuple(q.shape)) for n, q in bf16[2]]
    assert optimizer.schema == bf16[3].schema and optimizer_names == bf16[4]
    for point in points:
        step(backbone, quant, optimizer, point)
    moved = [n for (n, q), (_, b) in zip(named, bf16[2]) if not torch.equal(q.detach(), b.detach())]
    assert len(moved) > len(named) // 2, moved  # the QAT steps trained the latents (attention and experts)
    assert any('.self_attn.' in n for n in moved) and any('.mlp.experts.' in n for n in moved)
    assert optimizer.state_dict()['schema']
    saved = {'student_parameters': {n: t.clone() for n, t in backbone_trainable_state(named).items()},
             'optimizer': optimizer.state_dict(), 'optimizer_parameter_names': optimizer_names}
    saved = torch.load(_roundtrip(saved), weights_only=False)
    # A fresh process: new model, install, the same setup.
    backbone2, quant2, named2, optimizer2, optimizer_names2 = setup(install=True)
    assert saved['optimizer']['schema'] == optimizer2.state_dict()['schema']
    assert saved['optimizer_parameter_names'] == optimizer_names2
    from natlang_neuralese.train.text_warmup import copy_restored_values
    copy_restored_values({'backbone.' + n: t for n, t in plain_named_tensors(backbone2.hf).items()},
                         saved['student_parameters'])
    optimizer2.load_state_dict(saved['optimizer'])
    optimizer2.param_groups = optimizer2._groups()
    for (n, q), (n2, q2) in zip(named, named2):
        assert n == n2 and torch.equal(q.detach(), q2.detach()), n
    # Both continue identically at the q4 point.
    torch.manual_seed(5)
    step(backbone, quant, optimizer, points[0])
    torch.manual_seed(5)
    step(backbone2, quant2, optimizer2, points[0])
    for (n, q), (_, q2) in zip(named, named2):
        assert torch.allclose(q.detach(), q2.detach(), atol=1e-6), n


def _roundtrip(state):
    import io
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return buffer
