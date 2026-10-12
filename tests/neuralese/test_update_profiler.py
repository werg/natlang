"""The env-gated update profiler (common/profiling.py): a no-op when unset; on CUDA it attributes forward, checkpoint
recompute, backward (by the forward scope of each gradient) and the MuonSR Newton-Schulz step."""
import json

import pytest
import torch

from natlang_neuralese.common import profiling


def test_scopes_are_shared_noops_when_no_profile_records(tmp_path):
    assert not profiling.ENABLED
    assert profiling.scope('norm') is profiling.scope('moe') is profiling.region('update/forward')
    assert profiling.backward_phase() is profiling.scope('x')
    profiler = profiling.UpdateProfiler(tmp_path, value='')
    profiler.begin(1)
    profiler.maybe_finish(1)
    profiler.close(1)
    assert not profiler.active and not list(tmp_path.iterdir())
    assert profiling.parse_range('5:8') == (5, 8) and profiling.parse_range('3') == (3, 3)
    with pytest.raises(ValueError):
        profiling.parse_range('8:5')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='kernel attribution needs CUDA')
def test_cuda_profile_attributes_forward_recompute_backward_and_newton_schulz(tmp_path):
    from natlang_neuralese.maple.maple_port import MaplePortBackbone
    from natlang_neuralese.maple.model import MapleConfig, MapleModel
    from natlang_neuralese.train.optim import MuonSR

    torch.manual_seed(0)
    config = MapleConfig(vocab_size=64, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, moe_intermediate_size=32, num_experts=4,
                         num_experts_per_tok=2, sliding_window=8, layer_types=['sliding_attention', 'full_attention'],
                         model_type='mellum', partial_rotary_factor=1.0, swiglu_clamp=None, dense_experts=True)
    model = MapleModel(config).cuda().bfloat16()
    latents = [p for layer in model.layers for p in layer.mlp.experts.make_latent(quantize=False)]
    optimizer = MuonSR([{'params': latents}], lr=1e-4)
    profiler = profiling.UpdateProfiler(tmp_path, value='2:3')
    ids = torch.randint(0, 64, (1, 32), device='cuda')
    for update in (1, 2, 3, 4):
        profiler.begin(update)
        model.checkpoint_layers = True
        with profiling.region('update/forward'):
            loss = model(input_ids=ids).last_hidden_state.float().square().mean()
        with profiling.region('update/backward'), profiling.backward_phase():
            loss.backward()
        with profiling.region('update/optimizer'):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        profiler.update_done(update, 0.0)
        profiler.maybe_finish(update)
    assert not profiling.ENABLED
    report = json.loads((tmp_path / 'profile-2-3.json').read_text())
    assert (tmp_path / 'profile-2-3.trace.json.gz').is_file()
    assert sorted(report['updates']) == ['2', '3'], report.get('error')
    categories = report['by_category_gpu_seconds']
    for name in ('fwd/norm', 'fwd/moe.route', 'fwd/attention.sdpa', 'recompute/norm', 'optim/muonsr.newton_schulz'):
        assert categories.get(name, 0) > 0, (name, categories)
    assert any(k.startswith('bwd/') and not k.startswith('bwd/unscoped') for k in categories), categories
    summary = report['summary']
    assert summary['forward'] > 0 and summary['recompute'] > 0 and summary['backward'] > 0
    assert set(report['by_phase_gpu_seconds']) >= {'update/forward', 'update/backward', 'update/optimizer'}
