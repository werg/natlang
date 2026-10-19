"""Explicit tensor-core math must retain source state and graph semantics."""
import copy
import pytest
import torch

from natlang_neuralese.model.native_precision import set_native_linear_precision, same_source_precision_identity
from natlang_neuralese.model.shared_state import SharedStateModel


def test_precision_is_per_model_restorable_and_preserves_parameters():
    from test_shared_state import tiny
    first = SharedStateModel(tiny('lfm'), (60, 61, 62, 63), rank=4)
    second = SharedStateModel(tiny('lfm'), (60, 61, 62, 63), rank=4)
    parameters = dict(first.named_parameters())
    state = copy.deepcopy(first.state_dict())
    flags = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
    set_native_linear_precision(first, 'bf16')
    assert getattr(second, 'native_linear_precision', 'ieee') == 'ieee'
    assert flags == (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
    assert all(value is dict(first.named_parameters())[key] for key, value in parameters.items())
    assert all(torch.equal(value, first.state_dict()[key]) for key, value in state.items())
    layer = first.port.hf.model.layers[0].feed_forward.w1
    assert type(layer) is torch.nn.Linear
    assert 'forward' not in layer.__dict__
    set_native_linear_precision(first, 'ieee')
    assert not hasattr(layer, '_native_original_forward')
    assert same_source_precision_identity({'base': 'same'}, {'base': 'same', 'native_linear_precision': 'tf32'})
    assert not same_source_precision_identity({'base': 'same'}, {'base': 'other', 'native_linear_precision': 'tf32'})


def test_precision_checkpoint_exact_restore_and_identity(tmp_path):
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model, load_native_checkpoint, model_state, native_identity, CHECKPOINT_SCHEMA
    base = tmp_path / 'base'; tiny_source(base)
    model, _, _, config = load_native_model(base, dtype=torch.float32, native_linear_precision='bf16')
    source_weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    saved = tmp_path / 'native.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, saved)
    restored, _, _, restored_config, _ = load_native_checkpoint(saved)
    assert restored.native_linear_precision == 'bf16'
    assert restored_config == config
    assert all(torch.equal(value, dict(restored.named_parameters())[key]) for key, value in source_weights.items())
    strict_config = {**config, 'native_linear_precision': 'ieee'}
    before = native_identity(model, config, checkpoint_sha256='1'*64)
    with pytest.raises(ValueError, match='arithmetic'):
        native_identity(model, strict_config, checkpoint_sha256='1'*64)
    set_native_linear_precision(model, 'ieee')
    assert before != native_identity(model, strict_config, checkpoint_sha256='1'*64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA native arithmetic')
@pytest.mark.parametrize('mode', ['ieee', 'tf32', 'bf16', 'bf16x6'])
def test_fixed_linear_modes_preserve_row_batching_and_master_gradient(mode):
    from natlang_neuralese.maple.fused_moe import invariant_linear
    torch.manual_seed(81)
    x = torch.randn(133, 128, device='cuda', requires_grad=True)
    w = torch.randn(129, 128, device='cuda', requires_grad=True)
    y = invariant_linear(x, w, precision=mode)
    split = torch.cat([invariant_linear(x[:79], w, precision=mode), invariant_linear(x[79:], w, precision=mode)])
    torch.testing.assert_close(y, split, atol=0, rtol=0)
    dx, dw = torch.autograd.grad(y.square().mean(), (x, w))
    assert dx.dtype == dw.dtype == torch.float32
    assert torch.isfinite(dx).all() and torch.isfinite(dw).all()
    assert dx.norm() > 0 and dw.norm() > 0
    # Independent rounded-operand matmul protects the master-weight backward,
    # including non-contiguous transposes and incomplete tile tails.
    if mode == 'bf16':
        expected = (2*y.detach()/y.numel()).T @ x.detach().to(torch.bfloat16).float()
        torch.testing.assert_close(dw, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA native arithmetic')
def test_single_expert_linear_elides_index_fill_without_changing_any_matrix_result():
    """The shared one-expert route must match the old explicit expert-0 plan exactly.

    This covers forward, input-gradient, and weight-gradient GEMMs. The profile
    showed one CUDA int-fill per grouped dense launch; the explicit-plan side
    below retains that old routing representation as a numeric reference.
    """
    from types import SimpleNamespace
    import triton
    from natlang_neuralese.maple.fused_moe import (
        INVARIANT_BLOCK, _dense_mm, invariant_linear, invariant_weight_gradient,
    )

    torch.manual_seed(119)
    x = torch.randn(137, 192, device='cuda', dtype=torch.float32)
    weight = torch.randn(320, 192, device='cuda', dtype=torch.float32)
    output_adjoint = torch.randn(137, 320, device='cuda', dtype=torch.float32)
    expected_plan = SimpleNamespace(
        size=x.shape[0], blocks=triton.cdiv(x.shape[0], INVARIANT_BLOCK),
        block=INVARIANT_BLOCK,
        block_expert=torch.zeros(triton.cdiv(x.shape[0], INVARIANT_BLOCK),
                                 dtype=torch.int32, device='cuda'),
    )

    actual_output = invariant_linear(x, weight)
    explicit_output = _dense_mm(x, None, weight[None], expected_plan, transpose=False)
    assert torch.equal(actual_output, explicit_output)

    x_for_grad = x.detach().requires_grad_(True)
    w_for_grad = weight.detach().requires_grad_(True)
    actual = torch.autograd.grad((invariant_linear(x_for_grad, w_for_grad,
                                                  forward_storage_dtype=torch.float32) * output_adjoint).sum(),
                                 (x_for_grad, w_for_grad))
    expected_dx = _dense_mm(output_adjoint, None, weight[None], expected_plan, transpose=True)
    weight_plan = SimpleNamespace(
        size=weight.shape[0], blocks=triton.cdiv(weight.shape[0], INVARIANT_BLOCK),
        block=INVARIANT_BLOCK,
        block_expert=torch.zeros(triton.cdiv(weight.shape[0], INVARIANT_BLOCK),
                                 dtype=torch.int32, device='cuda'),
    )
    expected_dw = _dense_mm(output_adjoint.T.contiguous(), None, x.T[None], weight_plan,
                            transpose=False, output_dtype=torch.float32)
    assert torch.equal(actual[0], expected_dx)
    assert torch.equal(actual[1], expected_dw)
    assert torch.equal(invariant_weight_gradient(x, output_adjoint), expected_dw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA native graph precision')
@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('mode', ['tf32', 'bf16', 'bf16x6'])
def test_both_ports_flat_cached_consumer_gradients_under_selected_math(family, mode):
    from test_shared_state import tiny, graph
    model = SharedStateModel(tiny(family), (60, 61, 62, 63), rank=4, native_norm_execution='stable-v1').cuda()
    set_native_linear_precision(model, mode)
    model.port.checkpoint_layers = True
    torch.manual_seed(93)
    with torch.no_grad(): model.adapter.up.weight.normal_(std=.01)
    plan = graph()
    flat, cached = model(plan), model(plan, reference=True)
    # Consumer-only loss requires raw-state gradient flow to the producer.
    a = model.logits(flat['h_final'][:, -1]).float().square().mean()
    b = model.logits(cached['h_final'][:, -1]).float().square().mean()
    ga = torch.autograd.grad(a, model.adapter.up.weight)[0]
    gb = torch.autograd.grad(b, model.adapter.up.weight)[0]
    torch.testing.assert_close(flat['h_final'], cached['h_final'], atol=5e-6, rtol=2e-5)
    torch.testing.assert_close(ga, gb, atol=3e-6, rtol=3e-4)
    assert torch.isfinite(ga).all() and ga.norm() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA native arithmetic transition')
def test_real_training_precision_transition_restores_optimizer_rng_and_row_credits(tmp_path, monkeypatch):
    import json
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    base = tmp_path/'base'; tiny_source(base)
    _, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
    files = {}
    for split in ('train', 'test'):
        files[split] = tmp_path/(split+'.jsonl')
        files[split].write_text(json.dumps({**graph(split), 'split':split,
            'tokenizer_fingerprint':config['tokenizer_fingerprint']})+'\n')
    out = tmp_path/'run'
    argv = ['--base',str(base),'--dtype','float32','--rank','4','--device','cuda',
        '--native-graphs',str(files['train']),'--held-native-graphs',str(files['test']),
        '--epochs','3','--eval-every','100','--out',str(out),
        '--forward-atol','1e-4','--gradient-atol','1e-5']
    def stop(policy, step, **kwargs): policy.stop.requested=True; return False
    monkeypatch.setattr(trainer.CheckpointPolicy, 'due', stop)
    assert trainer.main(argv) == 0
    before = torch.load(out/'checkpoint.pt', weights_only=False)
    # Saved diagnostic modes must not evade production argparse choices through
    # an omitted option. Restoration is real, but no update may execute.
    for diagnostic_mode in ('tf32', 'tf32x3', 'bf16'):
        diagnostic = copy.deepcopy(before)
        diagnostic['model_config']['native_linear_precision'] = diagnostic_mode
        diagnostic['interface_state']['native_linear_precision'] = diagnostic_mode
        diagnostic['training']['identity']['config']['native_linear_precision'] = diagnostic_mode
        torch.save(diagnostic, out/'checkpoint.pt')
        with pytest.raises(ValueError, match='explicitly transition'):
            trainer.main(argv)
    optimizer_loads, optimizer_load_options, rng_restores = [], [], []
    actual_make, actual_restore = trainer.make_native_optimizer, trainer.restore_training_rng_state
    def make(*args, **kwargs):
        optimizer, named = actual_make(*args, **kwargs)
        actual_load = optimizer.load_state_dict
        def load(state, **load_options):
            optimizer_loads.append(copy.deepcopy(state))
            optimizer_load_options.append(copy.deepcopy(load_options))
            return actual_load(state, **load_options)
        optimizer.load_state_dict = load
        return optimizer, named
    def restore(state, *args): rng_restores.append(copy.deepcopy(state)); return actual_restore(state, *args)
    monkeypatch.setattr(trainer, 'make_native_optimizer', make)
    monkeypatch.setattr(trainer, 'restore_training_rng_state', restore)
    assert trainer.main(argv+['--native-linear-precision','bf16x6']) == 0
    after = torch.load(out/'checkpoint.pt', weights_only=False)
    assert len(optimizer_loads) == 1 and len(rng_restores) == 2
    def equal(left, right):
        if torch.is_tensor(left): assert torch.equal(left.cpu(), right.cpu())
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left: equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a,b in zip(left,right): equal(a,b)
        else: assert left == right
    equal(optimizer_loads[0], before['training']['optimizer'])
    assert optimizer_load_options == [{'native_optimizer_migration': None}]
    equal(rng_restores[0], before['training']['rng'])
    equal(rng_restores[1], rng_restores[0])
    assert after['training']['options']['native_precision_transition']['reference_check']['runtime_reference']['passed']
    assert before['training']['exposure_counts'] == {'train':1}
    assert after['training']['exposure_counts'] == {'train':2}
    assert after['training']['global_step'] == 2
    assert after['model_config']['dtype'] == 'torch.float32'
    assert after['model_config']['base_inventory_sha256'] == before['model_config']['base_inventory_sha256']
    assert after['model_config']['native_linear_precision'] == 'bf16x6'
    assert after['training']['options']['native_precision_transition']['from'] == 'bf16'


def test_failed_destination_transition_restores_rng_before_rejecting(monkeypatch):
    from natlang_neuralese.eval import native_state
    from natlang_neuralese.train.native_state_sft import qualify_precision_transition
    before = torch.random.get_rng_state().clone()
    def failed(*args, **kwargs):
        torch.rand(17)
        return {'passed': False, 'forward_max_abs': .01}
    monkeypatch.setattr(native_state, 'qualify_existing', failed)
    with pytest.raises(ValueError, match='unchanged cached-reference'):
        qualify_precision_transition(None, None, None, {'native_linear_precision': 'bf16x6'},
            previous_precision='ieee', checkpoint_sha256='a'*64,
            forward_atol=1e-4, gradient_atol=1e-5, device='cpu')
    assert torch.equal(before, torch.random.get_rng_state())
