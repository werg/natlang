"""Explicit tensor-core math must retain source state and graph semantics."""
import copy
import pytest
import torch

from natlang_neuralese.model.native_precision import set_native_linear_precision, same_source_precision_identity
from natlang_neuralese.model.shared_state import SharedStateModel


def assert_state_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_base_location_change_requires_consistent_full_byte_inventory():
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from natlang_neuralese.model.native_precision import base_location_transition
    inventory = {'config.json': 'a'*64, 'model.safetensors': 'b'*64, 'tokenizer.json': 'c'*64}
    before = {'base': '/old/base', 'base_inventory': inventory,
              'base_inventory_sha256': canonical_json_sha256_hex(inventory), 'rank': 8}
    after = {**before, 'base': '/internal/base'}
    assert same_source_precision_identity(before, after)
    assert base_location_transition(before, after)['input_bytes'] == 'identical'
    assert not same_source_precision_identity(before, {**after, 'rank': 9})
    assert not same_source_precision_identity(before, {**after, 'base_inventory_sha256': '0'*64})
    assert not same_source_precision_identity(before, {**after, 'base_inventory': {'config.json': 'a'*64}})


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
    assert same_source_precision_identity({'base': 'same', 'precision': {'cuda_matmul_tf32': False}},
                                          {'base': 'same', 'precision': {'cuda_matmul_tf32': True}})


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
    import shutil
    relocated = tmp_path / 'relocated'
    shutil.copytree(base, relocated)
    moved, _, _, moved_config, _ = load_native_checkpoint(saved, base=relocated)
    assert same_source_precision_identity(config, moved_config)
    assert all(torch.equal(value, dict(moved.named_parameters())[key]) for key, value in source_weights.items())
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
@pytest.mark.parametrize('family', ['lfm'])
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


def test_production_stock_arithmetic_uses_tf32_tensor_cores(monkeypatch):
    from natlang_neuralese.model.native_model import configure_native_precision
    monkeypatch.delenv('NATLANG_STOCK_ARITHMETIC', raising=False)
    try:
        assert configure_native_precision() == {'float32_matmul': 'high', 'cuda_matmul_tf32': True, 'cudnn_tf32': True, 'cuda_bf16_reduced_precision_reduction': False}
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        monkeypatch.setenv('NATLANG_BF16_REDUCTION', 'reduced')
        assert configure_native_precision()['cuda_bf16_reduced_precision_reduction'] is True
        monkeypatch.setenv('NATLANG_BF16_REDUCTION', 'invalid')
        with pytest.raises(ValueError, match='NATLANG_BF16_REDUCTION'):
            configure_native_precision()
        monkeypatch.delenv('NATLANG_BF16_REDUCTION')
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        monkeypatch.setenv('NATLANG_STOCK_ARITHMETIC', 'fp64')
        with pytest.raises(ValueError):
            configure_native_precision()
    finally:
        monkeypatch.setenv('NATLANG_STOCK_ARITHMETIC', 'ieee')
        configure_native_precision()

# Written for one optimizer update per packed batch; accumulation is tested in test_native_update_accumulation.py.
pytestmark = pytest.mark.usefixtures('per_batch_native_updates')


def test_checkpoint_backed_base_restore_requires_exact_inventory():
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from natlang_neuralese.model.native_precision import base_restore_transition
    inventory = {'config.json': 'a' * 64, 'tokenizer.json': 'b' * 64, 'model.safetensors': 'c' * 64}
    digest = canonical_json_sha256_hex(inventory)
    old = {'base': '/base', 'base_inventory': inventory, 'base_inventory_sha256': digest}
    receipt = {'schema': 'natlang.native-config-only-checkpoint-restore/1',
               'mode': 'checkpoint-backbone-state+verified-config-tokenizer',
               'pinned_base_inventory_sha256': digest,
               'verified_local_files': {'config.json': 'a' * 64, 'tokenizer.json': 'b' * 64}}
    new = {**old, 'base_restore': receipt}
    assert same_source_precision_identity(old, new)
    assert base_restore_transition(old, new)['base_inventory_sha256'] == digest
    assert not same_source_precision_identity(old, {**new, 'base_restore': {**receipt, 'verified_local_files': {'config.json': 'z' * 64}}})
    assert not same_source_precision_identity(old, {**new, 'base_inventory_sha256': '0' * 64})
    assert not same_source_precision_identity(old, {**new, 'rank': 4})
