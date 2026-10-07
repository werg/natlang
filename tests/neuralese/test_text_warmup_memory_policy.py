import pytest

from natlang_neuralese.train.memory_policy import (
    effective_cuda_free_bytes,
    plan_saved_activation_offload,
    text_warmup_update_geometry_bytes,
)
from natlang_neuralese.train.memory_estimator import geometry_bytes
from natlang_neuralese.train.memory_estimator import backbone_memory_layout
from natlang_neuralese.train.memory_estimator import AdaptiveGraphMemory
from natlang_neuralese.train.text_warmup import (
    _seed_warmup_memory_estimator,
    _warmup_update_floor_bytes,
)


def test_fitting_update_keeps_saved_activations_on_device():
    plan = plan_saved_activation_offload(
        predicted_update_increment_bytes=750,
        free_bytes=800,
        total_bytes=1000,
        eligible_saved_activation_bytes=1000,
    )
    # 5% reserve leaves 750 bytes; predicted update increment is only 750.
    assert plan.usable_free_bytes == 750
    assert plan.predicted_total_incremental_peak_bytes == 750
    assert plan.offload_budget_bytes == 0
    assert not plan.should_offload
    assert plan.predicted_fit


def test_over_budget_plan_requests_only_conservative_minimum():
    plan = plan_saved_activation_offload(
        predicted_update_increment_bytes=840,
        free_bytes=800,
        total_bytes=1000,
        eligible_saved_activation_bytes=400,
        assumed_gpu_bytes_freed_per_cpu_byte=0.5,
    )
    # Need 90 GPU bytes beyond the 750-byte usable budget; at 0.5 efficiency,
    # target exactly 180 CPU bytes for hooks to offload.
    assert plan.required_gpu_reduction_bytes == 90
    assert plan.offload_budget_bytes == 180
    assert plan.predicted_residual_overage_bytes == 0
    assert plan.should_offload and plan.predicted_fit


def test_ineligible_activation_cap_returns_explicit_preflight_refusal():
    plan = plan_saved_activation_offload(
        predicted_update_increment_bytes=840,
        free_bytes=800,
        total_bytes=1000,
        eligible_saved_activation_bytes=100,
        assumed_gpu_bytes_freed_per_cpu_byte=0.5,
    )
    assert plan.offload_budget_bytes == 100
    assert plan.predicted_residual_overage_bytes == 40
    assert not plan.predicted_fit


@pytest.mark.parametrize('kwargs', [
    {'predicted_update_increment_bytes': -1},
    {'free_bytes': 1001},
    {'headroom_fraction': 1.0},
    {'assumed_gpu_bytes_freed_per_cpu_byte': 0.0},
])
def test_invalid_resource_estimates_are_rejected(kwargs):
    args = dict(predicted_update_increment_bytes=1, free_bytes=10,
                total_bytes=20, eligible_saved_activation_bytes=4)
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        plan_saved_activation_offload(**args)


def test_text_warmup_geometry_uses_max_live_pass_overlap_not_pass_count_sum():
    full = dict(width=8, layers=4, intermediate=16, kv_width=8,
                dtype_bytes=4, checkpointed=True)
    shallow = dict(width=8, layers=2, intermediate=16, kv_width=4,
                   dtype_bytes=4, checkpointed=True)
    kwargs = dict(prefix_tokens=4, target_tokens=12, batch_size=2,
                  full_layout=full, shallow_layout=shallow, cutoff=2,
                  vocab_size=64, readout_chunk_tokens=4)
    one_pass = text_warmup_update_geometry_bytes(sequence_passes=1, **kwargs)
    two_passes = text_warmup_update_geometry_bytes(sequence_passes=2, **kwargs)
    three_passes = text_warmup_update_geometry_bytes(sequence_passes=3, **kwargs)
    assert two_passes > one_pass
    assert three_passes == two_passes
    assert three_passes == 2 * text_warmup_update_geometry_bytes(
        sequence_passes=3, **{**kwargs, 'batch_size': 1})


def test_effective_free_reclaims_torch_cache_and_keeps_external_usage():
    # cuda.mem_get_info reports 300 free; 500 cached bytes belong to the Torch
    # allocator and are reusable. Live allocations are not counted twice.
    assert effective_cuda_free_bytes(300, 1000, 600, 100) == 800
    # Live non-Torch allocations have already reduced device_free_bytes.
    assert effective_cuda_free_bytes(100, 1000, 100, 100) == 100


def test_update_floor_counts_active_gradients_and_only_lazy_optimizer_slots():
    import torch

    backbone = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    feedback = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    full = torch.optim.AdamW([backbone, feedback], lr=1e-3)
    # Active parameter gradients need 14 bytes. Both AdamW moments are FP32,
    # so the first optimizer step adds 56 bytes of lazy state.
    assert _warmup_update_floor_bytes(
        [('backbone.weight', backbone), ('heads.feedback.weight', feedback)],
        full, bootstrap=False) == 14 + 2 * (4 + 3) * 4
    # Projection-first updates freeze the backbone and only allocate active
    # feedback-head gradients/state.
    assert _warmup_update_floor_bytes(
        [('backbone.weight', backbone), ('heads.feedback.weight', feedback)],
        full, bootstrap=True) == 6 + 2 * 3 * 4
    for param in (backbone, feedback):
        full.state[param]['step'] = torch.tensor(1.)
        full.state[param]['exp_avg'] = torch.zeros_like(param, dtype=torch.float32)
        full.state[param]['exp_avg_sq'] = torch.zeros_like(param, dtype=torch.float32)
    assert _warmup_update_floor_bytes(
        [('backbone.weight', backbone), ('heads.feedback.weight', feedback)],
        full, bootstrap=False) == 14


def test_positive_budget_on_cpu_preserves_gradients_without_offloading():
    import torch
    from natlang_neuralese.train.memory import offload_attention_tensors

    source = torch.randn(128, 64, requires_grad=True)
    expected = torch.autograd.grad(torch.sin(source * 1.7).square().mean(), source)[0]
    source.grad = None
    with offload_attention_tensors(2**20, activations=True) as stats:
        torch.sin(source * 1.7).square().mean().backward()
    torch.testing.assert_close(source.grad, expected)
    assert stats['offloaded_bytes'] == 0


def test_text_warmup_geometry_tracks_actual_context_checkpoint_and_readout_chunk():
    full = dict(width=8, layers=4, intermediate=16, kv_width=8,
                dtype_bytes=4, checkpointed=True)
    shallow = dict(width=8, layers=2, intermediate=16, kv_width=4,
                   dtype_bytes=4, checkpointed=True)
    base = text_warmup_update_geometry_bytes(
        4, 8, 3, 1, full, shallow, cutoff=2, vocab_size=64,
        readout_chunk_tokens=4)
    longer = text_warmup_update_geometry_bytes(
        4, 16, 3, 1, full, shallow, cutoff=2, vocab_size=64,
        readout_chunk_tokens=4)
    plain = geometry_bytes(4 + 8 - 1, 0, target_tokens=4,
                           vocab_size=64, **full)
    assert longer > base > plain


@pytest.mark.parametrize('observed_prefix', [None, 1, 6])
def test_seed_memory_estimator_imports_geometry_and_skips_offloaded_peaks(tmp_path, observed_prefix):
    import json

    full = dict(width=8, layers=4, intermediate=16, kv_width=8,
                dtype_bytes=4, checkpointed=True)
    shallow = dict(width=8, layers=2, intermediate=16, kv_width=4,
                   dtype_bytes=4, checkpointed=True)
    rows = [
        {'positions': 8, 'batch': 1,
         'schedule': {'sequence_passes': 3, 'plateau_reached': True},
         'memory': {'start_allocated_bytes': 100, 'peak_allocated_bytes': 300}},
        {'positions': 8, 'batch': 1,
         'schedule': {'sequence_passes': 3, 'plateau_reached': True},
         'memory': {'start_allocated_bytes': 100, 'peak_allocated_bytes': 900,
                    'preflight': {'offload_budget_bytes': 50},
                    'offload': {'offloaded_tensors': 1}}},
    ]
    if observed_prefix is not None:
        rows[0]['memory']['preflight'] = {
            'context_tokens': observed_prefix + 8 - 1,
            'target_tokens': 8,
            'offload_budget_bytes': 0,
        }
    path = tmp_path / 'train.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    class EmptyOptimizer:
        param_groups = []
        state = {}

    estimator = AdaptiveGraphMemory()
    seeded = _seed_warmup_memory_estimator(
        estimator, path, prefix_tokens=4, full_layout=full,
        shallow_layout=shallow, cutoff=2, vocab_size=64,
        batch_size=1, named=[], optimizer=EmptyOptimizer())

    assert seeded == 1
    actual_prefix = 4 if observed_prefix is None else observed_prefix
    samples = estimator.samples[estimator.key(
        'text-warmup-complete-update-v1:batch1:passes3', actual_prefix + 7, 8)]
    assert samples == [200 / text_warmup_update_geometry_bytes(
        actual_prefix, 8, 3, 1, full, shallow, cutoff=2, vocab_size=64)]


def _fake_backbone(kind):
    from types import SimpleNamespace
    import torch

    if kind == 'lfm-dense':
        widths, attention = [8, 12, 16, 20], [True, False, True, True]
        layers = [SimpleNamespace(feed_forward=SimpleNamespace(
            w1=SimpleNamespace(out_features=64)),
            self_attn=SimpleNamespace(k_proj=SimpleNamespace(out_features=w)))
            for w in widths]
        config = SimpleNamespace(hidden_size=128)
    elif kind == 'qwen-dense':
        widths, attention = [4, 8, 12], [True, True, False]
        layers = [SimpleNamespace(mlp=SimpleNamespace(gate_proj=SimpleNamespace(
            out_features=80)),
            self_attn=SimpleNamespace(k_proj=SimpleNamespace(out_features=w)))
            for w in widths]
        config = SimpleNamespace(hidden_size=160)
    else:
        widths, attention = [16, 24, 32, 40], [True, True, False, True]
        layers = [SimpleNamespace(mlp=SimpleNamespace(gate_proj=SimpleNamespace(
            out_features=128)),
            self_attn=SimpleNamespace(k_proj=SimpleNamespace(out_features=w)))
            for w in widths]
        config = SimpleNamespace(hidden_size=256, num_experts=8,
                                 num_experts_per_tok=2, moe_intermediate_size=32)
    return SimpleNamespace(config=config, layers=layers, num_layers=len(layers),
        is_attention=lambda i: attention[i],
        embedding_weight=torch.empty(1, dtype=torch.bfloat16))


@pytest.mark.parametrize(('kind', 'expected_width', 'expected_intermediate', 'expected_full_kv', 'expected_shallow_kv'), [
    ('lfm-dense', 128, 64, 88, 16),
    ('qwen-dense', 160, 80, 24, 24),
    ('maple-moe', 256, 64, 160, 80),
])
def test_shared_backbone_memory_layouts(kind, expected_width, expected_intermediate,
                                       expected_full_kv, expected_shallow_kv):
    backbone = _fake_backbone(kind)
    full = backbone_memory_layout(backbone, checkpointed=True)
    shallow = backbone_memory_layout(backbone, depth=2, checkpointed=True,
                                     stage_group_size=3)
    empty = backbone_memory_layout(backbone, depth=0, checkpointed=False)

    assert (full['width'], full['intermediate'], full['layers']) == (
        expected_width, expected_intermediate, backbone.num_layers)
    assert full['kv_width'] == expected_full_kv
    assert (shallow['layers'], shallow['kv_width'], shallow['stage_group_size']) == (
        2, expected_shallow_kv, 3)
    assert empty['layers'] == empty['kv_width'] == 0
    assert empty['dtype_bytes'] == 2
