import pytest
import torch

from natlang_neuralese.model.native_gradient_partitions import GradientPartitions, gradient_partitions
from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope


@pytest.mark.parametrize('backend', ['cpu_file', 'device_fp32'])
@pytest.mark.parametrize('microbatches', [1, 2])
def test_ordered_dense_dwd_fusion_matches_each_original_kernel_add(backend, microbatches, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.maple.fused_moe import invariant_linear

    torch.manual_seed(1207)
    initial = torch.nn.Parameter(torch.randn((128, 96), device='cuda', dtype=torch.bfloat16))
    operands = []
    for frame in range(microbatches):
        frame_operands = []
        for invocation, rows in enumerate((17, 9, 23)):
            x = torch.randn((rows, 96), device='cuda', dtype=torch.bfloat16)
            dy = torch.randn((rows, 128), device='cuda', dtype=torch.bfloat16)
            frame_operands.append((x, dy, invocation))
        operands.append(frame_operands)

    def run(*, fused):
        weight = torch.nn.Parameter(initial.detach().clone())
        scope = ParameterCotangentScope([('weight', weight)], mode='update',
                                        accumulator_backend=backend,
                                        directory=tmp_path)
        if not fused:
            scope.defer_dense_linear_rows = lambda *_args: False
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    acc_events=True) as profile:
            with scope:
                outputs, input_values = [], []
                for frame, frame_operands in enumerate(operands):
                    if frame:
                        scope.new_backward()
                    loss = None
                    for x, dy, invocation in frame_operands:
                        x = x.detach().clone().requires_grad_(True)
                        input_values.append(x)
                        with gradient_partitions(GradientPartitions(
                                (x.shape[0],), invocation_index=invocation,
                                invocation_namespace=(frame,))):
                            output = invariant_linear(x, weight, forward_storage_dtype=torch.bfloat16,
                                                      cotangent_anchors=(scope.anchor_for(weight),))
                        part = (output * dy).sum()
                        outputs.append(output.detach().clone())
                        loss = part if loss is None else loss + part
                    loss.backward()
                scope.finish()
                result = weight.grad.detach().clone()
                count = scope.entries[id(weight)]['count']
                input_gradients = [item.grad.detach().clone() for item in input_values]
        kernel_counts = {event.key: event.count for event in profile.key_averages()
                         if event.key in {'_grouped_dense_mm', '_ordered_dense_weight_gradient'}}
        return result, count, outputs, input_gradients, kernel_counts

    actual = run(fused=True)
    expected = run(fused=False)
    assert torch.equal(actual[0], expected[0])
    assert actual[1] == expected[1] == 3 * microbatches
    assert all(torch.equal(left, right) for left, right in zip(actual[2], expected[2]))
    assert all(torch.equal(left, right) for left, right in zip(actual[3], expected[3]))
    assert expected[4].get('_grouped_dense_mm', 0) - actual[4].get('_grouped_dense_mm', 0) == 3 * microbatches
    assert actual[4].get('_ordered_dense_weight_gradient', 0) == microbatches


def test_ordered_dense_dwd_fusion_preserves_nonzero_prior_fp32_accumulator(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.maple.fused_moe import invariant_linear

    torch.manual_seed(991)
    expected = torch.nn.Parameter(torch.randn((64, 32), device='cuda', dtype=torch.bfloat16))
    actual = torch.nn.Parameter(expected.detach().clone())

    def run(weight, fused):
        scope = ParameterCotangentScope([('weight', weight)], mode='update',
                                        accumulator_backend='cpu_file', directory=tmp_path)
        if not fused:
            scope.defer_dense_linear_rows = lambda *_args: False
        with scope:
            for frame in range(2):
                if frame:
                    scope.new_backward()
                x = torch.randn((19, 32), generator=torch.Generator(device='cuda').manual_seed(88 + frame),
                                device='cuda', dtype=torch.bfloat16)
                dy = torch.randn((19, 64), generator=torch.Generator(device='cuda').manual_seed(98 + frame),
                                 device='cuda', dtype=torch.bfloat16)
                with gradient_partitions(GradientPartitions((19,), invocation_namespace=(frame,))):
                    out = invariant_linear(x, weight, forward_storage_dtype=torch.bfloat16,
                                           cotangent_anchors=(scope.anchor_for(weight),))
                (out * dy).sum().backward()
            scope.finish()
            return weight.grad.detach().clone()

    # Re-run both implementations from identical operands, but force a nonzero
    # accumulator before the two invocation records are fused in each backward.
    base = run(expected, fused=False)
    got = run(actual, fused=True)
    assert torch.equal(got, base)


@pytest.mark.parametrize('backend', ['cpu_file', 'device_fp32'])
def test_ordered_dense_dwd_fusion_preserves_returned_vjp(backend, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.maple.fused_moe import invariant_linear

    torch.manual_seed(992)
    initial = torch.nn.Parameter(torch.randn((64, 48), device='cuda', dtype=torch.bfloat16))
    operands = [(torch.randn((rows, 48), device='cuda', dtype=torch.bfloat16),
                 torch.randn((rows, 64), device='cuda', dtype=torch.bfloat16))
                for rows in (11, 17, 8)]

    def run(fused):
        weight = torch.nn.Parameter(initial.detach().clone())
        scope = ParameterCotangentScope([('weight', weight)], mode='vjp',
                                        accumulator_backend=backend, directory=tmp_path)
        if not fused:
            scope.defer_dense_linear_rows = lambda *_args: False
        with scope:
            loss = None
            for index, (x, dy) in enumerate(operands):
                with gradient_partitions(GradientPartitions((x.shape[0],), invocation_index=index,
                                                            invocation_namespace=(0,))):
                    output = invariant_linear(x, weight, forward_storage_dtype=torch.bfloat16,
                                              cotangent_anchors=(scope.anchor_for(weight),))
                part = (output * dy).sum()
                loss = part if loss is None else loss + part
            returned, = torch.autograd.grad(loss, weight)
            scope.finish()
            assert weight.grad is None
            return returned.detach().clone()

    assert torch.equal(run(True), run(False))


def test_ordered_dense_dwd_fusion_preserves_adapter_base_and_delta_vjps(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.maple.fused_moe import invariant_linear

    torch.manual_seed(997)
    initial_base = torch.nn.Parameter(torch.randn((48, 32), device='cuda', dtype=torch.bfloat16))
    initial_delta = torch.randn((48, 32), device='cuda', dtype=torch.float32, requires_grad=True)
    inputs = [(torch.randn((rows, 32), device='cuda', dtype=torch.bfloat16),
               torch.randn((rows, 48), device='cuda', dtype=torch.bfloat16))
              for rows in (11, 17, 8)]

    def run(fused):
        base = torch.nn.Parameter(initial_base.detach().clone())
        delta = initial_delta.detach().clone().requires_grad_(True)
        effective = base.float() + delta
        scope = ParameterCotangentScope([('base', base)], mode='vjp',
                                        accumulator_backend='cpu_file', directory=tmp_path,
                                        track_auxiliary=True)
        if not fused:
            scope.defer_dense_linear_rows = lambda *_args: False
        with scope:
            scope.register_effective_weight(effective, base, (delta,))
            anchors = scope.anchors_for_weight(effective)
            loss = None
            for index, (x, dy) in enumerate(inputs):
                with gradient_partitions(GradientPartitions((x.shape[0],), invocation_index=index,
                                                            invocation_namespace=(1,))):
                    output = invariant_linear(x, effective, forward_storage_dtype=torch.bfloat16,
                                              cotangent_anchors=anchors)
                part = (output * dy).sum()
                loss = part if loss is None else loss + part
            base_gradient, delta_gradient = torch.autograd.grad(loss, (base, delta))
            scope.finish()
            return base_gradient.detach().clone(), delta_gradient.detach().clone()

    actual = run(True)
    expected = run(False)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize('backend', ['cpu_file', 'device_fp32'])
def test_ordered_dense_dwd_fusion_preserves_disjoint_and_overlapping_view_vjps(backend, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.maple.fused_moe import invariant_linear

    torch.manual_seed(9971)
    initial = torch.nn.Parameter(torch.randn((32, 80), device='cuda', dtype=torch.bfloat16))
    calls = []
    for offset in (0, 16, 0):
        x = torch.randn((13, 64), device='cuda', dtype=torch.bfloat16)
        dy = torch.randn((13, 32), device='cuda', dtype=torch.bfloat16)
        calls.append((offset, x, dy))

    def run(fused):
        weight = torch.nn.Parameter(initial.detach().clone())
        scope = ParameterCotangentScope([('weight', weight)], mode='vjp',
                                        accumulator_backend=backend, directory=tmp_path)
        if not fused:
            scope.defer_dense_linear_rows = lambda *_args: False
        with scope:
            loss = None
            for invocation, (offset, x, dy) in enumerate(calls):
                view = weight[:, offset:offset + 64]
                with gradient_partitions(GradientPartitions((x.shape[0],), invocation_index=invocation,
                                                            invocation_namespace=(9,))):
                    output = invariant_linear(x, view, forward_storage_dtype=torch.bfloat16,
                                              cotangent_anchors=(scope.anchor_for(weight),))
                part = (output * dy).sum()
                loss = part if loss is None else loss + part
            (gradient,) = torch.autograd.grad(loss, weight)
            scope.finish()
            return gradient.detach().clone()

    assert torch.equal(run(True), run(False))


@pytest.mark.parametrize('backend', ['cpu_file', 'device_fp32'])
def test_ordered_dense_dwd_fusion_preserves_mixed_invocation_dtype_laws(backend, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.maple.fused_moe import invariant_linear

    torch.manual_seed(9972)
    initial = torch.nn.Parameter(torch.randn((48, 32), device='cuda', dtype=torch.bfloat16))
    calls = []
    for dtype in (torch.bfloat16, torch.float32, torch.bfloat16):
        x = torch.randn((15, 32), device='cuda', dtype=dtype)
        dy = torch.randn((15, 48), device='cuda', dtype=dtype)
        calls.append((x, dy, dtype))

    def run(fused):
        weight = torch.nn.Parameter(initial.detach().clone())
        scope = ParameterCotangentScope([('weight', weight)], mode='vjp',
                                        accumulator_backend=backend, directory=tmp_path)
        if not fused:
            scope.defer_dense_linear_rows = lambda *_args: False
        with scope:
            loss = None
            for invocation, (x, dy, dtype) in enumerate(calls):
                with gradient_partitions(GradientPartitions((x.shape[0],), invocation_index=invocation,
                                                            invocation_namespace=(10,))):
                    output = invariant_linear(x, weight,
                                              forward_storage_dtype=dtype,
                                              cotangent_anchors=(scope.anchor_for(weight),))
                part = (output * dy).sum()
                loss = part if loss is None else loss + part
            (gradient,) = torch.autograd.grad(loss, weight)
            scope.finish()
            return gradient.detach().clone()

    assert torch.equal(run(True), run(False))


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_ordered_dense_dwd_fusion_matches_shared_graph_training_update(family, monkeypatch):
    """Exercise the production VJP scope inside the real shared-graph trainer."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_shared_state import tiny, graph
    from natlang_neuralese.data.native_graph_schema import NativeGraphExample
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer, native_training_update

    torch.manual_seed(8403)
    cpu_model = SharedStateModel(tiny(family), (60, 61, 62, 63), rank=4)
    cpu_model.port.hf.to(dtype=torch.bfloat16)
    if family == 'mellum':
        cpu_model.port.hf.natlang_precision = 'bf16'
    cpu_model.port.checkpoint_layers = True
    cpu_model.port.ffn_chunk_tokens = 2
    initial_state = {name: value.detach().clone() for name, value in cpu_model.state_dict().items()}
    parameters = list(cpu_model.parameters())
    print('dense-dW CUDA preflight', {
        'family': family,
        'cpu_parameters': sum(parameter.numel() for parameter in parameters),
        'largest_parameter_numel': max(parameter.numel() for parameter in parameters),
        'largest_parameter_shape': max(parameters, key=lambda value: value.numel()).shape,
        'cuda_mem_info_bytes': torch.cuda.mem_get_info(),
        'cuda_allocated_bytes': torch.cuda.memory_allocated(),
        'cuda_reserved_bytes': torch.cuda.memory_reserved(),
        'cuda_fraction': torch.cuda.get_per_process_memory_fraction(),
    }, flush=True)
    model = cpu_model.to('cuda')
    actual_optimizer, _ = make_native_optimizer(model, lr=1e-4, weight_decay=0.)
    plan = graph()
    weights = [1.0] * plan.token_count
    for offset in plan.offsets:
        weights[offset] = 0.0
    example = NativeGraphExample('ordered-dw-shared-graph', 'train', ('fixture',), 'fixture-tokenizer',
                                 plan, tuple(weights), {})
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    actual_loss, _ = native_training_update(
        model, actual_optimizer, [example], checkpoint_layers=True, chunk_tokens=8)
    actual_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(initial_state)
    expected_optimizer, _ = make_native_optimizer(model, lr=1e-4, weight_decay=0.)
    original = ParameterCotangentScope.defer_dense_linear_rows
    monkeypatch.setattr(ParameterCotangentScope, 'defer_dense_linear_rows', lambda *_args: False)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    try:
        expected_loss, _ = native_training_update(
            model, expected_optimizer, [example], checkpoint_layers=True, chunk_tokens=8)
    finally:
        monkeypatch.setattr(ParameterCotangentScope, 'defer_dense_linear_rows', original)
    expected_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    assert torch.equal(actual_loss, expected_loss)
    for name, value in actual_state.items():
        assert torch.equal(value, expected_state[name]), name


@pytest.mark.parametrize('backend', ['cpu_file', 'device_fp32'])
def test_ordered_dense_dwd_fusion_preserves_partition_reduction_order(backend, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.maple.fused_moe import invariant_linear

    torch.manual_seed(2026)
    baseline = torch.nn.Parameter(torch.randn((32, 24), device='cuda', dtype=torch.bfloat16))
    fused = torch.nn.Parameter(baseline.detach().clone())
    x = torch.randn((39, 24), device='cuda', dtype=torch.bfloat16)
    dy = torch.randn((39, 32), device='cuda', dtype=torch.bfloat16)
    parts = GradientPartitions((11, 0, 13, 15), invocation_namespace=(71,))

    def run(weight, use_fusion):
        scope = ParameterCotangentScope([('weight', weight)], mode='update',
                                        accumulator_backend=backend, directory=tmp_path)
        if not use_fusion:
            scope.defer_dense_linear_rows = lambda *_args: False
        with scope:
            with gradient_partitions(parts):
                output = invariant_linear(x, weight, forward_storage_dtype=torch.bfloat16,
                                          cotangent_anchors=(scope.anchor_for(weight),))
            (output * dy).sum().backward()
            scope.finish()
            return weight.grad.detach().clone()

    assert torch.equal(run(fused, True), run(baseline, False))


def test_ordered_dense_dwd_fusion_reuses_checkpoint_captured_scope_and_partitions(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from torch.utils.checkpoint import checkpoint
    from natlang_neuralese.maple.fused_moe import invariant_linear
    from natlang_neuralese.model.native_gradient_partitions import capture_gradient_partitions
    from natlang_neuralese.model.native_weight_carrier import capture_cotangent_scope

    torch.manual_seed(91)
    start = torch.nn.Parameter(torch.randn((48, 32), device='cuda', dtype=torch.bfloat16))
    inputs = [torch.randn((13, 32), device='cuda', dtype=torch.bfloat16, requires_grad=True)
              for _ in range(2)]
    weights = torch.randn((13, 48), device='cuda', dtype=torch.bfloat16)

    def run(fused):
        weight = torch.nn.Parameter(start.detach().clone())
        scope = ParameterCotangentScope([('weight', weight)], mode='update',
                                        accumulator_backend='cpu_file', directory=tmp_path)
        if not fused:
            scope.defer_dense_linear_rows = lambda *_args: False
        with scope:
            outputs, input_values = [], []
            for frame, (source, dy) in enumerate(zip(inputs, weights)):
                value = source.detach().clone().requires_grad_(True)
                input_values.append(value)
                partitions = GradientPartitions((value.shape[0],), invocation_namespace=(frame,))
                anchor = scope.anchor_for(weight)
                def evaluate(tensor, anchor_input, frame_partitions=partitions):
                    with gradient_partitions(frame_partitions):
                        return invariant_linear(tensor, weight, forward_storage_dtype=torch.bfloat16,
                            cotangent_anchors=(anchor_input,))
                captured = capture_cotangent_scope(capture_gradient_partitions(evaluate))
                outputs.append(checkpoint(captured, value, anchor, use_reentrant=True))
                outputs[-1] = (outputs[-1] * dy).sum()
            sum(outputs).backward()
            scope.finish()
            return weight.grad.detach().clone(), [value.grad.detach().clone() for value in input_values]

    actual = run(True)
    expected = run(False)
    assert torch.equal(actual[0], expected[0])
    assert all(torch.equal(left, right) for left, right in zip(actual[1], expected[1]))


