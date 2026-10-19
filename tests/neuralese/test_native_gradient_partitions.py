"""Real midpoint regression; graph-forward integration is checked separately."""
import pytest
import torch
from natlang_neuralese.model.native_gradient_partitions import (
    GradientPartitions, active_gradient_partitions, gradient_partitions,
    capture_gradient_partitions, token_slices,
)
from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope, stored_linear


def test_partition_context_captures_checkpoint_and_restores_nested_context():
    parts = GradientPartitions((20, 0, 14))
    with gradient_partitions(parts):
        captured = capture_gradient_partitions(active_gradient_partitions)
    assert active_gradient_partitions() is None
    with gradient_partitions(GradientPartitions((1,))):
        assert captured() is parts
        assert active_gradient_partitions().lengths == (1,)
    assert [(s.start, s.stop) for s in token_slices(parts, 34)] == [(20, 34), (0, 20)]
    with pytest.raises(RuntimeError, match='provenance'):
        token_slices(parts, 33)


def test_actual_seed67_bf16_midpoint_matches_cached_delivered_gradient():
    torch.manual_seed(67)
    layer = torch.nn.Linear(96, 64, bias=False).to(torch.bfloat16)
    x = torch.randn(34, 96, dtype=torch.bfloat16, requires_grad=True)
    dy = (torch.randn(34, 64) * .01).to(torch.bfloat16)

    def run(*, split=False, partitions=None):
        with ParameterCotangentScope([('weight', layer.weight)], mode='vjp') as scope:
            with gradient_partitions(partitions):
                call = lambda z: stored_linear(z, layer.weight, storage_dtype=layer.weight.dtype)
                y = torch.cat([call(x[:20]), call(x[20:])]) if split else call(x)
            dx, dw = torch.autograd.grad((y * dy).sum(), (x, layer.weight))
            fp32 = scope.entries[id(layer.weight)]['accumulator'].clone()
            count = scope.entries[id(layer.weight)]['count']
            scope.finish()
        return y.detach(), dx, dw, fp32, count

    ordinary = run()
    cached = run(split=True)
    fixed = run(partitions=GradientPartitions((20, 14)))
    # This fixture must actually exercise the original delivered-gradient bug.
    assert not torch.equal(ordinary[2], cached[2])
    assert torch.equal(fixed[0], cached[0])
    assert torch.equal(fixed[1], cached[1])
    assert torch.equal(fixed[2], cached[2])
    assert torch.equal(fixed[3], cached[3])
    assert fixed[4] == cached[4] == 2
    assert layer.weight.grad is None


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_unpadded_expert_tile_pack_preserves_rows_across_4096_plus_3(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    """Plan padding is removed before adjacent chunks restore expert-local order."""
    from natlang_neuralese.maple.fused_moe import Plan, INVARIANT_BLOCK
    from natlang_neuralese.maple.fp8_experts import pack_grouped_expert_chunks

    torch.manual_seed(913)
    tokens, hidden, experts, top_k = 4099, 8, 3, 2
    ids = torch.arange(tokens, device=device)
    routes = torch.stack((ids.remainder(experts), (ids * 7 + 1).remainder(experts)), dim=1)
    activation = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    pair_gradient = torch.randn(tokens * top_k, 5, dtype=torch.bfloat16, device=device)

    def record(start, stop):
        plan = Plan(routes[start:stop], experts, stop - start, INVARIANT_BLOCK)
        padded_gradient = pair_gradient[start * top_k:stop * top_k].new_zeros(plan.size, 5)
        padded_gradient[plan.position] = pair_gradient[start * top_k:stop * top_k]
        chunk_activation = torch.cat((activation[start:stop], activation.new_zeros((1, hidden))), dim=0)
        return (padded_gradient, chunk_activation, plan.rows, plan.start, plan.count,
                plan.position.numel())

    full, first, tail = record(0, tokens), record(0, 4096), record(4096, tokens)
    expected_order = torch.argsort(routes.reshape(-1), stable=True)
    expected_ends = torch.searchsorted(routes.reshape(-1)[expected_order],
                                       torch.arange(experts, device=device), right=True).to(torch.int32)
    expected_gradient = pair_gradient[expected_order]
    expected_activation = activation[(expected_order // top_k)]
    packed_full = pack_grouped_expert_chunks([full], (experts, 5, hidden))
    packed_split = pack_grouped_expert_chunks([first, tail], (experts, 5, hidden))
    assert torch.equal(packed_full[0], expected_gradient)
    assert torch.equal(packed_split[0], expected_gradient)
    assert torch.equal(packed_full[1], expected_activation)
    assert torch.equal(packed_split[1], expected_activation)
    assert torch.equal(packed_full[2], expected_ends)
    assert torch.equal(packed_split[2], expected_ends)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_chunk_tile_reduction_preserves_exact_bf16_midpoint(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.maple.fp8_experts import (
        grouped_weight_gradient, grouped_weight_gradient_chunk_tiles)

    # The two products sum exactly to the midpoint between BF16 1.0 and its
    # successor. Preserve FP32 accumulation through the single final cast.
    records = []
    for activation_value in (1.0, 1.0 / 256.0):
        records.append({
            'gradient': torch.ones((1, 1), dtype=torch.bfloat16, device=device),
            'activation': torch.tensor([[activation_value]], dtype=torch.bfloat16, device=device),
            'row_indices': None,
            'starts': torch.tensor([0], dtype=torch.int32, device=device),
            'counts': torch.tensor([1], dtype=torch.int32, device=device),
            'pair_count': 1,
        })
    result = grouped_weight_gradient_chunk_tiles(records, (1, 1, 1))
    all_grad = torch.cat([item['gradient'] for item in records])
    all_activation = torch.cat([item['activation'] for item in records])
    baseline = grouped_weight_gradient(
        all_grad, all_activation, torch.tensor([2], dtype=torch.int32, device=device), (1, 1, 1))
    assert torch.equal(result, baseline)
    if device == 'cpu':
        assert torch.equal(result, torch.tensor([[[1.00390625]]], dtype=torch.float32))
        assert torch.equal(result.to(torch.bfloat16), torch.ones_like(result, dtype=torch.bfloat16))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_dense_grouped_expert_chunk_boundary_preserves_delivered_gradient(device):
    """The actual grouped projection path preserves dW when chunks reset the router."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.maple.fused_moe import dense_grouped_experts
    from natlang_neuralese.model.native_gradient_partitions import token_local_chunks

    torch.manual_seed(913)
    invocation_lengths = (4099, 39)
    tokens, hidden, experts, ff, top_k = sum(invocation_lengths), 8, 3, 8, 2
    ids = torch.arange(tokens, device=device)
    routes = torch.stack((ids.remainder(experts), (ids * 7 + 1).remainder(experts)), dim=1)
    combine = torch.full((tokens, top_k), 0.5, dtype=torch.float32, device=device)
    gate_up = torch.nn.Parameter((torch.randn(experts, 2 * ff, hidden, device=device) * 0.15).to(torch.bfloat16))
    down = torch.nn.Parameter((torch.randn(experts, hidden, ff, device=device) * 0.15).to(torch.bfloat16))
    inputs = (torch.randn(tokens, hidden, device=device) * 0.15).to(torch.bfloat16)
    cotangents = (torch.randn(tokens, hidden, device=device) * 0.1).to(torch.bfloat16)
    project = dense_grouped_experts

    def run(chunk_tokens, *, defer):
        x = inputs.detach().clone().requires_grad_()
        cursor = 0
        captured = {id(gate_up): [], id(down): []}
        def one_chunk(part):
            nonlocal cursor
            start, stop = cursor, cursor + part.shape[0]
            cursor = stop
            return project(part, routes[start:stop], combine[start:stop], gate_up, down, None)
        with ParameterCotangentScope([('experts.gate_up', gate_up), ('experts.down', down)], mode='vjp',
                accumulator_backend='device_fp32' if device == 'cuda' else 'cpu_file') as scope:
            if not defer:
                scope.defer_expert_chunk = lambda *args, **kwargs: False
            else:
                original_defer = scope.defer_expert_chunk
                def capture(weight, gradient, activation, row_indices, starts, counts,
                            pair_count, partitions, anchors):
                    captured[id(weight)].append((gradient.detach().clone(), activation.detach().clone(),
                        None if row_indices is None else row_indices.detach().clone(),
                        starts.detach().clone(), counts.detach().clone(), pair_count,
                        partitions.invocation_index, partitions.chunk_offset,
                        partitions.lengths[0]))
                    return original_defer(weight, gradient, activation, row_indices, starts,
                                          counts, pair_count, partitions, anchors)
                scope.defer_expert_chunk = capture
            with gradient_partitions(GradientPartitions(invocation_lengths)):
                output = token_local_chunks(x[None], chunk_tokens,
                    lambda part: one_chunk(part[0])[None]).squeeze(0)
            dx, dgate, ddown = torch.autograd.grad((output * cotangents).sum(),
                                                   (x, gate_up, down))
            acc_gate = scope.entries[id(gate_up)]['accumulator'].clone()
            acc_down = scope.entries[id(down)]['accumulator'].clone()
            scope.finish()
        return output.detach(), dx.detach(), dgate.detach(), ddown.detach(), acc_gate, acc_down, captured

    unchunked = run(0, defer=False)
    deferred_unchunked = run(0, defer=True)
    chunked = run(4096, defer=True)
    from natlang_neuralese.maple.fp8_experts import pack_grouped_expert_chunks
    for parameter in (gate_up, down):
        key = id(parameter)
        full_records = [record[:6] for record in deferred_unchunked[6][key]]
        split_records = [record[:6] for record in sorted(chunked[6][key], key=lambda item: item[7])]
        packed_full = pack_grouped_expert_chunks(full_records, parameter.shape)
        packed_split = pack_grouped_expert_chunks(split_records, parameter.shape)
        assert torch.equal(packed_full[1], packed_split[1]), 'routed activation row order changed'
        assert torch.equal(packed_full[2], packed_split[2]), 'expert count order changed'
        assert torch.equal(packed_full[0], packed_split[0]), (
            'per-route backward gradients changed', parameter.shape,
            float((packed_full[0].float() - packed_split[0].float()).abs().max()))
    for reference in (deferred_unchunked, chunked):
        for index in range(6):
            assert torch.equal(unchunked[index], reference[index]), (index, device)


@pytest.mark.parametrize('projection', ['grouped_linear', 'fp8_expert'])
def test_grouped_projection_custom_backward_flushes_chunk_anchor(projection):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if not hasattr(torch, '_grouped_mm'):
        pytest.skip('grouped GEMM unavailable')
    from natlang_neuralese.maple.fp8_experts import GroupedLinear
    from natlang_neuralese.maple.expert_gemm import BlockTable, Fp8ExpertProjection
    from natlang_neuralese.model.native_gradient_partitions import token_local_chunks

    torch.manual_seed(1047)
    device = 'cuda'
    invocation_lengths = (19, 5)
    tokens, experts, hidden, width = sum(invocation_lengths), 3, 8, 8
    route_ids = torch.arange(tokens, device=device)
    routes = (route_ids * 5 + 1).remainder(experts)
    inputs = (torch.randn(tokens, hidden, device=device) * .1).to(torch.bfloat16)
    weight = torch.nn.Parameter((torch.randn(experts, width, hidden, device=device) * .1).to(torch.bfloat16))

    def run(chunk_tokens, defer):
        x = inputs.detach().clone().requires_grad_()
        cursor = 0
        with ParameterCotangentScope([('experts.projection', weight)], mode='vjp',
                                     accumulator_backend='device_fp32') as scope:
            if not defer:
                scope.defer_expert_chunk = lambda *args, **kwargs: False
            def project(part):
                nonlocal cursor
                start, stop = cursor, cursor + part.shape[0]
                cursor = stop
                ids = routes[start:stop]
                order = torch.argsort(ids, stable=True)
                expert_ids = ids[order]
                rows = part[order]
                offsets = torch.searchsorted(
                    expert_ids, torch.arange(experts, device=device), right=True).to(torch.int32)
                if projection == 'grouped_linear':
                    result = GroupedLinear.apply(
                        rows, weight, offsets, False, *scope.anchors_for_weight(weight))
                else:
                    table = BlockTable(offsets, rows.shape[0])
                    result = Fp8ExpertProjection.apply(
                        rows, None, None, 1, weight, table, offsets, None,
                        *scope.anchors_for_weight(weight))
                return result[torch.argsort(order)]
            with gradient_partitions(GradientPartitions(invocation_lengths)):
                output = token_local_chunks(x[None], chunk_tokens,
                    lambda part: project(part[0])[None]).squeeze(0)
            dx, dweight = torch.autograd.grad(output.float().sum(), (x, weight))
            accumulated = scope.entries[id(weight)]['accumulator'].clone()
            scope.finish()
        return output.detach(), dx.detach(), dweight.detach(), accumulated

    reference = run(0, False)
    deferred_full = run(0, True)
    deferred_chunks = run(7, True)
    for candidate in (deferred_full, deferred_chunks):
        for expected, actual in zip(reference, candidate):
            assert torch.equal(expected, actual), (projection, expected.shape)


def test_grouped_linear_deferred_chunks_route_to_base_and_adapter_delta():
    if not torch.cuda.is_available() or not hasattr(torch, '_grouped_mm'):
        pytest.skip('CUDA grouped GEMM unavailable')
    from natlang_neuralese.maple.fp8_experts import GroupedLinear
    from natlang_neuralese.model.native_gradient_partitions import token_local_chunks

    torch.manual_seed(294)
    device = 'cuda'
    lengths = (19, 5)
    tokens, experts, hidden, width = sum(lengths), 3, 8, 8
    routes = ((torch.arange(tokens, device=device) * 5) + 1).remainder(experts)
    inputs = torch.randn(tokens, hidden, device=device).to(torch.bfloat16)
    base = torch.nn.Parameter((torch.randn(experts, width, hidden, device=device) * .1).to(torch.bfloat16))
    delta = torch.nn.Parameter((torch.randn(experts, width, hidden, device=device) * .01).to(torch.bfloat16))
    def run(chunk_tokens, deferred):
        x = inputs.detach().clone().requires_grad_()
        cursor = 0
        effective = (base.float() + delta.float()).to(torch.bfloat16)
        with ParameterCotangentScope([('experts.base', base)], mode='vjp',
                                     accumulator_backend='device_fp32') as scope:
            scope.register_effective_weight(effective, base, [delta])
            assert scope._effective_route_for(effective)[1] is not None
            if not deferred:
                scope.defer_expert_chunk = lambda *args, **kwargs: False
            def project(part):
                nonlocal cursor
                start, stop = cursor, cursor + part.shape[0]
                cursor = stop
                ids = routes[start:stop]
                order = torch.argsort(ids, stable=True)
                sorted_ids = ids[order]
                offsets = torch.searchsorted(
                    sorted_ids, torch.arange(experts, device=device), right=True).to(torch.int32)
                output = GroupedLinear.apply(
                    part[order], effective, offsets, False, *scope.anchors_for_weight(effective))
                return output[torch.argsort(order)]
            with gradient_partitions(GradientPartitions(lengths)):
                output = token_local_chunks(x[None], chunk_tokens,
                    lambda part: project(part[0])[None]).squeeze(0)
            dx, dbase, ddelta = torch.autograd.grad(output.float().square().sum(), (x, base, delta))
            accum_base = scope.entries[id(base)]['accumulator'].clone()
            accum_delta = scope._auxiliary_entries[id(delta)]['accumulator'].clone()
            scope.finish()
        return output.detach(), dx.detach(), dbase.detach(), ddelta.detach(), accum_base, accum_delta

    reference = run(0, False)
    full = run(0, True)
    chunks = run(7, True)
    for candidate in (full, chunks):
        for expected, actual in zip(reference, candidate):
            assert torch.equal(expected, actual), expected.shape


def test_deferred_expert_chunks_flush_before_next_microbatch_accumulation():
    from natlang_neuralese.maple.fused_moe import dense_grouped_experts
    from natlang_neuralese.model.native_gradient_partitions import token_local_chunks

    torch.manual_seed(331)
    device = 'cpu'
    lengths, experts, hidden, ff, top_k = (17, 11), 3, 8, 8, 2
    gate_up = torch.nn.Parameter(torch.randn(experts, 2 * ff, hidden).to(torch.bfloat16))
    down = torch.nn.Parameter(torch.randn(experts, hidden, ff).to(torch.bfloat16))
    xs = [torch.randn(length, hidden).to(torch.bfloat16) for length in lengths]
    cotangents = [torch.randn(length, hidden).to(torch.bfloat16) for length in lengths]
    routes = [torch.stack((torch.arange(length).remainder(experts),
                           (torch.arange(length) * 7 + 1).remainder(experts)), dim=1)
              for length in lengths]
    combine = [torch.full((length, top_k), .5) for length in lengths]

    def forward(x, route, weights, chunk_size, scope):
        cursor = 0
        def project(part):
            nonlocal cursor
            start, stop = cursor, cursor + part.shape[0]
            cursor = stop
            return dense_grouped_experts(
                part, route[start:stop], weights[start:stop], gate_up, down, None)
        with gradient_partitions(GradientPartitions((x.shape[0],))):
            return token_local_chunks(x[None], chunk_size,
                lambda part: project(part[0])[None]).squeeze(0)

    expected = []
    for x, route, mixing, cotangent in zip(xs, routes, combine, cotangents):
        with ParameterCotangentScope([('gate_up', gate_up), ('down', down)], mode='vjp') as scope:
            output = forward(x, route, mixing, 7, scope)
            torch.autograd.grad((output * cotangent).sum(), (gate_up, down))
            expected.append((scope.entries[id(gate_up)]['accumulator'].clone(),
                             scope.entries[id(down)]['accumulator'].clone()))

    with ParameterCotangentScope([('gate_up', gate_up), ('down', down)], mode='update') as scope:
        for index, (x, route, mixing, cotangent) in enumerate(zip(xs, routes, combine, cotangents)):
            output = forward(x.detach().clone().requires_grad_(), route, mixing, 7, scope)
            (output * cotangent).sum().backward()
            assert not scope._expert_chunk_buckets
            if index + 1 < len(xs):
                scope.new_backward()
        actual_gate = scope.entries[id(gate_up)]['accumulator'].clone()
        actual_down = scope.entries[id(down)]['accumulator'].clone()
        scope.finish()
    assert torch.equal(actual_gate, expected[0][0] + expected[1][0])
    assert torch.equal(actual_down, expected[0][1] + expected[1][1])
    assert torch.equal(gate_up.grad, actual_gate.to(gate_up.dtype))
    assert torch.equal(down.grad, actual_down.to(down.dtype))


@pytest.mark.parametrize('checkpointed', [False, True])
def test_shared_expert_forward_frames_survive_checkpoint_recompute(checkpointed):
    from torch.utils.checkpoint import checkpoint
    from natlang_neuralese.maple.fp8_experts import GroupedLinear
    from natlang_neuralese.model.native_gradient_partitions import (
        GradientPartitions, gradient_partitions, new_gradient_namespace, token_local_chunks)

    # torch._grouped_mm requires its matrix strides to be 16-byte aligned.
    # Every coordinate receives the same scalar product, keeping the midpoint
    # ordering witness while exercising the real grouped-linear backward.
    weight = torch.nn.Parameter(torch.zeros((1, 16, 16), dtype=torch.bfloat16))
    operands = ((1.0, 1.0), (2.0**-12, 2.0**-12), (2.0**-12, 2.0**-12))
    with ParameterCotangentScope([('tied_expert', weight)], mode='vjp',
                                 accumulator_backend='cpu_file') as scope:
        outputs = []
        frame_runs = []
        for invocation, (value, cotangent) in enumerate(operands):
            x = torch.full((1, 1, 16), value, dtype=torch.bfloat16, requires_grad=True)
            # The outer context models one graph forward frame. The actor's
            # invocation index is local to that frame; its namespace must stay
            # captured when checkpoint() later recomputes the first forward.
            frame = GradientPartitions((1,), invocation_namespace=new_gradient_namespace())
            with gradient_partitions(frame):
                provenance = GradientPartitions((1,), invocation_index=invocation,
                    invocation_namespace=frame.invocation_namespace+(0, invocation))
                runs = []

                def project(rows):
                    offsets = torch.tensor([rows.shape[0]], dtype=torch.int32)
                    return GroupedLinear.apply(rows, weight, offsets, False,
                                               *scope.anchors_for_weight(weight))

                def execute(current, captured_provenance=provenance, captured_runs=runs):
                    captured_runs.append(captured_provenance.invocation_namespace)
                    with gradient_partitions(captured_provenance):
                        return token_local_chunks(current, 1, lambda part: project(part[0])[None]).squeeze(0)

                output = checkpoint(execute, x, use_reentrant=False) if checkpointed and invocation == 0 else execute(x)
                frame_runs.append((provenance.invocation_namespace, runs))
            outputs.append(output)
        returned, = torch.autograd.grad(tuple(outputs), (weight,), grad_outputs=tuple(
            torch.full_like(output, cotangent) for output, (_, cotangent) in zip(outputs, operands)))
        actual = scope.entries[id(weight)]['accumulator'].clone()
        scope.finish()
    expected = torch.tensor(0.0, dtype=torch.float32)
    for value, cotangent in reversed(operands):
        expected = expected + torch.tensor(value, dtype=torch.float32) * torch.tensor(cotangent, dtype=torch.float32)
    expected_full = expected.expand_as(actual).clone()
    assert len({frame for frame, _runs in frame_runs}) == len(operands)
    for index, (frame, runs) in enumerate(frame_runs):
        assert runs and all(observed == frame for observed in runs)
        assert len(runs) >= (2 if checkpointed and index == 0 else 1)
    assert torch.equal(actual, expected_full)
    assert torch.equal(returned, expected_full.to(weight.dtype))


@pytest.mark.parametrize('checkpointed', [False, True])
def test_shared_expert_actor_groups_keep_frame_and_invocation_order(monkeypatch, checkpointed):
    from torch.utils.checkpoint import checkpoint
    from natlang_neuralese.maple.fp8_experts import GroupedLinear
    from natlang_neuralese.model import shared_state
    from natlang_neuralese.model.native_adapters import capture_native_checkpoint_execution
    from natlang_neuralese.model.native_gradient_partitions import (
        GradientPartitions, gradient_partitions, new_gradient_namespace, token_local_chunks)

    class Bank:
        def execute(self, _name, callback):
            return callback()

    weight = torch.nn.Parameter(torch.zeros((1, 16, 16), dtype=torch.bfloat16))
    def finish(_port, _layer, h, _out, _attention):
        def project(rows):
            offsets = torch.tensor([rows.shape[0]], dtype=torch.int32)
            return GroupedLinear.apply(rows, weight, offsets, False,
                                       *scope.anchors_for_weight(weight))
        return token_local_chunks(h, 1, lambda part: project(part[0])[None])
    monkeypatch.setattr(shared_state, '_finish', finish)
    bank = Bank()
    operands = ((1.0, 1.0), (2.0**-12, 2.0**-12), (2.0**-12, 2.0**-12))
    with ParameterCotangentScope([('tied_expert', weight)], mode='vjp',
                                 accumulator_backend='cpu_file') as scope:
        outputs = []
        frame_runs = []
        for frame_index, (value, cotangent) in enumerate(operands):
            x = torch.full((1, 2, 16), value, dtype=torch.bfloat16, requires_grad=True)
            groups = tuple((name, torch.tensor([row]), GradientPartitions((1,)), row)
                           for row, name in enumerate(('actor-a', 'actor-b')))
            frame = GradientPartitions((1, 1), invocation_namespace=new_gradient_namespace())
            runs = []

            def execute(current, captured_frame=frame, captured_groups=groups, captured_runs=runs):
                captured_runs.append(captured_frame.invocation_namespace)
                with gradient_partitions(captured_frame):
                    return shared_state._actor_finish(bank, captured_groups, None, None,
                        current, current, False, layer_index=0)

            with gradient_partitions(frame):
                captured = capture_native_checkpoint_execution(execute)
                output = checkpoint(captured, x, use_reentrant=False) if checkpointed and frame_index == 0 else captured(x)
            frame_runs.append((frame.invocation_namespace, runs))
            outputs.append(output)
        returned, = torch.autograd.grad(tuple(outputs), (weight,), grad_outputs=tuple(
            torch.full_like(output, cotangent) for output, (_, cotangent) in zip(outputs, operands)))
        actual = scope.entries[id(weight)]['accumulator'].clone()
        scope.finish()
    expected = torch.tensor(0.0, dtype=torch.float32)
    for value, cotangent in reversed(operands):
        for _actor_index in (1, 0):
            expected = expected + torch.tensor(value, dtype=torch.float32) * torch.tensor(cotangent, dtype=torch.float32)
    expected_full = expected.expand_as(actual).clone()
    assert len({frame for frame, _runs in frame_runs}) == len(operands)
    for index, (frame, runs) in enumerate(frame_runs):
        assert runs and all(observed == frame for observed in runs)
        assert len(runs) >= (2 if checkpointed and index == 0 else 1)
    assert torch.equal(actual, expected_full)
    assert torch.equal(returned, expected_full.to(weight.dtype))



@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('checkpointed', [False, True])
@pytest.mark.parametrize('ffn_chunk', [0, 2])
def test_branched_graph_complete_bf16_derivatives_against_cached_execution(
        family, device, checkpointed, ffn_chunk):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_shared_state import tiny, graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.eval.native_gradients import compare_complete_gradients
    import torch.nn.functional as F

    model = SharedStateModel(tiny(family), (58, 59, 60, 61), rank=4).to(device)
    model.port.hf.to(dtype=torch.bfloat16)
    if family == 'mellum':
        model.port.hf.natlang_precision = 'bf16'
    for parameter in model.port.parameters():
        parameter.requires_grad_(True)
    model.port.checkpoint_layers = checkpointed
    model.port.ffn_chunk_tokens = ffn_chunk
    with torch.no_grad():
        model.adapter.up.weight.normal_(std=.03)
    _, named = make_native_optimizer(model, lr=3e-6, in_backward=False)
    plan = graph()
    ids = plan.tensor([t for node in plan.invocations for t in node.token_ids], device)
    labels = ids.roll(-1)
    initial = model.embed(ids[None]).detach()
    outputs = []
    def make(reference):
        x = initial.clone().requires_grad_()
        scope = ParameterCotangentScope.for_model(model, mode='vjp')
        scope.__enter__()
        try:
            out = model(plan, embeddings=x, reference=reference)
            outputs.append(out['h_final'].detach())
            loss = F.cross_entropy(model.logits(out['h_final'])[0].float(), labels)
        except BaseException:
            scope.close()
            raise
        return loss, x, scope
    flat, x, scope = make(False)
    report = compare_complete_gradients(flat, None, named, x, None,
        cached_factory=lambda: make(True), flat_cotangent_scope=scope,
        atol=1e-5, rtol=1e-4)
    failures = {name: row for name, row in report['comparisons'].items() if not row['passed']}
    assert report['passed'], failures
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-4, rtol=1e-4)
    assert all(parameter.grad is None for _, parameter in named)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_graph_still_visits_each_physical_row_once_per_layer(monkeypatch, family):
    from test_shared_state import tiny, graph
    from natlang_neuralese.model import shared_state
    model = shared_state.SharedStateModel(tiny(family), (58, 59, 60, 61), rank=4)
    model.port.checkpoint_layers = False
    model.port.ffn_chunk_tokens = 2
    plan = graph()
    seen = []
    original = shared_state._actor_layer_inputs
    def observed(bank, groups, port, layer, h, attention):
        seen.append(h.shape[1])
        return original(bank, groups, port, layer, h, attention)
    def forbidden(*args, **kwargs):
        raise AssertionError('flat graph secretly called the cached graph oracle')
    monkeypatch.setattr(shared_state, '_actor_layer_inputs', observed)
    monkeypatch.setattr(shared_state, 'forward_reference', forbidden)
    out = model(plan)
    assert out['h_final'].shape[1] == plan.token_count
    assert seen == [plan.token_count] * model.port.num_layers


@pytest.mark.parametrize('device,backend', [('cpu','cpu_file'), ('cuda','device_fp32')])
def test_embedding_frequency_is_local_to_cached_invocation(device, backend):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.model.native_weight_carrier import stored_embedding
    weight = torch.nn.Parameter(torch.ones(9, 4, dtype=torch.bfloat16, device=device))
    ids = torch.tensor([[2, 2, 2, 2, 2]], device=device)
    def run(split):
        with ParameterCotangentScope([('weight',weight)], mode='vjp',
                accumulator_backend=backend) as scope:
            def call(selected):
                return stored_embedding(selected, weight, scale_grad_by_freq=True)
            if split:
                output = torch.cat([call(ids[:,:2]),call(ids[:,2:])],1)
            else:
                with gradient_partitions(GradientPartitions((2,3))):
                    output = call(ids)
            gradient, = torch.autograd.grad(output.sum(), (weight,))
            scope.finish()
        return gradient
    flat, cached = run(False), run(True)
    assert torch.equal(flat,cached)
    assert torch.equal(flat[2],torch.full_like(flat[2],2))



def test_first_order_checkpoint_context_crosses_worker_and_refuses_expired_scope():
    from concurrent.futures import ThreadPoolExecutor
    from natlang_neuralese.model.native_adapters import capture_native_checkpoint_execution
    from natlang_neuralese.model.native_weight_carrier import active_cotangent_scope
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    parts = GradientPartitions((1, 1))
    with ParameterCotangentScope([('weight',parameter)],mode='vjp') as scope:
        with gradient_partitions(parts):
            callback = capture_native_checkpoint_execution(
                lambda: (active_cotangent_scope(), active_gradient_partitions()))
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(active_cotangent_scope).result() is None
            assert pool.submit(callback).result() == (scope, parts)
            assert pool.submit(active_cotangent_scope).result() is None
        scope.finish()
    with pytest.raises(RuntimeError,match='outlived'):
        callback()
