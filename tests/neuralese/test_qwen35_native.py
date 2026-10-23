"""CPU contract tests for Qwen3.5's hybrid native Neuralese state path."""
from dataclasses import replace

import torch
from torch import nn

from natlang_neuralese.model.hf_port import QwenPortBackbone
from natlang_neuralese.model.lfm2_port import ControlTokens, DeltaState, PortCache
from natlang_neuralese.model.native_graph_plan import ImportSpan, Invocation, compile_graph
from natlang_neuralese.model.native_model import SharedStateModel
from natlang_neuralese.model.shared_state import forward_graph, forward_reference, native_step, splice_block
from natlang_neuralese.model.shared_state import _layer_inputs, _rotate_pair
from natlang_neuralese.model.native_gated_delta import replay
from natlang_neuralese.model.state_block import NativeStateBlock


def tiny_port(conv_kernel=3):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_conv_kernel_dim=conv_kernel, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=['full_attention', 'linear_attention'], max_position_embeddings=64)
    text = Qwen3_5TextModel(config).eval()

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = text

    class Checkpoint(nn.Module):
        base_model_prefix = 'model'

        def __init__(self):
            super().__init__()
            self.model = Outer()
            self.lm_head = nn.Linear(32, 64, bias=False)
            self.config = Qwen3_5Config(text_config=config)

        def get_output_embeddings(self):
            return self.lm_head

    return QwenPortBackbone(Checkpoint(), ControlTokens(60, 61), fast=False, markers=False)


def test_qwen35_text_stack_is_not_registered_twice():
    port = tiny_port()
    text_owners = [name for name, module in port.named_modules(remove_duplicate=False)
                   if module is port.text]
    assert text_owners == ['hf.model.language_model']
    assert not any(name.startswith('text.') for name in port.state_dict())


def native_block(result, port):
    return NativeStateBlock(text='source', token_ids=result['token_ids'],
        tokenizer_fingerprint='1' * 64, checkpoint_sha256='2' * 64,
        embedding_weight_sha256='3' * 64, architecture='qwen3_5',
        embeddings=result['embeddings'], positions=result['positions'],
        layers=tuple(result['layers'])).validate()


def test_qwen35_gated_delta_payload_roundtrip_and_incremental_splice():
    torch.manual_seed(17)
    port = tiny_port()
    model = SharedStateModel(port, (60, 61, 62, 63), rank=4)
    ids = torch.tensor([[4, 5, 6]])
    produced = native_step(model, ids, producing=True)
    block = native_block(produced, port)
    assert tuple(layer['kind'] for layer in block.layers) == ('attention', 'gated_delta')
    rebuilt = NativeStateBlock.from_tensors(block.metadata(), block.tensors())
    assert rebuilt.metadata() == block.metadata()
    assert set(rebuilt.tensors()) == set(block.tensors())
    for name, tensor in block.tensors().items():
        assert torch.equal(rebuilt.tensors()[name], tensor)

    identity = {'checkpoint_sha256': '2' * 64, 'tokenizer_fingerprint': '1' * 64,
                'embedding_weight_sha256': '3' * 64}
    restored = splice_block(model, PortCache.empty(port.num_layers), rebuilt, **identity)
    for left, right in zip(produced['cache'].states, restored.states):
        if hasattr(left, 'k'):
            assert torch.equal(left.k, right.k) and torch.equal(left.v, right.v)
        else:
            assert torch.equal(left.conv, right.conv)
            assert torch.equal(left.recurrent, right.recurrent)
    suffix = torch.tensor([[8, 9]])
    direct = native_step(model, suffix, produced['cache'])
    imported = native_step(model, suffix, restored)
    assert torch.equal(direct['h_final'], imported['h_final'])


def test_qwen35_graph_replays_imported_transitions_per_branch_with_gradients():
    torch.manual_seed(23)
    port = tiny_port()
    plan = compile_graph((
        Invocation('source', (4, 5, 6, 7), producing=((0, 4),)),
        Invocation('branch_a', (8, 9), imports=(ImportSpan(0, 'source', 1, 4),)),
        Invocation('branch_b', (10, 11), imports=(ImportSpan(0, 'source', 0, 2),
            ImportSpan(1, 'source', 2, 4))),
    ))
    graph = forward_graph(port, plan)
    reference = forward_reference(port, plan)
    assert torch.allclose(graph['h_final'], reference['h_final'], atol=3e-6, rtol=3e-6)
    graph['h_final'][:, 4:].float().square().sum().backward()
    gradient = port.embedding_weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient[[4, 5, 6, 7]]).all()
    assert gradient[[4, 5, 6, 7]].abs().sum() > 0


def test_qwen35_external_delta_import_receives_consumer_gradients():
    torch.manual_seed(29)
    port = tiny_port()
    model = SharedStateModel(port, (60, 61, 62, 63), rank=4)
    source = native_block(native_step(model, torch.tensor([[4, 5, 6]]), producing=True), port)
    layers = []
    for layer in source.layers:
        copied = dict(layer)
        for name, value in layer.items():
            if name != 'kind':
                copied[name] = value.detach().clone().requires_grad_(True)
        layers.append(copied)
    source = replace(source, layers=tuple(layers)).validate()
    plan = compile_graph((Invocation('consumer', (8, 9),
        imports=(ImportSpan(0, 'stored_source', 0, 3),)),), external_sizes={'stored_source': 3})
    identity = {'checkpoint_sha256': '2' * 64, 'tokenizer_fingerprint': '1' * 64,
                'embedding_weight_sha256': '3' * 64, 'architecture': 'qwen3_5'}
    result = forward_graph(port, plan, external_blocks={'stored_source': source}, external_identity=identity)
    result['h_final'].float().square().sum().backward()
    delta = source.layers[1]
    assert all(delta[name].grad is not None for name in ('mixed_qkv', 'z', 'b', 'a'))
    assert sum(delta[name].grad.abs().sum() for name in ('mixed_qkv', 'z', 'b', 'a')) > 0


def test_qwen35_delta_import_occurrences_obey_per_edge_stop_gradient():
    torch.manual_seed(33)
    stopped_plan = compile_graph((
        Invocation('source', (4, 5, 6), producing=((0, 3),)),
        Invocation('consumer', (8, 9, 10), imports=(ImportSpan(0, 'source', 0, 3,
            stop_gradient=True),)),
    ))
    mixed_plan = compile_graph((
        Invocation('source', (4, 5, 6), producing=((0, 3),)),
        Invocation('consumer', (8, 9, 10), imports=(ImportSpan(0, 'source', 0, 3,
            stop_gradient=True), ImportSpan(1, 'source', 0, 3))),
    ))
    for run in (forward_graph, forward_reference):
        stopped_port = tiny_port()
        stopped = run(stopped_port, stopped_plan)
        stopped['h_final'][:, 3:].float().square().sum().backward()
        assert stopped_port.embedding_weight.grad[[4, 5, 6]].abs().sum() == 0

        mixed_port = tiny_port()
        mixed = run(mixed_port, mixed_plan)
        mixed['h_final'][:, 3:].float().square().sum().backward()
        assert mixed_port.embedding_weight.grad[[4, 5, 6]].abs().sum() > 0


def test_qwen35_invocation_actor_substitutions_cover_delta_recurrence(monkeypatch):
    from natlang_neuralese.model.native_adapters import _Operation
    import natlang_neuralese.model.native_actor_graph as actor_graph

    torch.manual_seed(35)
    port = tiny_port()
    model = SharedStateModel(port, (60, 61, 62, 63), rank=4)
    plan = compile_graph((
        Invocation('producer', (4, 5, 6), producing=((0, 3),)),
        Invocation('consumer', (8, 9), imports=(ImportSpan(0, 'producer', 0, 3),)),
    ))
    model_parameters = dict(model.named_parameters())
    attention_names = [next(name for name in model_parameters if name.endswith(suffix))
                       for suffix in ('layers.0.self_attn.q_norm.weight',
                                      'layers.0.self_attn.k_norm.weight')]
    delta_projection_names = [next(name for name in model_parameters
                                   if name.endswith('layers.1.linear_attn.' + suffix))
                              for suffix in ('in_proj_qkv.weight', 'in_proj_z.weight',
                                             'in_proj_b.weight', 'in_proj_a.weight')]
    delta_recurrent_names = [next(name for name in model_parameters
                                  if name.endswith('layers.1.linear_attn.' + suffix))
                             for suffix in ('A_log', 'conv1d.weight', 'norm.weight', 'out_proj.weight')]
    actor_values = {
        'producer': {name: (model_parameters[name].detach() + 0.01 * (index + 1)).requires_grad_()
                     for index, name in enumerate(attention_names + delta_projection_names)},
        'consumer': {name: (model_parameters[name].detach() + 0.025 * (index + 1)).requires_grad_()
                     for index, name in enumerate(attention_names + delta_projection_names + delta_recurrent_names)},
    }

    class ActorBank:
        def __init__(self, values):
            self.values = values
            self.names = []

        def row_groups(self, _plan, _device):
            return []

        def execute(self, name, callback):
            self.names.append(name)
            return torch.func.functional_call(_Operation(model, callback),
                {'model.' + key: value for key, value in self.values[name].items()}, ())

        def receipt(self):
            return {'schema': 'test.qwen35.actor-bank/1'}

    for run in (forward_graph, forward_reference):
        bank = ActorBank(actor_values)
        monkeypatch.setattr(actor_graph, 'invocation_actor_bank', lambda _port, bank=bank: bank)
        token_ids = plan.tensor([token for node in plan.invocations for token in node.token_ids],
                                port.embedding_weight.device)
        embeddings = port.embed(token_ids[None]).detach().requires_grad_()
        result = run(port, plan, embeddings=embeddings)
        result['h_final'][:, 3:].float().square().sum().backward()
        assert set(bank.names) == {'producer', 'consumer'}
        assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
        assert embeddings.grad[:, :3].abs().sum() > 0
        zero_gradients = [f'{actor}:{name}' for actor, values in actor_values.items()
                          for name, value in values.items()
                          if value.grad is None or value.grad.abs().sum() == 0]
        producer_z = next(name for name in actor_values['producer'] if name.endswith('in_proj_z.weight'))
        assert zero_gradients == [f'producer:{producer_z}'], zero_gradients
        for actor in actor_values.values():
            for value in actor.values():
                value.grad = None


def test_qwen35_bf16_delta_parameters_use_shared_vjp_carriers():
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope

    torch.manual_seed(39)
    port = tiny_port()
    port.hf.to(dtype=torch.bfloat16)
    for parameter in port.hf.parameters():
        parameter.requires_grad_(False)
    delta = port.layers[1].linear_attn
    trainable = (delta.in_proj_qkv.weight, delta.conv1d.weight, delta.norm.weight,
                 delta.out_proj.weight, delta.A_log, delta.dt_bias,
                 port.layers[1].input_layernorm.weight)
    for parameter in trainable:
        parameter.requires_grad_(True)
    plan = compile_graph((
        Invocation('producer', (4, 5, 6), producing=((0, 3),)),
        Invocation('consumer', (8, 9, 10), imports=(ImportSpan(0, 'producer', 0, 3),)),
    ))
    with ParameterCotangentScope.for_model(SharedStateModel(port, (60, 61, 62, 63), rank=4), mode='vjp') as scope:
        result = forward_graph(port, plan)
        loss = result['h_final'][:, 3:].float().square().sum()
        gradients = torch.autograd.grad(loss, trainable)
        scope.finish()
    assert all(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
               for gradient in gradients)


def test_qwen35_bf16_qk_and_delta_scalar_vjps_match_fp32_reference():
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope

    torch.manual_seed(40)
    port = tiny_port()
    port.hf.to(dtype=torch.bfloat16)
    original = {name: value.detach().clone() for name, value in port.hf.state_dict().items()}
    for parameter in port.hf.parameters():
        parameter.requires_grad_(False)
    targets = (port.layers[0].self_attn.q_norm.weight, port.layers[0].self_attn.k_norm.weight,
               port.layers[1].linear_attn.A_log, port.layers[1].linear_attn.dt_bias)
    for parameter in targets:
        parameter.requires_grad_(True)
    plan = compile_graph((
        Invocation('producer', (4, 5, 6), producing=((0, 3),)),
        Invocation('consumer', (8, 9), imports=(ImportSpan(0, 'producer', 0, 3),)),
    ))
    model = SharedStateModel(port, (60, 61, 62, 63), rank=4)
    qnorm_contributions = []
    with ParameterCotangentScope.for_model(model, mode='vjp') as scope:
        add = scope.add
        def record_add(parameter, contribution):
            if parameter is targets[0]:
                qnorm_contributions.append(contribution.detach().float().clone())
            return add(parameter, contribution)
        scope.add = record_add
        actual = forward_graph(port, plan)
        loss = actual['h_final'][:, 3:].float().square().sum()
        actual_gradients = torch.autograd.grad(loss, targets)
        scope.finish()

    reference_port = tiny_port()
    reference_port.hf.load_state_dict(original)
    reference_port.hf.to(dtype=torch.bfloat16)
    reference_targets = (reference_port.layers[0].self_attn.q_norm.weight,
        reference_port.layers[0].self_attn.k_norm.weight,
        reference_port.layers[1].linear_attn.A_log, reference_port.layers[1].linear_attn.dt_bias)
    for parameter in reference_port.hf.parameters():
        parameter.requires_grad_(False)
    for parameter in reference_targets:
        parameter.data = parameter.detach().float()
        parameter.requires_grad_(True)
    expected = forward_graph(reference_port, plan)
    reference_loss = expected['h_final'][:, 3:].float().square().sum()
    expected_gradients = torch.autograd.grad(reference_loss, reference_targets)
    torch.testing.assert_close(actual['h_final'], expected['h_final'], atol=2e-3, rtol=2e-3)
    names = ('q_norm', 'k_norm', 'A_log', 'dt_bias')
    for name, target, actual_gradient, expected_gradient in zip(names, targets, actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient.float(), expected_gradient.float(), atol=2e-4, rtol=2e-3,
                                   msg=lambda message: f'{name}: {message}\nactual={actual_gradient.float()}\nexpected={expected_gradient.float()}\n'
                                   f'calls={len(qnorm_contributions)} contributions={[value.tolist() for value in qnorm_contributions]}')


def test_qwen35_rms_norm_raw_dw_and_repeated_call_fanin_match_source():
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope
    from natlang_neuralese.model.native_gradient_partitions import GradientPartitions, gradient_partitions
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    torch.manual_seed(41)
    for heads in (1, 2):
        stored = Qwen3_5RMSNorm(8).to(dtype=torch.bfloat16)
        stored.weight.data.copy_(torch.randn(8).bfloat16())
        x = torch.randn(1, 3, heads, 8).bfloat16()
        probe = torch.randn_like(x)
        source_module = Qwen3_5RMSNorm(8)
        source_module.weight = torch.nn.Parameter(stored.weight.detach().float())
        source_loss = (source_module(x) * probe).sum()
        source_dw, = torch.autograd.grad(source_loss, (source_module.weight,))
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + stored.eps)
        hand_dw = (probe.float() * normalized).sum(dim=tuple(range(probe.ndim - 1)))
        torch.testing.assert_close(hand_dw, source_dw, atol=2e-5, rtol=2e-4)

        container = nn.Module()
        container.norm = stored
        with ParameterCotangentScope.for_model(container, mode='vjp') as scope:
            outputs = [container.norm(x), container.norm(x.flip(1))]
            loss = (outputs[0] * probe).sum() + (outputs[1] * probe.flip(1)).sum()
            actual_dw, = torch.autograd.grad(loss, (stored.weight,))
            scope.finish()

        ref_module = Qwen3_5RMSNorm(8)
        ref_module.weight = torch.nn.Parameter(stored.weight.detach().float())
        reference_loss = ((ref_module(x) * probe).sum() +
                          (ref_module(x.flip(1)) * probe.flip(1)).sum())
        expected_dw, = torch.autograd.grad(reference_loss, (ref_module.weight,))
        torch.testing.assert_close(actual_dw.float(), expected_dw.bfloat16().float(), atol=0, rtol=0,
                                   msg=f'{heads} heads, two repeated calls; actual={actual_dw.float()} expected={expected_dw.float()}')

        partitioned = Qwen3_5RMSNorm(8).to(dtype=torch.bfloat16)
        partitioned.weight.data.copy_(stored.weight.detach())
        container = nn.Module()
        container.norm = partitioned
        xp = torch.randn(1, 5, heads, 8).bfloat16()
        pp = torch.randn_like(xp)
        with ParameterCotangentScope.for_model(container, mode='vjp'):
            with gradient_partitions(GradientPartitions((3, 2))):
                output = container.norm(xp)
            partitioned_gradient, = torch.autograd.grad((output * pp).sum(), partitioned.weight)
        partition_reference = Qwen3_5RMSNorm(8)
        partition_reference.weight = torch.nn.Parameter(partitioned.weight.detach().float())
        partition_expected, = torch.autograd.grad((partition_reference(xp) * pp).sum(),
                                                   (partition_reference.weight,))
        torch.testing.assert_close(partitioned_gradient.float(), partition_expected.bfloat16().float(),
                                   atol=0, rtol=0, msg=f'{heads} heads, 3+2 graph token partitions')


def test_qwen35_checkpointed_graph_recomputes_delta_gradients():
    torch.manual_seed(37)
    plan = compile_graph((Invocation('source', (4, 5, 6), producing=((0, 3),)),
                          Invocation('consumer', (8, 9), imports=(ImportSpan(0, 'source', 0, 3),))))
    baseline = tiny_port()
    checkpointed = tiny_port()
    checkpointed.hf.load_state_dict(baseline.hf.state_dict())
    checkpointed.checkpoint_layers = True
    expected = forward_graph(baseline, plan)['h_final']
    actual = forward_graph(checkpointed, plan)['h_final']
    assert torch.allclose(actual, expected, atol=3e-6, rtol=3e-6)
    actual[:, -2:].float().square().sum().backward()
    gradient = checkpointed.embedding_weight.grad
    assert gradient is not None and gradient[[4, 5, 6]].abs().sum() > 0


def test_qwen35_generic_sft_and_rl_objectives_train_importing_graph(tmp_path):
    import json
    from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
    from natlang_neuralese.rl.native_replay import NativePolicyObjective, graph_identity
    from natlang_neuralese.train.native_state_sft import packed_graph_loss

    torch.manual_seed(41)
    raw = {'schema': SCHEMA, 'id': 'qwen35-sft-rl', 'split': 'train',
        'source_groups': ['qwen35-cpu-contract'], 'tokenizer_fingerprint': '1' * 64,
        'nodes': [
            {'name': 'producer', 'token_ids': [4, 5, 6], 'producing': [[0, 3]],
             'target_weights': [0., 0., 0.]},
            {'name': 'consumer', 'token_ids': [8, 9, 10, 11], 'target_weights': [0., 1., 0., 1.],
             'imports': [{'at': 2, 'source': 'producer', 'start': 0, 'stop': 3}]}]}
    example = parse_graph(raw, tokenizer_fingerprint='1' * 64)
    identity = {'checkpoint_sha256': '2' * 64, 'tokenizer_fingerprint': '1' * 64,
        'embedding_weight_sha256': '3' * 64, 'architecture': 'qwen3_5'}

    sft_port = tiny_port()
    sft = SharedStateModel(sft_port, (60, 61, 62, 63), rank=4)
    ce, _ = packed_graph_loss(sft, [example], checkpoint_layers=True, chunk_tokens=2)
    ce.backward()
    assert torch.isfinite(ce) and sft_port.embedding_weight.grad[[4, 5, 6]].abs().sum() > 0

    rl_port = tiny_port()
    rl = SharedStateModel(rl_port, (60, 61, 62, 63), rank=4)
    objective_row = {'graph_sha256': None,
        'episode_turns': ['producer', 'consumer'], 'advantage': 0.,
        'actions': [{'query': 5, 'token_id': 11, 'temperature': 1., 'allowed_token_ids': None,
            'banned_token_ids': None, 'behavior_logp': 0.}]}
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    objective_row['graph_sha256'] = canonical_json_sha256_hex(graph_identity(example))
    spec = {'schema': 'natlang.native-policy-objective/1', 'kind': 'native-policy', 'mode': 'rft',
        'policy': identity, 'rows': {example.id: objective_row}, 'block_bundle': None}
    spec_path = tmp_path / 'objective.json'
    spec_path.write_text(json.dumps(spec))
    objective = NativePolicyObjective(spec_path)
    loss, details = packed_graph_loss(rl, [example], checkpoint_layers=True,
        chunk_tokens=2, objective=objective)
    loss.backward()
    assert details['sampled_tokens'] == 1 and torch.isfinite(loss)
    assert rl_port.embedding_weight.grad[[4, 5, 6]].abs().sum() > 0


def test_qwen35_native_optimizer_updates_nested_text_layer_from_import_graph():
    from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
    from natlang_neuralese.train.native_state_sft import make_native_optimizer, packed_graph_loss

    torch.manual_seed(42)
    port = tiny_port()
    model = SharedStateModel(port, (60, 61, 62, 63), rank=4)
    optimizer, named = make_native_optimizer(model, lr=1e-3, weight_decay=0., in_backward=False)
    names = {name for name, _ in named}
    assert names and any(name.startswith('backbone.model.language_model.layers.') for name in names)
    assert not any('.visual.' in name for name in names)
    assert any(name.endswith('layers.1.linear_attn.in_proj_qkv.weight') for name in names)
    assert all(not parameter.requires_grad for name, parameter in port.hf.named_parameters()
               if name.startswith('model.visual.'))

    raw = {'schema': SCHEMA, 'id': 'qwen35-real-optimizer', 'split': 'train',
        'source_groups': ['qwen35-optimizer-contract'], 'tokenizer_fingerprint': '1' * 64,
        'nodes': [
            {'name': 'producer', 'token_ids': [4, 5, 6], 'producing': [[0, 3]],
             'target_weights': [0., 0., 0.]},
            {'name': 'consumer', 'token_ids': [8, 9, 10, 11], 'target_weights': [0., 1., 0., 1.],
             'imports': [{'at': 2, 'source': 'producer', 'start': 0, 'stop': 3}]}]}
    example = parse_graph(raw, tokenizer_fingerprint='1' * 64)
    target = port.layers[1].linear_attn.in_proj_qkv.weight
    before = target.detach().clone()
    loss, _ = packed_graph_loss(model, [example], checkpoint_layers=False)
    loss.backward()
    assert target.grad is not None and torch.isfinite(target.grad).all() and target.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, target)


def test_qwen35_default_optimizer_updates_delta_controls_at_training_learning_rate():
    from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
    from natlang_neuralese.model.native_model import model_state, restore_model_state
    from natlang_neuralese.model.native_parameter_precision import (
        QWEN35_DELTA_PROFILE, configure_norm_precision)
    from natlang_neuralese.train.native_state_sft import make_native_optimizer, packed_graph_loss

    torch.manual_seed(43)
    port = tiny_port()
    port.hf.to(dtype=torch.bfloat16)
    model = SharedStateModel(port, (60, 61, 62, 63), rank=4)
    configure_norm_precision(model, QWEN35_DELTA_PROFILE)
    optimizer, _ = make_native_optimizer(model, lr=3e-6, weight_decay=.1, in_backward=False)
    delta = port.layers[1].linear_attn
    controls = (delta.A_log, delta.dt_bias)
    before = tuple(parameter.detach().clone() for parameter in controls)
    assert all(parameter.dtype == torch.float32 for parameter in controls)

    raw = {'schema': SCHEMA, 'id': 'qwen35-delta-controls-lr', 'split': 'train',
        'source_groups': ['qwen35-delta-control-update'], 'tokenizer_fingerprint': '1' * 64,
        'nodes': [
            {'name': 'producer', 'token_ids': [4, 5, 6], 'producing': [[0, 3]],
             'target_weights': [0., 0., 0.]},
            {'name': 'consumer', 'token_ids': [8, 9, 10, 11], 'target_weights': [0., 1., 0., 1.],
             'imports': [{'at': 2, 'source': 'producer', 'start': 0, 'stop': 3}]}]}
    example = parse_graph(raw, tokenizer_fingerprint='1' * 64)
    loss, _ = packed_graph_loss(model, [example], checkpoint_layers=False)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in controls)
    optimizer.step()
    updates = tuple({'changed': not torch.equal(parameter, old),
        'gradient': parameter.grad.detach().float().tolist(),
        'delta': (parameter.detach().float() - old.float()).tolist()}
        for parameter, old in zip(controls, before))
    assert all(item['changed'] for item in updates), updates

    saved = model_state(model)
    saved_optimizer = optimizer.state_dict()
    restored_port = tiny_port()
    restored_port.hf.to(dtype=torch.bfloat16)
    restored = SharedStateModel(restored_port, (60, 61, 62, 63), rank=4)
    configure_norm_precision(restored, QWEN35_DELTA_PROFILE)
    restore_model_state(restored, saved)
    restored_optimizer, _ = make_native_optimizer(restored, lr=3e-6, weight_decay=.1, in_backward=False)
    restored_optimizer.load_state_dict(saved_optimizer)
    restored_controls = (restored_port.layers[1].linear_attn.A_log,
                         restored_port.layers[1].linear_attn.dt_bias)
    assert all(torch.equal(parameter, old) for parameter, old in zip(restored_controls, controls))
    assert all(torch.equal(restored_optimizer.auxiliary.state[parameter]['exp_avg'],
                           optimizer.auxiliary.state[source]['exp_avg'])
               for parameter, source in zip(restored_controls, controls))


def test_qwen35_raw_policy_precision_inventory_rejects_bf16_delta_controls():
    from types import SimpleNamespace
    import pytest
    from natlang_neuralese.model.native_parameter_precision import (
        QWEN35_DELTA_PROFILE, ordinary_precision_state_dtypes, validate_norm_precision)

    port = tiny_port()
    state = port.hf.state_dict()
    controls = [name for name in state if name.endswith(('.linear_attn.A_log', '.linear_attn.dt_bias'))]
    assert len(controls) == 2
    proxy = SimpleNamespace(port=SimpleNamespace(architecture='qwen3_5'),
        native_ordinary_precision=QWEN35_DELTA_PROFILE,
        _native_ordinary_parameter_dtypes=ordinary_precision_state_dtypes(state, QWEN35_DELTA_PROFILE))
    config = {'native_ordinary_precision': QWEN35_DELTA_PROFILE}
    validate_norm_precision(proxy, config)
    bad_state = dict(state)
    bad_state[controls[0]] = bad_state[controls[0]].bfloat16()
    proxy._native_ordinary_parameter_dtypes = ordinary_precision_state_dtypes(
        bad_state, QWEN35_DELTA_PROFILE)
    with pytest.raises(ValueError, match='storage'):
        validate_norm_precision(proxy, config)


def test_qwen35_lora_injection_covers_nested_hybrid_layers():
    from natlang_neuralese.train.adapters import (
        adapter_layers, inject_lora, lora_state, lora_target_names, restore_lora_checkpoint,
    )
    from natlang_neuralese.train.warmup_export import _adapter_layer_indices

    port = tiny_port()
    targets = lora_target_names(port, [0, 1])
    assert 'model.language_model.layers.0.self_attn.q_proj' in targets
    assert 'model.language_model.layers.0.self_attn.o_proj' in targets
    assert 'model.language_model.layers.0.self_attn.out_proj' not in targets
    assert 'model.language_model.layers.1.linear_attn.in_proj_qkv' in targets
    assert 'model.language_model.layers.1.mlp.gate_proj' in targets
    inject_lora(port, [0, 1], rank=2, alpha=4)
    assert adapter_layers(port) == [0, 1]
    saved = lora_state(port)
    assert any('model.language_model.layers.1.linear_attn.in_proj_qkv.lora_A' in name for name in saved)
    assert not any('.visual.' in name for name in saved)
    assert _adapter_layer_indices(saved) == [0, 1]

    checkpoint = {'lora': saved, 'lora_layers': [0, 1], 'lora_rank': 2}
    restored_port = tiny_port()
    receipt = restore_lora_checkpoint(restored_port, checkpoint, alpha=4)
    assert receipt == {'layers': [0, 1], 'rank': 2, 'tensor_count': len(saved)}
    for name, value in saved.items():
        assert torch.equal(lora_state(restored_port)[name], value)

    # The native port calls the same HF modules, so adapted projections must be
    # visible on its real forward path and expose a gradient to the adapter.
    restored_port.hf.zero_grad(set_to_none=True)
    output = restored_port.forward_ids(torch.tensor([[4, 5, 6]]), logits=False)['h_final']
    output.square().mean().backward()
    exposed = [parameter for name, parameter in restored_port.hf.named_parameters()
               if '.lora_' in name and parameter.requires_grad]
    assert exposed and any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                           for parameter in exposed)


def test_qwen35_batched_prefill_incremental_cache_and_ragged_padding():
    from natlang_neuralese.serve.batch import split_rows

    torch.manual_seed(43)
    port = tiny_port()
    ids = torch.tensor([[4, 5, 6], [8, 9, 10]])
    batched_prefill = port.forward_ids(ids, logits=False)
    output, batch_cache = batched_prefill['h_final'], batched_prefill['cache']
    for row in range(2):
        single = port.forward_ids(ids[row:row + 1], logits=False)
        torch.testing.assert_close(output[row:row + 1], single['h_final'], atol=3e-6, rtol=3e-6)
        for actual, expected in zip(batch_cache.states, single['cache'].states):
            if isinstance(actual, DeltaState):
                torch.testing.assert_close(actual.conv[row:row + 1], expected.conv)
                torch.testing.assert_close(actual.recurrent[row:row + 1], expected.recurrent)
            else:
                torch.testing.assert_close(actual.k[row:row + 1], expected.k)
                torch.testing.assert_close(actual.v[row:row + 1], expected.v)

    selected = batch_cache.select([1, 0])
    assert selected.states[1].recurrent.shape[0] == 2
    repeated = batch_cache.select([0]).repeat_interleave(2)
    torch.testing.assert_close(repeated.states[1].recurrent[0], repeated.states[1].recurrent[1])
    next_ids = torch.tensor([[7], [11]])
    next_batched = port.forward_ids(next_ids, cache=batch_cache, logits=False)
    for row in range(2):
        prior = port.forward_ids(ids[row:row + 1], logits=False)['cache']
        expected = port.forward_ids(next_ids[row:row + 1], cache=prior, logits=False)
        torch.testing.assert_close(next_batched['h_final'][row:row + 1], expected['h_final'],
                                   atol=3e-6, rtol=3e-6)

    padded_ids = torch.tensor([[0, 4, 5], [8, 9, 10]])
    pad = torch.tensor([1, 0])
    padded = port.forward_ids(padded_ids, left_pad=pad, logits=False)
    split = split_rows(padded['cache'], pad)
    for row, real_ids in enumerate((torch.tensor([[4, 5]]), torch.tensor([[8, 9, 10]]))):
        expected = port.forward_ids(real_ids, logits=False)
        real_start = int(pad[row])
        torch.testing.assert_close(padded['h_final'][row:row + 1, real_start:], expected['h_final'],
                                   atol=3e-6, rtol=3e-6)
        for actual, target in zip(split[row].states, expected['cache'].states):
            if isinstance(actual, DeltaState):
                torch.testing.assert_close(actual.conv, target.conv)
                torch.testing.assert_close(actual.recurrent, target.recurrent)
            else:
                torch.testing.assert_close(actual.k, target.k)
                torch.testing.assert_close(actual.v, target.v)


def test_qwen35_kernel_one_convolution_tail_and_incremental_decode():
    torch.manual_seed(47)
    port = tiny_port(conv_kernel=1)
    ids = torch.tensor([[4, 5, 6]])
    full = port.forward_ids(ids, logits=False)
    prefix = port.forward_ids(ids[:, :2], logits=False)
    assert prefix['cache'].states[1].conv.shape[-1] == 0
    suffix = port.forward_ids(ids[:, 2:], cache=prefix['cache'], logits=False)
    torch.testing.assert_close(suffix['h_final'], full['h_final'][:, 2:], atol=3e-6, rtol=3e-6)


def test_qwen35_delta_state_only_replay_has_empty_readout():
    port = tiny_port()
    module = port.layers[1].linear_attn
    payload = {'mixed_qkv': torch.randn(3, module.conv_dim),
        'z': torch.randn(3, module.num_v_heads * module.head_v_dim),
        'b': torch.randn(3, module.num_v_heads), 'a': torch.randn(3, module.num_v_heads)}
    output, conv, recurrent = replay(module, payload, output_positions=[])
    assert output.shape == (1, 0, module.hidden_size)
    assert conv.shape == (1, module.conv_dim, module.conv_kernel_size - 1)
    assert recurrent is not None


def test_qwen35_full_attention_matches_hf_with_causal_mask_and_sdpa():
    torch.manual_seed(31)
    port = tiny_port()
    ids = torch.tensor([[4, 5, 6, 7]])
    embeddings = port.embedding_weight[ids]
    positions = torch.arange(ids.shape[1])
    causal = torch.zeros((1, 1, ids.shape[1], ids.shape[1]), dtype=embeddings.dtype)
    causal.masked_fill_(~torch.tril(torch.ones((ids.shape[1], ids.shape[1]), dtype=torch.bool))[None, None],
                        torch.finfo(embeddings.dtype).min)
    expected = port.text(inputs_embeds=embeddings, attention_mask=causal, use_cache=False).last_hidden_state
    actual, _ = port.run_layers(embeddings, range(port.num_layers), PortCache.empty(port.num_layers))
    actual = port.final_norm(actual)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward

    layer = port.layers[0]
    query, key, value, gate = _layer_inputs(port, layer, embeddings, True)
    query, key = _rotate_pair(port, query, key, positions, 0)
    eager, _ = eager_attention_forward(layer.self_attn, query, key, value, causal,
                                        scaling=layer.self_attn.scaling)
    sdpa, _ = ALL_ATTENTION_FUNCTIONS['sdpa'](layer.self_attn, query, key, value, causal,
        dropout=0.0, scaling=layer.self_attn.scaling)
    eager = layer.self_attn.o_proj(eager.reshape(1, ids.shape[1], -1) * gate.sigmoid())
    sdpa = layer.self_attn.o_proj(sdpa.reshape(1, ids.shape[1], -1) * gate.sigmoid())
    assert torch.allclose(eager, sdpa, atol=2e-6, rtol=2e-6)


def test_qwen35_nested_auto_loader_role_provision_and_native_restore(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    from natlang_neuralese.model.native_backbone_restore import restore_backbone
    from natlang_neuralese.model.native_model import load_native_model, model_state

    text_config = Qwen3_5TextConfig(vocab_size=4, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_conv_kernel_dim=3, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=['full_attention', 'linear_attention'], max_position_embeddings=32)
    vision_config = Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=32,
        num_heads=2, patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
        out_hidden_size=16, num_position_embeddings=16)
    checkpoint = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
        text_config=text_config, vision_config=vision_config,
        image_token_id=28, video_token_id=29, vision_start_token_id=30, vision_end_token_id=31))
    checkpoint.save_pretrained(tmp_path, safe_serialization=True)
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'sample': 1}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
    hf_tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\\n{% endfor %}"
    hf_tokenizer.save_pretrained(tmp_path)

    model, tokenizer, roles, config = load_native_model(tmp_path, rank=4, dtype=torch.float32)
    assert config['architecture'] == 'qwen3_5'
    assert config['native_ordinary_precision'] == 'fp32-norms-qwen35-delta-controls-v1'
    assert model.port.layers[1].block_type == 'linear_attention'
    assert model.port.embedding_weight.shape[0] > text_config.vocab_size
    from natlang_neuralese.model.native_parameter_precision import norm_parameters
    assert any(name.startswith('model.language_model.layers.0.') for name in norm_parameters(model))
    assert len(set(roles.ids)) == 4
    assert all(tokenizer.convert_ids_to_tokens(token) for token in roles.ids)
    state = model_state(model)['backbone_state']
    restored, _ = restore_backbone(tmp_path, 'qwen3_5', state, device='cpu', dtype=torch.float32)
    assert restored.config.model_type == 'qwen3_5'
    assert restored.model.language_model.config.vocab_size == model.port.embedding_weight.shape[0]
    assert all(torch.equal(state[name], restored.state_dict()[name]) for name in state)
