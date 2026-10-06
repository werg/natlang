"""Strict per-sketch credit, including attention/short-convolution history."""
import pytest
import torch

from natlang_neuralese.model.heads import PortHeads
from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone
from natlang_neuralese.train.execution import prefill_write_contexts, unroll_write, write_generated


def test_cache_branching_preserves_prefix_storage_and_row_adjoints():
    from natlang_neuralese.model.lfm2_port import AttentionState, ConvState, PortCache
    from natlang_neuralese.model.lfm2_port import append_kv
    prefix = torch.randn(1, 2, 7, 4, requires_grad=True)
    window = torch.randn(1, 8, 2, requires_grad=True)
    cache = PortCache((AttentionState(prefix, prefix * 2), ConvState(window)),
                      (7, 7), torch.tensor([2]), (2,))
    branched = cache.repeat_interleave(4)
    assert branched.states[0].tail_k.untyped_storage().data_ptr() == prefix.untyped_storage().data_ptr()
    assert branched.pad_offsets == (2, 2, 2, 2)
    assert branched.lengths == cache.lengths
    new = torch.randn(4, 2, 1, 4)
    appended = append_kv(branched.states[0], new, new, static=False)
    assert cache.states[0].length == 7 and appended.length == 8
    (branched.states[0].k.sum() + branched.states[1].window.sum()).backward()
    torch.testing.assert_close(prefix.grad, torch.full_like(prefix, 4))
    torch.testing.assert_close(window.grad, torch.full_like(window, 4))


def tiny(kind):
    torch.manual_seed(21)
    if kind == 'lfm':
        from transformers import Lfm2Config, Lfm2ForCausalLM
        config = Lfm2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                            block_multiple_of=8, block_auto_adjust_ff_dim=False,
                            layer_types=['conv', 'full_attention', 'conv', 'full_attention'])
        model = Lfm2ForCausalLM(config).eval()
        backbone = PortBackbone(model, ControlTokens(62, 63), fast=False)
    else:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from natlang_neuralese.model.hf_port import QwenPortBackbone
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=8, max_position_embeddings=128)).eval()
        backbone = QwenPortBackbone(model, ControlTokens(62, 63), fast=False)
    for p in model.parameters():
        p.requires_grad_(False)
    heads = PortHeads(backbone, cutoff=2, max_length=8, profile='latent-sketch-v2').eval()
    # Nonzero residual makes consumer credit independent of argmax coincidences.
    with torch.no_grad():
        heads.content.proj.weight.normal_(0, .02)
    return backbone, heads


@pytest.mark.parametrize('kind', ['lfm', 'qwen'])
@pytest.mark.parametrize('checkpointed', [False, True])
@pytest.mark.parametrize('length', [1, 5])
def test_local_stage_preserves_primal_and_has_diagonal_sketch_credit(kind, checkpointed, length):
    backbone, heads = tiny(kind)
    backbone.checkpoint_layers = checkpointed
    contexts = [torch.randn(n, 32, requires_grad=True) for n in (7, 4)]
    pre = prefill_write_contexts(backbone, heads, contexts)
    lengths = torch.tensor([length, max(1, length - 2)])
    with torch.no_grad():
        expected = unroll_write(backbone, heads, pre, lengths=lengths)
    branches = []
    def capture(module, args, output):
        if torch.is_grad_enabled():
            branches.append(output)
    handle = heads.feedback.register_forward_hook(capture)
    try:
        actual = write_generated(backbone, heads, pre, 'local_stage', lengths=lengths)
    finally:
        handle.remove()
    for field in ('inputs', 'shallow', 'final', 'payload', 'stop_logits'):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=2e-5, rtol=2e-5)
    assert torch.equal(actual.lengths, expected.lengths)
    assert torch.equal(actual.truncated, expected.truncated)
    assert torch.isfinite(actual.sketch_target_loss)
    sketches, auxiliary = branches[::2], branches[1::2]
    assert len(sketches) == length
    for j in range(length):
        gradients = torch.autograd.grad(actual.final[:, j].square().sum(),
                                        sketches + contexts, allow_unused=True, retain_graph=True)
        for i, grad in enumerate(gradients[:length]):
            if i == j:
                assert grad is not None and grad.abs().sum() > 0
            else:
                assert grad is None or not grad.any(), (j, i)
        assert all(g is not None and g.abs().sum() > 0 for g in gradients[length:])
    # Targets are detached: auxiliary loss does not enter consumer sketches.
    target_grads = torch.autograd.grad(actual.sketch_target_loss, sketches,
                                      allow_unused=True, retain_graph=True)
    assert all(g is None or not g.any() for g in target_grads)
    if length > 1:
        # p[j+1] receives s[j], not s[j+1] (autoregressive slot alignment).
        grads = torch.autograd.grad(actual.payload[:, 1].square().sum(), sketches,
                                    allow_unused=True, retain_graph=True)
        assert grads[0] is not None and grads[0].abs().sum() > 0
        assert all(g is None or not g.any() for g in grads[1:])
    (actual.payload.square().mean() + actual.sketch_target_loss).backward()
    assert heads.feedback.correction.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('scale', [0., .05, 1.])
def test_auxiliary_scope_scale_and_full_projection_credit(scale):
    backbone, heads = tiny('lfm')
    context = torch.randn(4, 32, requires_grad=True)
    pre = prefill_write_contexts(backbone, heads, [context])
    actual = write_generated(backbone, heads, pre, 'local_stage', length=3,
                             sketch_target_backbone_scale=scale)
    grad, = torch.autograd.grad(actual.sketch_target_loss, [context], retain_graph=True, allow_unused=True)
    if scale == 0:
        assert grad is None or not grad.any()
    else:
        assert grad is not None and grad.abs().sum() > 0
    actual.sketch_target_loss.backward()
    assert heads.feedback.correction.weight.grad.abs().sum() > 0


def test_local_stage_rejects_active_dropout():
    backbone, heads = tiny('lfm')
    pre = prefill_write_contexts(backbone, heads, [torch.randn(4, 32)])
    backbone.hf.add_module('test_dropout', torch.nn.Dropout(.1).train())
    with pytest.raises(ValueError, match='dropout'):
        write_generated(backbone, heads, pre, 'local_stage', length=3)


@pytest.mark.parametrize('temperature', [0., .5])
def test_no_grad_primal_skips_replay_and_preserves_samples_and_targets(temperature):
    backbone, heads = tiny('lfm')
    context = torch.randn(1, 4, 32)
    calls = []
    hook = heads.feedback.register_forward_hook(lambda *args: calls.append(1))
    try:
        with torch.no_grad():
            pre = prefill_write_contexts(backbone, heads, [context[0]])
            actual = write_generated(backbone, heads, pre, 'local_stage', length=5,
                                     local_stage_batch_size=8, temperature=temperature,
                                     generator=torch.Generator().manual_seed(8))
    finally:
        hook.remove()
    assert len(calls) == 5  # only greedy recurrence, no local branch replay
    with torch.no_grad():
        expected = unroll_write(backbone, heads, pre, length=5, temperature=temperature,
                                generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(actual.payload, expected.payload, rtol=0, atol=0)
    training = write_generated(backbone, heads, pre, 'local_stage', length=5,
                               local_stage_batch_size=8)
    torch.testing.assert_close(actual.sketch_target_loss, training.sketch_target_loss, rtol=0, atol=0)
    assert actual.local_replay_max_abs_error is None


@pytest.mark.parametrize('kind', ['lfm', 'qwen'])
@pytest.mark.parametrize('stage_batch_size', [2, 8])
def test_grouped_stages_match_sequential_adjoints_and_keep_diagonal_support(kind, stage_batch_size):
    backbone, heads = tiny(kind)
    backbone.checkpoint_layers = True
    backbone_parameter = next(p for name, p in backbone.hf.named_parameters()
                              if 'layers.0.' in name and p.ndim == 2)
    backbone_parameter.requires_grad_(True)
    context = torch.randn(2, 7, 32, requires_grad=True)
    lengths = torch.tensor([5, 3])
    def run(width, capture=None):
        pre = prefill_write_contexts(backbone, heads, [context[0], context[1, :4]])
        hook = heads.feedback.register_forward_hook(capture) if capture else None
        try:
            return write_generated(backbone, heads, pre, 'local_stage', lengths=lengths,
                                   local_stage_batch_size=width)
        finally:
            if hook:
                hook.remove()
    branches = []
    def capture(module, inputs, output):
        if torch.is_grad_enabled():
            branches.append(output)
    sequential = run(1)
    grouped = run(stage_batch_size, capture)
    torch.testing.assert_close(grouped.payload, sequential.payload, rtol=0, atol=0)
    sketches = branches[::2]
    for j in range(5):
        grads = torch.autograd.grad(grouped.final[:, j].square().sum(), sketches,
                                    allow_unused=True, retain_graph=True)
        reached, offset = [], 0
        for value, grad in zip(sketches, grads):
            if grad is not None:
                reached.extend(offset + i for i in range(value.shape[1]) if grad[:, i].abs().sum() > 0)
            offset += value.shape[1]
        assert reached == [j]
    parameters = [context, backbone_parameter, *heads.feedback.parameters(), *heads.content.proj.parameters()]
    adjoint = torch.linspace(-1, 1, grouped.final.numel()).reshape_as(grouped.final)
    def loss(written):
        return (written.final * adjoint).sum() + .1 * written.sketch_target_loss
    expected = torch.autograd.grad(loss(sequential), parameters, allow_unused=True)
    actual = torch.autograd.grad(loss(grouped), parameters, allow_unused=True)
    assert expected[1] is not None and expected[1].abs().sum() > 0
    for got, wanted in zip(actual, expected):
        if wanted is None:
            assert got is None
        else:
            torch.testing.assert_close(got, wanted, atol=2e-4, rtol=3e-4)


def test_batched_writer_self_target_keeps_per_producer_normalization():
    backbone, heads = tiny('lfm')
    context = torch.randn(2, 7, 32, requires_grad=True)
    contexts = [context[0], context[1, :4]]
    lengths = torch.tensor([5, 3])
    parameters = [context, *heads.feedback.parameters(), *heads.content.proj.parameters()]

    def compute_batch():
        pre = prefill_write_contexts(backbone, heads, contexts)
        written = write_generated(backbone, heads, pre, 'local_stage', lengths=lengths,
                                  local_stage_batch_size=2)
        return ([written.payload[row, :length] for row, length in enumerate((5, 3))],
                [[written.sketch_target_loss_by_row[row]] for row in range(2)])

    pre = prefill_write_contexts(backbone, heads, contexts)
    grouped = write_generated(backbone, heads, pre, 'local_stage', lengths=lengths,
                              local_stage_batch_size=2)
    assert grouped.sketch_target_loss_by_row.shape == (2,)
    # The legacy scalar remains token-weighted across a multirow Written.
    torch.testing.assert_close(
        grouped.sketch_target_loss,
        (5 * grouped.sketch_target_loss_by_row[0] + 3 * grouped.sketch_target_loss_by_row[1]) / 8)
    grouped_grads = torch.autograd.grad(grouped.sketch_target_loss_by_row.mean(), parameters,
                                        allow_unused=True)

    from natlang_neuralese.train.staging import StagedWrites
    staged = StagedWrites(collect=lambda: None)
    staged.add_batch(compute_batch, auxiliaries=[None, None])
    staged.backward(penalty_weight=1., scale=1.)
    staged_grads = [None if parameter.grad is None else parameter.grad.detach().clone()
                    for parameter in parameters]
    staged.clear()
    for parameter in parameters:
        parameter.grad = None

    singleton_losses = []
    for row, length in enumerate((5, 3)):
        pre = prefill_write_contexts(backbone, heads, [contexts[row]])
        written = write_generated(backbone, heads, pre, 'local_stage', length=length,
                                  local_stage_batch_size=1)
        assert written.sketch_target_loss_by_row.shape == (1,)
        torch.testing.assert_close(written.sketch_target_loss, written.sketch_target_loss_by_row[0])
        singleton_losses.append(written.sketch_target_loss)
    singleton_grads = torch.autograd.grad(sum(singleton_losses) / 2, parameters,
                                         allow_unused=True)
    for actual in (grouped_grads, staged_grads):
        for got, wanted in zip(actual, singleton_grads):
            if wanted is None:
                assert got is None
            else:
                torch.testing.assert_close(got, wanted, atol=2e-4, rtol=3e-4)
