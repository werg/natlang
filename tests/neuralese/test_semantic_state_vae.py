"""Cross-tokenizer VAE and joint gradients through actual native consumers."""
import pytest
import torch

from natlang_neuralese.model.semantic_vae import StateLayout, SemanticStateVAE, alignment_weights
from natlang_neuralese.model.shared_state import SharedStateModel, Invocation, ImportSpan, compile_graph, capture_block
from test_shared_state import tiny


def model_and_pair(family, tokens, key):
    model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
    identity = {'checkpoint_sha256': key * 64, 'tokenizer_fingerprint': key * 64,
                'embedding_weight_sha256': key * 64, 'architecture': model.port.config.model_type}
    plan = compile_graph([Invocation('writer', (1, 2, *tokens), producing=((2, 2+len(tokens)),))])
    result = model(plan)
    reference = capture_block(result, 'writer', 2, 2+len(tokens), text='abcd', **identity)
    baseline_plan = compile_graph([Invocation('writer', tuple(tokens), producing=((0, len(tokens)),))])
    baseline = capture_block(model(baseline_plan), 'writer', 0, len(tokens), text='abcd', **identity)
    return model, reference, baseline, identity


def test_sparse_token_alignment_preserves_overlap_and_marks_fallback():
    matrix, receipt = alignment_weights([(0, 2), (2, 4)], [(0, 1), (1, 3), (3, 4)])
    torch.testing.assert_close(matrix.to_dense(), torch.tensor([[1., 0.], [.5, .5], [0., 1.]]))
    assert not receipt['best_effort']
    with pytest.raises(ValueError, match='alignment repair'):
        alignment_weights([(0, 2)], [(3, 4)])
    _, approximate = alignment_weights([(0, 2)], [(3, 4)], allow_nearest=True)
    assert approximate['best_effort'] and approximate['nearest_fallback_target_tokens'] == [0]


def test_joint_vae_consumer_loss_reaches_source_writer_and_shared_space():
    source_model, source, source_baseline, _ = model_and_pair('lfm', (5, 6), '1')
    target_model, target, target_baseline, target_identity = model_and_pair('mellum', (8, 9, 10), '2')
    vae = SemanticStateVAE({'lfm': StateLayout.from_block(source), 'mellum': StateLayout.from_block(target)},
                            latent_width=6, hidden_width=12)
    forward, _ = alignment_weights([(0, 2), (2, 4)], [(0, 1), (1, 3), (3, 4)])
    backward, _ = alignment_weights([(0, 1), (1, 3), (3, 4)], [(0, 2), (2, 4)])
    vae_loss, terms = vae.pair_loss('lfm', source, source_baseline, 'mellum', target, target_baseline,
                                   source_to_target=forward, target_to_source=backward)
    decoded = terms['target_block']
    plan = compile_graph([Invocation('consumer', (60, 61, 12), imports=(ImportSpan(1, 'wire', 0, 3),))],
                          external_sizes={'wire': 3})
    result = target_model(plan, external_blocks={'wire': decoded}, external_identity=target_identity)
    behavior = torch.nn.functional.cross_entropy(target_model.logits(result['h_final'][:, -1]).float(),
                                                torch.tensor([13]))
    # Include the producer closing-marker behavior in the same differentiable
    # update, rather than optimizing reconstruction while leaving write-out frozen.
    writer_plan = compile_graph([Invocation('write', (3, 58, 5, 6, 59), producing=((2, 4),))])
    writer = source_model(writer_plan)
    control = torch.nn.functional.cross_entropy(source_model.logits(writer['h_final'][0, [1, 2, 3]]),
                                               torch.tensor([5, 6, 59]))
    total = vae_loss + behavior + control
    total.backward()
    assert torch.isfinite(total)
    assert source_model.adapter.up.weight.grad.norm() > 0
    assert target_model.input_rows.grad is not None
    assert vae.posterior[-1].weight.grad.norm() > 0
    assert vae.output_adapters[vae.keys['mellum']].weight.grad.norm() > 0
    assert decoded.checkpoint_sha256 == target.checkpoint_sha256


def test_model_addition_preserves_existing_parameter_names_and_restore():
    _, first, _, _ = model_and_pair('lfm', (5, 6), '1')
    layout = StateLayout.from_block(first)
    model = SemanticStateVAE({'b': layout, 'c': layout}, latent_width=4, hidden_width=8)
    before = dict(model.named_parameters())
    added = model.add_model('a', layout)
    assert added and all(dict(model.named_parameters())[name] is value for name, value in before.items())
    restored = SemanticStateVAE.from_specification(model.specification())
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.specification() == model.specification()
