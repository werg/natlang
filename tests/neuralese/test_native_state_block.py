"""Native shared-state provenance, shape and gradient preservation."""
import pytest
import torch

from natlang_neuralese.model.state_block import NativeStateBlock


def block():
    return NativeStateBlock(text='hello world', token_ids=torch.tensor([7, 8]),
        tokenizer_fingerprint='1' * 64, checkpoint_sha256='2' * 64,
        embedding_weight_sha256='3' * 64, architecture='hybrid-test',
        embeddings=torch.randn(2, 4, requires_grad=True), positions=torch.tensor([5, 6]),
        layers=({'kind': 'attention', 'raw_k': torch.randn(2, 2, 3, requires_grad=True),
                 'v': torch.randn(2, 2, 3, requires_grad=True)},
                {'kind': 'finite_convolution', 'gated_inputs': torch.randn(2, 4, requires_grad=True)}))


def test_state_inventory_roundtrip_keeps_exact_tensors_and_gradients():
    source = block()
    rebuilt = NativeStateBlock.from_tensors(source.metadata(), source.tensors())
    assert rebuilt.metadata() == source.metadata()
    assert set(rebuilt.tensors()) == set(source.tensors())
    for name, value in source.tensors().items():
        assert rebuilt.tensors()[name] is value
    sum(t.sum() for t in rebuilt.tensors().values() if t.requires_grad).backward()
    for tensor in source.tensors().values():
        if tensor.requires_grad:
            assert torch.equal(tensor.grad, torch.ones_like(tensor))


@pytest.mark.parametrize('field', ['checkpoint_sha256', 'tokenizer_fingerprint',
                                   'embedding_weight_sha256', 'architecture'])
def test_native_reuse_rejects_each_foreign_identity(field):
    source = block()
    identity = {key: getattr(source, key) for key in ('checkpoint_sha256', 'tokenizer_fingerprint',
                                                     'embedding_weight_sha256', 'architecture')}
    source.require_native(**identity, layer_kinds=['attention', 'finite_convolution'])
    identity[field] = '4' * 64 if field != 'architecture' else 'another-backbone'
    with pytest.raises(ValueError, match='adapt or reencode'):
        source.require_native(**identity)


@pytest.mark.parametrize('mutation', ['position_gap', 'wrong_token_axis', 'nonfinite', 'unsupported_state'])
def test_invalid_states_are_rejected(mutation):
    source = block()
    if mutation == 'position_gap':
        source.positions = torch.tensor([5, 7])
    elif mutation == 'wrong_token_axis':
        source.layers[0]['raw_k'] = torch.zeros(3, 2, 2)
    elif mutation == 'nonfinite':
        source.embeddings = torch.full((2, 4), float('nan'))
    else:
        source.layers[1]['kind'] = 'unimplemented_linear_recurrence'
    with pytest.raises(ValueError):
        source.validate()


def test_deserialization_rejects_undeclared_tensor():
    source = block()
    tensors = dict(source.tensors(), accidental_projection=torch.ones(2, 4))
    with pytest.raises(ValueError, match='inventory'):
        NativeStateBlock.from_tensors(source.metadata(), tensors)
