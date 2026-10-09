"""Identity controls are prerequisites, not evidence that compression works."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from natlang_neuralese.eval.foundation import optional_channel_diagnostics, reference_next_embedding
from natlang_neuralese.model.heads import ContentProjection, InterfaceNorm, PortHeads
from natlang_neuralese.serve.engine import Engine


def test_fresh_content_projection_preserves_raw_embedding_exactly():
    torch.manual_seed(4)
    embedding = torch.randn(1, 7, 16)
    output = torch.randn_like(embedding) * 12
    projection = ContentProjection(16)
    assert torch.equal(projection(embedding, output), embedding)
    # Matching the average embedding scale is not an identity transform.
    interface = InterfaceNorm(embedding[0])
    assert not torch.equal(interface(embedding), embedding)


def test_full_output_reference_selects_next_token_not_current_token():
    table = torch.tensor([[1., 0.], [0., 2.], [-3., 0.]])
    backbone = SimpleNamespace(embed=lambda ids: table[ids], logits=lambda states: states)
    logits = torch.tensor([[[0., 9., 1.], [5., 1., 0.]]])
    assert torch.equal(reference_next_embedding(backbone, logits), table[torch.tensor([[1, 0]])])
    mixture = logits.softmax(-1) @ table
    assert not torch.equal(mixture, reference_next_embedding(backbone, logits))


def test_vocabulary_free_feedback_skips_only_optional_channel_diagnostics():
    heads = SimpleNamespace(profile='latent-sketch-v2', feedback=object())
    result = optional_channel_diagnostics(heads, backbone=None, raw=None, h_cut=None, logits=None, chosen=None)
    assert result['head_profile'] == 'latent-sketch-v2'
    assert result['learned_channel_diagnostic'] == 'not-applicable-no-vocabulary-readout'
    assert 'learned_feedback_raw_greedy_embedding_relative_mse' not in result


def test_transparent_reader_preserves_positions_and_payload_without_markers():
    engine = Engine.__new__(Engine)
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    payload = table[2:4].clone()
    engine.backbone = SimpleNamespace(embedding_weight=table, embed=lambda ids: table[ids],
                                     controls=SimpleNamespace(open_id=4, close_id=5))
    engine.heads = SimpleNamespace(interface=InterfaceNorm(table), read_markers=True)
    engine.heads.read_embeddings = lambda base, values: PortHeads.read_embeddings(engine.heads, base, values)
    engine.device = 'cpu'
    engine._template = None
    engine._specials = ()
    engine.lookup = lambda identifier: SimpleNamespace(payload=payload)
    engine._template_tokens = lambda segment, nonce: [0] if segment == 'before' else [1]
    rendered = SimpleNamespace(blocks=['value'], segments=['before', 0, 'after'], escape_nonce=None)
    with patch('natlang_neuralese.serve.engine.render_messages', return_value=rendered):
        raw = engine.prompt_embeddings([], None, block_mode='transparent')
        assert torch.equal(raw, table[torch.tensor([[0, 2, 3, 1]])])
        legacy = engine.prompt_embeddings([], None)
        assert legacy.shape[1] == raw.shape[1] + 2
    with pytest.raises(Exception, match='block mode must'):
        engine.prompt_embeddings([], None, block_mode='unknown')
