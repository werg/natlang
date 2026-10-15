"""Real two-port compact wire/consumer parity, without a second model loader."""
from types import SimpleNamespace

import pytest
import torch

from test_native_full_updates import source
from natlang_neuralese.model.native_model import load_native_model, native_identity
from natlang_neuralese.model.semantic_channel import SemanticChannel
from natlang_neuralese.model.semantic_vae import SemanticStateVAE, StateLayout
from natlang_neuralese.model.shared_state import capture_block
from natlang_neuralese.eval.native_semantic_runtime import qualify_semantic_transport


def fixture(tmp_path):
    participants, layouts, identities, tokenizers = {}, {}, {}, {}
    for name in ('lfm', 'mellum'):
        base = tmp_path/name
        dtype = source(base, name)
        model, tokenizer, roles, config = load_native_model(base, dtype=dtype, rank=4)
        identity = native_identity(model, config, checkpoint_sha256='1'*64)
        with torch.no_grad():
            from natlang_neuralese.model.shared_state import Invocation, compile_graph
            result = model(compile_graph([Invocation('write', (roles.produce_open, 4, roles.produce_close),
                                                    producing=((1,2),))]))
            block = capture_block(result, 'write', 1, 2, text='t4', **identity)
        participants[name] = SimpleNamespace(model=model, tokenizer=tokenizer, config=config)
        layouts[name], identities[name], tokenizers[name] = StateLayout.from_block(block), identity, tokenizer
    vae = SemanticStateVAE(layouts, latent_width=4, hidden_width=8)
    channel = SemanticChannel(vae, identities, tokenizers, adapter_sha256='2'*64)
    return participants, channel


def test_actual_both_port_directions_use_compact_learned_decode_and_preserve_state(tmp_path):
    participants, channel = fixture(tmp_path)
    participants['lfm'].model.train()
    participants['mellum'].model.eval()
    channel.vae.train()
    modes = [p.model.training for p in participants.values()] + [channel.vae.training]
    rng = torch.get_rng_state().clone()
    certificate = qualify_semantic_transport(participants, channel,
        texts={name:'t4' for name in participants}, rtol=1e-5, atol=1e-6)
    assert certificate['passed']
    assert {(row['source'],row['target']) for row in certificate['directions']} == {('lfm','mellum'),('mellum','lfm')}
    assert all(row['wire_layer_tensors'] == 0 and row['mode'] == 'learned-semantic-state-deltas'
               and row['artifact_id'].startswith('nzs1_') and row['next_cache'] for row in certificate['directions'])
    assert modes == [p.model.training for p in participants.values()] + [channel.vae.training]
    assert torch.equal(rng, torch.get_rng_state())


def test_stale_actual_weight_identity_is_rejected_and_modes_restored(tmp_path):
    participants, channel = fixture(tmp_path)
    with torch.no_grad():
        participants['lfm'].model.input_rows.add_(.125)
    assert participants['lfm'].model.training
    with pytest.raises(ValueError, match='actual exported model identity'):
        qualify_semantic_transport(participants, channel,
            texts={name:'t4' for name in participants}, rtol=1e-5, atol=1e-6)
    assert participants['lfm'].model.training and channel.vae.training


def test_text_fallback_cannot_receive_a_learned_transport_certificate(tmp_path, monkeypatch):
    participants, channel = fixture(tmp_path)
    from natlang_neuralese.serve.engine import Engine
    original = Engine._native_consume
    def wrong_mode(self, sequence, identifier):
        original(self, sequence, identifier)
        sequence.native_imports[-1]['mode'] = 'best-effort-text-retokenization'
    monkeypatch.setattr(Engine, '_native_consume', wrong_mode)
    with pytest.raises(ValueError, match='bypassed learned decoding'):
        qualify_semantic_transport(participants, channel,
            texts={name:'t4' for name in participants}, rtol=1e-5, atol=1e-6)


def test_corrupted_consumer_state_fails_exact_reference_without_relaxing_tolerances(tmp_path, monkeypatch):
    participants, channel = fixture(tmp_path)
    from natlang_neuralese.serve.engine import Engine
    original = Engine._native_consume
    def corrupted(self, sequence, identifier):
        original(self, sequence, identifier)
        sequence.logits = sequence.logits + .25
    monkeypatch.setattr(Engine, '_native_consume', corrupted)
    certificate = qualify_semantic_transport(participants, channel,
        texts={name:'t4' for name in participants}, rtol=0., atol=0.)
    assert not certificate['passed']
    assert all(not row['logits']['passed'] and row['logits']['max_abs'] >= .24
               for row in certificate['directions'])
