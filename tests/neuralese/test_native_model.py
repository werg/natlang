"""Real native model restore, including newly added role vocabulary rows."""
import torch
import pytest

from natlang_neuralese.model.native_model import (
    CHECKPOINT_SCHEMA, load_native_model, load_native_checkpoint, model_state, native_identity,
)
from natlang_neuralese.model.shared_state import Invocation, ImportSpan, compile_graph


def tiny_source(path):
    from transformers import Lfm2Config, Lfm2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    config = Lfm2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        layer_types=['conv', 'full_attention'], conv_L_cache=4, max_position_embeddings=128)
    torch.manual_seed(713)
    Lfm2ForCausalLM(config).save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f't{i}': i for i in range(64)}, unk_token='t0')), unk_token='t0',
        bos_token='t1', eos_token='t2', pad_token='t3')
    tokenizer.save_pretrained(path)


def test_native_checkpoint_restores_role_rows_adapter_and_backbone_exactly(tmp_path):
    base = tmp_path / 'base'
    tiny_source(base)
    model, tokenizer, roles, config = load_native_model(base, dtype=torch.float32, rank=4)
    assert config['precision'] == {'float32_matmul': 'highest', 'cuda_matmul_tf32': False, 'cudnn_tf32': False}
    assert torch.get_float32_matmul_precision() == 'highest'
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert min(roles.ids) == 64  # no reserved legacy marker names exist in this tokenizer
    assert model.port.embedding_weight.shape[0] == 68
    assert not any('control_rows' in name for name, _ in model.named_parameters())
    with torch.no_grad():
        model.input_rows.add_(0.2)
        model.adapter.up.weight.add_(0.1)
        next(model.port.hf.model.layers[0].parameters()).add_(0.03)
    plan = compile_graph([Invocation('producer', (5, roles.produce_open, 7, roles.produce_close),
                                     producing=((2, 3),))])
    expected = model.logits(model(plan)['h_final'])
    checkpoint = tmp_path / 'native.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, checkpoint)
    restored, restored_tokenizer, restored_roles, restored_config, _ = load_native_checkpoint(checkpoint)
    assert restored_roles.ids == roles.ids
    assert restored_config == config
    assert restored_tokenizer.get_vocab() == tokenizer.get_vocab()
    torch.testing.assert_close(restored.logits(restored(plan)['h_final']), expected, rtol=0, atol=0)
    original_identity = native_identity(model, config, checkpoint_sha256='1' * 64)
    assert native_identity(restored, restored_config, checkpoint_sha256='1' * 64) == original_identity
    with torch.no_grad():
        restored.adapter.up.weight.add_(0.01)
    assert native_identity(restored, restored_config, checkpoint_sha256='1' * 64) != original_identity
    assert not any('projection' in name or 'heads' in name for name in model_state(restored))
    with pytest.raises(ValueError, match='inventory'):
        load_native_model(base, dtype=torch.float32, inventory={'wrong': '0' * 64})


def test_native_loader_rejects_legacy_checkpoint(tmp_path):
    checkpoint = tmp_path / 'legacy.pt'
    torch.save({'schema': 'natlang.neuralese-direct/1'}, checkpoint)
    with pytest.raises(ValueError, match='legacy projection'):
        load_native_checkpoint(checkpoint)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_native_weights_only_child_export_keeps_exact_model_and_refuses_resume(tmp_path, family):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.train.checkpoint_offload import child_copy, load_state
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.serve import load_native_engine
    engine = engine_fixture(tmp_path, family)
    full = tmp_path/'full-parent.pt'
    view = tmp_path/'inference.pt'
    state = load_state(engine.native_checkpoint)
    state.update(step=5000, training={'optimizer': {'state': {'moment': torch.arange(100000)}},
        'rng': {'cpu': torch.get_rng_state()}, 'exposure_counts': {'exact-source-row': 7}})
    torch.save(state, full)
    original_sha = sha256_file_hex_cached(full)
    record = child_copy(full, view, prune=True)  # actual direct-file worker import path
    assert record['optimizer_pruned'] and record['bytes'] < record['source_bytes']
    written = load_state(view)
    assert written['optimizer_pruned_keys'] == ['training.optimizer']
    assert 'optimizer' not in written['training']
    assert written['training']['exposure_counts'] == state['training']['exposure_counts']
    assert torch.equal(written['training']['rng']['cpu'], state['training']['rng']['cpu'])
    assert written['inference_source']['training_checkpoint_sha256'] == original_sha
    assert sha256_file_hex_cached(full) == original_sha
    with pytest.raises(ValueError, match='weights-only checkpoint'):
        load_native_checkpoint(view)
    restored = load_native_engine(str(view), device='cpu')
    try:
        assert restored.rl_policy_binding['inference_source'] == written['inference_source']
        assert restored.native_identity['checkpoint_sha256'] == record['sha256']
        assert restored.native_identity['checkpoint_sha256'] != original_sha
        plan = compile_graph([Invocation('producer',(4,5,6,7), producing=((1,3),)),
            Invocation('consumer',(8,9),imports=(ImportSpan(1,'producer',1,3),))])
        with torch.no_grad():
            assert torch.equal(engine.shared_model.logits(engine.shared_model(plan)['h_final']),
                               restored.shared_model.logits(restored.shared_model(plan)['h_final']))
        for name, parameter in engine.shared_model.named_parameters():
            assert torch.equal(parameter, dict(restored.shared_model.named_parameters())[name])
    finally:
        engine.store.close();restored.store.close()


def test_native_composite_checkpoint_applies_named_patch_and_has_restorable_effective_identity(tmp_path):
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.model.native_composite import write_native_composite_checkpoint

    base = tmp_path / 'base'
    tiny_source(base)
    model, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
    base_checkpoint = tmp_path / 'base-native.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, base_checkpoint)
    model, _, _, config, _ = load_native_checkpoint(base_checkpoint, base=base, dtype=torch.float32)
    base_sha = sha256_file_hex_cached(base_checkpoint)
    parent_identity = native_identity(model, config, checkpoint_sha256=base_sha)
    with torch.no_grad():
        model.input_rows.add_(0.125)
        model.adapter.up.weight.add_(0.025)
    expected_input = model.input_rows.detach().clone()
    expected_up = model.adapter.up.weight.detach().clone()
    receipt = write_native_composite_checkpoint(base_checkpoint, tmp_path / 'composite', model, config,
        base_identity=parent_identity, optimizer={'name': 'adam', 'step': 7, 'lr': 0.001},
        optimizer_tensors={'m.adapter.up.weight': torch.ones_like(model.adapter.up.weight),
                           'v.adapter.up.weight': torch.full_like(model.adapter.up.weight, 2.)},
        update_provenance={'step': 7, 'reason': 'test optimizer update'})
    # An HF source override selects model files; it must not replace the
    # immutable native parent checkpoint named inside the composite manifest.
    restored, _, _, restored_config, state = load_native_checkpoint(receipt['path'], base=base, dtype=torch.float32)
    assert state['schema'] == 'natlang.neuralese-native-composite-checkpoint/1'
    assert state['optimizer'] == {'name': 'adam', 'step': 7, 'lr': 0.001}
    torch.testing.assert_close(state['optimizer_tensors']['m.adapter.up.weight'],
                               torch.ones_like(model.adapter.up.weight))
    assert receipt['native_identity']['checkpoint_sha256'] == sha256_file_hex_cached(receipt['path'])
    assert torch.equal(restored.input_rows, expected_input)
    assert torch.equal(restored.adapter.up.weight, expected_up)
    assert native_identity(restored, restored_config,
        checkpoint_sha256=receipt['native_identity']['checkpoint_sha256']) == receipt['native_identity']
    assert sha256_file_hex_cached(base_checkpoint) == base_sha


def test_native_composite_refuses_backbone_mutation_under_interface_only_patch(tmp_path):
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.model.native_composite import write_native_composite_checkpoint

    base = tmp_path / 'base'
    tiny_source(base)
    model, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
    base_checkpoint = tmp_path / 'base-native.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, base_checkpoint)
    model, _, _, config, _ = load_native_checkpoint(base_checkpoint, base=base, dtype=torch.float32)
    identity = native_identity(model, config, checkpoint_sha256=sha256_file_hex_cached(base_checkpoint))
    with torch.no_grad():
        next(model.port.hf.parameters()).add_(0.001)
    with pytest.raises(ValueError, match='unchanged backbone'):
        write_native_composite_checkpoint(base_checkpoint, tmp_path / 'composite', model, config,
                                          base_identity=identity)
