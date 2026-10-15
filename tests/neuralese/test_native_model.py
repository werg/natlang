"""Real native model restore, including newly added role vocabulary rows."""
import torch
import pytest

from natlang_neuralese.model.native_model import (
    CHECKPOINT_SCHEMA, load_native_model, load_native_checkpoint, model_state, native_identity,
)
from natlang_neuralese.model.shared_state import Invocation, compile_graph


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
