"""Run in the training image; the lightweight harness environment has no torch."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('peft')
from transformers import Lfm2Config, Lfm2ForCausalLM
from scripts.train_lora import completion_loss


@pytest.mark.parametrize('prompt_len,completion_len', [(1, 1), (9, 1), (9, 7)])
def test_completion_projection_preserves_loss_and_gradients(prompt_len, completion_len):
    torch.manual_seed(7)
    config = Lfm2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                       full_attn_idxs=[0, 1], use_cache=False)
    model = Lfm2ForCausalLM(config).eval()
    x = torch.randint(0, 32, (1, prompt_len + completion_len))
    full_labels = x.clone()
    full_labels[:, :prompt_len] = -100
    full = model(input_ids=x, labels=full_labels).loss
    full.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    trimmed = completion_loss(model, (x, full_labels[:, -(completion_len + 1):]))
    trimmed.backward()
    torch.testing.assert_close(full, trimmed)
    for n, p in model.named_parameters():
        if n in gradients:
            torch.testing.assert_close(p.grad, gradients[n], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('checkpointing', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_padded_microbatch_preserves_per_example_gradients(checkpointing, device, monkeypatch):
    import importlib.util
    if device == "cpu" and importlib.util.find_spec("causal_conv1d") is not None:
        pytest.skip("Installed optimized causal-conv1d requires CUDA; CPU fallback tested in baseline image")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from scripts.train_lora import collate_completions, batch_completion_loss
    torch.manual_seed(17)
    # Include convolution blocks: padding must not change their causal history.
    config = Lfm2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                       full_attn_idxs=[1], use_cache=False)
    model = Lfm2ForCausalLM(config).to(device).train()
    if checkpointing:
        model.gradient_checkpointing_enable()
    examples = [([1, 2], [3]), ([4, 5, 6, 7, 8], [9, 10, 11]), ([12], [13, 14])]
    separate = sum(batch_completion_loss(model, collate_completions([e], device=device))
                   for e in examples) / len(examples)
    separate.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    batched = batch_completion_loss(model, collate_completions(examples, device=device))
    batched.backward()
    torch.testing.assert_close(batched, separate, atol=1e-6, rtol=1e-5)
    for n, p in model.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], atol=1e-6, rtol=1e-4)


def test_microbatch_token_budget_and_cache_identity(tmp_path):
    from scripts.train_lora import microbatches, TokenCache
    examples = [([1] * n, [2]) for n in [7, 1, 3, 12, 4]]
    batches = list(microbatches(examples, 3, 12))
    assert sum(map(len, batches)) == len(examples)
    assert all(len(b) == 1 or max(len(x) + len(y) for x, y in b) * len(b) <= 12 for b in batches)
    cache = TokenCache(tmp_path / 'tokens.db', {'data': 'one'})
    cache.put(10, [[1], [2], 'family'])
    cache.db.commit()
    cache.db.close()
    cache = TokenCache(tmp_path / 'tokens.db', {'data': 'one'})
    assert cache.get(10) == [[1], [2], 'family']
    assert cache.get(11) is None
    cache.db.close()
    with pytest.raises(ValueError, match='mismatch'):
        TokenCache(tmp_path / 'tokens.db', {'data': 'two'})


def test_partial_layer_checkpointing_retains_sparse_activations():
    from scripts.train_lora import set_layer_checkpointing
    config = Lfm2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                       full_attn_idxs=[1, 3], use_cache=False)
    model = Lfm2ForCausalLM(config)
    total, active = set_layer_checkpointing(model, True, retain_every_n_layers=2)
    assert (total, active) == (4, 2)
    assert model.is_gradient_checkpointing
    set_layer_checkpointing(model, False)
    assert not model.is_gradient_checkpointing
