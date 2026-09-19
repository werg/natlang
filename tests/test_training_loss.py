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
