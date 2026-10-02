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


def test_custom_checkpointing_does_not_silently_ignore_partial_request():
    from scripts.train_lora import set_layer_checkpointing

    class NativeCheckpointModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enabled = False

        def gradient_checkpointing_enable(self):
            self.enabled = True

        def gradient_checkpointing_disable(self):
            self.enabled = False

    model = NativeCheckpointModel()
    with pytest.raises(ValueError, match='per-layer'):
        set_layer_checkpointing(model, True, retain_every_n_layers=2)
    assert not model.enabled
    assert set_layer_checkpointing(model, True) == (0, 0)
    assert model.enabled
    set_layer_checkpointing(model, False)
    assert not model.enabled


def test_adapter_coverage_names_linear_layers_left_without_an_adapter():
    import pytest
    import torch.nn as nn
    from scripts.train_lora import require_adapter_coverage

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict()

    model = nn.Module()
    model.attn = nn.Module(); model.attn.q_proj = Wrapped()
    model.conv = nn.Module(); model.conv.in_proj = nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="in_proj"):
        require_adapter_coverage(model, ["q_proj", "in_proj"])
    model.conv.in_proj = Wrapped()
    require_adapter_coverage(model, ["q_proj", "in_proj"])
    with pytest.raises(RuntimeError, match="match no layer: w1"):
        require_adapter_coverage(model, ["q_proj", "w1"])


def test_adapter_coverage_skips_excluded_layers():
    import torch.nn as nn
    from scripts.train_lora import require_adapter_coverage

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict()

    model = nn.Module()
    model.mlp = nn.Module(); model.mlp.gate_proj = Wrapped()
    model.mlp.experts = nn.ModuleList([nn.Module()]); model.mlp.experts[0].gate_proj = nn.Linear(2, 2)
    require_adapter_coverage(model, ["gate_proj"], r".*\.experts\.\d+\..*")


def test_adapter_coverage_counts_lora_on_stacked_expert_parameters():
    import re
    import torch.nn as nn
    from scripts.train_lora import EXPERTS, require_adapter_coverage

    class ParamWrapper(nn.Module):
        def __init__(self, parameter_name):
            super().__init__()
            self.lora_A = nn.ModuleDict()
            self.parameter_name = parameter_name

    model = nn.Module()
    model.feed_forward = nn.Module()
    model.feed_forward.experts = ParamWrapper("down_proj")
    model.feed_forward.experts.base_layer = ParamWrapper("gate_up_proj")
    require_adapter_coverage(model, ["gate_up_proj", "down_proj"])
    assert re.match(rf"(.*\.)?({EXPERTS})$", "model.layers.3.feed_forward.experts.gate_up_proj")
    assert re.match(rf"(.*\.)?({EXPERTS})$", "model.layers.3.feed_forward.experts.12.gate_proj")
    assert not re.match(rf"(.*\.)?({EXPERTS})$", "model.layers.3.feed_forward.shared_experts.gate_proj")
    assert not re.match(rf"(.*\.)?({EXPERTS})$", "model.layers.3.self_attn.q_proj")


def test_expert_rank_applies_to_individual_experts_with_peft():
    from peft import LoraConfig, get_peft_model
    from scripts.train_lora import EXPERTS

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
            for expert in self.experts:
                expert.gate_proj = torch.nn.Linear(4, 8)
            self.shared_experts = torch.nn.Module()
            self.shared_experts.gate_proj = torch.nn.Linear(4, 8)

    model = get_peft_model(Model(), LoraConfig(
        r=4, lora_alpha=8, target_modules=['gate_proj'],
        rank_pattern={EXPERTS: 2}, alpha_pattern={EXPERTS: 4},
    ))
    for expert in model.base_model.model.experts:
        assert expert.gate_proj.r['default'] == 2
        assert expert.gate_proj.lora_alpha['default'] == 4
    shared = model.base_model.model.shared_experts.gate_proj
    assert shared.r['default'] == 4
    assert shared.lora_alpha['default'] == 8


def test_split_targets_separates_linear_layers_from_stacked_expert_weights():
    import torch
    import torch.nn as nn
    from scripts.train_lora import split_targets

    model = nn.Module()
    model.shared = nn.Module(); model.shared.down_proj = nn.Linear(4, 4)
    model.experts = nn.Module()
    model.experts.gate_up_proj = nn.Parameter(torch.zeros(3, 8, 4))
    model.experts.down_proj = nn.Parameter(torch.zeros(3, 4, 4))
    layers, stacked = split_targets(model, ["down_proj", "gate_up_proj", "q_proj"])
    assert layers == ["down_proj"]
    assert stacked == ["experts.down_proj", "experts.gate_up_proj"]


def test_a_merged_conversation_trains_each_completion_as_its_own_turn_does():
    """A later turn's prompt is the earlier prompt and completion plus new context, so on a causal model each
    completion token has the same loss in the merged sequence as in its own turn."""
    from scripts.train_lora import collate_completions, batch_completion_loss
    torch.manual_seed(5)
    config = Lfm2Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, full_attn_idxs=[1], use_cache=False)
    model = Lfm2ForCausalLM(config).eval()
    p1, c1, d2, c2 = [1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11]
    turns = [(p1, c1), (p1 + c1 + d2, c2)]
    merged = (p1, c1 + d2 + c2, [True] * len(c1) + [False] * len(d2) + [True] * len(c2))
    encoded = collate_completions([merged], device="cpu")
    assert encoded["labels"][0].tolist()[1:] == c1 + [-100] * len(d2) + c2
    with torch.no_grad():
        by_turn = sum(batch_completion_loss(model, collate_completions([t], device="cpu")) * len(t[1])
                      for t in turns) / (len(c1) + len(c2))
        torch.testing.assert_close(batch_completion_loss(model, encoded), by_turn, atol=1e-6, rtol=1e-5)
