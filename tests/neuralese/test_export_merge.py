"""Export merges LoRA adapters (a phase-F port adapter) into plain weights: inference servers load base weights."""

import pytest
import torch

peft = pytest.importorskip("peft")

from natlang_neuralese.export.gguf import merged_state_dict


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(6, 4)
        self.out = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.out(torch.relu(self.proj(x)))


def test_merged_state_dict_matches_the_adapted_model_and_has_plain_names():
    torch.manual_seed(0)
    model = Tiny()
    assert merged_state_dict(model) is None
    config = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], lora_dropout=0.0)
    peft.inject_adapter_in_model(config, model, adapter_name="neuralese")
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param)
    x = torch.randn(5, 6)
    adapted = model(x)
    state = merged_state_dict(model)
    assert sorted(state) == sorted(Tiny().state_dict())
    plain = Tiny()
    plain.load_state_dict(state)
    assert torch.allclose(plain(x), adapted, atol=1e-5)
    assert not torch.allclose(state["proj.weight"], model.proj.base_layer.weight)
