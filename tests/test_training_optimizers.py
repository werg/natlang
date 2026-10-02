import copy
import random

import pytest
import torch

from scripts.training_optimizers import MuonWithAdamW, make_muon_optimizer
from scripts.train_lora import (capture_rng_state, restore_rng_state,
                                write_checkpoint_directory)

pytestmark = pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="PyTorch Muon unavailable")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 4)
        self.hidden = torch.nn.Linear(4, 4)
        self.lm_head = torch.nn.Linear(4, 11)

    def forward(self, inputs):
        return self.lm_head(torch.tanh(self.hidden(self.embed_tokens(inputs))))

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def advance(model, optimizer, scheduler, count):
    for _ in range(count):
        inputs = torch.randint(0, 11, (2, 5))
        labels = torch.randint(0, 11, (2, 5))
        loss = torch.nn.functional.cross_entropy(model(inputs).reshape(-1, 11), labels.reshape(-1))
        loss = loss * (0.5 + random.random())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_partition_and_schedule_shared_with_children():
    model = TinyModel()
    optimizer = make_muon_optimizer(model, lr=0.001)
    partition = {item["name"]: item["optimizer"] for item in optimizer.schema}
    assert partition["hidden.weight"] == "muon"
    assert all(partition[name] == "adamw" for name in
               ("hidden.bias", "embed_tokens.weight", "lm_head.weight", "lm_head.bias"))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9 ** step)
    advance(model, optimizer, scheduler, 2)
    assert optimizer.param_groups[0] is optimizer.muon.param_groups[0]
    assert optimizer.param_groups[1] is optimizer.auxiliary.param_groups[0]
    assert optimizer.muon.param_groups[0]["lr"] == pytest.approx(0.00081)
    assert optimizer.muon.state and optimizer.auxiliary.state


def test_complete_checkpoint_exact_resume(tmp_path):
    torch.manual_seed(42)
    random.seed(42)
    model = TinyModel()
    optimizer = make_muon_optimizer(model, lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9 ** step)
    advance(model, optimizer, scheduler, 3)
    write_checkpoint_directory(
        tmp_path, lambda path: (path.mkdir(), torch.save(model.state_dict(), path / "model.pt")),
        optimizer.state_dict(), scheduler.state_dict(), capture_rng_state(),
        {"step": 3, "cursor": 24})
    advance(model, optimizer, scheduler, 3)
    expected = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    resumed = TinyModel()
    resumed.load_state_dict(torch.load(tmp_path / "checkpoint/weights/model.pt", weights_only=True))
    resumed_optimizer = make_muon_optimizer(resumed, lr=0.001)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda step: 0.9 ** step)
    resumed_optimizer.load_state_dict(torch.load(tmp_path / "checkpoint/optimizer.pt", weights_only=True))
    resumed_scheduler.load_state_dict(torch.load(tmp_path / "checkpoint/scheduler.pt", weights_only=True))
    restore_rng_state(torch.load(tmp_path / "checkpoint/rng.pt", weights_only=False))
    advance(resumed, resumed_optimizer, resumed_scheduler, 3)
    assert_same(expected, (resumed.state_dict(), resumed_optimizer.state_dict(), resumed_scheduler.state_dict()))


def test_reordered_parameters_and_wrong_optimizer_rejected():
    model = TinyModel()
    optimizer = make_muon_optimizer(model, lr=0.001)
    reordered = MuonWithAdamW(list(model.named_parameters())[::-1], lr=0.001)
    with pytest.raises(ValueError, match="parameter names"):
        reordered.load_state_dict(optimizer.state_dict())
    with pytest.raises(ValueError, match="parameter names"):
        optimizer.load_state_dict(torch.optim.AdamW(model.parameters()).state_dict())


def test_stacked_experts_use_auxiliary_without_invalid_2d_update():
    model = torch.nn.Module()
    model.register_parameter("stacked", torch.nn.Parameter(torch.randn(3, 4, 4)))
    model.register_parameter("matrix", torch.nn.Parameter(torch.randn(4, 4)))
    optimizer = make_muon_optimizer(model, lr=0.001)
    assert optimizer.schema[0]["optimizer"] == "adamw"
    assert optimizer.schema[1]["optimizer"] == "muon"


def test_all_auxiliary_rejected_and_frozen_parameters_ignored():
    model = TinyModel()
    model.hidden.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="no eligible"):
        make_muon_optimizer(model, lr=0.001)


def test_zero_initialized_lora_factor_and_muon_only_resume():
    model = torch.nn.Module()
    model.register_parameter("lora_A", torch.nn.Parameter(torch.randn(2, 4)))
    model.register_parameter("lora_B", torch.nn.Parameter(torch.zeros(4, 2)))
    optimizer = make_muon_optimizer(model, lr=0.001)
    initial_a = model.lora_A.detach().clone()
    (model.lora_B @ model.lora_A).sum().backward()
    optimizer.step()
    assert torch.isfinite(model.lora_A).all() and torch.isfinite(model.lora_B).all()
    torch.testing.assert_close(model.lora_A, initial_a, rtol=0, atol=0)
    assert torch.count_nonzero(model.lora_B)
    saved = copy.deepcopy(optimizer.state_dict())
    restored = make_muon_optimizer(model, lr=0.001)
    restored.load_state_dict(saved)
    assert restored.auxiliary is None
    assert_same(saved, restored.state_dict())
