import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.periodic_heldout import (append_periodic_metric, evaluate_fixed_subset,
                                      periodic_metric_already_complete,
                                      read_periodic_metrics, select_fixed_subset,
                                      trainable_weights_sha256)
from scripts.train_lora import (batch_completion_loss, capture_rng_state,
                                collate_completions, restore_rng_state)


class TinyDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(self.dropout(x))


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 5)
        self.dropout = nn.Dropout(0.35)
        self.output = nn.Linear(5, 12)

    def forward(self, input_ids, labels=None, attention_mask=None, logits_to_keep=None):
        logits = self.output(self.dropout(self.embedding(input_ids)))
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        loss = None
        if labels is not None:
            shifted_labels = labels[:, 1:]
            loss = nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                shifted_labels.reshape(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)


def _training_update(model, optimizer, scheduler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x = torch.ones((4, 3))
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def test_periodic_eval_preserves_rng_mode_optimizer_and_next_training_update():
    torch.manual_seed(19)
    random.seed(29)
    expected = TinyDropoutModel()
    observed = copy.deepcopy(expected)
    expected_opt = torch.optim.AdamW(expected.parameters(), lr=0.01)
    observed_opt = torch.optim.AdamW(observed.parameters(), lr=0.01)
    expected_sched = torch.optim.lr_scheduler.LambdaLR(expected_opt, lambda step: 1.0 / (step + 1))
    observed_sched = torch.optim.lr_scheduler.LambdaLR(observed_opt, lambda step: 1.0 / (step + 1))

    training_rng = capture_rng_state()
    restore_rng_state(training_rng)
    _training_update(expected, expected_opt, expected_sched)
    _training_update(expected, expected_opt, expected_sched)
    expected_weights = copy.deepcopy(expected.state_dict())
    expected_optimizer = copy.deepcopy(expected_opt.state_dict())
    expected_scheduler = copy.deepcopy(expected_sched.state_dict())

    restore_rng_state(training_rng)
    _training_update(observed, observed_opt, observed_sched)
    eval_rng_before = capture_rng_state()
    rows = [
        {"id": "held-a", "task_family": "class-a", "source_ids": ["s1"], "source_groups": ["g1"]},
        {"id": "held-b", "task_family": "class-b", "source_ids": ["s2", "s3"], "source_groups": ["g2"]},
    ]

    def score(model, encoded):
        # Model evaluation and an independent random operation must not consume
        # randomness that changes the following training step.
        torch.rand(5)
        return float(model(encoded).square().sum().item()), 3

    result = evaluate_fixed_subset(
        model=observed, rows=rows, encode=lambda row: torch.ones((2, 3)),
        token_loss_sum=score, capture_rng=capture_rng_state,
        restore_rng=restore_rng_state, should_stop=lambda: False)
    assert result["evaluated_examples"] == 2
    assert result["supervised_tokens"] == 6
    assert result["class_support"] == {"class-a": 1, "class-b": 1}
    assert result["source_support"] == {"s1": 1, "s2": 1, "s3": 1}
    assert result["source_group_support"] == {"g1": 1, "g2": 1}
    assert observed.training
    assert torch.equal(torch.get_rng_state(), eval_rng_before["torch_cpu"])
    assert random.getstate() == eval_rng_before["python"]

    _training_update(observed, observed_opt, observed_sched)
    assert all(torch.equal(observed.state_dict()[key], value)
               for key, value in expected_weights.items())
    assert observed_opt.state_dict()["state"].keys() == expected_optimizer["state"].keys()
    for key, value in observed_opt.state_dict()["state"].items():
        for name, item in value.items():
            expected_item = expected_optimizer["state"][key][name]
            if isinstance(item, torch.Tensor):
                assert torch.equal(item, expected_item)
            else:
                assert item == expected_item
    assert observed_sched.state_dict() == expected_scheduler


def test_periodic_eval_stop_restores_rng_and_mode_without_record():
    torch.manual_seed(123)
    model = TinyDropoutModel()
    model.train()
    rng = capture_rng_state()
    result = evaluate_fixed_subset(
        model=model, rows=[{"id": "a"}], encode=lambda _: torch.rand((2, 3)),
        token_loss_sum=lambda m, x: (float(torch.rand(())), 1),
        capture_rng=capture_rng_state, restore_rng=restore_rng_state,
        should_stop=lambda: True)
    assert result is None
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng["torch_cpu"])


def test_periodic_metric_resume_identity_and_no_duplicate(tmp_path):
    model = TinyDropoutModel()
    weights = trainable_weights_sha256(model)
    path = tmp_path / "heldout-periodic.jsonl"
    item = {
        "schema": "natlang.periodic_heldout_loss/1", "status": "completed",
        "data_sha256": "d" * 64,
        "split_sha256": "s" * 64, "model_revision": "rev",
        "model_identity_sha256": "m" * 64, "step": 500,
        "subset_ids_sha256": "i" * 64, "trainable_weights_sha256": weights,
        "loss": 1.25, "supervised_tokens": 3,
    }
    append_periodic_metric(path, item)
    loaded = read_periodic_metrics(path, data_sha256="d" * 64,
                                   split_sha256="s" * 64, model_identity_sha256="m" * 64)
    assert loaded[(500, "i" * 64)] == item
    assert periodic_metric_already_complete(loaded, step=500,
                                            subset_ids_sha256="i" * 64,
                                            trainable_weights_sha256=weights)
    assert not periodic_metric_already_complete(loaded, step=500,
                                               subset_ids_sha256="different",
                                               trainable_weights_sha256=weights)
    with pytest.raises(ValueError, match="does not match checkpoint weights"):
        periodic_metric_already_complete(loaded, step=500,
                                         subset_ids_sha256="i" * 64,
                                         trainable_weights_sha256="wrong")
    with pytest.raises(ValueError, match="already exists"):
        append_periodic_metric(path, item)
    with pytest.raises(ValueError, match="another run identity"):
        read_periodic_metrics(path, data_sha256="x" * 64,
                              split_sha256="s" * 64, model_identity_sha256="m" * 64)


def test_periodic_subset_is_hash_ranked_not_input_order():
    rows = [{"id": f"row-{i}"} for i in range(25)]
    first = select_fixed_subset(rows, 8, 42)
    second = select_fixed_subset(list(reversed(rows)), 8, 42)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert first == select_fixed_subset(rows, 8, 42)


def test_production_single_example_loss_adapter_counts_only_masked_completion_tokens():
    torch.manual_seed(7)
    model = TinyCausalLM().eval()
    example = ([1, 2, 3], [4, 5])
    encoded = collate_completions([example], pad_id=0, device="cpu")
    token_count = int((encoded["labels"][:, 1:] != -100).sum().item())
    mean_loss = batch_completion_loss(model, encoded)
    logits = model(input_ids=encoded["input_ids"],
                   attention_mask=encoded["attention_mask"],
                   logits_to_keep=encoded["labels"].shape[-1]).logits[:, :-1].float()
    labels = encoded["labels"][:, 1:]
    manual_sum = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=-100, reduction="sum")
    assert token_count == 2
    torch.testing.assert_close(mean_loss * token_count, manual_sum)


def test_production_adapter_periodic_eval_preserves_next_causal_update_and_scheduler():
    torch.manual_seed(41)
    random.seed(43)
    expected = TinyCausalLM()
    observed = copy.deepcopy(expected)
    expected_opt = torch.optim.AdamW(expected.parameters(), lr=0.005)
    observed_opt = torch.optim.AdamW(observed.parameters(), lr=0.005)
    expected_sched = torch.optim.lr_scheduler.LambdaLR(expected_opt, lambda step: 1.0 / (step + 1))
    observed_sched = torch.optim.lr_scheduler.LambdaLR(observed_opt, lambda step: 1.0 / (step + 1))
    row = {"id": "eval-row", "task_family": "tiny", "source_ids": ["source-a"]}
    example = ([1, 2, 3], [4, 5])

    def update(model, optimizer, scheduler):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = batch_completion_loss(model, collate_completions([example], pad_id=0, device="cpu"))
        loss.backward()
        optimizer.step()
        scheduler.step()

    training_rng = capture_rng_state()
    update(expected, expected_opt, expected_sched)
    update(expected, expected_opt, expected_sched)
    expected_weights = copy.deepcopy(expected.state_dict())
    expected_opt_state = copy.deepcopy(expected_opt.state_dict())
    expected_sched_state = copy.deepcopy(expected_sched.state_dict())

    restore_rng_state(training_rng)
    update(observed, observed_opt, observed_sched)
    result = evaluate_fixed_subset(
        model=observed, rows=[row],
        encode=lambda _: collate_completions([example], pad_id=0, device="cpu"),
        token_loss_sum=lambda model, encoded: (
            float(batch_completion_loss(model, encoded).item()) *
            int((encoded["labels"][:, 1:] != -100).sum().item()),
            int((encoded["labels"][:, 1:] != -100).sum().item())),
        capture_rng=capture_rng_state, restore_rng=restore_rng_state,
        should_stop=lambda: False)
    assert result["supervised_tokens"] == 2
    update(observed, observed_opt, observed_sched)
    assert all(torch.equal(observed.state_dict()[key], value)
               for key, value in expected_weights.items())
    assert observed_opt.state_dict()["param_groups"] == expected_opt_state["param_groups"]
    for key, value in observed_opt.state_dict()["state"].items():
        for name, item in value.items():
            expected_item = expected_opt_state["state"][key][name]
            if isinstance(item, torch.Tensor):
                assert torch.equal(item, expected_item)
            else:
                assert item == expected_item
    assert observed_sched.state_dict() == expected_sched_state


def test_partial_metric_tail_is_preserved_and_due_step_can_be_retried(tmp_path):
    path = tmp_path / "heldout-periodic.jsonl"
    item = {
        "schema": "natlang.periodic_heldout_loss/1", "status": "completed",
        "data_sha256": "d" * 64, "split_sha256": "s" * 64,
        "model_identity_sha256": "m" * 64, "step": 500,
        "subset_ids_sha256": "i" * 64, "trainable_weights_sha256": "w" * 64,
    }
    append_periodic_metric(path, item)
    partial = b'{"schema":"natlang.periodic_heldout_loss/1","step":1000'
    with path.open("ab") as stream:
        stream.write(partial)
        stream.flush()
    loaded = read_periodic_metrics(path, data_sha256="d" * 64,
                                   split_sha256="s" * 64, model_identity_sha256="m" * 64)
    assert list(loaded) == [(500, "i" * 64)]
    assert path.read_bytes() == (json.dumps(item, ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":")) + "\n").encode()
    recovery_files = list(tmp_path.glob("*.partial-*.recovered"))
    assert len(recovery_files) == 1
    assert recovery_files[0].read_bytes() == partial
