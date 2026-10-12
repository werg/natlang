import importlib.util
import json
import importlib.util
from pathlib import Path

import torch
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_fp8_checkpoint_resume.py"
SPEC = importlib.util.spec_from_file_location("fp8_checkpoint_resume", SCRIPT)
WATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCH)
COMPARATOR_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_fp8_training_checkpoints.py"
COMPARATOR_SPEC = importlib.util.spec_from_file_location("compare_fp8_training_checkpoints", COMPARATOR_SCRIPT)
COMPARATOR = importlib.util.module_from_spec(COMPARATOR_SPEC)
COMPARATOR_SPEC.loader.exec_module(COMPARATOR)


def test_train_steps_ignores_partial_or_malformed_jsonl(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"step": 1}) + "\n" + "{\"step\":2" + "\n" +
                    json.dumps({"step": 3}) + "\n" + json.dumps({"step": True}) + "\n")
    assert WATCH.train_steps(path) == [1, 3]


def test_expected_signal_exit_requires_the_durable_requested_step():
    complete = {"status": "checkpointed_on_signal", "steps": 4}
    assert WATCH.expected_checkpoint_interrupt(143, complete, 3)
    assert not WATCH.expected_checkpoint_interrupt(143, {"status": "failed", "steps": 4}, 3)
    assert not WATCH.expected_checkpoint_interrupt(143, {"status": "checkpointed_on_signal", "steps": 2}, 3)
    assert not WATCH.expected_checkpoint_interrupt(0, complete, 3)


def test_file_signature_requires_complete_file(tmp_path):
    path = tmp_path / "checkpoint.pt"
    assert WATCH.file_signature(path) is None
    path.write_bytes(b"checkpoint")
    signature = WATCH.file_signature(path)
    assert signature == (10, path.stat().st_mtime_ns)


def test_runtime_handoff_is_needed_only_until_plan_pins_current_code(tmp_path):
    path = tmp_path / "direct-stage-plan.json"
    path.write_text(json.dumps({"frozen_code": {"train/direct_sft.py": "a" * 64}}))
    assert WATCH.runtime_handoff_needed(path, {"train/direct_sft.py": "b" * 64})
    assert not WATCH.runtime_handoff_needed(path, {"train/direct_sft.py": "a" * 64})
    path.write_text(json.dumps({"frozen_code": []}))
    with pytest.raises(ValueError, match="frozen code inventory"):
        WATCH.runtime_handoff_needed(path, {})


def test_training_checkpoint_load_allowlists_torch_version_only(tmp_path):
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    path = tmp_path / "training.pt"
    torch.save({"version": torch.torch_version.TorchVersion("2.14.0"),
                "weight": torch.tensor([1.0])}, path)
    state = CheckpointPolicy.load(path)
    assert str(state["version"]) == "2.14.0"
    assert torch.equal(state["weight"], torch.tensor([1.0]))


def test_watcher_receipt_write_is_atomic_and_replaces_progress(tmp_path):
    path = tmp_path / "receipt.json"
    WATCH.atomic_receipt(path, {"status": "running", "step": 3})
    WATCH.atomic_receipt(path, {"status": "complete", "step": 12})
    assert json.loads(path.read_text()) == {"status": "complete", "step": 12}
    assert not path.with_name(path.name + ".tmp").exists()


def test_checkpoint_comparison_reports_all_training_state_families(tmp_path):
    state = {
        "schema": "natlang.neuralese-direct/1", "step": 12,
        "student_parameters": {"layer.weight": torch.tensor([1.0, 2.0])},
        "neuralese_projection": {"weight": torch.tensor([3.0])},
        "optimizer": {"state": {0: {"momentum": torch.tensor([4.0])}}, "param_groups": []},
        "rng": {"cpu": torch.tensor([1, 2], dtype=torch.uint8)},
        "fp8_training": {"scale": torch.tensor([0.5])},
    }
    left, right = tmp_path / "left.pt", tmp_path / "right.pt"
    torch.save(state, left)
    torch.save(state, right)
    result = COMPARATOR.compare(left, right)
    assert result["left_step"] == result["right_step"] == 12
    assert result["families"]["fp8_training"]["exact_tensor_entries"] == 1
    assert result["families"]["optimizer"]["exact_tensor_entries"] == 1


def test_checkpoint_comparison_compares_numpy_rng_and_ignores_only_output_path(tmp_path):
    import numpy as np

    common = {"schema": "natlang.neuralese-direct/1", "step": 12,
              "identity": {"options": {"out": "/arm/a", "seed": 5}, "inputs": {"data": "abc"}},
              "rng": {"numpy": ("MT19937", np.array([1, 2, 3], dtype=np.uint32), 4, 0, 0.0)}}
    left, right = tmp_path / "left-rng.pt", tmp_path / "right-rng.pt"
    torch.save(common, left)
    other = dict(common)
    other["identity"] = {"options": {"out": "/arm/b", "seed": 5}, "inputs": {"data": "abc"}}
    torch.save(other, right)
    result = COMPARATOR.compare(left, right)
    assert result["identity_match"] == {"options_match_ignoring_output_path": True, "inputs_match": True}
    assert result["families"]["rng"]["all_entries_exact"] is True
