import importlib.util
import json
import importlib.util
from pathlib import Path

import torch


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


def test_file_signature_requires_complete_file(tmp_path):
    path = tmp_path / "checkpoint.pt"
    assert WATCH.file_signature(path) is None
    path.write_bytes(b"checkpoint")
    signature = WATCH.file_signature(path)
    assert signature == (10, path.stat().st_mtime_ns)


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
