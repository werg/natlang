import json
from pathlib import Path
import subprocess
import sys

from scripts.prepare_training_stages import prepare_streaming


def _write_rows(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_repartition_unfrozen_train_groups_preserves_group_and_reserved_test_splits(tmp_path):
    rows = [
        {"id": f"row-{i}", "program_id": f"group-{i:03d}", "split": "train",
         "prompt": f"prompt-{i}", "completion": f"answer-{i}"}
        for i in range(200)
    ]
    # These two source-linked rows must make the same split even though each
    # arrived with the default explicit train marker.
    rows.extend([
        {"id": "linked-a", "program_id": "linked-a", "source_groups": ["joint"], "split": "train",
         "prompt": "prompt-linked-a", "completion": "answer-linked-a"},
        {"id": "linked-b", "program_id": "linked-b", "source_groups": ["joint"], "split": "train",
         "prompt": "prompt-linked-b", "completion": "answer-linked-b"},
        {"id": "explicit-test", "program_id": "explicit-test", "split": "test",
         "prompt": "prompt-explicit-test", "completion": "answer-explicit-test"},
        {"id": "reserved", "program_id": "s102:probe:1", "split": "train",
         "prompt": "prompt-reserved", "completion": "answer-reserved"},
        {"id": "fixed-train", "program_id": "fixed-train", "split": "train",
         "prompt": "prompt-fixed-train", "completion": "answer-fixed-train"},
        {"id": "fixed-test", "program_id": "fixed-test", "split": "train",
         "prompt": "prompt-fixed-test", "completion": "answer-fixed-test"},
    ])
    source = tmp_path / "teacher.jsonl"
    _write_rows(source, rows)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"groups": {"fixed-train": "train", "fixed-test": "test"}}))
    out = tmp_path / "out"

    manifest = prepare_streaming(out, teacher_paths=[source], registry=registry,
                                 seed=42, repartition_unfrozen_train=True)
    prepared = [json.loads(line) for line in (out / "teacher.jsonl").read_text().splitlines()]
    by_id = {row["id"]: row for row in prepared}

    assert manifest["identity"]["train_split_policy"] == "hash-unfrozen-explicit-train-groups"
    assert by_id["explicit-test"]["split"] == "test"
    assert by_id["reserved"]["split"] == "test"
    assert by_id["fixed-train"]["split"] == "train"
    assert by_id["fixed-test"]["split"] == "test"
    assert by_id["linked-a"]["split"] == by_id["linked-b"]["split"]
    random_test = [row for row in prepared if row["id"].startswith("row-") and row["split"] == "test"]
    assert 2 <= len(random_test) <= 25


def test_repartition_policy_is_bound_into_immutable_output_identity(tmp_path):
    source = tmp_path / "teacher.jsonl"
    _write_rows(source, [{"id": "row", "program_id": "program", "split": "train"}])
    out = tmp_path / "out"

    prepare_streaming(out, teacher_paths=[source], seed=42, repartition_unfrozen_train=True)
    try:
        prepare_streaming(out, teacher_paths=[source], seed=42, repartition_unfrozen_train=False)
    except ValueError as exc:
        assert "inputs/settings changed" in str(exc)
    else:
        raise AssertionError("changing split policy must require a new output directory")


def test_streaming_cli_accepts_repartition_group_flag(tmp_path):
    source = tmp_path / "teacher.jsonl"
    _write_rows(source, [
        {"id": f"row-{i}", "program_id": f"g-{i}", "split": "train",
         "prompt": f"prompt-{i}", "completion": f"answer-{i}"}
        for i in range(80)
    ])
    output = tmp_path / "cli-out"
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_training_stages.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output), "--teacher", str(source),
         "--streaming", "--repartition-unfrozen-train-groups"],
        check=True, capture_output=True, text=True)
    assert "hash-unfrozen-explicit-train-groups" in (output / "manifest.json").read_text()
    prepared = [json.loads(line) for line in (output / "teacher.jsonl").read_text().splitlines()]
    assert any(row["split"] == "test" for row in prepared)
