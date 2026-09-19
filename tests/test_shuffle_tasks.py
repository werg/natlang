import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "shuffle_tasks", Path(__file__).resolve().parents[1] / "scripts" / "shuffle_tasks.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_disk_shuffle_is_deterministic_and_preserves_records(tmp_path):
    rows = [{"id": f"id-{i}", "split": "train", "state": str(i)} for i in range(30)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    manifest = module.shuffle(source, first, "seed")
    module.shuffle(source, second, "seed")
    assert first.read_bytes() == second.read_bytes()
    assert {json.loads(line)["id"] for line in first.read_text().splitlines()} == {row["id"] for row in rows}
    assert manifest["counts"] == {"train": 30}
    assert not first.with_suffix(".jsonl.sort.sqlite").exists()
