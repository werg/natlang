import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_fp8_checkpoint_resume.py"
SPEC = importlib.util.spec_from_file_location("fp8_checkpoint_resume", SCRIPT)
WATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCH)


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
    WATCH.write_json(path, {"status": "running", "step": 3})
    WATCH.write_json(path, {"status": "complete", "step": 12})
    assert json.loads(path.read_text()) == {"status": "complete", "step": 12}
    assert not path.with_name(path.name + ".tmp").exists()
