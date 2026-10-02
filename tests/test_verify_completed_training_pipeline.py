import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_completed_training_pipeline import verify


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    run = tmp_path / "parent-run"
    run.mkdir()
    source = tmp_path / "input.jsonl"
    script = tmp_path / "builder.py"
    output = run / "output.jsonl"
    source.write_text('{"input":1}\n')
    script.write_text("# current helper\n")
    output.write_text('{"output":1}\n')
    config = {"version": "natlang.training_pipeline/1", "run_directory": str(run)}
    config_path = tmp_path / "parent.json"
    config_path.write_text(json.dumps(config))
    state = {
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "status": "complete",
        "stages": {"native-reference": {
            "status": "complete", "exit_code": 0,
            "inputs": {str(source): _sha(source), str(script): "0" * 64},
            "outputs": {str(output): _sha(output)},
        }},
    }
    (run / "pipeline-state.json").write_text(json.dumps(state))
    return config_path, run, source, output


def test_verifies_completed_parent_and_artifact_hashes(tmp_path):
    config, run, _, _ = _fixture(tmp_path)
    proof = verify(config, run, ["native-reference"])
    assert proof["verified"] is True
    assert proof["required_stages"]["native-reference"]["exit_code"] == 0
    assert proof["historical_code_input_deltas"]


@pytest.mark.parametrize("which", ["input", "output"])
def test_rejects_changed_parent_inputs_or_outputs(tmp_path, which):
    config, run, source, output = _fixture(tmp_path)
    target = source if which == "input" else output
    target.write_text(target.read_text() + "changed\n")
    with pytest.raises(ValueError, match="changed|missing"):
        verify(config, run, ["native-reference"])


def test_rejects_incomplete_parent_state(tmp_path):
    config, run, _, _ = _fixture(tmp_path)
    state_path = run / "pipeline-state.json"
    state = json.loads(state_path.read_text())
    state["status"] = "running"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="not complete"):
        verify(config, run, ["native-reference"])
