import gzip
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/training_resource_preflight.py"


def invoke(tmp_path, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          text=True, capture_output=True, check=False)


def test_resource_floor_succeeds_with_exact_small_inputs(tmp_path):
    plain = tmp_path / "turns.jsonl"
    plain.write_bytes(b'{"x":1}\n')
    output = tmp_path / "ready.json"
    result = invoke(tmp_path, "--path", tmp_path, "--input", plain,
                    "--multiplier", "1", "--reserve-bytes", "0", "--output", output)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["ready"] is True
    assert report["input_bytes"] == len(b'{"x":1}\n')


def test_resource_floor_failure_persists_blocking_report(tmp_path):
    plain = tmp_path / "turns.jsonl"
    plain.write_bytes(b'{"x":1}\n')
    output = tmp_path / "blocked.json"
    result = invoke(tmp_path, "--path", tmp_path, "--input", plain,
                    "--multiplier", "1", "--reserve-bytes", "999999999999",
                    "--output", output)
    assert result.returncode == 75
    report = json.loads(output.read_text())
    assert report["ready"] is False
    assert report["required_free_bytes"] > report["available_free_bytes"]


def test_gzip_expansion_and_materialization_growth_are_explicit(tmp_path):
    plain = tmp_path / "source.jsonl.gz"
    payload = b'{"value":"visible evidence"}\n' * 11
    with gzip.open(plain, "wb") as stream:
        stream.write(payload)
    output = tmp_path / "expanded.json"
    result = invoke(tmp_path, "--path", tmp_path, "--input", plain,
                    "--gzip-expanded-input", plain,
                    "--extra-estimated-input-bytes", "1234",
                    "--estimate-note", "representative materialization-growth scenario",
                    "--multiplier", "1", "--reserve-bytes", "0", "--output", output)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["gzip_expanded_inputs"][0]["expanded_bytes"] == len(payload)
    assert report["estimated_materialization_growth_bytes"] == 1234
    assert report["input_bytes"] == plain.stat().st_size + len(payload) + 1234
    assert report["estimate_is_peak_guarantee"] is False
