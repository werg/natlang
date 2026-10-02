import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/concat_jsonl_stream.py"


def run(output: Path, *inputs: Path):
    return subprocess.run([sys.executable, str(SCRIPT), "--output", str(output), *map(str, inputs)],
                          text=True, capture_output=True, check=False)


def test_concatenates_in_input_order_and_adds_missing_final_newline(tmp_path):
    first, second, output = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "out.jsonl"
    first.write_bytes(b'{"text":"caf\xc3\xa9"}\n')
    second.write_bytes(b'{"n":2}')
    result = run(output, first, second)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == b'{"text":"caf\xc3\xa9"}\n{"n":2}\n'
    assert json.loads(result.stdout)["rows"] == 2


def test_malformed_row_keeps_existing_output_and_cleans_temporary_file(tmp_path):
    good, bad, output = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "out.jsonl"
    good.write_text('{"ok":true}\n')
    bad.write_text('[]\n')
    output.write_text("prior committed data\n")
    result = run(output, good, bad)
    assert result.returncode != 0
    assert output.read_text() == "prior committed data\n"
    assert list(tmp_path.glob("out.jsonl.*.pending")) == []
