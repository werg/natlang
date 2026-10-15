import importlib.util
from pathlib import Path
import subprocess
import sys


def test_render_parts_stream_reads_relocated_source(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts' / 'neuralese_data'))
    sys.path.insert(0, str(root / 'training' / 'neuralese'))
    spec = importlib.util.spec_from_file_location('render_parts_symlink_test', root / 'scripts' / 'neuralese_data' / 'render_parts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b'{"id":"one"}\n{"id":"two"}\n'
    physical = tmp_path / 'physical.jsonl.zst'
    physical.write_bytes(subprocess.run(['zstd', '-q', '-c'], input=payload, capture_output=True, check=True).stdout)
    logical = tmp_path / 'logical.jsonl.zst'
    logical.symlink_to(physical)
    assert b''.join(module.stream([logical])) == payload
    assert logical.is_symlink()
