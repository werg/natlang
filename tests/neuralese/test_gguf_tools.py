import json
from pathlib import Path

from natlang_neuralese.common import gguf_tools


def test_gguf_tool_root_uses_pin_when_no_override(monkeypatch):
    monkeypatch.delenv('NATLANG_LLAMA_NEURALESE', raising=False)
    expected = Path(json.loads(gguf_tools.PIN_FILE.read_text())['path'])
    if not expected.is_absolute():
        expected = (gguf_tools.PIN_FILE.parents[2] / expected).resolve()
    assert gguf_tools.llama_cpp_root() == expected


def test_gguf_tool_root_honors_explicit_toolchain_override(monkeypatch, tmp_path):
    monkeypatch.setenv('NATLANG_LLAMA_NEURALESE', str(tmp_path / 'toolchain'))
    assert gguf_tools.llama_cpp_root() == (tmp_path / 'toolchain').resolve()
