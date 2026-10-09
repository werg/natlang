import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/neuralese"))


import pytest  # noqa: E402
from natlang_neuralese.common import paths # noqa: E402


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for name in paths.DEFAULTS:
        monkeypatch.delenv(f"NATLANG_{name.upper()}", raising=False)
    monkeypatch.setenv(paths.PROFILE_ENV, str(tmp_path / "machine.toml"))


def test_defaults_are_todays_literals():
    assert paths.resolve_str("models", "maple-preview-bf16") == "/home/werg/data/models/maple-preview-bf16"
    assert paths.resolve_str("llama_cpp", "gguf-py") == "/home/werg/llama.cpp-neuralese/gguf-py"
    assert str(paths.root("data_hdd")) == "/mnt/external"
    assert str(paths.root("repo")) == "/home/werg/natlang"


def test_profile_overrides_default(tmp_path):
    (tmp_path / "machine.toml").write_text('models = "/x/models"\n')
    assert str(paths.root("models")) == "/x/models"
    (tmp_path / "machine.toml").write_text('[roots]\narchive = "/a"\n')
    assert str(paths.root("archive")) == "/a"


def test_env_beats_profile(tmp_path, monkeypatch):
    (tmp_path / "machine.toml").write_text('models = "/x/models"\n')
    monkeypatch.setenv("NATLANG_MODELS", "/env/models")
    assert str(paths.root("models")) == "/env/models"


def test_bad_profile_ignored(tmp_path):
    (tmp_path / "machine.toml").write_text("not = [toml")
    assert str(paths.root("models")) == "/home/werg/data/models"


def test_unknown_root():
    with pytest.raises(KeyError):
        paths.root("nope")
