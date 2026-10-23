"""The active native prompt-rendering contract is behaviorally fingerprinted."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
from natlang_neuralese.data import prompt_rendering, rendering_contract  # noqa: E402
from natlang_neuralese.serve import chat  # noqa: E402


def test_active_identity_binds_mode_fixtures_and_behavior():
    for mode in prompt_rendering.PROMPT_RENDERINGS:
        identity = prompt_rendering.prompt_rendering_identity(mode)
        assert identity == rendering_contract.compute_identity(mode)
        assert identity["schema"] == rendering_contract.SCHEMA
        assert identity["mode"] == mode
        assert identity["fixtures_sha256"] == hashlib.sha256(
            rendering_contract.FIXTURES_PATH.read_bytes()).hexdigest()
        assert identity["pieces_sha256"] == hashlib.sha256(
            rendering_contract.PIECES_PATH.read_bytes()).hexdigest()
        assert identity["identity_sha256"] == hashlib.sha256("\0".join((
            identity["schema"], mode, identity["fixtures_sha256"],
            identity["rendered_sha256"], identity["pieces_sha256"])).encode()).hexdigest()


@pytest.mark.parametrize("module, name, replacement", [
    (prompt_rendering, "render_system_prompt", lambda text, mode="full", pieces=(): text if mode == "full" else text[:200]),
    (chat, "escape_specials", lambda text, specials, nonce="": text),
])
def test_rendering_change_alters_identity(monkeypatch, module, name, replacement):
    before = {mode: rendering_contract.compute_identity(mode)["identity_sha256"]
              for mode in prompt_rendering.PROMPT_RENDERINGS}
    monkeypatch.setattr(module, name, replacement)
    after = {mode: rendering_contract.compute_identity(mode)["identity_sha256"]
             for mode in prompt_rendering.PROMPT_RENDERINGS}
    assert after["program-header"] != before["program-header"]
    if name == "escape_specials":
        assert after["full"] != before["full"]


def test_piece_table_change_alters_identity(monkeypatch, tmp_path):
    before = rendering_contract.compute_identity("program-header")["identity_sha256"]
    pieces = tmp_path / "pieces.json"
    pieces.write_bytes(rendering_contract.PIECES_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(rendering_contract, "PIECES_PATH", pieces)
    assert rendering_contract.compute_identity("program-header")["identity_sha256"] != before


def test_current_rendering_fixtures_are_valid_json_and_include_both_modes():
    fixtures = json.loads(rendering_contract.FIXTURES_PATH.read_text(encoding="utf-8"))
    assert fixtures["conversations"] and fixtures["system_prompts"]
    assert set(prompt_rendering.PROMPT_RENDERINGS) == {"full", "program-header"}
