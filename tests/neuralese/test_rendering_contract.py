"""Behavioural prompt-rendering identity (/2): refactors keep it, rendering changes alter it, /1 exports restore."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
sys.path.insert(0, str(ROOT / "scripts" / "neuralese_data"))
import prompt_rendering_equivalence as equivalence  # noqa: E402
from natlang_neuralese.data import prompt_rendering, rendering_contract  # noqa: E402
from natlang_neuralese.serve import chat  # noqa: E402

# Pop's LFM step-10006 export (runs/pop-lfm-muon-main-v2-export-step10006/manifest.json) and step 9000: bound by /1
# at 058a5269's sources, before 07630d34 changed prompt_rendering.py's bytes.
POP_U10006_V1 = "420ac1c71b622fe05c4c954832db830c5bf2143353e6937692fabe8de86e478e"


def current_blobs():
    package = ROOT / equivalence.PACKAGE
    return {name: (package / name).read_bytes() for name in equivalence.V1_SOURCES}


def test_unrelated_edit_keeps_identity():
    current = {mode: rendering_contract.compute_identity(mode)["identity_sha256"] for mode in equivalence.MODES}

    def edit(package):
        for name in equivalence.V1_SOURCES:
            path = package / name
            path.write_text(path.read_text() + "\n\ndef unrelated_helper(x):\n    return [x, x]\n")
    edited = equivalence.v2_identities(current_blobs(), (ROOT / equivalence.PIECES).read_bytes(), edit)
    assert {mode: edited[mode]["identity_sha256"] for mode in equivalence.MODES} == current


@pytest.mark.parametrize("module, name, replacement", [
    (prompt_rendering, "render_system_prompt", lambda text, mode="full", pieces=(): text if mode == "full" else text[:200]),
    (chat, "escape_specials", lambda text, specials, nonce="": text),
])
def test_rendering_change_alters_identity(monkeypatch, module, name, replacement):
    before = {mode: rendering_contract.compute_identity(mode)["identity_sha256"] for mode in equivalence.MODES}
    monkeypatch.setattr(module, name, replacement)
    after = {mode: rendering_contract.compute_identity(mode)["identity_sha256"] for mode in equivalence.MODES}
    assert after["program-header"] != before["program-header"]
    if name == "escape_specials":
        assert after["full"] != before["full"]


def test_piece_table_change_alters_identity(monkeypatch, tmp_path):
    before = rendering_contract.compute_identity("program-header")["identity_sha256"]
    pieces = tmp_path / "pieces.json"
    pieces.write_bytes(rendering_contract.PIECES_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(rendering_contract, "PIECES_PATH", pieces)
    assert rendering_contract.compute_identity("program-header")["identity_sha256"] != before


def test_old_v1_export_binding_restores():
    for mode in equivalence.MODES:
        assert prompt_rendering.prompt_rendering_binding_accepted(
            prompt_rendering.prompt_rendering_identity(mode)["identity_sha256"], mode)
    assert prompt_rendering.prompt_rendering_binding_accepted(POP_U10006_V1, "program-header")
    assert not prompt_rendering.prompt_rendering_binding_accepted(POP_U10006_V1, "full")
    assert not prompt_rendering.prompt_rendering_binding_accepted("0" * 64, "program-header")
    # Evaluation comparison keys recorded under /1 keep comparing while behaviour is unchanged.
    assert rendering_contract.comparison_binding("program-header")["identity_sha256"] == POP_U10006_V1


def test_pre_95bc6eca_binding_restores_only_through_its_declared_adoption(monkeypatch):
    # 95bc6eca (program-header serving keeps supplied tool schemas) is a real rendering change ...
    entry = json.loads(rendering_contract.EQUIVALENCE_PATH.read_text())["identities"][POP_U10006_V1]
    assert entry["v2_identity_sha256"] != rendering_contract.behavioural_identity("program-header")["identity_sha256"]
    # ... that the table declares applies to artifacts bound before it.
    assert prompt_rendering.prompt_rendering_binding_accepted(POP_U10006_V1, "program-header")
    monkeypatch.setattr(rendering_contract, "adoptions", lambda: [])
    assert not prompt_rendering.prompt_rendering_binding_accepted(POP_U10006_V1, "program-header")
    # Full mode did not change: its /1 bindings map directly.
    full = [k for k, v in json.loads(rendering_contract.EQUIVALENCE_PATH.read_text())["identities"].items()
            if v["mode"] == "full"]
    assert full and all(prompt_rendering.prompt_rendering_binding_accepted(k, "full") for k in full)


def test_old_v1_binding_refused_when_behaviour_changed(monkeypatch):
    monkeypatch.setattr(prompt_rendering, "render_system_prompt", lambda text, mode="full", pieces=(): "")
    monkeypatch.setattr(rendering_contract, "_CACHE", {})
    assert not prompt_rendering.prompt_rendering_binding_accepted(POP_U10006_V1, "program-header")


def test_equivalence_table_reverifies_from_git():
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", "058a5269^{commit}"],
                      capture_output=True).returncode:
        pytest.skip("history of the /1 sources is not available in this checkout")
    data = json.loads(rendering_contract.EQUIVALENCE_PATH.read_text())
    table = data["identities"]
    assert data["adoptions"] == equivalence.adoption_entries()
    assert POP_U10006_V1 in table
    commits = sorted({entry["commit"] for entry in table.values()})
    for commit in commits:
        blobs, pieces = equivalence.blobs_at(commit)
        v2 = equivalence.v2_identities(blobs, pieces)
        for mode in equivalence.MODES:
            v1 = equivalence.v1_identity(blobs, pieces, mode)
            entry = table[v1["identity_sha256"]]
            assert entry["identity"] == v1 and entry["mode"] == mode
            assert entry["v2_identity_sha256"] == v2[mode]["identity_sha256"]
