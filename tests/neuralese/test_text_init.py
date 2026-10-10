"""In-context text initialisation of soft bodies (text_init.py, owner 2026-10-10): the soft call with the initialised body
reproduces the text-instructed call exactly on heads whose read transport is the identity, the legacy RMS profile is
reported as inexact, and the bare-encode initialisation is gone from the library builder."""

import pytest
import torch

from natlang_neuralese.serve.store import TensorStore

DIALECT = "nd:natlang@1"
TEXT = "Translate the text into French and keep the names unchanged."


def _engine(loaded, profile):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.serve.engine import Engine

    _, tokenizer, backbone = loaded
    torch.manual_seed(1)
    heads = PortHeads(backbone, cutoff=6, max_length=8, profile=profile).eval()
    for p in heads.parameters():
        p.requires_grad_(False)
    return Engine(backbone, heads, tokenizer, TensorStore(), DIALECT, max_block=4)


def _soft_call(placeholder, after=" Answer with return_result."):
    return [{"role": "system", "content": [{"type": "text", "text": "You run one function.\n\nInstructions:\n"},
                                           {"type": "neuralese", "id": placeholder},
                                           {"type": "text", "text": after}]},
            {"role": "user", "content": "text: Anna meets Paul in Lyon on Tuesday."}]


@pytest.fixture(scope="module")
def raw_engine(loaded):
    return _engine(loaded, "raw-token-v1")


def test_raw_token_body_reproduces_the_text_instructed_call_exactly(raw_engine):
    from natlang_neuralese.serve.grad import embed_text
    from natlang_neuralese.text_init import init_gate, instruction_body

    placeholder = embed_text(raw_engine, "placeholder", "Neuralese<(text: string) => string>").id
    block, info = instruction_body(raw_engine, _soft_call(placeholder), None, placeholder, TEXT,
                                   "Neuralese<(text: string) => string>")
    assert info["exact"] and info["aligned"] and info["exact_transport"]
    assert block.producer["kind"] == "text-init-in-context"
    gate = init_gate(raw_engine, _soft_call(block.id), None, block.id, TEXT, reply_tokens=4)
    assert gate["passed"] and gate["agreement"] == 1.0 and gate["bit_exact"], gate


def test_a_merged_boundary_token_is_near_exact_and_gated(raw_engine):
    """The text call tokenizes the instructions with the next text run (`.` + `\n` -> `.\n`); the soft call cannot
    merge across its block, so the body is the span's own tokens and the gate measures the boundary's effect."""
    from natlang_neuralese.serve.grad import embed_text
    from natlang_neuralese.text_init import init_gate, instruction_body

    after = "\nAnswer with return_result."
    placeholder = embed_text(raw_engine, "placeholder").id
    block, info = instruction_body(raw_engine, _soft_call(placeholder, after), None, placeholder, TEXT)
    assert info["exact_transport"] and not info["aligned"] and not info["exact"]
    gate = init_gate(raw_engine, _soft_call(block.id, after), None, block.id, TEXT, reply_tokens=4)
    assert gate["positions"] >= 1 and 0.0 <= gate["kl"] and "passed" in gate


def test_legacy_rms_profile_is_reported_inexact(loaded):
    from natlang_neuralese.serve.grad import embed_text
    from natlang_neuralese.text_init import instruction_body

    from natlang_neuralese.text_init import init_gate

    engine = _engine(loaded, "legacy-rms-v1")
    placeholder = embed_text(engine, "placeholder").id
    block, info = instruction_body(engine, _soft_call(placeholder), None, placeholder, TEXT)
    assert info["exact_transport"] is False and info["exact"] is False
    gate = init_gate(engine, _soft_call(block.id), None, block.id, TEXT, reply_tokens=2)
    assert not gate["bit_exact"], "read markers and the interface norm change what the call reads"


def test_the_placeholder_must_appear_once(raw_engine):
    from natlang_neuralese.serve.grad import embed_text
    from natlang_neuralese.text_init import instruction_body

    placeholder = embed_text(raw_engine, "placeholder").id
    other = embed_text(raw_engine, "other").id
    with pytest.raises(ValueError, match="exactly once"):
        instruction_body(raw_engine, _soft_call(other), None, placeholder, TEXT)


def test_library_builder_has_no_bare_encode_method(raw_engine, tmp_path):
    from natlang_neuralese.stdlib import build_text_library

    with pytest.raises(ValueError, match="unknown text initialisation"):
        build_text_library(raw_engine, tmp_path / "x.nz", method="encode")
    with pytest.raises(ValueError, match="rendered call"):
        build_text_library(raw_engine, tmp_path / "x.nz")
