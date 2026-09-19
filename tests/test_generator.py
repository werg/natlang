"""The synthetic generator and reference policy: every family runs through the real harness, reaches the
expected value, and every emitted turn is accepted by the grammar of its turn."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate import run_program  # noqa: E402

from natlang.gen.programs import FAMILIES


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_generates_valid_trajectories(family):
    rng = random.Random(7)
    for _ in range(3):
        samples, episodes = run_program(FAMILIES[family](rng))     # asserts value == expected, grammar accepts
        assert samples and samples[-1]["skill"] == "reply" or any(s["skill"] == "reply" for s in samples)
        for s in samples:
            assert s["messages"][0]["role"] == "system" and s["messages"][1]["role"] == "user"
            user = s["messages"][1]["content"]
            assert "Workspace" not in user                          # data never rides in the user message


def test_dependent_calls_are_in_separate_turns():
    rng = random.Random(3)
    prog = FAMILIES["map_then_count"](rng)
    samples, _ = run_program(prog)
    root_task = samples[0]["messages"][1]["content"]           # the root episode comes first, whatever its wording
    root_turns = [s["skill"] for s in samples if s["messages"][1]["content"] == root_task]
    assert root_turns == ["call", "call", "reply"]        # the second call reads the local the first one made


@pytest.mark.parametrize("shape", ["ticket_report", "review_digest", "expense_audit", "nested_assessment"])
def test_synthesized_programs_run_and_match_their_twin(shape):
    """Pseudocode programs with code bases: every reference call is accepted by the harness and by the turn's
    grammar, and the result equals what the Python twin computed from the hidden world."""
    from natlang.gen.synth import SHAPES
    rng = random.Random(7)
    for _ in range(6):
        prog = SHAPES[shape](rng)
        samples, episodes = run_program(prog)
        assert episodes >= 4 and any(s["skill"] == "call" for s in samples)
        assert samples[0]["messages"][1]["content"].startswith("function ")
        assert "Functions you can call:" in samples[0]["messages"][1]["content"]
