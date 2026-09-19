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
    assert root_turns == ["write", "write", "run", "reply"]
