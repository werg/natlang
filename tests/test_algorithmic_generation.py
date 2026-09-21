import random
import sys
from pathlib import Path

import pytest

from natlang.gen.algorithms import ALGORITHMS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_synthetic_ir import freeze  # noqa: E402
from generate import run_program  # noqa: E402
from program_ir import lower  # noqa: E402


@pytest.mark.parametrize("family", sorted(ALGORITHMS))
def test_algorithmic_families_execute_against_independent_oracles(family):
    skills = []
    for seed in range(6):
        samples, _ = run_program(ALGORITHMS[family](random.Random(seed)))
        skills.extend(sample["skill"] for sample in samples)
    if family == "array_kernel":
        assert "run_code" in skills and "write" in skills
    elif family == "staged_ranking":
        assert skills.count("call") >= 18
    else:
        assert skills.count("call") >= 12
        assert "run_code" in skills and "write" in skills


@pytest.mark.parametrize("family", sorted(ALGORITHMS))
def test_algorithmic_families_round_trip_through_program_ir(family):
    program = ALGORITHMS[family](random.Random(17))
    record = freeze(program, f"fixture:{family}", {"sources": "test"})
    samples, _ = run_program(lower(record))
    assert samples
