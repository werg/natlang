import json

import pytest

from scripts.select_sft import distribution, select


def row(identity, program, skill):
    return {"id": identity, "program_id": program, "skill": skill,
            "prompt": identity, "completion": "target"}


def test_selection_keeps_rare_behavior_and_caps_common_turns():
    rows = ([row(f"w{i}", "p", "write") for i in range(8)] +
            [row(f"t{i}", "p", "reply") for i in range(4)] +
            [row("code", "p", "run_code"), row("call", "p", "call"), row("edit", "p", "edit")])
    kept, dropped = select(rows, max_per_program=8, max_writes=2, max_terminals=1)
    skills = [r["skill"] for r in kept]
    assert {"run_code", "call", "edit"} <= set(skills)
    assert skills.count("write") == 2
    assert skills.count("reply") == 1
    assert sum(dropped.values()) == len(rows) - len(kept)


def test_selection_never_mixes_program_caps():
    rows = [row(f"{p}-{i}", p, "call") for p in ("a", "b") for i in range(5)]
    kept, _ = select(rows, max_per_program=3, max_writes=1, max_terminals=1)
    assert {p: sum(r["program_id"] == p for r in kept) for p in ("a", "b")} == {"a": 3, "b": 3}


def test_distribution_exposes_algorithm_and_target_length_mix():
    rows = [row("922:array_kernel:1:0", "p", "run_code"), row("plain", "q", "write")]
    rows[0]["completion"] = "<think>reason</think>target"
    stats = distribution(rows)
    assert stats["algorithmic_rows"] == 1
    assert stats["algorithmic_fraction"] == .5
    assert stats["reasoning_rows"] == 1
    assert stats["completion_chars"]["max"] == len(rows[0]["completion"])
