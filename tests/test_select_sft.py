import json

import pytest

from scripts.select_sft import balance_targets, difficulty, distribution, select


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


def test_selection_prefers_dependent_and_iterated_calls():
    simple = row("simple", "p", "call")
    simple["native_target"] = "<|tool_call_start|>[call(function='f', to='return', inputs={'x': 'args/x'})]"
    dependent = row("dependent", "p", "call")
    dependent["native_target"] = "<|tool_call_start|>[call(function='f', to='return', inputs={'x': 'let/x'})]"
    iterated = row("iterated", "p", "call")
    iterated["native_target"] = "<|tool_call_start|>[call(function='f', to='return', over='args/x')]"
    kept, _ = select([simple, dependent, iterated], max_per_program=2,
                     max_writes=1, max_terminals=1)
    assert {r["id"] for r in kept} == {"dependent", "iterated"}
    assert difficulty(iterated)[0] > difficulty(simple)[0]


def test_selection_preserves_minimum_contrast_and_state_access():
    rows = [row("read", "p", "read"), row("write-a", "p", "write"),
            row("write-b", "p", "write"), row("reply", "p", "reply"),
            row("error", "p", "report_error"), row("edit", "p", "edit"),
            row("call", "p", "call")]
    kept, _ = select(rows, max_per_program=2, max_writes=1, max_terminals=1)
    skills = {r["skill"] for r in kept}
    assert {"read", "write", "reply", "report_error", "edit"} <= skills
    assert sum(r["skill"] == "write" for r in kept) == 1


def test_selection_keeps_all_failure_contrasts():
    rows = [row(f"valid-{i}", "p", "write") for i in range(4)]
    for item in rows:
        item["family"] = "failure_bounds_valid"
    kept, _ = select(rows, max_per_program=1, max_writes=1, max_terminals=1)
    assert kept == rows


def test_target_balance_caps_common_and_replicates_rare_without_split_leakage():
    rows = [row(f"common-{i}", f"p{i}", "call") for i in range(5)]
    for item in rows:
        item["native_target"] = "call-common"
    rare = row("rare", "rare-program", "write")
    rare["native_target"] = "write-source"
    balanced, added, removed = balance_targets(rows + [rare], minimum=3, maximum=4)
    targets = [item["native_target"] for item in balanced]
    replicas = [item for item in balanced if item["native_target"] == "write-source"]
    assert targets.count("call-common") == 4
    assert len(replicas) == 3
    assert {item["program_id"] for item in replicas} == {"rare-program"}
    assert len({item["id"] for item in replicas}) == 3
    assert (added, removed) == (2, 1)
