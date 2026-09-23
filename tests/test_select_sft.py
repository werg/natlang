import json

import pytest

from scripts.select_sft import balance_targets, difficulty, distribution, select


def row(identity, program, skill):
    return {"id": identity, "program_id": program, "skill": skill,
            "prompt": identity, "completion": "target"}


def test_selection_keeps_rare_behavior_and_caps_common_turns():
    rows = ([row(f"w{i}", "p", "write_file") for i in range(8)] +
            [row(f"t{i}", "p", "return_result" if i % 2 else "reply") for i in range(4)] +
            [row("code", "p", "eval"), row("fn", "p", "read_function"), row("edit", "p", "edit_function")])
    kept, dropped = select(rows, max_per_program=8, max_writes=2, max_terminals=1)
    skills = [r["skill"] for r in kept]
    assert {"eval", "read_function", "edit_function"} <= set(skills)
    assert skills.count("write_file") == 2
    assert skills.count("reply") + skills.count("return_result") == 1
    assert sum(dropped.values()) == len(rows) - len(kept)


def test_selection_never_mixes_program_caps():
    rows = [row(f"{p}-{i}", p, "eval") for p in ("a", "b") for i in range(5)]
    kept, _ = select(rows, max_per_program=3, max_writes=1, max_terminals=1)
    assert {p: sum(r["program_id"] == p for r in kept) for p in ("a", "b")} == {"a": 3, "b": 3}


def test_distribution_exposes_algorithm_and_target_length_mix():
    rows = [row("922:array_kernel:1:0", "p", "eval"), row("plain", "q", "write_file")]
    rows[0]["completion"] = "<think>reason</think>target"
    stats = distribution(rows)
    assert stats["algorithmic_rows"] == 1
    assert stats["algorithmic_fraction"] == .5
    assert stats["reasoning_rows"] == 1
    assert stats["completion_chars"]["max"] == len(rows[0]["completion"])


def test_selection_prefers_iterated_function_calls():
    simple = row("simple", "p", "eval")
    simple["native_target"] = "eval(code='return x + 1')"
    called = row("called", "p", "eval")
    called["native_target"] = "eval(code='return await f(x)')"
    iterated = row("iterated", "p", "eval")
    iterated["native_target"] = "eval(code='return await Promise.all(xs.map(x => f(x)))')"
    kept, _ = select([simple, called, iterated], max_per_program=2,
                     max_writes=1, max_terminals=1)
    assert {r["id"] for r in kept} == {"called", "iterated"}
    assert difficulty(iterated)[0] > difficulty(simple)[0]


def test_selection_preserves_minimum_contrast_and_state_access():
    rows = [row("read", "p", "read_page"), row("write-a", "p", "write_file"),
            row("write-b", "p", "write_file"), row("reply", "p", "reply"),
            row("error", "p", "failed"), row("edit", "p", "edit_function"),
            row("code", "p", "eval")]
    kept, _ = select(rows, max_per_program=2, max_writes=1, max_terminals=1)
    skills = {r["skill"] for r in kept}
    assert {"read_page", "write_file", "reply", "failed", "edit_function"} <= skills
    assert sum(r["skill"] == "write_file" for r in kept) == 1


def test_selection_keeps_all_failure_contrasts():
    rows = [row(f"valid-{i}", "p", "write_file") for i in range(4)]
    for item in rows:
        item["family"] = "failure_bounds_valid"
    kept, _ = select(rows, max_per_program=1, max_writes=1, max_terminals=1)
    assert kept == rows


def test_scope_eval_skills_receive_current_curriculum_weights():
    loop = row("loop", "p", "eval")
    loop["native_target"] = "eval(code='const ys = await Promise.all(xs.map(x => f(x))); ys')"
    literal = row("literal", "p", "return_result")
    read = row("inspect", "p", "read_file")
    repair = row("repair", "p", "edit_file")
    kept, _ = select([loop, literal, read, repair], max_per_program=2,
                     max_writes=1, max_terminals=1)
    assert {item["id"] for item in kept} >= {"repair", "inspect", "literal"}
    score, features = difficulty(loop)
    assert score >= 120 and {"iteration", "algorithmic_glue"} <= set(features)


def test_target_balance_caps_common_and_replicates_rare_without_split_leakage():
    rows = [row(f"common-{i}", f"p{i}", "eval") for i in range(5)]
    for item in rows:
        item["native_target"] = "call-common"
    rare = row("rare", "rare-program", "write_file")
    rare["native_target"] = "write-source"
    balanced, added, removed = balance_targets(rows + [rare], minimum=3, maximum=4)
    targets = [item["native_target"] for item in balanced]
    replicas = [item for item in balanced if item["native_target"] == "write-source"]
    assert targets.count("call-common") == 4
    assert len(replicas) == 3
    assert {item["program_id"] for item in replicas} == {"rare-program"}
    assert len({item["id"] for item in replicas}) == 3
    assert (added, removed) == (2, 1)
