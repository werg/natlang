import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from episode_lib import balance_choice_positions, case_record, held_split, write_packet  # noqa: E402


def choice(answer):
    return {"args": ["q", {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}],
            "expected": {"kind": "choice", "answer": answer, "options": ["A", "B", "C", "D"]}}


def test_balancing_rotates_answer_labels_and_keeps_the_right_text():
    cases = [choice("C") for _ in range(8)]
    balanced = balance_choice_positions(cases, "salt")
    answers = [c["expected"]["answer"] for c in balanced]
    assert sorted(answers) == ["A", "A", "B", "B", "C", "C", "D", "D"]
    for case in balanced:
        options = case["args"][1]
        assert options[case["expected"]["answer"]] == "gamma"
        assert sorted(options.values()) == ["alpha", "beta", "delta", "gamma"]


def test_held_split_schedule_and_case_identity():
    assert [held_split(n, 10) for n in range(10)] == ["train"] * 8 + ["validation", "test"]
    a, b = case_record("src", "k1", [1], 2), case_record("src", "k1", [9], 3)
    assert a["id"] == b["id"] and a["group"] == b["group"]


def test_write_packet_refuses_repeated_groups_and_records_its_hash(tmp_path):
    episode = lambda i, group: {"id": f"e{i}", "split": "train", "support": {"cases": [{"group": group}]},
                                "query": {"cases": [{"group": f"q{i}"}]}}
    manifest = write_packet(str(tmp_path), "p", [episode(0, "g0"), episode(1, "g1")], {"schema": "x", "episodes": None})
    assert list(manifest) == ["schema", "episodes", "splits", "sha256"] and manifest["episodes"] == 2
    assert json.loads((tmp_path / "p.manifest.json").read_text())["sha256"] == manifest["sha256"]
    try:
        write_packet(str(tmp_path / "bad"), "p", [episode(0, "g"), episode(1, "g")], {})
    except ValueError:
        pass
    else:
        raise AssertionError("repeated groups must be refused")
