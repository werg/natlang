"""S7 task-success rollouts: reward mapping, train-only derivation, sampled queue and spread routing."""
import hashlib
import json

import pytest

from natlang_neuralese.rl.task_rollouts import authored_root_row, build_queue, reward_of, score


def _row(split="train", code="return await f(task);"):
    return {"id": f"row-{split}", "split": split, "family": "fam",
            "curriculum": {"reference": {"root": [["eval", {"code": code}], ["return_result", {}]]}}}


def test_reward_is_binary_task_success_and_infrastructure_is_unscored():
    assert reward_of("task_success") == 1.0
    for outcome in ("invalid_action", "repetition", "budget_exhaustion", "model_incomplete", "semantic_mismatch"):
        assert reward_of(outcome) == 0.0
    assert reward_of("infrastructure_error") is None and reward_of("unresolved") is None


def test_authored_root_derivation_is_train_only_and_hash_bound():
    derived = authored_root_row(_row())
    contract = derived["curriculum"]["trajectory_contract"]
    assert contract["authored_root_eval"] is True
    assert contract["authored_root_sha256"] == hashlib.sha256(b"return await f(task);").hexdigest()
    with pytest.raises(ValueError, match="non-train"):
        authored_root_row(_row(split="test"))


def test_queue_samples_distinct_seeds_and_refuses_greedy_groups(tmp_path):
    entries = build_queue(tmp_path / "p.jsonl", 2, 3, tmp_path, temperature=1.0, case_seconds=60,
                          max_model_requests=8, max_turns=4, seed_base=10)
    assert len(entries) == 6 and len({e["seed"] for e in entries}) == 6 and len({e["key"] for e in entries}) == 6
    assert all(e["count"] == 1 and e["temperature"] == 1.0 for e in entries)
    with pytest.raises(ValueError):
        build_queue(tmp_path / "p.jsonl", 1, 3, tmp_path, temperature=0.0, case_seconds=60,
                    max_model_requests=8, max_turns=4, seed_base=0)


def test_score_routes_groups_by_spread(tmp_path):
    packet = [authored_root_row(_row()), {**authored_root_row(_row()), "id": "row-2"}]
    (tmp_path / "packet.jsonl").write_text("".join(json.dumps(r) + "\n" for r in packet))
    entries = build_queue(tmp_path / "packet.jsonl", 2, 2, tmp_path, temperature=1.0, case_seconds=60,
                          max_model_requests=8, max_turns=4, seed_base=0)
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    done = {"outcome": {"status": "done", "accepted": True}}
    failed = {"outcome": {"status": "failed", "detail": "model_incomplete: terminated_without_result"}}
    for entry, result in zip(entries, [done, failed, failed, failed]):
        output = tmp_path / entry["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result) + "\n")
    report = score(tmp_path)
    routes = [g["route"] for g in report["groups"]]
    assert routes == ["grpo", "rejection-sampling"]
    assert report["successes"] == 1 and report["groups_with_spread"] == 1
