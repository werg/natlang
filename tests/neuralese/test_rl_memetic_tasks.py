"""Memetic optimiser on task success: pure parts (guidance channel, failure digest without gold, chooseNext policy)."""
import json
import random
from types import SimpleNamespace

import natlang_neuralese.rl.memetic_tasks as memetic


def _fit(tmp_path, fitness, spread=0, stops=0):
    return {"fitness": fitness, "budget_stops": stops, "groups_with_spread": spread, "per_row": [], "scored": 4,
            "dir": str(tmp_path)}


def test_guidance_uses_the_runtime_wrapper_and_empty_guidance_leaves_the_prompt():
    assert memetic.guided_system("BASE", "") == "BASE"
    assert memetic.guided_system("BASE", "Check fields.") == \
        "BASE\n\n<natlang_program_guidance>\nCheck fields.\n</natlang_program_guidance>\n"


def test_failure_digest_reports_failures_but_never_gold(tmp_path):
    (tmp_path / "rewards.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"group": 0, "sample": 0, "family": "fam", "outcome": "semantic_mismatch", "reward": 0.0},
        {"group": 1, "sample": 0, "family": "fam", "outcome": "task_success", "reward": 1.0}]))
    case = tmp_path / "row-000" / "sample-00"
    case.mkdir(parents=True)
    (case / "results.jsonl").write_text(json.dumps({"task": {"program_ir": {"curriculum": {"reference": "GOLD-ANSWER"}}},
                                                    "outcome": {"checks": {"answer": False, "files": True}, "value": "wrong",
                                                                "scope_failures": [{"message": "TypeError: x"}]}}) + "\n")
    digest = memetic.failure_digest({"dir": str(tmp_path)})
    assert "semantic_mismatch" in digest and "['answer']" in digest and "TypeError" in digest and '"wrong"' in digest
    assert "GOLD" not in digest and digest.count("\n") == 0  # one failed rollout, the success is not listed


def test_choose_next_crisp_tries_untried_operators_and_needs_spread_for_the_gradient(tmp_path):
    args = SimpleNamespace(policy="crisp")
    flat = [{"id": "ind-0", "guidance": "", "fit": _fit(tmp_path, 0.3)}]
    assert memetic.choose_next(args, flat, [], random.Random(0))["operator"] == "propose"
    spread = [{"id": "ind-0", "guidance": "", "fit": _fit(tmp_path, 0.3, spread=2)}]
    assert memetic.choose_next(args, spread, [{"operator": "propose", "gain": 0.0}], random.Random(0))["operator"] == "refine-adapter"


def test_choose_next_nl_answer_is_validated_and_an_invalid_one_falls_back(tmp_path, monkeypatch):
    args = SimpleNamespace(policy="nl", author="http://author", author_model="m")
    population = [{"id": "ind-0", "guidance": "", "fit": _fit(tmp_path, 0.3)}]
    monkeypatch.setattr(memetic, "author_call", lambda *a, **k: '{"parent": "ind-0", "operator": "propose", "why": "first"}')
    assert memetic.choose_next(args, population, [], random.Random(0)) == \
        {"parent": "ind-0", "operator": "propose", "policy": "nl", "why": "first"}
    monkeypatch.setattr(memetic, "author_call", lambda *a, **k: '{"parent": "ind-0", "operator": "refine-adapter"}')
    choice = memetic.choose_next(args, population, [], random.Random(0))
    assert choice["policy"] == "crisp-fallback" and choice["operator"] == "propose" and "not allowed" in choice["nl_error"]
