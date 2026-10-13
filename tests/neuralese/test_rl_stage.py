"""S7 `neuralese_rl` stage: RFT data selection, recipe wiring onto the shared direct trainer, argument forwarding."""
import json

import pytest

from natlang_neuralese.rl import stage as rl_stage
from natlang_neuralese.rl.stage import held_task, select_rft_documents
from natlang_neuralese.train import recipe


def _rollout(task, quality, turns=1):
    turn = {"messages": [{"role": "user", "content": task}], "tools": None,
            "target": {"role": "assistant", "content": "ok"}}
    return {"task_id": task, "reward": None if quality is None else {"quality": quality}, "turns": [turn] * turns}


def test_rft_selects_successes_of_sometimes_solved_tasks_only():
    rollouts = ([_rollout("mixed", 1.0, 2), _rollout("mixed", 0.0), _rollout("mixed", None)]
                + [_rollout("always", 1.0)] * 4 + [_rollout("never", 0.0)] * 4)
    documents, receipt = select_rft_documents(rollouts, max_success_rate=0.75, held_fraction=0.0)
    assert {d["task_id"] for d in documents} == {"mixed"} and len(documents) == 2  # two turns of one success
    assert receipt["skipped"] == {"never_solved": 1, "always_solved_above_rate": 1}
    assert receipt["tasks_selected"][0]["success_rate"] == 0.5  # the unscored rollout does not count


def test_held_split_is_by_task_and_deterministic():
    tasks = [f"task-{i}" for i in range(400)]
    held = [t for t in tasks if held_task(t, 0.125)]
    assert held == [t for t in tasks if held_task(t, 0.125)] and 20 < len(held) < 80
    assert not any(held_task(t, 0.0) for t in tasks)


def test_rl_handler_extends_the_direct_trainer():
    rl, direct = recipe.HANDLERS["neuralese_rl"], recipe.HANDLERS["neuralese_direct"]
    assert direct["parameters"] <= rl["parameters"] and recipe.RL_PARAMETERS <= rl["parameters"]
    assert recipe.missing_stage_inputs("neuralese_rl", {"rollouts"}) == {
        "continue_from", "continue_identity", "continue_monitor_reference"}
    lineage = {"stages": [{"id": "direct_sft", "kind": "neuralese_direct", "requires": []},
                          {"id": "rl_rft", "kind": "neuralese_rl", "requires": ["direct_sft"],
                           "inputs": {"rollouts": "s7_rollouts", "heads": "base_heads"}}]}
    assert recipe.direct_phase_predecessor(lineage, lineage["stages"][1]) == "direct_sft"
    recipe.check_direct_phase_continuation(lineage, lineage["stages"][1])  # continuation roles bound implicitly
    parameters = recipe.effective_stage_parameters(
        {"kind": "neuralese_rl", "parameters": {"rft_max_success_rate": 0.5}}, [],
        {"stage_parameter_defaults": {"neuralese_direct": {"latent_lr": 3e-6, "epochs": 1.0},
                                      "neuralese_rl": {"epochs": 2.0}}})
    assert parameters == {"latent_lr": 3e-6, "epochs": 2.0, "rft_max_success_rate": 0.5}
    assert "rollouts" in " ".join(recipe.stage_input_args({"rollouts": {"path": "/r.jsonl"}}, "neuralese_rl"))


def test_stage_forwards_to_direct_trainer_with_fresh_reference(tmp_path, monkeypatch):
    rollouts = tmp_path / "r.jsonl"
    rows = [_rollout(f"t{i}", q) for i in range(40) for q in (1.0, 0.0)]
    rollouts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    calls = {}
    monkeypatch.setattr(rl_stage, "render_rows", lambda documents, tokenizer, mode: [
        {"text": "x" + d["task_id"], "split": d["split"], "source_groups": [d["task_id"]]} for d in documents])
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda path: object()))
    from natlang_neuralese.train import direct_sft
    monkeypatch.setattr(direct_sft, "main", lambda argv: calls.setdefault("argv", argv))
    rl_stage.main(["--rollouts", str(rollouts), "--rl-tokenizer", str(tmp_path), "--out", str(tmp_path / "out"),
                   "--heads", "/h.pt", "--continue-from", "/c.pt", "--epochs", "1"])
    argv = calls["argv"]
    assert argv[:6] == ["--heads", "/h.pt", "--continue-from", "/c.pt", "--epochs", "1"]
    assert "--fresh-monitor-reference" in argv and argv[argv.index("--text-data") + 1].endswith("rl-text-data.jsonl")
    receipt = json.loads((tmp_path / "out" / "rl-data-receipt.json").read_text())
    assert receipt["documents"] == 40 and len(receipt["tasks_selected"]) == 40


def test_stage_refuses_without_train_and_held_documents(tmp_path):
    rollouts = tmp_path / "r.jsonl"
    rollouts.write_text(json.dumps(_rollout("never", 0.0)) + "\n")
    with pytest.raises(SystemExit):
        rl_stage.main(["--rollouts", str(rollouts), "--rl-tokenizer", str(tmp_path), "--out", str(tmp_path / "o")])
