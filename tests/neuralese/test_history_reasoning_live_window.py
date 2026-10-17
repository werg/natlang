import hashlib
import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import pytest


SCRIPT = (Path(__file__).resolve().parents[2] / "scripts/neuralese_data" /
          "run_history_reasoning_live_window.py")
SPEC = importlib.util.spec_from_file_location("history_reasoning_live_window", SCRIPT)
WINDOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WINDOW)


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(index: int):
    return {"case_id": f"case-{index}", "task_id": f"program-{index}",
            "program_id": f"program-{index}", "source_row_id": f"row-{index}",
            "source_row_sha256": f"{index:064x}", "trajectory_index": index,
            "raw_response_sha256": f"{index + 1:064x}",
            "validation_proof_path": f"/proof/{index}",
            "validation_proof_sha256": f"{index + 2:064x}"}


def test_baseline_prefix_reuse_requires_exact_case_and_run_bindings(tmp_path):
    cases_path = tmp_path / "old-cases.jsonl"
    cases = [_case(i) for i in range(48)]
    cases_path.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases))
    results_path = tmp_path / "old-results.jsonl"
    records = [{"case_id": case["case_id"], "task_id": case["task_id"],
                "profile": "last_turn_only", "source_row_id": case["source_row_id"],
                "source_row_sha256": case["source_row_sha256"],
                "generation": {"content": "answer"}, "likelihood": {"target_token_count": 1}}
               for case in cases[:45]]
    results_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    run_path = tmp_path / "old-run.json"
    _write_json(run_path, {"schema": WINDOW.SCHEMA, "checkpoint_sha256": "a" * 64,
        "baseline_profile": "last_turn_only", "candidate_profile": "keep",
        "weights_mutated": False, "training_admission": False,
        "cases": {"current_protection_sha256": "b" * 64,
                  "case_packet_path": str(cases_path.resolve()),
                  "case_packet_sha256": _sha(cases_path)}})
    enriched = [{**case, "messages": [{"role": "user", "content": "task"}],
                 "target": {"content": "answer"}, "tools": None} for case in cases]

    prefix_records, binding = WINDOW.validate_baseline_prefix(
        prefix_run_path=run_path, prefix_results_path=results_path,
        prefix_cases_path=cases_path, cases=enriched, checkpoint_sha256="a" * 64,
        baseline_profile="last_turn_only", current_protection_sha256="b" * 64)
    assert len(prefix_records) == 45
    assert [record["generation"] for record in prefix_records] == [record["generation"] for record in records]
    assert binding["results_sha256"] == _sha(results_path)

    changed = list(cases)
    changed[44] = {**changed[44], "source_row_sha256": "c" * 64}
    changed = [{**case, "messages": [], "target": {}, "tools": None} for case in changed]
    with pytest.raises(ValueError, match="changed baseline-prefix case"):
        WINDOW.validate_baseline_prefix(
            prefix_run_path=run_path, prefix_results_path=results_path,
            prefix_cases_path=cases_path, cases=changed, checkpoint_sha256="a" * 64,
            baseline_profile="last_turn_only", current_protection_sha256="b" * 64)


def test_baseline_prefix_rejects_nonbaseline_or_unbound_result(tmp_path):
    cases_path = tmp_path / "old-cases.jsonl"
    cases = [_case(i) for i in range(48)]
    cases_path.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases))
    results_path = tmp_path / "old-results.jsonl"
    record = {"case_id": cases[0]["case_id"], "task_id": cases[0]["task_id"],
              "profile": "keep", "source_row_id": cases[0]["source_row_id"],
              "source_row_sha256": cases[0]["source_row_sha256"],
              "generation": {}, "likelihood": {}}
    results_path.write_text(json.dumps(record) + "\n")
    run_path = tmp_path / "old-run.json"
    _write_json(run_path, {"schema": WINDOW.SCHEMA, "checkpoint_sha256": "a" * 64,
        "baseline_profile": "last_turn_only", "candidate_profile": "keep",
        "weights_mutated": False, "training_admission": False,
        "cases": {"current_protection_sha256": "b" * 64,
                  "case_packet_path": str(cases_path.resolve()),
                  "case_packet_sha256": _sha(cases_path)}})
    enriched = [{**case, "messages": [], "target": {}, "tools": None} for case in cases]
    with pytest.raises(ValueError, match="contains no baseline-profile rows"):
        WINDOW.validate_baseline_prefix(
            prefix_run_path=run_path, prefix_results_path=results_path,
            prefix_cases_path=cases_path, cases=enriched, checkpoint_sha256="a" * 64,
            baseline_profile="last_turn_only", current_protection_sha256="b" * 64)


def test_candidate_prefix_reuses_success_and_records_exact_request_failure(tmp_path):
    cases_path = tmp_path / "candidate-cases.jsonl"
    cases = [_case(i) for i in range(48)]
    cases_path.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases))
    results_path = tmp_path / "candidate-results.jsonl"
    records = [{"case_id": case["case_id"], "task_id": case["task_id"],
                "profile": "keep", "source_row_id": case["source_row_id"],
                "source_row_sha256": case["source_row_sha256"],
                "generation": {"content": "answer"},
                "likelihood": {"target_token_count": 1}}
               for case in cases[:2]]
    records[1] = {**records[1], "generation": None,
                  "generation_error": {"type": "RequestError",
                      "code": "neuralese-native-state-open",
                      "message": "neuralese-native-state-open: generation stopped inside a produce span"}}
    results_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    parent_results_path = tmp_path / "original-results.jsonl"
    parent_results_path.write_text("{}\n")
    parent_run_path = tmp_path / "original-run.json"
    _write_json(parent_run_path, {"schema": WINDOW.SCHEMA, "checkpoint_sha256": "a" * 64})
    journal_path = tmp_path / "attempt-journal.txt"
    journal_path.write_text("neuralese-native-state-open: generation stopped inside a produce span\n")
    error = records[1]["generation_error"]
    proof = {"schema": "natlang.history-reasoning-sampled-failure-evidence/1",
        "parent_run_sha256": _sha(parent_run_path), "parent_results_sha256": _sha(parent_results_path),
        "case_packet_sha256": _sha(cases_path), "checkpoint_sha256": "a" * 64,
        "profile": "keep", "seed": 0, "case_index": 1,
        "case_id": cases[1]["case_id"], "task_id": cases[1]["task_id"],
        "source_row_id": cases[1]["source_row_id"], "source_row_sha256": cases[1]["source_row_sha256"],
        "error": error, "journal_path": str(journal_path), "journal_sha256": _sha(journal_path)}
    run_path = tmp_path / "candidate-run.json"
    _write_json(run_path, {"schema": WINDOW.SCHEMA, "checkpoint_sha256": "a" * 64,
        "baseline_profile": "last_turn_only", "candidate_profile": "keep",
        "seed": 0, "prefix_parent": {"run_path": str(parent_run_path),
            "run_sha256": _sha(parent_run_path), "results_path": str(parent_results_path),
            "results_sha256": _sha(parent_results_path)},
        "sampled_output_failures": [proof],
        "weights_mutated": False, "training_admission": False,
        "cases": {"current_protection_sha256": "b" * 64,
                  "case_packet_path": str(cases_path.resolve()),
                  "case_packet_sha256": _sha(cases_path)}})
    enriched = [{**case, "messages": [], "target": {}, "tools": None} for case in cases]
    prefix, binding = WINDOW.validate_candidate_prefix(
        prefix_run_path=run_path, prefix_results_path=results_path,
        prefix_cases_path=cases_path, cases=enriched, checkpoint_sha256="a" * 64,
        candidate_profile="keep", current_protection_sha256="b" * 64, expected_seed=0)
    assert len(prefix) == 2 and binding["rows"] == 2
    assert prefix[1]["generation"] is None
    assert prefix[1]["generation_error"]["code"] == "neuralese-native-state-open"

    changed = list(enriched)
    changed[1] = {**changed[1], "source_row_sha256": "c" * 64}
    with pytest.raises(ValueError, match="changed candidate-prefix case"):
        WINDOW.validate_candidate_prefix(
            prefix_run_path=run_path, prefix_results_path=results_path,
            prefix_cases_path=cases_path, cases=changed, checkpoint_sha256="a" * 64,
            candidate_profile="keep", current_protection_sha256="b" * 64, expected_seed=0)


def test_record_request_error_requires_and_preserves_exact_code():
    error = ValueError("neuralese-native-state-open: generation stopped inside a produce span")
    error.code = "neuralese-native-state-open"
    assert WINDOW.record_request_error(error) == {
        "type": "ValueError", "code": "neuralese-native-state-open",
        "message": str(error)}
    infrastructure = ValueError("unknown actor state")
    infrastructure.code = "neuralese-native-state-unknown"
    with pytest.raises(ValueError, match="only a sampled-output marker failure"):
        WINDOW.record_request_error(infrastructure)
    with pytest.raises(ValueError, match="only a sampled-output marker failure"):
        WINDOW.record_request_error(ValueError("uncoded infrastructure failure"))


@pytest.mark.parametrize("baseline_prefix_count", [45, 48])
def test_compare_reuses_prefix_generations_and_runs_only_missing_baseline_cases(
        tmp_path, monkeypatch, baseline_prefix_count):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = _sha(checkpoint)
    old_cases_path = tmp_path / "old-cases.jsonl"
    old_cases = [_case(i) for i in range(48)]
    old_cases_path.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in old_cases))
    cases = [{**case, "messages": [{"role": "user", "content": "task"}],
              "target": {"content": "answer"}, "tools": None} for case in old_cases]
    results_path = tmp_path / "old-results.jsonl"
    prior_records = [{"case_id": case["case_id"], "task_id": case["task_id"],
        "profile": "last_turn_only", "source_row_id": case["source_row_id"],
        "source_row_sha256": case["source_row_sha256"],
        "generation": {"content": f"old-{index}"},
        "likelihood": {"target_token_count": 1, "target_log_likelihood": -1.0}}
        for index, case in enumerate(cases[:baseline_prefix_count])]
    results_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in prior_records))
    run_path = tmp_path / "old-run.json"
    _write_json(run_path, {"schema": WINDOW.SCHEMA, "checkpoint_sha256": checkpoint_sha,
        "baseline_profile": "last_turn_only", "candidate_profile": "keep",
        "weights_mutated": False, "training_admission": False,
        "cases": {"current_protection_sha256": "b" * 64,
                  "case_packet_path": str(old_cases_path.resolve()),
                  "case_packet_sha256": _sha(old_cases_path)}})

    candidate_parent_results_path = tmp_path / "candidate-parent-results.jsonl"
    candidate_parent_records = [{"case_id": case["case_id"], "task_id": case["task_id"],
        "profile": "keep", "source_row_id": case["source_row_id"],
        "source_row_sha256": case["source_row_sha256"], "generation": {"content": f"keep-{index}"},
        "likelihood": {"target_token_count": 1, "target_log_likelihood": -1.0}}
        for index, case in enumerate(cases[:10])]
    candidate_parent_results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
        for row in candidate_parent_records))
    candidate_parent_run_path = tmp_path / "candidate-parent-run.json"
    _write_json(candidate_parent_run_path, {"schema": WINDOW.SCHEMA,
        "checkpoint_sha256": checkpoint_sha, "candidate_profile": "keep"})
    marker_error = {"type": "RequestError", "code": "neuralese-native-state-open",
        "message": "neuralese-native-state-open: generation stopped inside a produce span"}
    candidate_prefix_rows = [*candidate_parent_records,
        {"case_id": cases[10]["case_id"], "task_id": cases[10]["task_id"],
         "profile": "keep", "source_row_id": cases[10]["source_row_id"],
         "source_row_sha256": cases[10]["source_row_sha256"], "generation": None,
         "generation_error": marker_error,
         "likelihood": {"target_token_count": 1, "target_log_likelihood": -1.0}}]
    candidate_results_path = tmp_path / "candidate-prefix-results.jsonl"
    candidate_results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
        for row in candidate_prefix_rows))
    journal_path = tmp_path / "sampled-failure-journal.txt"
    journal_path.write_text(marker_error["message"] + "\n")
    candidate_proof = {"schema": "natlang.history-reasoning-sampled-failure-evidence/1",
        "parent_run_sha256": _sha(candidate_parent_run_path),
        "parent_results_sha256": _sha(candidate_parent_results_path),
        "case_packet_sha256": _sha(old_cases_path), "checkpoint_sha256": checkpoint_sha,
        "profile": "keep", "seed": 0, "case_index": 10,
        "case_id": cases[10]["case_id"], "task_id": cases[10]["task_id"],
        "source_row_id": cases[10]["source_row_id"],
        "source_row_sha256": cases[10]["source_row_sha256"], "error": marker_error,
        "journal_path": str(journal_path), "journal_sha256": _sha(journal_path)}
    candidate_run_path = tmp_path / "candidate-prefix-run.json"
    _write_json(candidate_run_path, {"schema": WINDOW.SCHEMA,
        "checkpoint_sha256": checkpoint_sha, "baseline_profile": "last_turn_only",
        "candidate_profile": "keep", "seed": 0,
        "prefix_parent": {"run_path": str(candidate_parent_run_path),
            "run_sha256": _sha(candidate_parent_run_path),
            "results_path": str(candidate_parent_results_path),
            "results_sha256": _sha(candidate_parent_results_path)},
        "sampled_output_failures": [candidate_proof],
        "weights_mutated": False, "training_admission": False,
        "cases": {"current_protection_sha256": "b" * 64,
            "case_packet_path": str(old_cases_path.resolve()),
            "case_packet_sha256": _sha(old_cases_path)}})

    monkeypatch.setattr(WINDOW, "validate_case_packet",
        lambda *args, **kwargs: (cases, {"case_count": 48}))
    likelihood_policies = []
    monkeypatch.setattr(WINDOW, "target_log_likelihood",
        lambda engine, *args, **kwargs: (likelihood_policies.append(engine.tokenizer.policy) or
                                         {"target_token_count": 1, "target_log_likelihood": -2.0}))

    import natlang_neuralese.serve as serve
    import natlang_neuralese.serve.chat as chat
    import natlang_neuralese.serve.engine as engine_module

    class FakeModel:
        def eval(self):
            return self
        def parameters(self):
            return iter(())

    class FakeEngine:
        def __init__(self):
            self.tokenizer = SimpleNamespace(policy="last_turn_only",
                natlang_history_reasoning={"policy": "last_turn_only"})
            self.native_identity = {"checkpoint_sha256": checkpoint_sha}
            self.shared_model = FakeModel()
            self.generated = []
        def generate(self, request):
            self.generated.append(self.tokenizer.policy)
            return {"choices": [{"message": {"content": self.tokenizer.policy},
                                  "finish_reason": "stop"}], "usage": {"total_tokens": 1}}

    fake_engine = FakeEngine()
    monkeypatch.setattr(serve, "load_native_engine", lambda *args, **kwargs: fake_engine)
    monkeypatch.setattr(chat, "bind_history_reasoning",
        lambda tokenizer, policy, allow_transition=False: setattr(tokenizer, "policy", policy) or tokenizer)
    monkeypatch.setattr(engine_module, "GenerationRequest", lambda **kwargs: SimpleNamespace(**kwargs))

    args = SimpleNamespace(cases="unused", exclusion_proof="unused", exclusion_proof_sha256="c" * 64,
        current_protection_sha256="b" * 64, cases_per_variant=48, checkpoint=str(checkpoint),
        output=str(tmp_path / "successor"), baseline="last_turn_only", base=None, device="cpu",
        prompt_rendering="full", max_tokens=8, seed=0,
        baseline_prefix_run=str(run_path), baseline_prefix_results=str(results_path),
        baseline_prefix_cases=str(old_cases_path),
        candidate_prefix_run=str(candidate_run_path),
        candidate_prefix_results=str(candidate_results_path),
        candidate_prefix_cases=str(old_cases_path))
    summary = WINDOW.run(args)
    output_rows = [json.loads(line) for line in (tmp_path / "successor" / "results.jsonl").open()]
    assert summary["result_counts"] == {"last_turn_only": 48, "keep": 48}
    assert len(output_rows) == 96
    assert [row["generation"] for row in output_rows[:baseline_prefix_count]] == [
        record["generation"] for record in prior_records]
    assert fake_engine.generated == ["last_turn_only"] * (48 - baseline_prefix_count) + ["keep"] * 37
    assert likelihood_policies == ["last_turn_only"] * 48 + ["keep"] * 48
    assert output_rows[48 + 10]["generation_error"] == marker_error
