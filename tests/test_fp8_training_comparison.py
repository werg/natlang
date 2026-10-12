import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "training/neuralese/natlang_neuralese/eval/fp8_training_comparison.py"
SPEC = importlib.util.spec_from_file_location("fp8_training_comparison", MODULE)
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


def _summary(recipe_id, backend, update_seconds, tokens_per_second, windows):
    return {
        "recipe_id": recipe_id,
        "measurements": {
            "warm_median_update_seconds": update_seconds,
            "warm_median_supervised_tokens_per_second": tokens_per_second,
            "peak_observed_incremental_bytes": 100,
            "peak_cuda_allocated_gb": 2.0,
            "first_update_loss": 1.0,
            "first_evaluation": [],
            "last_evaluation": [],
            "gradient_audit_first_update": None,
        },
        "execution_plans": windows,
    }


def test_comparison_result_reuses_immutable_prepared_runs(tmp_path):
    path = tmp_path / "comparison-plan.json"
    path.write_text("{}\n")
    windows = [{"step": 1, "source_documents": ["doc-a"], "mode": "sequential",
                "context_tokens": 6144, "target_tokens": 256}]
    runs = [
        _summary("pair-short-bf16", "bf16", 2.0, 100.0, windows),
        _summary("pair-short-te-fp8", "te_fp8", 1.0, 200.0, windows),
    ]
    result = COMPARISON.comparison_result(tmp_path, path, runs)
    assert result["paired_profiles"]["short"]["warm_median_update_speedup"] == 2.0
    assert result["paired_profiles"]["short"]["same_execution_windows"] is True
    assert (tmp_path / "comparison-result.json").is_file()


def test_comparison_result_refuses_different_window_schedules(tmp_path):
    path = tmp_path / "comparison-plan.json"
    path.write_text("{}\n")
    left = [{"step": 1, "source_documents": ["doc-a"], "mode": "sequential",
             "context_tokens": 6144, "target_tokens": 256}]
    right = [{"step": 1, "source_documents": ["doc-b"], "mode": "sequential",
              "context_tokens": 6144, "target_tokens": 256}]
    runs = [_summary("pair-long-bf16", "bf16", 2.0, 100.0, left),
            _summary("pair-long-te-fp8", "te_fp8", 1.0, 200.0, right)]
    with pytest.raises(RuntimeError, match="sampled different update windows"):
        COMPARISON.comparison_result(tmp_path, path, runs)


def test_warm_aggregate_rate_uses_same_updates_and_actual_supervised_count(tmp_path):
    output = tmp_path / "direct_sft"
    output.mkdir()
    (output / "train.jsonl").write_text("\n".join([
        '{"step":1,"seconds":100,"supervised_tokens":1}',
        '{"step":2,"seconds":100,"supervised_tokens":1}',
        '{"step":3,"seconds":100,"supervised_tokens":1}',
        '{"step":4,"seconds":2,"supervised_tokens":100}',
        '{"step":5,"seconds":3,"supervised_tokens":300}',
    ]) + "\n")
    measured = COMPARISON.summarize_run(tmp_path)
    assert measured["warm_aggregate_wall_seconds"] == 5
    assert measured["warm_aggregate_supervised_tokens"] == 400
    assert measured["warm_aggregate_supervised_tokens_per_second"] == 80
