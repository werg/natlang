from natlang_neuralese.eval.task_packet import TASK_OUTCOMES, classify_case


def test_repeated_exact_empty_eos_is_model_incomplete_not_infrastructure_or_budget():
    result = {"outcome": {"status": "quiesced", "accepted": False,
                          "detail": "model_incomplete: terminated_without_result (repeated immediate EOS after one recovery turn)"}}
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "model_incomplete"
    assert "model_incomplete" in TASK_OUTCOMES


def test_empty_reply_without_explicit_eos_classification_stays_unresolved_or_semantic():
    result = {"outcome": {"status": "quiesced", "accepted": False, "detail": "empty reply"}}
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "semantic_mismatch"
    assert classify_case(None, {"status": "timeout", "resource_limit_reason": "case_wall_budget"}) == "budget_exhaustion"
