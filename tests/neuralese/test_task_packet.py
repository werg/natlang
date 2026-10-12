from natlang_neuralese.eval.task_packet import TASK_OUTCOMES, classify_case


def test_repeated_exact_empty_eos_is_model_incomplete_not_infrastructure_or_budget():
    result = {"outcome": {"status": "quiesced", "accepted": False,
                          "detail": "source-authored root eval did not finish",
                          "invocation_ledger": [{"completion_detail": "model_incomplete: terminated_without_result (repeated immediate EOS after one recovery turn)"}],
                          "action_ledger": [], "scope_failures": []}}
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "model_incomplete"
    assert "model_incomplete" in TASK_OUTCOMES


def test_empty_reply_without_explicit_eos_classification_stays_unresolved_or_semantic():
    result = {"outcome": {"status": "quiesced", "accepted": False, "detail": "empty reply"}}
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "semantic_mismatch"
    assert classify_case(None, {"status": "timeout", "resource_limit_reason": "case_wall_budget"}) == "budget_exhaustion"


def test_model_incomplete_phrase_in_generated_action_text_is_not_protocol_evidence():
    result = {"outcome": {"status": "quiesced", "accepted": False, "detail": "root eval did not finish",
                          "invocation_ledger": [],
                          "action_ledger": [{"arguments": {"code": "return 'model_incomplete: terminated_without_result'"}}]}}
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "semantic_mismatch"


def test_nested_episode_budget_exhaustion_is_not_semantic_mismatch():
    result = {"outcome": {"status": "quiesced", "accepted": False,
                          "detail": "source-authored root eval did not finish",
                          "invocation_ledger": [{"completion_status": "quiesced",
                                                 "completion_source": "execution_graph",
                                                 "completion_detail": "episode turn, token, or wall-clock budget exhausted"}],
                          "action_ledger": [], "scope_failures": []}}
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "budget_exhaustion"


def test_recovered_nested_host_abort_is_infrastructure_error():
    result = {"outcome": {"status": "quiesced", "accepted": False,
                          "invocation_ledger": [{"completion_status": "failed",
                                                 "completion_source": "runtime_invocation_trace",
                                                 "completion_detail": "This operation was aborted"}],
                          "scope_failures": [{"failure_kind": "boundary",
                                              "message": "This operation was aborted"}]}}
    recovered_finish = {"status": "complete", "resource_limit_reason": None}
    assert classify_case(result, recovered_finish) == "infrastructure_error"
