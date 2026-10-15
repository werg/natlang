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


def test_explicit_wrong_blocked_return_is_scored_semantic_failure():
    result = {"outcome": {"status": "quiesced", "accepted": False,
                          "rejection_reasons": ["expected_status", "answer"],
                          "scope_failures": [{"failure_kind": "runtime", "message": "sample not delivered"}],
                          "action_ledger": [{"name": "return_result", "outcome": "blocked",
                                             "arguments": {"status": "blocked", "reason": "sample not delivered"}}]}}
    finish = {"status": "complete", "resource_limit_reason": None}
    assert classify_case(result, finish) == "semantic_mismatch"


def test_missing_result_with_explicit_transport_accounting_is_unscored_infrastructure():
    finish = {'status': 'incomplete', 'completion_tokens': 8192, 'output_accounting': {'job_states': [
        {'state': 'transport_failed', 'terminal_error': {'error': 'TypeError: terminated', 'sha256': 'a' * 64}}
    ]}}
    assert classify_case(None, finish) == 'infrastructure_error'
    finish['output_accounting']['job_states'][0]['state'] = 'failed'
    assert classify_case(None, finish) == 'unresolved'
    assert classify_case(None, {'status': 'incomplete', 'completion_tokens': 8192}) == 'unresolved'
