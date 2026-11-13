from natlang_neuralese.eval.task_packet import TASK_OUTCOMES, accounted, classify_case, score_rows
import copy
import json
import pytest


@pytest.mark.parametrize("history", [
    {"action_ledger": [{"outcome": "invalid", "diagnostics": ["Expression expected"]}]},
    {"scope_failures": [{"message": "TypeError: earlier invalid eval"}]},
])
def test_completed_wrong_answer_does_not_inherit_recovered_action_failure(history):
    result = {"outcome": {"status": "done", "accepted": False,
                          "checks": {"expected_status": True, "answer": False},
                          "value": {"verdict": "entailed", "exception": None}, **history}}
    assert classify_case(result, {"status": "complete"}) == "semantic_mismatch"
    result["outcome"]["accepted"] = True
    assert classify_case(result, {"status": "complete"}) == "task_success"


def test_actual_terminal_invalid_action_remains_invalid():
    result = {"outcome": {"status": "failed", "accepted": False,
                          "checks": {"answer": False},
                          "action_ledger": [{"outcome": "invalid"}]}}
    assert classify_case(result, {"status": "complete"}) == "invalid_action"


def native_open_finish(termination):
    return {"status": "incomplete", "output_accounting": {
        "manifest_valid": True, "source_sha256": "c"*64, "expected_jobs": 1, "job_states": [{
            "state": "failed", "digest": "b"*64, "terminal_error": {"sha256": "a" * 64,
                "record_binding_verified": True, "program_ir_sha256": "b"*64, "source_sha256": "c"*64,
                "error": "Error: model stream error: " + json.dumps({
                    "code": "neuralese-native-state-open",
                    "message": f"neuralese-native-state-open: generation {termination} inside a produce span"})}}]}}


@pytest.mark.parametrize("termination,expected", [("ended", "budget_exhaustion"),
                                                  ("stopped", "model_incomplete")])
def test_manifest_bound_native_open_termination(termination, expected):
    assert classify_case(None, native_open_finish(termination)) == expected


@pytest.mark.parametrize("mutation", ["manifest", "sha", "mixed", "unknown", "malformed", "no_prefix"])
def test_native_termination_requires_exact_terminal_accounting(mutation):
    finish = native_open_finish("stopped")
    accounting = finish["output_accounting"]
    error = accounting["job_states"][0]["terminal_error"]
    if mutation == "manifest":
        accounting["manifest_valid"] = False
    elif mutation == "sha":
        error["sha256"] = ""
    elif mutation == "mixed":
        accounting["job_states"].append({"state": "failed"})
    elif mutation == "unknown":
        error["error"] = error["error"].replace("stopped", "crashed")
    elif mutation == "malformed":
        error["error"] += " trailing text"
    else:
        error["error"] = error["error"].removeprefix("Error: model stream error: ")
    assert classify_case(None, finish) == "unresolved"


def test_transport_and_explicit_limits_take_precedence_over_native_termination():
    finish = native_open_finish("stopped")
    transport = copy.deepcopy(finish)
    transport["output_accounting"]["job_states"].append({"state": "transport_failed"})
    assert classify_case(None, transport) == "infrastructure_error"
    finish["resource_limit_reason"] = "model_request_budget"
    assert classify_case(None, finish) == "budget_exhaustion"


def test_accounted_native_failure_does_not_fabricate_an_exported_result():
    score = score_rows([{"id": "case"}], [], {"key": native_open_finish("stopped")}, {0: "key"})
    assert accounted(score)
    assert score["outcome_counts"]["model_incomplete"] == 1
    assert score["task_successes"] == 0
    assert score["cases_with_exact_terminal_result"] == 0
    assert score["complete"] is False
    assert score["eligible_for_best"] is False


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
    assert classify_case(None, {"status": "timeout", "resource_limit_reason": "case_wall_budget"}) == "unresolved"


def test_model_incomplete_phrase_in_generated_action_text_is_not_protocol_evidence():
    result = {"outcome": {"status": "quiesced", "accepted": False, "detail": "root eval did not finish",
                          "invocation_ledger": [],
                          "action_ledger": [{"arguments": {"code": "return 'model_incomplete: terminated_without_result'"}}]}}
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "semantic_mismatch"


def test_ambiguous_nested_wall_budget_is_unscored_but_explicit_token_budget_is_known():
    result = {"outcome": {"status": "quiesced", "accepted": False,
                          "detail": "source-authored root eval did not finish",
                          "invocation_ledger": [{"completion_status": "quiesced",
                                                 "completion_source": "execution_graph",
                                                 "completion_detail": "episode turn, token, or wall-clock budget exhausted"}],
                          "action_ledger": [], "scope_failures": []}}
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "unresolved"
    result['outcome']['resource_limit_reason']='token_budget'
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "budget_exhaustion"


def test_wall_timeout_cannot_be_used_as_measured_behavioral_noise():
    score=score_rows([{'id':'case'}],[],{'key':{'status':'timeout',
        'resource_limit_reason':'case_wall_budget','stuckness':'unknown','decode_activity':True}},{0:'key'})
    assert score['outcome_counts']['unresolved']==1
    assert not accounted(score) and not score['eligible_for_best']
    # An already exported accepted final result remains actual success.
    assert classify_case({'outcome':{'status':'done','accepted':True}},
                         {'status':'timeout','resource_limit_reason':'case_wall_budget'})=='task_success'


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


@pytest.mark.parametrize('mutation',['unverified','missing_record','missing_source','wrong_record','wrong_source','missing_current_source'])
def test_native_termination_rejects_unbound_or_mismatched_record_source(mutation):
    finish=native_open_finish('stopped');accounting=finish['output_accounting']
    error=accounting['job_states'][0]['terminal_error']
    if mutation=='unverified':error.pop('record_binding_verified')
    elif mutation=='missing_record':error.pop('program_ir_sha256')
    elif mutation=='missing_source':error.pop('source_sha256')
    elif mutation=='wrong_record':error['program_ir_sha256']='d'*64
    elif mutation=='wrong_source':error['source_sha256']='d'*64
    else:accounting.pop('source_sha256')
    assert classify_case(None,finish)=='unresolved'


def accepted_blocked_result():
    return {'outcome': {'status': 'quiesced', 'accepted': True,
            'oracle': {'accepted': True, 'level': 'exact'}, 'rejection_reasons': [],
            'checks': {'expected_status': True, 'honest_stop': True, 'stop_evidence': True,
                       'effects': True, 'answer': True},
            'detail': 'blocked: external operation was rejected; no authorized repair exists',
            'scope_failures': [{'message': 'TypeError in an earlier recovered action'}],
            'action_ledger': [{'name': 'eval', 'outcome': 'rejected'},
                              {'name': 'return_result', 'outcome': 'blocked',
                               'arguments': {'status': 'blocked', 'reason': 'external operation rejected'}}]}}


def test_verified_expected_blocked_stop_earns_task_success_despite_action_history():
    result = accepted_blocked_result()
    assert classify_case(result, {'status': 'complete'}) == 'task_success'
    from natlang_neuralese.rl.task_rollouts import reward_of
    assert reward_of(classify_case(result, {'status': 'complete'})) == 1.


@pytest.mark.parametrize('missing', ['expected_status', 'honest_stop', 'stop_evidence'])
def test_accepted_flag_alone_does_not_qualify_an_unproven_stop(missing):
    result = accepted_blocked_result()
    del result['outcome']['checks'][missing]
    assert classify_case(result, {'status': 'complete'}) != 'task_success'


def test_expected_stop_cannot_hide_resource_exhaustion_or_rejected_checks():
    result = accepted_blocked_result()
    assert classify_case(result, {'status': 'timeout', 'resource_limit_reason': 'case_wall_budget'}) == 'unresolved'
    result['outcome']['checks']['effects'] = False
    assert classify_case(result, {'status': 'complete'}) != 'task_success'


def test_verified_expected_failed_stop_and_accepted_complete_result_are_distinct():
    result = accepted_blocked_result()
    result['outcome']['status'] = 'failed'
    assert classify_case(result, {'status': 'complete'}) == 'task_success'
    result['outcome']['oracle']['accepted'] = False
    assert classify_case(result, {'status': 'complete'}) != 'task_success'


def completed_wrong(**outcome):
    return {"outcome": {"status": "done", "accepted": False, "checks": {"expected_status": True, "answer": False},
                        "value": "wrong", **outcome}}


@pytest.mark.parametrize("literal", ["HTTP 500", "connection refused", "ECONNRESET", "server unavailable",
                                     "provider request retry exhausted", "repeated the same action again"])
def test_model_authored_text_cannot_make_a_completed_wrong_answer_unscored(literal):
    finish = {"status": "complete", "resource_limit_reason": None}
    result = completed_wrong(
        action_ledger=[{"name": "eval", "outcome": "ok", "arguments": {"code": f"return 'boom: {literal}'"}}],
        invocation_ledger=[{"completion_status": "done", "completion_source": "execution_graph",
                            "completion_detail": f'"{literal}"'}], detail=f'"{literal}"')
    assert classify_case(result, finish) == "semantic_mismatch"
    assert classify_case(result, {"status": "complete_with_skips"}) == "semantic_mismatch"


@pytest.mark.parametrize("phrase", ["wall budget exhausted", "episode wall-clock budget exhausted",
                                    "model request budget exhausted"])
def test_model_terminal_reason_naming_a_budget_is_not_budget_evidence(phrase):
    finish = {"status": "complete", "resource_limit_reason": None}
    # completed wrong answer whose returned value / nested stop reason quote a budget phrase
    result = completed_wrong(detail=f'"{phrase}"', invocation_ledger=[
        {"completion_status": "quiesced", "completion_source": "execution_graph",
         "completion_detail": f"blocked: {phrase}"}])
    assert classify_case(result, finish) == "semantic_mismatch"
    # an explicit blocked/failed stop reason is model text even when the episode is not completed
    for prefix in ("blocked", "error"):
        stopped = {"outcome": {"status": "quiesced", "accepted": False, "detail": f"{prefix}: {phrase}",
                               "invocation_ledger": [], "action_ledger": [], "scope_failures": []}}
        assert classify_case(stopped, finish) == "semantic_mismatch"


def test_host_written_wall_budget_detail_stays_unscored():
    result = {"outcome": {"status": "quiesced", "accepted": False, "detail": "episode wall-clock budget exhausted",
                          "invocation_ledger": [], "action_ledger": [], "scope_failures": []}}
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "unresolved"


def test_host_transport_and_abort_evidence_still_classify_infrastructure():
    finish = {"status": "complete", "resource_limit_reason": None}
    unfinished = {"status": "quiesced", "accepted": False, "detail": "root eval did not finish",
                  "invocation_ledger": [], "action_ledger": [], "scope_failures": []}
    assert classify_case({"error": "ProviderRequestRetriesExhaustedError: HTTP 503", "outcome": unfinished},
                         finish) == "infrastructure_error"
    transport = {"status": "incomplete", "output_accounting": {"job_states": [{"state": "transport_failed"}]}}
    assert classify_case({"outcome": unfinished}, transport) == "infrastructure_error"
    # ...but a finished answer the oracle rejected stays a model failure
    assert classify_case({"error": "HTTP 503", **completed_wrong()}, finish) == "semantic_mismatch"
    aborted = completed_wrong(invocation_ledger=[{"completion_status": "failed",
                                                  "completion_source": "runtime_invocation_trace",
                                                  "completion_detail": "This operation was aborted"}])
    assert classify_case(aborted, finish) == "infrastructure_error"


def test_failed_terminal_is_scored_like_blocked():
    finish = {"status": "complete", "resource_limit_reason": None}
    for status in ("blocked", "failed"):
        result = {"outcome": {"status": "quiesced", "accepted": False, "rejection_reasons": ["expected_status", "answer"],
                              "scope_failures": [{"failure_kind": "runtime", "message": "sample not delivered"}],
                              "action_ledger": [{"name": "return_result", "outcome": "blocked",
                                                 "arguments": {"status": status, "reason": "sample not delivered"}}]}}
        assert classify_case(result, finish) == "semantic_mismatch", status


def test_no_action_stop_is_model_incomplete_even_after_scope_failures():
    # ts-host 9f13cce63: four complete replies without a tool call or result end the call.
    result = {"outcome": {"status": "quiesced", "accepted": False,
                          "detail": "model_incomplete: no_action (4 replies in a row called no tool and returned no result)",
                          "invocation_ledger": [], "action_ledger": [],
                          "scope_failures": [{"failure_kind": "runtime", "message": "x is not iterable"}]}}
    assert classify_case(result, {"status": "complete", "resource_limit_reason": None}) == "model_incomplete"


def test_stream_cut_in_a_child_call_stays_infrastructure_through_its_call_error_wrapper():
    finish = {"status": "complete", "resource_limit_reason": None}
    for message in ("terminated", "checkItem: failed: terminated", "nl@eval:1: failed: terminated",
                    "outer: failed: nl@eval:2: failed: terminated"):
        result = {"outcome": {"status": "quiesced", "accepted": False, "detail": "root eval did not finish",
                              "invocation_ledger": [], "action_ledger": [],
                              "scope_failures": [{"failure_kind": "runtime", "message": message}]}}
        assert classify_case(result, finish) == "infrastructure_error", message
    result["outcome"]["scope_failures"] = [{"failure_kind": "runtime", "message": "checkItem: failed: not terminated"}]
    assert classify_case(result, finish) != "infrastructure_error"
