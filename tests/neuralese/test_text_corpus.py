import hashlib
import json

from natlang_neuralese.data.text_corpus import gold_text_rows


class _Backend:
    def to_str(self):
        return "{}"


class _Tokenizer:
    chat_template = "fixture-template"
    all_special_tokens = []
    backend_tokenizer = _Backend()

    def get_vocab(self):
        return {"x": 0}

    def apply_chat_template(self, messages, **_kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) for char in text]}


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_gold_text_hydrates_marker_output_writer_and_preserves_exact_code():
    block_id = "nz1_" + "a" * 52
    body = "Supported facts from this pass; unresolved fields remain pending."
    marker = f"<|neuralese|>{body}<|/neuralese|>"
    code = f"const notes: Neuralese<string> = {marker}; return notes;"
    raw = json.dumps({"code": code, "finish": True}, ensure_ascii=False, separators=(",", ":"))
    sidecar = {
        "schema": "natlang.neuralese-code/1", "mode": "marker-output", "code_sha256": _sha(code),
        "parts": [
            {"type": "text", "text": "const notes: Neuralese<string> = "},
            {"$write": {"name": f"soft-state:{block_id}", "type": "Neuralese<string>",
                        "source": body, "code_source": marker}},
            {"type": "text", "text": "; return notes;"},
        ],
        "sites": [{"name": f"soft-state:{block_id}", "purpose": "validated-runtime-soft-state-writer",
                    "block_id": block_id, "body_sha256": _sha(body),
                    "writer_call_id": "eval-1", "writer_node": "eval-1#1",
                    "writer_action": "eval-finish-true"}],
    }
    group = ["case:shared-source-group"]
    common = {"split": "test", "source_groups": group,
              "training_admission": {"approved": True},
              "decision": {"training_approved": True, "failed_action": False}}
    writer = {**common, "id": "writer", "source_ref": {"source_row_sha256": "1" * 64},
              "messages": [{"role": "user", "content": "Write the accumulated note."}],
              "target": {"role": "assistant", "tool_calls": [{"id": "eval-1",
                  "function": {"name": "eval", "arguments": raw}, "neuralese_code": sidecar}]}}
    reader = {**common, "id": "reader", "source_ref": {"source_row_sha256": "1" * 64},
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": "Prior notes: "},
                  {"type": "neuralese", "id": block_id},
              ]}],
              "target": {"role": "assistant", "content": "Continue from the prior notes."}}
    train = {**common, "id": "train-anchor", "split": "train", "source_groups": ["case:train-anchor"],
             "messages": [{"role": "user", "content": "Independent training anchor."}],
             "target": {"role": "assistant", "content": "Anchor answer."}}

    rows, receipt, omissions, provenance = gold_text_rows([writer, reader, train], [], tokenizer=_Tokenizer())

    assert not omissions
    assert len(rows) == 3
    writer_text = next(row["text"] for row in rows if row["id"] == "writer")
    reader_text = next(row["text"] for row in rows if row["id"] == "reader")
    assert code in writer_text
    assert f"<|neuralese|>{body}<|/neuralese|>" in reader_text
    reader_provenance = next(row for row in provenance if row["id"] == "reader")
    assert reader_provenance["neuralese_context_attestations"][0]["writer_record_id"] == "writer"
    assert receipt["task_or_trajectory_admission_granted"] is False


def test_provider_expanded_soft_read_is_crisp_and_provenance_bound_to_existing_writer():
    block_id = "nz1_" + "c" * 52
    name = f"soft-state:{block_id}"
    body = "Observed note from the successful earlier producer."
    group = ["case:provider-read"]
    common = {"split": "test", "source_groups": group,
              "training_admission": {"approved": True},
              "decision": {"training_approved": True, "failed_action": False}}
    writer = {**common, "id": "provider-writer", "source_ref": {"source_row_sha256": "1" * 64},
              "messages": [{"role": "user", "content": "Write a note."}],
              "target": {"role": "assistant", "tool_calls": [{"function": {"name": "return_result",
                  "arguments": json.dumps({"status": "success", "value": {"$write": {
                      "name": name, "type": "Neuralese<string>", "source": body}}})}}]}}
    reader = {**common, "id": "provider-reader", "source_ref": {
                  "source_row_sha256": "2" * 64, "invocation_id": "call-reader"},
              "neuralese_conversion": {"external_context_inputs": [{
                  "schema": "natlang.external-context-input/1", "origin": "same-run-producer",
                  "learner_representation": "typed-read-linked-to-existing-writer",
                  "block_id": block_id, "type": "Neuralese<string>",
                  "body_sha256": _sha(body), "invocation_id": "call-reader",
                  "source_row_sha256": "2" * 64, "read_node": "call-reader#7",
                  "model_turn_node": "call-reader#turn1", "producer_write_node": "call-writer#6",
                  "transport_provenance_sha256": "3" * 64, "learned_vectors": False,
                  "qualification_certificate": False, "training_admission": False}]},
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": "Prior note: "},
                  {"type": "read", "name": name, "source": body}]}],
              "target": {"role": "assistant", "content": "Continue."}}
    anchor = {**common, "id": "provider-train-anchor", "split": "train",
              "source_groups": ["case:independent-anchor"],
              "messages": [{"role": "user", "content": "Independent training anchor."}],
              "target": {"role": "assistant", "content": "Anchor answer."}}

    rows, receipt, omissions, provenance = gold_text_rows([writer, reader, anchor], [], tokenizer=_Tokenizer())

    assert not omissions
    rendered = next(row["text"] for row in rows if row["id"] == "provider-reader")
    assert "Prior note: " + body in rendered
    assert "<|neuralese|>" not in rendered
    reader_provenance = next(row for row in provenance if row["id"] == "provider-reader")
    attest = reader_provenance["neuralese_context_attestations"]
    assert len(attest) == 1
    assert attest[0]["source_kind"] == "provider-expanded-same-run-read"
    assert attest[0]["writer_record_id"] == "provider-writer"
    assert attest[0]["read_node"] == "call-reader#7"
    assert receipt["hash_bound_reader_context_blocks"] == 1

    corrupt = json.loads(json.dumps(reader))
    corrupt["messages"][0]["content"][1]["source"] = "different body"
    _, _, rejected, _ = gold_text_rows([writer, corrupt, anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == "provider-reader" and "digest mismatch" in item.get("detail", "")
               for item in rejected)


def test_provider_expanded_context_only_read_hydrates_without_selected_writer_target():
    block_id = "nz1_" + "d" * 52
    body = "Exact body from an authenticated same-run typed input."
    body_sha = _sha(body)
    reader_invocation, writer_invocation = "reader-call", "writer-call"
    read_node, turn_node, write_node = "reader-call#4", "reader-call#turn1", "writer-call#8"
    row_sha, trace_sha = "a" * 64, "b" * 64
    raw_sha, rendered_sha, transport_sha = "c" * 64, "d" * 64, "e" * 64
    read = {"kind": "block_read", "call_id": reader_invocation, "node": read_node, "turn": turn_node, "seq": 10,
            "block": block_id, "inputs": [{"node": write_node, "block": block_id, "port": "block"}]}
    turn = {"kind": "model_turn", "call_id": reader_invocation, "node": turn_node,
            "inputs": [{"node": read_node, "port": "read", "block": block_id}]}
    read2 = {"kind": "block_read", "call_id": reader_invocation, "node": "reader-call#12", "seq": 14,
             "turn": "reader-call#turn2",
             "block": block_id, "inputs": [{"node": write_node, "block": block_id, "port": "block"}]}
    turn2 = {"kind": "model_turn", "call_id": reader_invocation, "node": "reader-call#turn2",
             "inputs": [{"node": read2["node"], "port": "read", "block": block_id}]}
    write = {"kind": "block_write", "call_id": writer_invocation, "node": write_node, "seq": 8,
             "block": block_id, "truncated": False, "producer": "text-marker-emulation",
             "source_kind": "typed-text-result", "result_type": "Neuralese<string>",
             "text_body_sha256": body_sha}
    provider_receipt = {"schema": "natlang.provider-expanded-read-context/2", "origin": "same-run-producer",
        "invocation_id": reader_invocation, "parent_invocation_id": "root-call", "source_row_sha256": row_sha,
        "trace_sha256": trace_sha, "transport_provenance_sha256": transport_sha,
        "raw_request_sha256": raw_sha, "rendered_request_sha256": rendered_sha,
        "block": {"id": block_id, "type": "Neuralese<string>", "body": body,
                  "body_sha256": body_sha, "learned_vectors": False},
        "block_read": read, "model_turn": turn,
        "additional_read_turn_pairs": [{"block_read": read2, "model_turn": turn2}], "context_occurrences": 1,
        "producer_write": write, "writer_source_class": "modern-typed-text-result",
        "writer_target_selected": False, "learned_vectors": False,
        "qualification_certificate": False, "training_admission": False}
    common = {"split": "test", "source_groups": ["case:context-only"],
              "training_admission": {"approved": True},
              "decision": {"training_approved": True, "failed_action": False}}
    reader = {**common, "id": "context-only-reader", "source_ref": {"source_row_sha256": row_sha,
        "trajectory_id": "run-context-only", "invocation_id": reader_invocation,
        "parent_invocation_id": "root-call", "provider_expanded_read_contexts": [provider_receipt]},
        "provenance": {"trace_sha256": trace_sha},
        "neuralese_conversion": {"external_context_inputs": [{
            "schema": "natlang.external-context-input/1", "origin": "same-run-producer",
            "learner_representation": "typed-read-from-authenticated-runtime-writer-event-context-only",
            "block_id": block_id, "type": "Neuralese<string>", "body_sha256": body_sha,
            "invocation_id": reader_invocation, "parent_invocation_id": "root-call",
            "source_row_sha256": row_sha, "trace_sha256": trace_sha,
            "read_node": read_node, "model_turn_node": turn_node, "producer_write_node": write_node,
            "additional_read_nodes": [read2["node"]], "additional_model_turn_nodes": [turn2["node"]],
            "producer_call_id": writer_invocation, "writer_source_class": "modern-typed-text-result",
            "writer_target_selected": False, "context_occurrences": 1,
            "transport_provenance_sha256": transport_sha, "raw_request_sha256": raw_sha,
            "rendered_request_sha256": rendered_sha, "learned_vectors": False,
            "qualification_certificate": False, "training_admission": False}]},
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Prior notes: "},
            {"type": "read", "name": f"soft-state:{block_id}", "source": body}]}],
        "target": {"role": "assistant", "content": "Continue from the exact note."}}
    anchor = {**common, "id": "context-anchor", "split": "train", "source_groups": ["case:anchor"],
        "messages": [{"role": "user", "content": "Independent anchor."}],
        "target": {"role": "assistant", "content": "Anchor."}}
    held_anchor = {**common, "id": "context-held-anchor", "split": "test", "source_groups": ["case:held-anchor"],
        "messages": [{"role": "user", "content": "Independent held anchor."}],
        "target": {"role": "assistant", "content": "Held anchor."}}

    rows, _, omissions, provenance = gold_text_rows([reader, anchor], [], tokenizer=_Tokenizer())
    assert not omissions
    rendered = next(row["text"] for row in rows if row["id"] == reader["id"])
    assert "Prior notes: " + body in rendered
    attest = next(row for row in provenance if row["id"] == reader["id"])["neuralese_context_attestations"][0]
    assert attest["source_kind"] == "provider-expanded-context-only-same-run-read"
    assert attest["writer_record_id"] is None
    assert attest["writer_target_selected"] is False

    typed_input = json.loads(json.dumps(reader))
    typed_input["id"] = "context-only-typed-message-reader"
    typed_input["messages"] = [{"role": "user", "content": [
        {"type": "text", "text": "Typed prior note: "},
        {"type": "neuralese", "id": block_id},
    ]}]
    typed_rows, _, typed_omissions, typed_provenance = gold_text_rows(
        [typed_input, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert not typed_omissions
    typed_rendered = next(row["text"] for row in typed_rows if row["id"] == typed_input["id"])
    assert f"Typed prior note: <|neuralese|>{body}<|/neuralese|>" in typed_rendered
    typed_attests = next(row for row in typed_provenance if row["id"] == typed_input["id"])[
        "neuralese_context_attestations"]
    assert len(typed_attests) == 1
    assert typed_attests[0]["source_kind"] == "provider-expanded-context-only-same-run-read"
    assert typed_attests[0]["writer_target_selected"] is False

    wrong_typed_id = json.loads(json.dumps(typed_input))
    wrong_typed_id["messages"][0]["content"][1]["id"] = "nz1_" + "f" * 52
    _, _, wrong_id_omissions, _ = gold_text_rows(
        [wrong_typed_id, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == typed_input["id"] for item in wrong_id_omissions)

    wrong_typed_body = json.loads(json.dumps(typed_input))
    wrong_typed_body["source_ref"]["provider_expanded_read_contexts"][0]["block"]["body_sha256"] = "f" * 64
    _, _, wrong_body_omissions, _ = gold_text_rows(
        [wrong_typed_body, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == typed_input["id"] for item in wrong_body_omissions)

    wrong_typed_turn = json.loads(json.dumps(typed_input))
    wrong_typed_turn["neuralese_conversion"]["external_context_inputs"][0]["model_turn_node"] = "reader-call#turn9"
    _, _, wrong_turn_omissions, _ = gold_text_rows(
        [wrong_typed_turn, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == typed_input["id"] for item in wrong_turn_omissions)

    repeated = json.loads(json.dumps(reader))
    repeated["source_ref"]["provider_expanded_read_contexts"][0]["context_occurrences"] = 2
    repeated["neuralese_conversion"]["external_context_inputs"][0]["context_occurrences"] = 2
    repeated["messages"].append({"role": "assistant", "content": None, "tool_calls": [{"id": "scope_0",
        "type": "function", "function": {"name": "scope", "arguments": json.dumps([
            {"type": "neuralese", "id": block_id}])}}]})
    repeated_rows, _, repeated_omissions, repeated_provenance = gold_text_rows(
        [repeated, anchor], [], tokenizer=_Tokenizer())
    assert not repeated_omissions
    repeated_text = next(row["text"] for row in repeated_rows if row["id"] == reader["id"])
    assert repeated_text.count(body) >= 2, "both authenticated input occurrences hydrate from the same exact body"
    repeated_attests = next(row for row in repeated_provenance if row["id"] == reader["id"])[
        "neuralese_context_attestations"]
    assert any(item.get("writer_record_id") is None and item.get("writer_target_selected") is False
               for item in repeated_attests), "the added occurrence stays context-only, with no selected writer target"

    bad_occurrence = json.loads(json.dumps(repeated))
    bad_occurrence["source_ref"]["provider_expanded_read_contexts"][0]["context_occurrences"] = 1
    bad_occurrence["neuralese_conversion"]["external_context_inputs"][0]["context_occurrences"] = 1
    _, _, wrong_count_omissions, _ = gold_text_rows(
        [bad_occurrence, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == reader["id"] and "occurrence count" in item.get("detail", "")
               for item in wrong_count_omissions)

    legacy = json.loads(json.dumps(reader))
    legacy_receipt = legacy["source_ref"]["provider_expanded_read_contexts"][0]
    legacy_receipt["writer_source_class"] = "legacy-text-marker-standin-eval-code"
    legacy_write = legacy_receipt["producer_write"]
    legacy_write.pop("producer")
    legacy_write.pop("source_kind")
    legacy_write.update({"emulation_version": "text-marker-standin/2", "marker_context": "eval-code",
                         "learned_vectors": False})
    legacy["neuralese_conversion"]["external_context_inputs"][0]["writer_source_class"] = \
        "legacy-text-marker-standin-eval-code"
    legacy_rows, _, legacy_omissions, legacy_provenance = gold_text_rows([legacy, anchor], [], tokenizer=_Tokenizer())
    assert not legacy_omissions
    legacy_attest = next(row for row in legacy_provenance if row["id"] == reader["id"])[
        "neuralese_context_attestations"][0]
    assert legacy_attest["source_kind"] == "provider-expanded-context-only-same-run-read"

    eval_return = json.loads(json.dumps(reader))
    eval_return_receipt = eval_return["source_ref"]["provider_expanded_read_contexts"][0]
    eval_return_receipt["writer_source_class"] = "legacy-text-marker-standin-eval-return"
    eval_return_receipt["producer_write"].update({"source": "eval-return", "marker_context": "return-result"})
    eval_return_receipt["writer_witness"] = {
        "kind": "completed-eval-return-host-reference", "source": "eval-return",
        "host_result_call_id": writer_invocation, "host_result_type": "Neuralese<string>",
        "host_result_value_sha256": "f" * 64, "completion_status": "done",
        "completion_source": "execution_graph", "completion_detail": f"\uE000{block_id}\uE001",
    }
    eval_return["neuralese_conversion"]["external_context_inputs"][0]["writer_source_class"] = \
        "legacy-text-marker-standin-eval-return"
    eval_return["neuralese_conversion"]["external_context_inputs"][0]["writer_witness"] = \
        eval_return_receipt["writer_witness"]
    target_sha = hashlib.sha256(_canonical(eval_return["target"]).encode("utf-8")).hexdigest()
    eval_return["decision"] = {"index": 8, "training_approved": True, "failed_action": False}
    for owner in (eval_return_receipt, eval_return["neuralese_conversion"]["external_context_inputs"][0]):
        owner.update({"source_action_target_sha256": target_sha, "source_trajectory_index": 8,
                      "source_request_sha256": "1" * 64, "source_response_sha256": "2" * 64})
    eval_rows, _, eval_omissions, eval_provenance = gold_text_rows(
        [eval_return, anchor], [], tokenizer=_Tokenizer())
    assert not eval_omissions
    eval_attest = next(row for row in eval_provenance if row["id"] == reader["id"])[
        "neuralese_context_attestations"][0]
    assert eval_attest["source_kind"] == "provider-expanded-context-only-same-run-read"
    assert "Prior notes: " + body in next(row["text"] for row in eval_rows if row["id"] == reader["id"])

    mismatched_action = json.loads(json.dumps(eval_return))
    mismatched_action["target"]["content"] = "Different selected action."
    _, _, action_omissions, _ = gold_text_rows([mismatched_action, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == reader["id"] for item in action_omissions)

    mismatched_request = json.loads(json.dumps(eval_return))
    mismatched_request["neuralese_conversion"]["external_context_inputs"][0]["source_request_sha256"] = "3" * 64
    _, _, request_omissions, _ = gold_text_rows(
        [mismatched_request, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == reader["id"] for item in request_omissions)

    forged_eval_return = json.loads(json.dumps(eval_return))
    forged_eval_return["source_ref"]["provider_expanded_read_contexts"][0]["writer_witness"][
        "completion_detail"] = f"\uE000{'nz1_' + 'e' * 52}\uE001"
    _, _, forged_omissions, _ = gold_text_rows(
        [forged_eval_return, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == reader["id"] for item in forged_omissions)

    # V4 modern schema/2 receipts can predate converter-side class and
    # occurrence metadata. Recover only the exact single typed-reference case.
    modern_v2 = json.loads(json.dumps(reader))
    modern_v2["source_ref"]["provider_expanded_read_contexts"][0].pop("writer_source_class")
    modern_v2["neuralese_conversion"]["external_context_inputs"][0].pop("writer_source_class")
    modern_v2["neuralese_conversion"]["external_context_inputs"][0].pop("context_occurrences")
    v2_rows, _, v2_omissions, v2_provenance = gold_text_rows(
        [modern_v2, anchor], [], tokenizer=_Tokenizer())
    assert not v2_omissions
    assert any(item["id"] == reader["id"] for item in v2_rows)
    v2_attest = next(row for row in v2_provenance if row["id"] == reader["id"])[
        "neuralese_context_attestations"][0]
    assert v2_attest.get("writer_source_class") is None

    wrong_v2_count = json.loads(json.dumps(modern_v2))
    wrong_v2_count["source_ref"]["provider_expanded_read_contexts"][0]["context_occurrences"] = 2
    _, _, wrong_v2_omissions, _ = gold_text_rows(
        [wrong_v2_count, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == reader["id"] and "single exact typed reference" in item.get("detail", "")
               for item in wrong_v2_omissions)

    forged_v2 = json.loads(json.dumps(modern_v2))
    forged_v2["source_ref"]["provider_expanded_read_contexts"][0]["producer_write"]["source_kind"] = "untyped"
    _, _, forged_v2_omissions, _ = gold_text_rows(
        [forged_v2, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == reader["id"] and "authenticate exact body and graph" in item.get("detail", "")
               for item in forged_v2_omissions)

    for source_kind, writer_class, witness in [
        ("return_result", "legacy-text-marker-standin-return-result", {
            "kind": "raw-return-result-value-equals-expanded-body", "source": "return_result",
            "host_result_call_id": writer_invocation, "host_result_type": "Neuralese<string>",
            "host_result_value_sha256": "6" * 64, "raw_response_sha256": "7" * 64}),
        ("eval-finish", "legacy-text-marker-standin-eval-finish", {
            "kind": "completed-eval-finish-host-reference", "source": "eval-finish",
            "host_result_call_id": writer_invocation, "host_result_type": "Neuralese<string>",
            "host_result_value_sha256": "8" * 64}),
    ]:
        old_result = json.loads(json.dumps(reader))
        old_receipt = old_result["source_ref"]["provider_expanded_read_contexts"][0]
        old_receipt["writer_source_class"] = writer_class
        old_receipt["writer_witness"] = witness
        old_receipt["producer_write"].update({"producer": "text-marker-emulation",
            "source_kind": "typed-text-result", "source": source_kind, "marker_context": "return-result"})
        old_result["neuralese_conversion"]["external_context_inputs"][0].update(
            {"writer_source_class": writer_class, "writer_witness": witness})
        old_rows, _, old_omissions, _ = gold_text_rows([old_result, anchor], [], tokenizer=_Tokenizer())
        assert not old_omissions
        assert any(item["id"] == reader["id"] for item in old_rows)
        bad_witness = json.loads(json.dumps(old_result))
        bad_witness["source_ref"]["provider_expanded_read_contexts"][0]["writer_witness"][
            "host_result_call_id"] = "other-call"
        _, _, bad_omissions, _ = gold_text_rows([bad_witness, anchor, held_anchor], [], tokenizer=_Tokenizer())
        assert any(item["id"] == reader["id"] for item in bad_omissions)

    corrupt = json.loads(json.dumps(reader))
    corrupt["source_ref"]["provider_expanded_read_contexts"][0]["block"]["body_sha256"] = "f" * 64
    _, _, rejected, _ = gold_text_rows([corrupt, anchor, held_anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == reader["id"] and "body and graph" in item.get("detail", "")
               for item in rejected)


def test_protected_target_sidecar_equivalence_is_exact_and_narrow():
    from natlang_neuralese.data.text_corpus import _canonical, _protected_target_sidecar_equivalence, _sha

    block_id = "nz1_" + "a" * 52
    prefix, body_source, code_source, suffix = "const fn = nl.with({input})`", "approved", "\ue000" + block_id + "\ue001", "`;"
    code = prefix + code_source + suffix
    site_name = "inline-site:test"
    start = len(prefix) - 1
    end = len(prefix) + len(code_source) + 1
    plan = {"capture_binding_plan": {"schema": "natlang.inline-capture-binding-plan/1", "syntax": "nl.with",
        "body_block_id": block_id, "body_source_sha256": _sha(body_source.encode()),
        "parent_invocation_id": "parent", "parent_scope_sha256": "1" * 64,
        "child_scope_sha256": "2" * 64,
        "captures": [{"name": "input", "type": "string", "mode": "snapshot", "value": "visible"}]}}
    sidecar = {"schema": "natlang.inline-instruction-code/1", "code_sha256": _sha(code.encode()),
               "parts": [{"type": "text", "text": prefix},
                         {"$write": {"name": site_name, "type": "Neuralese<string>", "source": body_source,
                                     "code_source": code_source}},
                         {"type": "text", "text": suffix}],
               "sites": [{"name": site_name, "code_span": {"start": start, "end": end}, "plan": plan}]}
    call = {"id": "scope-sidecar", "type": "function", "neuralese_code": sidecar,
            "function": {"name": "scope", "arguments": json.dumps({"code": code})}}
    target = {"role": "assistant", "content": None, "tool_calls": [call]}
    projected = json.loads(json.dumps(target))
    del projected["tool_calls"][0]["neuralese_code"]
    projected_sha = _sha(_canonical(projected).encode())
    sidecar_sha = _sha(_canonical(sidecar).encode())
    adapter = {"schema": "natlang.protected-target-inline-sidecar-equivalence/1",
               "kind": "remove-one-validated-neuralese-code-sidecar",
               "protected_target_sha256": _sha(_canonical(target).encode()),
               "materializer_target_sha256": projected_sha, "projection_sha256": projected_sha,
               "call_id": call["id"], "sidecar_code_sha256": sidecar["code_sha256"],
               "sidecar_sha256": sidecar_sha}
    receipt = {"source_action_target_sha256": projected_sha, "target_binding_adapter": adapter}
    metadata = {"source_action_target_sha256": projected_sha, "target_binding_adapter": adapter}
    record = {"target": target}
    assert _protected_target_sidecar_equivalence(record, receipt, metadata)

    wrong_arguments = json.loads(json.dumps(record))
    wrong_arguments["target"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"code": "return false"})
    assert not _protected_target_sidecar_equivalence(wrong_arguments, receipt, metadata)
    wrong_call = json.loads(json.dumps(receipt))
    wrong_call["target_binding_adapter"]["call_id"] = "wrong-call"
    assert not _protected_target_sidecar_equivalence(record, wrong_call, metadata)
    wrong_projection = json.loads(json.dumps(receipt))
    wrong_projection["target_binding_adapter"]["projection_sha256"] = "f" * 64
    assert not _protected_target_sidecar_equivalence(record, wrong_projection, metadata)
    extra_field = json.loads(json.dumps(record))
    extra_field["target"]["tool_calls"][0]["unexpected"] = True
    assert not _protected_target_sidecar_equivalence(extra_field, receipt, metadata)

    def retarget(changed_sidecar):
        changed = json.loads(json.dumps(record))
        changed["target"]["tool_calls"][0]["neuralese_code"] = changed_sidecar
        projected_changed = json.loads(json.dumps(changed["target"]))
        del projected_changed["tool_calls"][0]["neuralese_code"]
        projection_digest = _sha(_canonical(projected_changed).encode())
        changed_adapter = json.loads(json.dumps(adapter))
        changed_adapter.update({"protected_target_sha256": _sha(_canonical(changed["target"]).encode()),
            "materializer_target_sha256": projection_digest, "projection_sha256": projection_digest,
            "sidecar_sha256": _sha(_canonical(changed_sidecar).encode())})
        return changed, {"source_action_target_sha256": projection_digest,
                         "target_binding_adapter": changed_adapter}, {"source_action_target_sha256": projection_digest,
                         "target_binding_adapter": changed_adapter}

    for tamper in ("name", "type", "span"):
        changed_sidecar = json.loads(json.dumps(sidecar))
        if tamper == "name":
            changed_sidecar["parts"][1]["$write"]["name"] = "renamed-site"
        elif tamper == "type":
            changed_sidecar["parts"][1]["$write"]["type"] = "Neuralese<number>"
        else:
            changed_sidecar["sites"][0]["code_span"]["start"] += 1
        bad_record, bad_receipt, bad_metadata = retarget(changed_sidecar)
        assert not _protected_target_sidecar_equivalence(bad_record, bad_receipt, bad_metadata), tamper
