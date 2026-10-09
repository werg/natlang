import importlib.util
from pathlib import Path
import hashlib
import json


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_neuralese_gold_text_delta.py"
SPEC = importlib.util.spec_from_file_location("gold_text_delta", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_anchor_omission_is_not_selected_delta_coverage():
    anchor = {"id": "held-anchor"}
    omissions = [{"id": "held-anchor", "reason": "unresolved_reference"},
                 {"id": "delta-row", "reason": "separate_delta_omission"}]
    receipt = {"excluded_train_exact_held_complete_documents": 0}
    result = MODULE.anchor_qualification(anchor, [], omissions, receipt)
    assert result["qualified"] is False
    assert result["reason"] == "anchor_omitted_by_shared_text_renderer"
    assert result["omission"]["id"] == "held-anchor"


def test_anchor_must_be_present_in_rendered_rows():
    anchor = {"id": "held-anchor"}
    result = MODULE.anchor_qualification(anchor, [{"id": "delta-row", "source_record_ids": ["delta-row"]}],
                                         [], {"excluded_train_exact_held_complete_documents": 0})
    assert result == {"qualified": False, "reason": "anchor_not_present_in_rendered_rows"}


def test_base_anchor_omission_is_separate_from_delta_omissions():
    omissions = [{"id": "held-anchor", "reason": "anchor_only"},
                 {"id": "delta-row", "reason": "selected_delta"}]
    assert MODULE.selected_delta_omissions(omissions, {"delta-row"}) == [
        {"id": "delta-row", "reason": "selected_delta"}]


def test_configured_read_context_must_exactly_match_frozen_combinator_source():
    source = """export const COMBINATORS = {
  read: { type: '(v: Neuralese<unknown>) => unknown',
    text: 'Read the value that v holds and return it exactly.' },
} as const;
"""
    definition_type, definition_text = MODULE.extract_combinator_definition(source, "read")
    body_hash = hashlib.sha256(definition_text.encode()).hexdigest()
    block_id = "nz1_revision1234567890"
    receipt = {
        "schema": "natlang.provider-expanded-read-context/1",
        "origin": "configured-function-definition",
        "definition": {"id": "nz-fn:" + block_id, "revision": "revision123456"},
        "block": {"id": block_id, "type": f"Neuralese<{definition_type}>",
                  "body": definition_text, "body_sha256": body_hash},
        "readout": {"schema": "natlang.text-template-readout/1", "call": "return_result",
                    "value": "decode", "value_type": "string", "read_body_id": block_id,
                    "read_source_sha256": body_hash, "learned_vectors": False,
                    "qualification_certificate": False, "training_admission": False},
    }
    MODULE.validate_configured_read_context(receipt, definition_type, definition_text)

    # Simulate a truncated body with every local digest updated consistently.
    truncated = definition_text[:-1]
    truncated_hash = hashlib.sha256(truncated.encode()).hexdigest()
    forged = json.loads(json.dumps(receipt))
    forged["block"]["body"] = truncated
    forged["block"]["body_sha256"] = truncated_hash
    forged["readout"]["read_source_sha256"] = truncated_hash
    try:
        MODULE.validate_configured_read_context(forged, definition_type, definition_text)
    except ValueError as exc:
        assert "exact frozen-runtime source" in str(exc)
    else:
        raise AssertionError("internally rehashed truncated instructions must not be accepted")


def _runtime_pin_fixture(tmp_path):
    runtime_dir = tmp_path / "runtime-v1"
    runtime_dir.mkdir()
    runtime_path = runtime_dir / "frozen-runtime.json"
    runtime_path.write_text('{"schema":"test-runtime"}\n')
    return runtime_path, _sha(runtime_path)


def test_runtime_manifest_can_be_bound_directly_from_approval_pin(tmp_path):
    runtime_path, runtime_sha = _runtime_pin_fixture(tmp_path)
    approval = {"input_pins": {"runtime-v1/frozen-runtime.json": {"sha256": runtime_sha}}}
    assert MODULE.pinned_runtime_manifest(approval, tmp_path) == runtime_path.resolve()


def test_runtime_manifest_can_be_bound_through_pinned_materialization_receipt(tmp_path):
    runtime_path, runtime_sha = _runtime_pin_fixture(tmp_path)
    receipt = tmp_path / "materialization" / "receipt.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({
        "converter": {
            "runtime_manifest_path": "runtime-v1/frozen-runtime.json",
            "runtime_manifest_sha256": runtime_sha,
        }
    }) + "\n")
    approval = {"input_pins": {"materialization/receipt.json": {"sha256": _sha(receipt)}}}
    assert MODULE.pinned_runtime_manifest(approval, tmp_path) == runtime_path.resolve()


def test_runtime_manifest_provenance_rejects_changed_receipt_or_runtime(tmp_path):
    runtime_path, runtime_sha = _runtime_pin_fixture(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"converter": {
        "runtime_manifest_path": "runtime-v1/frozen-runtime.json",
        "runtime_manifest_sha256": runtime_sha,
    }}) + "\n")
    pin = {"sha256": _sha(receipt)}
    approval = {"input_pins": {"receipt.json": pin}}
    receipt.write_text(receipt.read_text() + " ")
    try:
        MODULE.pinned_runtime_manifest(approval, tmp_path)
    except ValueError as exc:
        assert "input pin changed" in str(exc)
    else:
        raise AssertionError("a changed intermediate receipt must not be trusted")

    receipt.write_text(json.dumps({"converter": {
        "runtime_manifest_path": "runtime-v1/frozen-runtime.json",
        "runtime_manifest_sha256": runtime_sha,
    }}) + "\n")
    approval["input_pins"]["receipt.json"]["sha256"] = _sha(receipt)
    runtime_path.write_text('{"schema":"changed-runtime"}\n')
    try:
        MODULE.pinned_runtime_manifest(approval, tmp_path)
    except ValueError as exc:
        assert "runtime manifest hash mismatch" in str(exc)
    else:
        raise AssertionError("a changed runtime manifest must not be trusted")


def test_runtime_manifest_provenance_rejects_ambiguous_chains(tmp_path):
    first_path, first_sha = _runtime_pin_fixture(tmp_path)
    second_dir = tmp_path / "runtime-v2"
    second_dir.mkdir()
    second_path = second_dir / "frozen-runtime.json"
    second_path.write_text('{"schema":"second-runtime"}\n')
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"converters": [
        {"runtime_manifest_path": "runtime-v1/frozen-runtime.json",
         "runtime_manifest_sha256": first_sha},
        {"runtime_manifest_path": "runtime-v2/frozen-runtime.json",
         "runtime_manifest_sha256": _sha(second_path)},
    ]}) + "\n")
    approval = {"input_pins": {"receipt.json": {"sha256": _sha(receipt)}}}
    try:
        MODULE.pinned_runtime_manifest(approval, tmp_path)
    except ValueError as exc:
        assert "more than one runtime manifest" in str(exc)
    else:
        raise AssertionError("conflicting runtime provenance must fail closed")


def test_runtime_hash_without_a_path_is_not_treated_as_a_manifest_binding(tmp_path):
    runtime_path, runtime_sha = _runtime_pin_fixture(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"collector": {"runtime_manifest_sha256": runtime_sha}}) + "\n")
    approval = {"input_pins": {"receipt.json": {"sha256": _sha(receipt)}}}
    assert MODULE.pinned_runtime_manifest(approval, tmp_path) is None

    receipt.write_text(json.dumps({"collector": {"runtime_manifest_path": "runtime-v1/frozen-runtime.json"}}) + "\n")
    approval["input_pins"]["receipt.json"]["sha256"] = _sha(receipt)
    try:
        MODULE.pinned_runtime_manifest(approval, tmp_path)
    except ValueError as exc:
        assert "malformed runtime-manifest provenance" in str(exc)
    else:
        raise AssertionError("a path without its manifest digest must fail closed")


def test_missing_same_run_writer_witness_omits_only_its_record():
    block_id = "nz1_" + "b" * 52
    body = "Authenticated read context body."
    body_hash = hashlib.sha256(body.encode()).hexdigest()
    invocation = "sampled-call/4"
    target = {"tool_calls": []}
    target_hash = hashlib.sha256(json.dumps(target, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    response_hash = "d" * 64
    read = {"version": "reduction-trace/1", "kind": "block_read", "call_id": invocation,
            "node": "read-node", "block": block_id}
    turn = {"version": "reduction-trace/1", "kind": "model_turn", "call_id": invocation,
            "node": "turn-node", "inputs": [{"node": "read-node", "block": block_id}]}
    receipt = {
        "schema": "natlang.provider-expanded-read-context/2", "origin": "same-run-producer",
        "invocation_id": invocation, "parent_invocation_id": "parent-call/2",
        "source_row_sha256": "a" * 64, "trace_sha256": "e" * 64,
        "transport_provenance_sha256": "f" * 64, "raw_request_sha256": "1" * 64,
        "rendered_request_sha256": "2" * 64, "source_request_sha256": "3" * 64,
        "source_response_sha256": response_hash, "source_action_target_sha256": target_hash,
        "source_trajectory_index": 4, "context_occurrences": 1,
        "learned_vectors": False, "qualification_certificate": False, "training_admission": False,
        "writer_target_selected": False, "writer_witness": None,
        "block": {"id": block_id, "type": "Neuralese<string>", "body": body, "body_sha256": body_hash},
        "block_read": read, "model_turn": turn,
        "producer_write": {"call_id": "producer-call/2", "node": "write-node"},
    }
    affected = {"id": "affected", "target": target, "messages": [{"type": "neuralese", "id": block_id}],
                "decision": {"source_raw_response_sha256": response_hash},
                "source_ref": {"invocation_id": invocation, "source_row_sha256": "a" * 64,
                                "provider_expanded_read_contexts": [receipt]}}
    unrelated = {"id": "unrelated", "target": {}, "messages": [], "source_ref": {}}
    bindings = MODULE.bind_exact_provider_contexts([affected, unrelated], {}, Path.cwd())
    assert affected["_text_context_omissions"][0]["reason"] == "missing_authenticated_writer_witness"
    assert not unrelated.get("_text_context_omissions")
    assert bindings == [{"record_id": "affected", "status": "omitted_unbound_provider_context",
                         "omissions": affected["_text_context_omissions"]}]


def test_top_level_provider_read_accepts_only_matching_complete_root_capture():
    block_id = "nz1_" + "c" * 52
    body = "Authenticated top-level read context."
    invocation = "root-call/9"
    target = {"tool_calls": []}
    target_hash = hashlib.sha256(json.dumps(target, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    response_hash = "d" * 64
    receipt = {
        "schema": "natlang.provider-expanded-read-context/2", "origin": "same-run-producer",
        "invocation_id": invocation, "parent_invocation_id": None,
        "source_row_sha256": "a" * 64, "trace_sha256": "e" * 64,
        "transport_provenance_sha256": "f" * 64, "raw_request_sha256": "1" * 64,
        "rendered_request_sha256": "2" * 64, "source_request_sha256": "3" * 64,
        "source_response_sha256": response_hash, "source_action_target_sha256": target_hash,
        "source_trajectory_index": 4, "context_occurrences": 2,
        "learned_vectors": False, "qualification_certificate": False, "training_admission": False,
        "writer_target_selected": False, "writer_witness": {"schema": "test-witness"},
        "block": {"id": block_id, "type": "Neuralese<string>", "body": body,
                  "body_sha256": hashlib.sha256(body.encode()).hexdigest()},
        "block_read": {"version": "reduction-trace/1", "kind": "block_read", "call_id": invocation,
                       "node": "read-node", "block": block_id},
        "model_turn": {"version": "reduction-trace/1", "kind": "model_turn", "call_id": invocation,
                       "node": "turn-node", "inputs": [{"node": "read-node", "block": block_id}]},
        "producer_write": {"call_id": "producer-call/2", "node": "write-node"},
    }
    record = {
        "id": "root-read", "target": target, "messages": [
            {"type": "neuralese", "id": block_id}, {"type": "read", "name": "soft-state:" + block_id}],
        "decision": {"source_raw_response_sha256": response_hash},
        "neuralese_conversion": {"external_context_inputs": [{
            "block_id": block_id, "target_write_name": "soft-state:" + block_id}]},
        "source_ref": {
            "invocation_id": invocation, "source_row_sha256": "a" * 64,
            "provider_expanded_read_contexts": [receipt],
            "host_result_capture": {"capture": {
                "call_id": invocation, "capture_kind": "invocation_output", "complete": True,
                "parent_call_id": None}, "validation": {"valid": True}},
        },
    }
    bindings = MODULE.bind_exact_provider_contexts([record], {}, Path.cwd())
    assert bindings[0]["record_id"] == "root-read"
    bound = record["neuralese_conversion"]["external_context_inputs"][-1]
    assert bound["parent_invocation_id"] is None
    assert bound["context_occurrences"] == 2

    forged = json.loads(json.dumps(record))
    forged["source_ref"]["host_result_capture"]["capture"]["parent_call_id"] = "unbound-child"
    try:
        MODULE.bind_exact_provider_contexts([forged], {}, Path.cwd())
    except ValueError as exc:
        assert "malformed or mismatched exact provider context receipt" in str(exc)
    else:
        raise AssertionError("a parentless receipt without a matching root capture must remain held")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integration_adoption_fixture(tmp_path):
    artifacts = {}
    for key, name in {
        "assembled_native_records": "native.jsonl",
        "assembled_recurrence_records": "recurrence.jsonl",
        "assembled_native_pieces": "native-pieces.jsonl",
        "assembled_recurrence_pieces": "recurrence-pieces.jsonl",
        "cumulative_text": "text.jsonl",
        "cumulative_text_provenance": "provenance.jsonl",
    }.items():
        path = tmp_path / name
        path.write_text(name + "\n")
        artifacts[key] = {"path": str(path.relative_to(tmp_path)), "sha256": _sha(path)}
    text_path = tmp_path / "text.jsonl"
    manifest = tmp_path / "assembly-manifest.json"
    manifest.write_text(json.dumps({
        "outputs": {"text.jsonl": {"sha256": _sha(text_path), "bytes": text_path.stat().st_size}},
        "composition": {"text": {"rows": 4612, "train": 2908, "test": 1704,
                                  "tokenizer_sha256": "e" * 64}},
    }))
    artifacts["assembly_manifest"] = {"path": manifest.name, "sha256": _sha(manifest)}
    integration = tmp_path / "integration-review.json"
    integration.write_text(json.dumps({
        "schema": "natlang.v18-action45-admitted84-native-text-integration-review/1",
        "status": "verified exact admitted native/text delta; ready for registry publication and DGX sync",
        "scope": "Exact84 root-admitted native SFT actions plus faithful shared-renderer ordinary text only; no new recurrence rows and no active GPU input selection.",
        "counts": {
            "native": {"selected_delta": 7, "total": 5374, "train": 3339, "test": 2035},
            "recurrence": {"added_rows": 0, "total": 4961},
            "text": {"selected_delta_actions_covered": 7, "omissions_for_delta": 0,
                     "total": 4612, "train": 2908, "test": 1704},
        },
        "limits": {"no_child_or_parent_trajectory_admission": True,
                   "no_qualification_claim": True, "no_gpu_launch_or_input_change": True},
        "admission": {"active_training_inputs_changed": False,
                      "learned_writer_qualification": False, "recurrence_admission": False,
                      "whole_trajectories": 0},
        "artifacts": artifacts,
    }))
    adoption = {
        "schema": "natlang.root-integration-adoption/1",
        "decision": "approve-exact-native-and-text-assembly-for-next-reviewed-run",
        "scope": {"native_action_SFT": True, "ordinary_text_derivation": True,
                  "active_GPU_inputs_changed": False, "new_recurrence_admission": False,
                  "learned_writer_qualification": False, "whole_trajectory_admission": False,
                  "runtime_qualification": False, "DPO_admission": False},
        "integration_receipt": integration.name,
        "integration_receipt_sha256": _sha(integration),
        "independent_checks": {"artifact_pins_verified": 6,
                               "native_base_prefix_byte_identical": True,
                               "text_base_prefix_byte_identical": True,
                               "text_provenance_base_prefix_byte_identical": True,
                               "delta_omissions": 0,
                               "suffix_decode_mismatches": 0,
                               "exact_admitted_target_message_split_group_matches": 7,
                               "native_rows": 5374, "native_train": 3339, "native_test": 2035,
                               "text_rows": 4612, "text_train": 2908, "text_test": 1704},
    }
    return adoption, artifacts


def test_root_integration_adoption_normalizes_exact_prefix_pins(tmp_path):
    adoption, artifacts = _integration_adoption_fixture(tmp_path)
    normalized = MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    assert normalized["schema"] == "natlang.root-integration-adoption/1"
    assert normalized["artifacts"]["text"]["sha256"] == artifacts["cumulative_text"]["sha256"]
    assert normalized["artifacts"]["provenance"]["sha256"] == artifacts["cumulative_text_provenance"]["sha256"]
    assert normalized["artifacts"]["recurrence"]["sha256"] == artifacts["assembled_recurrence_records"]["sha256"]


def test_adopted_text_prefix_metadata_uses_reviewed_manifest_hash_count_and_tokenizer(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    binding = MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    metadata = MODULE.adopted_text_prefix_metadata(binding, root=tmp_path)
    assert metadata["documents"] == 4612
    assert metadata["train_documents"] == 2908
    assert metadata["test_documents"] == 1704
    assert metadata["tokenizer_sha256"] == "e" * 64


def test_adopted_text_prefix_metadata_rejects_manifest_hash_mismatch(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    binding = MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    manifest = tmp_path / "assembly-manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    try:
        MODULE.adopted_text_prefix_metadata(binding, root=tmp_path)
    except ValueError as exc:
        assert "manifest" in str(exc)
    else:
        raise AssertionError("an unpinned assembly manifest must be rejected")


def test_adopted_text_prefix_metadata_rejects_manifest_count_and_tokenizer_mismatch(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    binding = MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    manifest = tmp_path / "assembly-manifest.json"
    data = json.loads(manifest.read_text())
    data["composition"]["text"]["rows"] = 1
    data["composition"]["text"].pop("tokenizer_sha256")
    manifest.write_text(json.dumps(data))
    review = tmp_path / adoption["integration_receipt"]
    receipt = json.loads(review.read_text())
    receipt["artifacts"]["assembly_manifest"]["sha256"] = _sha(manifest)
    review.write_text(json.dumps(receipt))
    adoption["integration_receipt_sha256"] = _sha(review)
    binding = MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    try:
        MODULE.adopted_text_prefix_metadata(binding, root=tmp_path)
    except ValueError as exc:
        assert "tokenizer, or counts" in str(exc)
    else:
        raise AssertionError("manifest text counts and tokenizer must agree with the root adoption review")


def test_root_adoption_checks_legacy_sibling_receipt_metadata(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    binding = MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"documents": 4612, "train_documents": 2908,
                                   "test_documents": 1704, "tokenizer_sha256": "e" * 64,
                                   "renderer_code": {"old": "pin"}}))
    metadata = MODULE.resolve_base_text_prefix_metadata(binding, receipt, root=tmp_path)
    assert metadata["source_assembly_manifest"]["sha256"] == _sha(tmp_path / "assembly-manifest.json")
    assert metadata["legacy_sibling_receipt_metadata"]["renderer_code"] == {"old": "pin"}
    assert metadata["omitted_records"] is None


def test_root_adoption_rejects_legacy_sibling_metadata_override(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    binding = MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"documents": 4612, "train_documents": 2908,
                                   "test_documents": 1704, "tokenizer_sha256": "0" * 64}))
    try:
        MODULE.resolve_base_text_prefix_metadata(binding, receipt, root=tmp_path)
    except ValueError as exc:
        assert "conflicts with root-adopted" in str(exc)
    else:
        raise AssertionError("a mutable sibling receipt cannot override root-adopted tokenizer facts")


def test_legacy_root_adoption_without_assembly_manifest_keeps_receipt_path(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    integration_path = tmp_path / adoption["integration_receipt"]
    integration = json.loads(integration_path.read_text())
    integration["artifacts"].pop("assembly_manifest")
    integration_path.write_text(json.dumps(integration))
    adoption["integration_receipt_sha256"] = _sha(integration_path)
    binding = MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt = {"documents": 4612, "train_documents": 2908, "test_documents": 1704,
               "tokenizer_sha256": "e" * 64, "omitted_records": 7,
               "unresolved_omissions": [{"id": "old-omission"}]}
    receipt_path.write_text(json.dumps(receipt))
    metadata = MODULE.resolve_base_text_prefix_metadata(binding, receipt_path, root=tmp_path)
    assert metadata == receipt


def test_unknown_assembled_base_omission_inventory_stays_unknown_when_composed():
    assert MODULE.accumulate_known(None, 2) is None
    assert MODULE.extend_known(None, [{"id": "delta-omission"}]) is None
    assert MODULE.accumulate_known(5, 2) == 7
    assert MODULE.extend_known([{"id": "base-omission"}], [{"id": "delta-omission"}]) == [
        {"id": "base-omission"}, {"id": "delta-omission"}]


def test_root_adopted_assembly_does_not_claim_unreviewed_prefix_dedup_or_exclusion_counts(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    binding = MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    metadata = MODULE.adopted_text_prefix_metadata(binding, root=tmp_path)
    assert metadata["omitted_records"] is None
    assert metadata["duplicate_same_split_documents_deduplicated"] is None
    assert metadata["excluded_train_exact_held_complete_documents"] is None


def _mixed_root_action_receipt(tmp_path):
    review = tmp_path / "review.json"
    review.write_text('{"review":"pinned"}')
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"input":true}\n')
    rows = []
    dispositions = ["admit-ordinary-native-action"] * 14 + [
        "hold-source-required-neuralese-reader-contract"] * 3 + [
        "hold-ambiguous-source-read-scope"] * 2 + [
        "reject-action-failed"] * 3 + [
        "exclude-already-admitted-case01-duplicate"] * 3
    for index, decision in enumerate(dispositions):
        rows.append({"native_id": f"native-{index}", "decision": decision,
                     "training_admission": decision == "admit-ordinary-native-action",
                     "split": "train", "source_group": "source-group",
                     "source_groups": ["source-group"]})
    return {
        "schema": "natlang.root-per-action-training-admission/1",
        "review_path": "review.json", "review_sha256": _sha(review),
        "input_pins": {"input.jsonl": {"sha256": _sha(input_path), "bytes": input_path.stat().st_size}},
        "rows": rows, "admitted_native_count": 14,
        "held_source_contract_final_count": 3, "held_ambiguous_source_read_scope_count": 2,
        "failed_count": 3, "already_adopted_count": 3,
        "whole_trajectory_admission": False, "runtime_qualification": False,
        "active_gpu_inputs_changed": False, "new_world_credit": False,
    }


def test_mixed_root_per_action_receipt_selects_only_14_exactly_admitted_rows(tmp_path):
    receipt = _mixed_root_action_receipt(tmp_path)
    admitted = MODULE.admitted_root_per_action_rows(receipt, root=tmp_path)
    assert len(admitted) == 14
    assert all(row["training_admission"] is True
               and row["decision"] == "admit-ordinary-native-action" for row in admitted)


def test_per_action_group_binding_compares_complete_group_list():
    row = {"source_groups": ["inline-curriculum:case-a", "v17:case-a"]}
    assert MODULE.admission_source_groups_match(
        {"source_groups": ["inline-curriculum:case-a", "v17:case-a"]}, row)
    assert MODULE.admission_source_groups_match(
        {"source_group": "inline-curriculum:case-a"},
        {"source_groups": ["inline-curriculum:case-a"]})
    assert not MODULE.admission_source_groups_match(
        {"source_group": "inline-curriculum:case-a"}, row)
    assert not MODULE.admission_source_groups_match(
        {"source_groups": ["v17:case-a"]}, row)


def test_mixed_root_per_action_receipt_rejects_true_flag_on_held_row(tmp_path):
    receipt = _mixed_root_action_receipt(tmp_path)
    receipt["rows"][14]["training_admission"] = True
    try:
        MODULE.admitted_root_per_action_rows(receipt, root=tmp_path)
    except ValueError as exc:
        assert "conflicts with decision" in str(exc)
    else:
        raise AssertionError("a held row with a true admission flag must be rejected")


def test_mixed_root_per_action_receipt_rejects_admitted_flag_with_wrong_decision(tmp_path):
    receipt = _mixed_root_action_receipt(tmp_path)
    receipt["rows"][0]["decision"] = "hold-ambiguous-source-read-scope"
    try:
        MODULE.admitted_root_per_action_rows(receipt, root=tmp_path)
    except ValueError as exc:
        assert "conflicts with decision" in str(exc)
    else:
        raise AssertionError("an admission flag cannot override a non-admit decision")


def test_mixed_root_per_action_receipt_rejects_count_conflict(tmp_path):
    receipt = _mixed_root_action_receipt(tmp_path)
    receipt["admitted_native_count"] = 13
    try:
        MODULE.admitted_root_per_action_rows(receipt, root=tmp_path)
    except ValueError as exc:
        assert "admitted_native_count" in str(exc)
    else:
        raise AssertionError("the receipt's admitted count must match its admitted rows")


def test_mixed_root_per_action_receipt_cannot_overlay_a_held_target(tmp_path):
    receipt = _mixed_root_action_receipt(tmp_path)
    admitted_ids = {row["native_id"] for row in receipt["rows"][:14]}
    admitted_ids.add(receipt["rows"][14]["native_id"])
    try:
        MODULE.admitted_root_per_action_rows(receipt, delta_ids=admitted_ids, root=tmp_path)
    except ValueError as exc:
        assert "do not equal root-admitted" in str(exc)
    else:
        raise AssertionError("a held target must not enter the admitted delta")


def test_tokenizers_backend_snapshot_fallback_preserves_serialized_fast_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    original = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]",
                                       chat_template="{{ messages[0]['content'] }}")
    original.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "TokenizersBackend"
    config["clean_up_tokenization_spaces"] = False
    config["padding_side"] = "left"
    config["truncation_side"] = "left"
    config["model_max_length"] = 12345
    config_path.write_text(json.dumps(config))

    loaded = MODULE.load_pinned_tokenizer(tmp_path)
    assert loaded.get_vocab() == original.get_vocab()
    assert loaded.all_special_tokens == original.all_special_tokens
    assert loaded.chat_template == original.chat_template
    assert loaded.encode("hello") == original.encode("hello")
    assert loaded.clean_up_tokenization_spaces is False
    assert loaded.padding_side == "left"
    assert loaded.truncation_side == "left"
    assert loaded.model_max_length == 12345


def test_tokenizers_backend_fallback_requires_exact_class_and_serialized_backend(tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    original = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    original.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "OtherBackend"
    config_path.write_text(json.dumps(config))
    try:
        MODULE.load_pinned_tokenizer(tmp_path)
    except ValueError as exc:
        assert "OtherBackend" in str(exc)
    else:
        raise AssertionError("unknown tokenizer classes must not enter the compatibility fallback")

    config["tokenizer_class"] = "TokenizersBackend"
    config_path.write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").unlink()
    try:
        MODULE.load_pinned_tokenizer(tmp_path)
    except ValueError as exc:
        assert "TokenizersBackend" in str(exc)
    else:
        raise AssertionError("missing serialized backend must fail instead of falling back loosely")


def test_pinned_tokenizer_class_and_backend_are_checked_after_auto_load_success(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    loaded = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    loaded.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "OtherBackend"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *_args, **_kwargs: loaded))
    try:
        MODULE.load_pinned_tokenizer(tmp_path)
    except ValueError as exc:
        assert "OtherBackend" in str(exc)
        assert "class mismatch" in str(exc)
    else:
        raise AssertionError("AutoTokenizer success must not bypass the exact configured class")

    config["tokenizer_class"] = "TokenizersBackend"
    config_path.write_text(json.dumps(config))
    other_backend = Tokenizer(models.WordLevel({"[UNK]": 0, "different": 1}, unk_token="[UNK]"))
    tmp_backend = tmp_path / "other-tokenizer.json"
    other_backend.save(str(tmp_backend))
    (tmp_path / "tokenizer.json").write_bytes(tmp_backend.read_bytes())
    try:
        MODULE.load_pinned_tokenizer(tmp_path)
    except ValueError as exc:
        assert "differs from pinned tokenizer.json" in str(exc)
    else:
        raise AssertionError("matching wrapper class must not bypass a changed serialized backend")


def test_root_integration_adoption_rejects_mismatched_artifact(tmp_path):
    adoption, artifacts = _integration_adoption_fixture(tmp_path)
    artifacts["cumulative_text"]["sha256"] = "0" * 64
    integration = tmp_path / adoption["integration_receipt"]
    review = json.loads(integration.read_text())
    review["artifacts"] = artifacts
    integration.write_text(json.dumps(review))
    adoption["integration_receipt_sha256"] = _sha(integration)
    try:
        MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    except ValueError as exc:
        assert "cumulative_text" in str(exc)
    else:
        raise AssertionError("mismatched cumulative text must be rejected")


def test_root_integration_adoption_rejects_unverified_review_counts(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    integration = tmp_path / adoption["integration_receipt"]
    review = json.loads(integration.read_text())
    review["counts"]["recurrence"]["added_rows"] = 1
    integration.write_text(json.dumps(review))
    adoption["integration_receipt_sha256"] = _sha(integration)
    try:
        MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    except ValueError as exc:
        assert "inconsistent action, split, recurrence, or text counts" in str(exc)
    else:
        raise AssertionError("an integration review adding recurrence rows must be rejected")


def test_root_integration_adoption_rejects_receipt_path_escape(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    adoption["integration_receipt"] = "../outside-review.json"
    try:
        MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    except ValueError as exc:
        assert "escapes the repository root" in str(exc)
    else:
        raise AssertionError("integration receipts must stay inside the canonical repository")
