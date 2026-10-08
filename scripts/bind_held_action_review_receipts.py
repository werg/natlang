#!/usr/bin/env python3
"""Bind held action selections and raw receipts to exact review-document records.

The output is a new, additive receipt pair. It never edits provider outcomes or
the source review. The review-document, case-record, item-record, and derived
selection-record digests have distinct fields from the raw result file/row
digests.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_sha(value) -> str:
    return sha(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def lines(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bind(selections, receipts, summary, review_path: Path):
    review_bytes = review_path.read_bytes()
    review_sha = sha(review_bytes)
    review = json.loads(review_bytes)
    if summary.get("source_review_sha256") != review_sha:
        raise ValueError("selection summary does not bind the supplied review document")
    if summary.get("source_review_path") != str(review_path):
        # Repository-relative path identities are accepted when the summary uses one.
        resolved = review_path.resolve()
        if Path(summary.get("source_review_path", "")).name != review_path.name:
            raise ValueError("selection summary review path differs from supplied review document")
        if not resolved.is_file():
            raise ValueError("review document is unavailable")
    cases = review.get("cases")
    if not isinstance(cases, list) or len(selections) != len(receipts):
        raise ValueError("review cases or selection/receipt cardinality is invalid")

    bound_selections, bound_receipts = [], []
    for index, (selection, receipt) in enumerate(zip(selections, receipts)):
        record = selection.get("source_review_record")
        if not isinstance(record, dict):
            raise ValueError(f"selection {index} lacks source_review_record")
        case_index = record.get("case_index")
        if not isinstance(case_index, int) or isinstance(case_index, bool) or not 0 <= case_index < len(cases):
            raise ValueError(f"selection {index} has invalid case_index")
        case = cases[case_index]
        if case.get("case_index") != case_index or case.get("case_id") != record.get("case_id"):
            raise ValueError(f"selection {index} case identity differs from review document")
        items = [item for item in case.get("items", []) if item.get("item_key") == record.get("item_key")]
        if len(items) != 1:
            raise ValueError(f"selection {index} item key is not unique in its review case")
        item = items[0]
        bound_invocations = [entry for entry in item.get("question_bound_child_invocations", [])
                             if entry.get("invocation_id") == record.get("invocation_id")]
        if len(bound_invocations) != 1:
            raise ValueError(f"selection {index} invocation is not uniquely bound to this reviewed item")
        bound_invocation = bound_invocations[0]
        expected = {
            "source_id": item.get("source_record_id"),
            "source_group": (item.get("source_groups") or [None])[0],
            "split": item.get("source_split"),
            "item_path": item.get("item_path"),
            "item_body_sha256": item.get("item_body_sha256"),
            "question_sha256": item.get("question_sha256"),
            "evidence_sha256": item.get("evidence_sha256"),
            "expected_value": item.get("gold_value"),
            "actual_parent_value": item.get("actual_parent_outcome_value"),
            "actual_parent_value_sha256": item.get("parent_item_value_sha256"),
            "source_quality_disposition": item.get("review_disposition"),
            "source_result_sha256": case.get("result_sha256"),
            "source_row_sha256": case.get("source_row_sha256"),
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"selection {index} review record mismatch at {key}")
        if receipt.get("candidate_key") != selection.get("candidate_key"):
            raise ValueError(f"selection {index} candidate key differs from raw receipt")
        invocation_id = record.get("invocation_id")
        if receipt.get("invocation_id") != invocation_id:
            raise ValueError(f"selection {index} invocation differs from raw receipt")

        raw_path = Path(receipt["raw_result_path"])
        raw_bytes = raw_path.read_bytes()
        if sha(raw_bytes) != receipt.get("raw_result_file_sha256") or sha(raw_bytes) != case.get("result_sha256"):
            raise ValueError(f"selection {index} raw result file is not the reviewed result")
        matching_lines = [line for line in raw_bytes.splitlines() if line.strip()
                          and sha(line.rstrip(b"\r")) == receipt.get("raw_result_row_sha256")]
        if len(matching_lines) != 1:
            raise ValueError(f"selection {index} raw result row digest does not match exactly once")
        raw_row = json.loads(matching_lines[0])
        roles = receipt.get("provider_action_roles") or {}
        generation = roles.get("target_generation_turn") or roles.get("selected_output_action")
        if not isinstance(generation, dict):
            raise ValueError(f"selection {index} has no selected generation turn")
        trajectory_index = generation.get("trajectory_index")
        trajectory = raw_row.get("trajectory", [])
        if (not isinstance(trajectory_index, int) or isinstance(trajectory_index, bool) or
                not 0 <= trajectory_index < len(trajectory)):
            raise ValueError(f"selection {index} has invalid generation turn index")
        turn = trajectory[trajectory_index]
        if (turn.get("invocation_id") != invocation_id or
                turn.get("request_sha256") != generation.get("request_sha256") or
                turn.get("raw_response_sha256") != generation.get("raw_response_sha256")):
            raise ValueError(f"selection {index} generation turn differs from raw trajectory")
        review_turns = [entry for entry in bound_invocation.get("turns", [])
                        if entry.get("trajectory_index") == trajectory_index]
        if len(review_turns) != 1 or review_turns[0].get("raw_response_sha256") != generation.get("raw_response_sha256"):
            raise ValueError(f"selection {index} generation turn differs from reviewed item binding")
        turn_context_sha = sha(json.dumps(turn.get("context"), ensure_ascii=False,
                                          sort_keys=True, separators=(",", ":")).encode("utf-8"))
        if (review_turns[0].get("context_sha256") != turn_context_sha or
                receipt.get("context_scope", {}).get("target_turn_context_sha256") != turn_context_sha):
            raise ValueError(f"selection {index} context differs from reviewed and raw generation turn")

        host_rows = [entry for entry in raw_row.get("outcome", {}).get("invocation_ledger", [])
                     if entry.get("invocation_id") == invocation_id and entry.get("completion_status") == "done"]
        if len(host_rows) != 1:
            raise ValueError(f"selection {index} lacks one successful host result for selected invocation")
        host = host_rows[0].get("host_result") or {}
        host_value = host.get("value")
        host_value_sha = sha(json.dumps(host_value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if (bound_invocation.get("completion_status") != "done" or
                bound_invocation.get("host_result", {}).get("value_sha256") != host.get("value_sha256") or
                host.get("value_sha256") != record.get("actual_parent_value_sha256") or
                host_value != item.get("actual_parent_outcome_value") or host_value_sha != host.get("value_sha256") or
                generation.get("host_result_value_sha256") != host.get("value_sha256")):
            raise ValueError(f"selection {index} actual host result differs from reviewed parent item value")
        if generation.get("target_kind") == "assistant_plain_text":
            response_text = turn.get("model_response", {}).get("text")
            if (response_text != host_value or
                    sha(str(response_text).encode("utf-8")) != generation.get("response_text_sha256") or
                    generation.get("host_value_link_kind") != "provider_plain_text_equals_observed_host_result"):
                raise ValueError(f"selection {index} plain-text target does not exactly equal its host result")
        elif generation.get("target_kind") == "assistant_tool_call":
            raw_calls = turn.get("model_response", {}).get("raw_calls", [])
            expected_ids = set(generation.get("terminal_child_action_tool_call_ids", []))
            action_events = [event for event in raw_row.get("outcome", {}).get("action_ledger", [])
                             if event.get("call_id") == invocation_id and event.get("tool_call_id") in expected_ids
                             and event.get("seq") == host.get("terminal_action_seq") and
                             event.get("outcome") == "completed"]
            matching_raw_calls = [call for call in raw_calls if call.get("id") in expected_ids]
            if len(action_events) != 1 or len(matching_raw_calls) != 1:
                raise ValueError(f"selection {index} tool target lacks one exact terminal action/host binding")
            raw_function = matching_raw_calls[0].get("function") or {}
            if (action_events[0].get("name") != raw_function.get("name") or
                    action_events[0].get("arguments") != json.loads(raw_function.get("arguments", "{}")) or
                    generation.get("host_value_link_kind") != "terminal_child_action_matches_provider_tool_call_id_and_host_capture"):
                raise ValueError(f"selection {index} raw tool call differs from its terminal action binding")
        else:
            raise ValueError(f"selection {index} has unsupported target kind")

        review_binding = {
            "schema": "s1.held-action-review-binding/1",
            "review_document_path": str(review_path),
            "review_document_sha256": review_sha,
            "review_case_index": case_index,
            "review_case_id": case.get("case_id"),
            "review_case_record_sha256": compact_sha(case),
            "review_item_key": item.get("item_key"),
            "review_item_record_sha256": compact_sha(item),
            "selected_review_record_sha256": compact_sha(record),
            "raw_result_path": str(raw_path),
            "raw_result_file_sha256": sha(raw_bytes),
            "raw_result_row_sha256": receipt["raw_result_row_sha256"],
        }
        bound_selection = dict(selection)
        bound_selection["source_review_binding"] = review_binding
        bound_receipt = dict(receipt)
        legacy_value = bound_receipt.pop("source_quality_review_sha256", None)
        bound_receipt["legacy_mislabeled_source_quality_review_sha256"] = legacy_value
        bound_receipt["source_quality_review_sha256"] = review_sha
        bound_receipt["source_quality_review_case_record_sha256"] = review_binding["review_case_record_sha256"]
        bound_receipt["source_quality_review_item_record_sha256"] = review_binding["review_item_record_sha256"]
        bound_receipt["selected_source_review_record_sha256"] = review_binding["selected_review_record_sha256"]
        bound_receipt["source_result_file_sha256"] = receipt["raw_result_file_sha256"]
        bound_receipt["source_review_binding"] = review_binding
        bound_selections.append(bound_selection)
        bound_receipts.append(bound_receipt)
    return bound_selections, bound_receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out-selections", type=Path, required=True)
    parser.add_argument("--out-receipts", type=Path, required=True)
    args = parser.parse_args()
    selections, receipts = bind(lines(args.selections), lines(args.receipts),
                                json.loads(args.summary.read_text(encoding="utf-8")), args.review)
    for path, rows in ((args.out_selections, selections), (args.out_receipts, receipts)):
        if path.exists():
            raise ValueError(f"output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                     for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(selections), "review_document_sha256":
                      selections[0]["source_review_binding"]["review_document_sha256"] if selections else None,
                      "corrected_receipt_schema": "s1.held-action-review-binding/1"}, indent=2))


if __name__ == "__main__":
    main()
