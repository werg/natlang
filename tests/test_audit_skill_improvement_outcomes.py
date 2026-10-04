import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_skill_improvement_outcomes import audit, trace_kind


def validation(source, quality=1.0, passed=1, total=1):
    return {"source": source, "split": "validation", "quality": quality, "gatesPassed": quality == 1.0,
            "passed": passed, "total": total, "modelCalls": 3, "evidence": "ev-hash", "sourceBytes": 42,
            "scores": [{"caseId": "case-1", "quality": quality}]}


def artifact(**updates):
    value = {"version": "natlang.skill-authoring-trajectory/1", "episode": "ep-1", "family": "research",
             "source_groups": ["g1", "g2"], "baseline": "same", "selected": "same", "disposition": "not-promoted",
             "positive": False, "search": {"disposition": "baseline-retained", "state": {"iteration": 2,
               "history": [{"reason": "MODEL CLAIM: failed badly", "accepted": False}], "stopReason": "finished"},
               "baseline": validation("same"), "validation": validation("same")},
             "searchDefinition": {"cases": [{"id": "case-1", "group": "g1", "split": "train", "args": ["prompt"],
                   "expected": {"host": "gold"}, "services": {"research": "source"}}]},
             "selectedFiles": {"solve.nl": "program"}, "skillDiagnostics": [], "traces": [],
             "authorExchanges": [{"recording_version": "natlang.effective-model-turn/1", "request": {
                 "messages": [{"role": "user", "content": "private prompt"}], "tools": [], "seed": 0, "max_tokens": 20},
                 "turn": {"calls": []}, "wireExchanges": []}], "executorExchanges": [], "query": None, "transfer": None}
    value.update(updates)
    return value


class SkillImprovementOutcomesAuditTests(unittest.TestCase):
    def write(self, root, relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_no_headroom_uses_host_validation_not_history_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, "task/attempt-001/result.json", artifact())
            result = audit(root)
            row = result["artifacts"][0]
            self.assertIn("no_observed_headroom_at_support_ceiling", row["classification"])
            self.assertEqual(row["search"]["baseline_support_validation"]["quality"], 1.0)
            self.assertEqual(row["search"]["selected_support_validation"]["passed"], 1)
            self.assertEqual(row["search"]["history_record_count_unverified"], 1)
            self.assertNotIn("MODEL CLAIM", json.dumps(result))
            self.assertFalse(result["interpretation_policy"]["history_is_oracle"])
            self.assertEqual(result["request_fingerprint_index"]["dpo_pairs_admitted"], 0)

    def test_incomplete_oracle_mismatch_is_not_quality_negative(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            failed = artifact(disposition="incomplete", search={"disposition": "incomplete-search", "error":
                "Error: reported quality does not match independently executed selected source", "state": {}, "baseline": None,
                "validation": None})
            self.write(root, "task/attempt-001/result.json", failed)
            row = audit(root)["artifacts"][0]
            self.assertIn("model_claimed_score_disagrees_with_host_measurement", row["classification"])
            self.assertIn("collection_incomplete_or_failed", row["classification"])
            self.assertIsNone(row["final_query"])
            self.assertFalse(audit(root)["interpretation_policy"]["incomplete_or_invalid_contract_are_quality_negatives"])

    def test_changed_candidate_with_invalid_metadata_is_review_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            changed = artifact(selected="new-source", disposition="incomplete", skillDiagnostics=[
                {"path": "solve/skills/x/SKILL.md", "code": "skill-frontmatter", "severity": "error", "message": "raw detail"}])
            self.write(root, "task/attempt-001/result.json", changed)
            row = audit(root)["artifacts"][0]
            self.assertIn("invalid_skill_metadata_or_contract", row["classification"])
            self.assertIn("edited_candidate_invalid_skill_metadata", row["classification"])
            self.assertEqual(row["revision_review"]["status"], "invalid_skill_metadata")
            self.assertNotIn("raw detail", json.dumps(row))
            self.assertEqual(row["selected_files"][0]["path"], "solve.nl")

    def test_support_reference_and_request_fingerprints_are_content_free(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root, "task/attempt-001/result.json", artifact())
            result = audit(root)
            row = result["artifacts"][0]
            ref = row["support_evidence_refs"][0]
            self.assertEqual((ref["case_id"], ref["group"], ref["split"]), ("case-1", "g1", "train"))
            self.assertTrue(ref["input_sha256"])
            self.assertTrue(ref["expected_sha256"])
            self.assertTrue(row["author_request_fingerprints"][0]["request_fingerprint"])
            self.assertNotIn("private prompt", json.dumps(result))
            self.assertNotIn("gold", json.dumps(result))

    def test_trace_categories_separate_budget_quiescence_and_bad_calls(self):
        self.assertEqual(trace_kind({"outcome": "quiesced", "detail": "Not enough turns remaining"}),
                         "turn_or_output_budget_exhaustion")
        self.assertEqual(trace_kind({"outcome": "error", "detail": "malformed tool arguments"}),
                         "tool_protocol_or_schema_error")
        self.assertEqual(trace_kind({"outcome": "done"}), "completed_tool_exchange")


if __name__ == "__main__":
    unittest.main()
