import unittest
import hashlib
import json
from pathlib import Path
import tempfile

from scripts.training_exclusion import (bind_transition_manifest_sha, exclusion_target,
                                        filter_training_order, pending_source_review_entry,
                                        transition_state,
                                        validate_exclusion_checkpoint_resume)
from scripts.training_readiness import validate_training_inventory_audit_frozen_alias
from scripts.train_lora import validate_heldout_rows


class TrainingExclusionTests(unittest.TestCase):
    def test_exclusion_gate_preserves_complete_heldout_split(self):
        heldout = [{"id": f"held-{index}"} for index in range(5410)]
        split = {"held_turns": 5410}
        validate_heldout_rows(heldout, split)
        with self.assertRaisesRegex(ValueError, "count differs"):
            validate_heldout_rows(heldout[:2], split)
        with self.assertRaisesRegex(ValueError, "overwritten"):
            validate_heldout_rows([{"source_id": f"hold-{index}"} for index in range(5410)], split)

    def test_pending_source_review_checks_the_exact_entry(self):
        text = '''const reviews = [
          {"id": "source-a", "status": "resolved"},
          { id: 'source-b', status: 'pending', reason: 'review' },
        ];\n'''
        self.assertFalse(pending_source_review_entry(text, "source-a"))
        self.assertTrue(pending_source_review_entry(text, "source-b"))
        self.assertFalse(pending_source_review_entry(text, "missing"))

    def test_removes_consumed_and_future_rows_preserving_exact_remaining_order(self):
        ordered = [{"id": f"r{i}"} for i in range(12)]
        result = filter_training_order(ordered, ["r1", "r8", "r10"], cursor=5)
        self.assertEqual([row["id"] for row in result.rows],
                         ["r0", "r2", "r3", "r4", "r5", "r6", "r7", "r9", "r11"])
        self.assertEqual(result.cursor, 4)
        self.assertEqual(result.consumed_removed_ids, ("r1",))
        self.assertEqual(result.future_removed_ids, ("r8", "r10"))
        self.assertEqual([row["id"] for row in result.rows[result.cursor:]],
                         ["r5", "r6", "r7", "r9", "r11"])

    def test_target_subtracts_only_future_exclusions_and_keeps_lr_horizon(self):
        self.assertEqual(exclusion_target(105317, 9600, 7), 105310)
        state = {"step": 1200, "cursor": 5, "trained_examples": 5, "skipped": 0,
                 "corpus": {"target_examples": 105317, "steps": 13165, "optimizer": "muon"},
                 "log": [[1200, 1.4]]}
        order = filter_training_order([{"id": f"r{i}"} for i in range(10)], ["r2", "r9"], 5)
        updated = transition_state(state, exclusion_manifest_sha256="a" * 64,
                                   order=order, target_examples=105316)
        self.assertEqual(updated["step"], 1200)
        self.assertEqual(updated["trained_examples"], 5)
        self.assertEqual(updated["cursor"], 4)
        self.assertEqual(updated["corpus"]["steps"], 13165)
        self.assertEqual(updated["corpus"]["target_examples"], 105316)
        self.assertEqual(updated["exclusion_transition"]["manifest_sha256"], "a" * 64)
        self.assertTrue(updated["exclusion_transition"]["optimizer_scheduler_rng_files_unchanged"])
        self.assertEqual(updated["log"], state["log"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, content in (("optimizer.pt", b"optimizer slots"),
                                  ("scheduler.pt", b"scheduler state"),
                                  ("rng.pt", b"rng state"),
                                  ("weights.bin", b"model weights")):
                (root / name).write_bytes(content)
            before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in root.iterdir()}
            bound = bind_transition_manifest_sha(updated, "a" * 64)
            (root / "state.json").write_text(json.dumps(bound))
            after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in root.iterdir() if path.name != "state.json"}
            self.assertEqual(after, before)

    def test_rejects_skips_diverged_examples_or_nontraining_exclusion_ids(self):
        ordered = [{"id": f"r{i}"} for i in range(4)]
        with self.assertRaisesRegex(ValueError, "not in the training split"):
            filter_training_order(ordered, ["test-row"], 2)
        state = {"step": 1, "cursor": 2, "trained_examples": 1, "skipped": 1,
                 "corpus": {"target_examples": 4, "steps": 1}}
        order = filter_training_order(ordered, ["r3"], 2)
        with self.assertRaisesRegex(ValueError, "zero skipped"):
            transition_state(state, exclusion_manifest_sha256="b" * 64,
                             order=order, target_examples=3)

    def test_resume_accepts_mutated_muon_files_after_first_transition_checkpoint(self):
        parent_files = {"weights/adapter.safetensors": "a" * 64,
                        "optimizer.pt": "b" * 64, "scheduler.pt": "c" * 64,
                        "rng.pt": "d" * 64, "state.json": "e" * 64}
        manifest_sha = "f" * 64
        manifest = {"checkpoint": {"files": parent_files, "step": 10, "cursor": 80,
                                    "trained_examples": 80,
                                    "state_sha256": "e" * 64,
                                    "parent_corpus_identity": {"steps": 13165}}}
        state = {"step": 10, "cursor": 78, "trained_examples": 80, "skipped": 0,
                 "exclusion_transition": {"parent_step": 10, "parent_cursor": 80,
                     "cursor": 78, "trained_examples_preserved": 80,
                     "parent_state_sha256": "e" * 64, "manifest_sha256": manifest_sha,
                     "consumed_removed_ids": ["r2", "r3"]}}
        exact_parent_files = {k: v for k, v in parent_files.items() if k != "state.json"}
        # The newly staged transition checkpoint must retain every parent file byte.
        self.assertTrue(validate_exclusion_checkpoint_resume(
            manifest, state, exact_parent_files, accumulation=8, manifest_sha256=manifest_sha))

        # After one Muon optimizer step, those stateful files legitimately differ.
        continued = {**state, "step": 11, "cursor": 86, "trained_examples": 88}
        changed = {"weights/adapter.safetensors": "1" * 64,
                   "optimizer.pt": "2" * 64, "scheduler.pt": "3" * 64,
                   "rng.pt": "4" * 64}
        self.assertFalse(validate_exclusion_checkpoint_resume(
            manifest, continued, changed, accumulation=8, manifest_sha256=manifest_sha))
        with self.assertRaisesRegex(ValueError, "cursor and trained-example"):
            validate_exclusion_checkpoint_resume(
                manifest, {**continued, "cursor": 87}, changed,
                accumulation=8, manifest_sha256=manifest_sha)
        with self.assertRaisesRegex(ValueError, "file layout"):
            validate_exclusion_checkpoint_resume(
                manifest, continued, {**changed, "extra.bin": "5" * 64},
                accumulation=8, manifest_sha256=manifest_sha)

    def test_frozen_policy_alias_preserves_original_ready_report_path_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "canonical-policy.json"
            alias = root / "sealed-policy.json"
            report_path = root / "inventory-report.json"
            ready_path = root / "inventory-ready.json"
            policy_bytes = b'{"policy":1}\n'
            original.write_bytes(policy_bytes)
            alias.write_bytes(policy_bytes)
            policy_sha = hashlib.sha256(policy_bytes).hexdigest()
            report = {"version": "natlang.training_data_inventory/1",
                      "policy": str(original), "policy_sha256": policy_sha,
                      "missing_required_default_inputs": [], "included_quality_blockers": []}
            report_bytes = (json.dumps(report, sort_keys=True) + "\n").encode()
            report_path.write_bytes(report_bytes)
            ready_path.write_text(json.dumps({"ready": True, "report": str(report_path),
                                              "sha256": hashlib.sha256(report_bytes).hexdigest(),
                                              "policy_sha256": policy_sha}))
            proof = validate_training_inventory_audit_frozen_alias(
                ready_path, alias, original, policy_sha)
            self.assertEqual(proof["policy_sha256"], policy_sha)
            self.assertEqual(proof["report_policy_path"], str(original))
            alias.write_bytes(b'{"policy":2}\n')
            with self.assertRaisesRegex(ValueError, "frozen inventory alias bytes"):
                validate_training_inventory_audit_frozen_alias(
                    ready_path, alias, original, policy_sha)


if __name__ == "__main__":
    unittest.main()
