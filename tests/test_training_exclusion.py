import unittest
import hashlib
import json
from pathlib import Path
import tempfile

from scripts.training_exclusion import (bind_transition_manifest_sha, exclusion_target,
                                        filter_training_order, transition_state)


class TrainingExclusionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
