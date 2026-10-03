import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.corpus import digest, index_pairs, split_programs
from scripts.prepare_training_append_intake import prepare_append_intake
from scripts.training_append import append_train_order, cosine_extension_multiplier, ids_digest


def write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainingAppendIntakeTests(unittest.TestCase):
    def test_append_order_keeps_old_permutation_as_prefix(self):
        base_order = [{"id": "b", "offset": 2}, {"id": "a", "offset": 1}]
        candidate = [{"id": "c", "offset": 5}]
        combined = [{"id": "a", "offset": 0}, {"id": "b", "offset": 2}, candidate[0]]
        ordered = append_train_order(base_order, candidate, combined,
                                     base_order_sha256=ids_digest(["b", "a"]),
                                     candidate_order_sha256=ids_digest(["c"]))
        self.assertEqual([row["id"] for row in ordered], ["b", "a", "c"])

    def test_append_order_rejects_candidate_order_drift(self):
        with self.assertRaisesRegex(ValueError, "candidate physical row order"):
            append_train_order([{"id": "a"}], [{"id": "c"}, {"id": "b"}],
                               [{"id": "a"}, {"id": "b"}, {"id": "c"}],
                               base_order_sha256=ids_digest(["a"]),
                               candidate_order_sha256=ids_digest(["b", "c"]))

    def test_extended_cosine_is_continuous_and_reaches_zero(self):
        self.assertEqual(cosine_extension_multiplier(40, start_step=40, end_step=100), 1.0)
        self.assertAlmostEqual(cosine_extension_multiplier(70, start_step=40, end_step=100), 0.5)
        self.assertEqual(cosine_extension_multiplier(100, start_step=40, end_step=100), 0.0)

    def fixture(self, root: Path, candidate_groups=("g-new",), cursor=0):
        base = root / "base.jsonl"
        candidate = root / "candidate.jsonl"
        renderer = {"model": "test", "revision": "r1", "tokenizer_fingerprint_sha256": "t"}
        base_rows = [
            {"id": "a1", "program_id": "a", "source_groups": ["g-a"], "split": "train", "prompt": "p", "completion": "c"},
            {"id": "b1", "program_id": "b", "source_groups": ["g-b"], "split": "train", "prompt": "p", "completion": "c"},
            {"id": "t1", "program_id": "t", "source_groups": ["g-test"], "split": "test", "prompt": "p", "completion": "c"},
        ]
        candidate_rows = [
            {"id": f"new{i}", "program_id": f"new{i}", "source_groups": [group], "split": "train", "prompt": "p", "completion": "c"}
            for i, group in enumerate(candidate_groups)
        ]
        write_rows(base, base_rows)
        write_rows(candidate, candidate_rows)
        base_audit = root / "base.manifest.json"
        candidate_audit = root / "candidate.manifest.json"
        for path, data in ((base_audit, base), (candidate_audit, candidate)):
            write_json(path, {"sha256": sha(data), "rows": len(data.read_text().splitlines()),
                              "audit": {"ready": True, "max_len": 128}, "renderer": renderer})
        _, base_train, split = split_programs(index_pairs(base), 1, 3)
        split_path = root / "split.json"
        write_json(split_path, split)
        checkpoint = root / "checkpoint.json"
        write_json(checkpoint, {
            "step": 0, "cursor": cursor, "trained_examples": cursor,
            "corpus": {"data_sha256": sha(base), "split_sha256": digest(split),
                       "target_examples": len(base_train), "steps": 1,
                       "optimizer": "muon"},
        })
        gate = root / "gate.json"
        candidate_ids = [row["id"] for row in candidate_rows]
        write_json(gate, {
            "schema": "lfm-training-append-gates/1",
            "candidate_sha256": sha(candidate), "candidate_rows": len(candidate_rows),
            "candidate_row_ids_sha256": digest(candidate_ids),
            "current_policy_sha256": "a" * 64,
            "current_admission_passed": True, "source_conversion_passed": True,
            "source_review_passed": True, "native_materialization_passed": True,
        })
        return base, base_audit, split_path, checkpoint, candidate, candidate_audit, gate

    def invoke(self, root, fixture):
        base, base_audit, split, checkpoint, candidate, candidate_audit, gate = fixture
        output = root / "intake.json"
        combined = root / "combined.jsonl"
        result = prepare_append_intake(
            base_path=base, base_audit_path=base_audit, base_split_path=split,
            checkpoint_path=checkpoint, candidate_path=candidate,
            candidate_audit_path=candidate_audit, gate_receipt_path=gate,
            combined_path=combined,
            output_path=output,
        )
        return result, output

    def test_valid_append_keeps_old_prefix_and_optimizer_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, output = self.invoke(root, self.fixture(root))
            self.assertTrue(output.is_file())
            self.assertEqual(result["status"], "prepared_for_append_aware_resume")
            self.assertTrue(result["resume_requirements"]["preserve_optimizer_state"])
            self.assertEqual(result["combined_split"]["candidate_rows_in_train"], 1)
            self.assertEqual(result["checkpoint"]["optimizer"], "muon")
            self.assertEqual(len(index_pairs(Path(result["combined"]["path"]))), 4)

    def test_rejects_candidate_that_joins_protected_test_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self.fixture(root, candidate_groups=("g-test",))
            with self.assertRaisesRegex(ValueError, "conflicting explicit train/test"):
                self.invoke(root, fixture)

    def test_rejects_resume_after_old_order_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self.fixture(root, cursor=2)
            with self.assertRaisesRegex(ValueError, "old training order is exhausted"):
                self.invoke(root, fixture)

    def test_rejects_unadmitted_candidate_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self.fixture(root)
            gate = json.loads(fixture[-1].read_text())
            gate["source_review_passed"] = False
            write_json(fixture[-1], gate)
            with self.assertRaisesRegex(ValueError, "missing/failed current-policy gates"):
                self.invoke(root, fixture)
            self.assertFalse((root / "combined.jsonl").exists())
            self.assertFalse((root / "intake.json").exists())

    def test_rejects_candidate_audit_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self.fixture(root)
            rows = [json.loads(fixture[4].read_text().splitlines()[0])]
            rows[0]["completion"] = "changed"
            write_rows(fixture[4], rows)
            with self.assertRaisesRegex(ValueError, "hash differs"):
                self.invoke(root, fixture)


if __name__ == "__main__":
    unittest.main()
