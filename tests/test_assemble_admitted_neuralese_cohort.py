import importlib.util
import json
from pathlib import Path
import hashlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/assemble_admitted_neuralese_cohort.py"
SPEC = importlib.util.spec_from_file_location("assemble_admitted_neuralese_cohort", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class AssemblerInvariantTests(unittest.TestCase):
    def _mixed_admission(self, root):
        target = {"role": "assistant", "content": "approved"}
        source = root / "source.jsonl"
        source.write_text(json.dumps({"id": "selected", "target": target}) + "\n")
        review = root / "review.json"
        review.write_text('{"review":"pinned"}\n')
        approval = {
            "schema": "natlang.root-per-action-training-admission/1",
            "review_path": "review.json", "review_sha256": builder.sha(review),
            "input_pins": {"source.jsonl": {"sha256": builder.sha(source), "bytes": source.stat().st_size}},
            "rows": [
                {"native_id": "selected", "decision": "admit-ordinary-native-action", "training_admission": True,
                 "split": "train", "source_group": "g", "target_sha256": builder.target_digest({"target": target})},
                {"native_id": "held", "decision": "hold-source-required-neuralese-reader-contract", "training_admission": False},
                {"native_id": "failed", "decision": "reject-action-failed", "training_admission": False},
            ],
            "admitted_native_count": 1, "held_source_contract_final_count": 1,
            "held_ambiguous_source_read_scope_count": 0, "failed_count": 1, "already_adopted_count": 0,
            "whole_trajectory_admission": False, "runtime_qualification": False,
            "active_gpu_inputs_changed": False, "new_world_credit": False,
        }
        return approval, [{"id": "selected", "target": target, "split": "train", "source_groups": ["g"]}]

    def test_mixed_root_admission_selects_only_true_ordinary_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approval, delta = self._mixed_admission(root)
            selected = builder.validate_root_per_action_approval(approval, delta, root=root)
            self.assertEqual(list(selected), ["selected"])
            self.assertNotIn("held", selected)

    def test_mixed_root_admission_rejects_flag_decision_count_and_held_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approval, delta = self._mixed_admission(root)
            bad = json.loads(json.dumps(approval)); bad['rows'][1]['training_admission'] = True
            with self.assertRaisesRegex(ValueError, 'flag conflicts'):
                builder.validate_root_per_action_approval(bad, delta, root=root)
            bad = json.loads(json.dumps(approval)); bad['rows'][0]['decision'] = 'hold-source-required-neuralese-reader-contract'
            with self.assertRaisesRegex(ValueError, 'flag conflicts'):
                builder.validate_root_per_action_approval(bad, delta, root=root)
            bad = json.loads(json.dumps(approval)); bad['admitted_native_count'] = 2
            with self.assertRaisesRegex(ValueError, 'count conflicts'):
                builder.validate_root_per_action_approval(bad, delta, root=root)
            with self.assertRaisesRegex(ValueError, 'do not equal root-admitted'):
                builder.validate_root_per_action_approval(approval, delta + [{"id": "held"}], root=root)

    def test_append_preserves_prefix_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "prefix.jsonl"
            output = root / "out.jsonl"
            prefix.write_bytes(b'{"id":"old"}\r\n')
            builder.write_append(prefix, output, [{"id": "new", "split": "train"}])
            self.assertTrue(output.read_bytes().startswith(prefix.read_bytes()))
            self.assertEqual(len(list(builder.rows(output))), 2)

    def test_piece_conflict_is_rejected_without_overwriting_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, delta, output = (root / x for x in ("base", "delta", "out"))
            base.write_text('{"name":"p","text":"base"}\n')
            delta.write_text('{"name":"p","text":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "piece name conflict"):
                builder.merge_pieces(base, delta, output)
            self.assertFalse(output.exists())

    def test_group_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in [
                {"id": "a", "split": "train", "source_groups": ["g"], "source_ids": ["s1"]},
                {"id": "b", "split": "test", "source_groups": ["g"], "source_ids": ["s2"]},
            ]))
            with self.assertRaisesRegex(ValueError, "source groups cross splits"):
                builder.audit_splits(path)

    def test_root_native_facet_appends_without_recurrence_additions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native-records.jsonl"
            recurrence = root / "recurrence-records.jsonl"
            native_pieces = root / "native-pieces.jsonl"
            recurrence_pieces = root / "recurrence-pieces.jsonl"
            base = {"id": "base", "split": "train", "source_groups": ["base-world"], "source_ids": ["base-source"]}
            native.write_text(json.dumps(base) + "\n")
            recurrence.write_text('{"id":"base-r","links":[]}\n')
            native_pieces.write_text(""); recurrence_pieces.write_text("")
            delta_pieces = root / "delta-pieces.jsonl"; delta_pieces.write_text("")
            delta_recurrence = root / "delta-recurrence.jsonl"; delta_recurrence.write_text("")
            target = {"role": "assistant", "content": "approved target"}
            delta = {"id": "selected", "split": "train", "source_groups": ["selected-world"],
                     "source_ids": ["selected-source"], "target": target,
                     "messages": [], "decision": {"failed_action": False, "training_approved": False},
                     "training_admission": {"approved": False}}
            delta_native = root / "delta-native.jsonl"; delta_native.write_text(json.dumps(delta) + "\n")
            approval_row = {"native_id": "selected", "split": "train", "source_groups": ["selected-world"],
                            "target_sha256": hashlib.sha256(json.dumps(target, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                            "decision": "admit-exact-selected-native-action-SFT-only"}
            approval = root / "admission.json"
            approval.write_text(json.dumps({"schema": "natlang.root-selected-action-admission/1",
                "rows": [approval_row], "counts": {"native_SFT_train_actions": 1, "whole_trajectories": 0},
                "qualifications": {"recurrence": False, "learned_writer": False}}))
            base_receipt = root / "base-receipt.json"
            base_receipt.write_text(json.dumps({"schema": "natlang.corpus-prefix-binding/1",
                "status": "verified-exact-prefix", "training_admission": False, "files": {
                "native-records.jsonl": {"sha256": builder.sha(native)},
                "recurrence-records.jsonl": {"sha256": builder.sha(recurrence)},
                "native-pieces.jsonl": {"sha256": builder.sha(native_pieces)},
                "recurrence-pieces.jsonl": {"sha256": builder.sha(recurrence_pieces)}}}))
            out = root / "out"
            argv = ["assembler", "--base-native", str(native), "--base-recurrence", str(recurrence),
                    "--base-receipt", str(base_receipt), "--delta-native", str(delta_native),
                    "--delta-recurrence", str(delta_recurrence), "--delta-pieces", str(delta_pieces),
                    "--base-native-pieces", str(native_pieces), "--base-recurrence-pieces", str(recurrence_pieces),
                    "--approval", str(approval), "--out", str(out)]
            def fake_audit(command, **kwargs):
                Path(command[-1]).write_text(json.dumps({"structurally_closed": True, "linked_edges": 0}))
            with patch.object(sys, "argv", argv), patch.object(builder.subprocess, "run", side_effect=fake_audit):
                builder.main()
            admitted = list(builder.rows(out / "delta-native-records.jsonl"))
            self.assertEqual(len(admitted), 1)
            self.assertEqual(admitted[0]["target"], target)
            self.assertTrue(admitted[0]["training_admission"]["approved"])
            self.assertEqual((out / "recurrence-records.jsonl").read_bytes(), recurrence.read_bytes())
            self.assertEqual((out / "recurrence-pieces.jsonl").read_bytes(), recurrence_pieces.read_bytes())
            self.assertTrue(os.path.samefile(out / "recurrence-records.jsonl", recurrence))
            self.assertTrue(os.path.samefile(out / "recurrence-pieces.jsonl", recurrence_pieces))
            manifest = json.loads((out / "proposal-manifest.json").read_text())
            self.assertEqual(manifest["unchanged_artifact_transport"]["recurrence"]["method"], "hardlink")
            self.assertEqual(manifest["unchanged_artifact_transport"]["recurrence_pieces"]["method"], "hardlink")
            self.assertEqual(manifest["admitted_facets"],
                             {"native": True, "recurrence": False, "native_only_receipt": True})

    def test_selected_action_receipt_rejects_changed_target(self):
        target = {"role": "assistant", "content": "original"}
        row = {"id": "selected", "target": target}
        digest = builder.target_digest(row)
        row["target"] = {"role": "assistant", "content": "changed"}
        self.assertNotEqual(builder.target_digest(row), digest)


if __name__ == "__main__":
    unittest.main()
