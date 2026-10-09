"""The review-packet scripts read the natlang row reviewer's receipts in place of an outside annotation file."""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts import bind_held_action_review_receipts as bind
from scripts import build_step5_native_action_review_packet as build
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import source_review as sr  # noqa: E402
from test_source_review import FIXED, reviewer, row_input

SHA = sr.sha256_hex


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PacketFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = self.root / "registry"
        shutil.copytree(sr.REGISTRY, self.registry, ignore=shutil.ignore_patterns("receipts"))
        self.receipts = self.registry / "receipts"

    def row_receipt(self, identifier, decision=None, status="equivalent", numeric=True):
        precheck = {"exact_match": False, "actual_is_exact_source_span": False, "numerically_equal": numeric}
        output = {"kind": "row", "input": row_input(identifier, precheck=precheck), "reviewer": reviewer("reviewSourceRow"),
                  "recommendation": {"status": status, "rationale": "Stated in the evidence.",
                                     "evidence": [{"quote": "The mean is 3977.75"}], "confidence": "high"}}
        built = sr.build_receipt(output, self.registry)
        sr.save_receipt(built, self.receipts)
        if decision:
            sr.decide(built["id"], decision, "owner", registry=self.registry, receipts=self.receipts,
                      manifests=self.root / "manifests", now=FIXED)
        return built


class BuildPacketTests(PacketFixture):
    def run_build(self, *extra):
        files = {}
        for name in ("result", "trace", "native-rows", "materialization-manifest", "capture-snapshot", "runtime-manifest"):
            files[name] = self.root / f"{name}.json"
        files["result"].write_text(json.dumps({"outcome": {"action_ledger": [], "invocation_ledger": [], "status": "done", "accepted": True},
                                               "provenance": {}}), encoding="utf-8")
        files["trace"].write_text("", encoding="utf-8")
        write_jsonl(files["native-rows"], [{"id": "row-1", "split": "test", "source_groups": ["g"], "source_ref": {"invocation_id": "i1"},
                                            "decision": {"index": 0, "assistant": {"calls": []}}, "target": {"tool_calls": []},
                                            "messages": [], "tools": [], "training_admission": None}])
        for name in ("materialization-manifest", "capture-snapshot", "runtime-manifest"):
            files[name].write_text("{}", encoding="utf-8")
        cases = self.root / "cases.jsonl"
        line = json.dumps({"id": "case", "source_groups": ["g"], "split": "test"})
        cases.write_text(line + "\n", encoding="utf-8")
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"source": {"path": "cases.jsonl", "sha256": SHA(cases.read_bytes()), "index": 0,
                                               "row_sha256": SHA(line.encode())}}), encoding="utf-8")
        output = self.root / "packet.json"
        argv = ["build"] + [arg for name, path in files.items() for arg in (f"--{name}", str(path))] + [
            "--source-cases", str(cases), "--launch-plan", str(plan), "--output", str(output), "--source-row-index", "0",
            "--controller-provenance", "authored", *extra]
        with mock.patch.object(build, "ROOT", self.root), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            build.main()
        return json.loads(output.read_text(encoding="utf-8"))

    def test_decided_row_receipts_supply_the_dispositions_and_are_pinned(self):
        self.row_receipt("row-1", decision="clear", numeric=False)
        packet = self.run_build("--row-reviews", str(self.receipts))
        assessment = packet["rows"][0]["assessment"]
        self.assertEqual(assessment["disposition"], "candidate")
        self.assertEqual(assessment["v7_category"], "receipts")
        self.assertIs(assessment["training_admission"], False)
        self.assertEqual(packet["counts"]["candidate_recommendations"], 1)
        self.assertIn("row_reviews", packet["pins"])
        self.assertNotIn("annotations", packet["pins"])
        self.assertIs(packet["limits"]["training_admission"], False)

    def test_a_hold_decision_gives_a_held_row(self):
        self.row_receipt("row-1", decision="hold", status="mismatch")
        packet = self.run_build("--row-reviews", str(self.receipts))
        self.assertEqual(packet["rows"][0]["assessment"]["disposition"], "hold")
        self.assertEqual(packet["counts"]["held_recommendations"], 1)

    def test_a_row_without_an_explicit_decision_is_refused(self):
        self.row_receipt("row-1")
        with self.assertRaisesRegex(sr.ReviewError, "no explicit hold or clear decision"):
            self.run_build("--row-reviews", str(self.receipts))

    def test_a_row_without_a_receipt_is_refused(self):
        self.row_receipt("another-row", decision="clear")
        with self.assertRaisesRegex(ValueError, "missing explicit candidate/hold assessment for row-1"):
            self.run_build("--row-reviews", str(self.receipts))

    def test_exactly_one_review_source_is_given(self):
        with self.assertRaisesRegex(ValueError, "exactly one of"):
            self.run_build()
        annotations = self.root / "annotations.json"
        annotations.write_text(json.dumps({"rows": {"row-1": {"disposition": "candidate"}}}), encoding="utf-8")
        self.assertEqual(self.run_build("--annotations", str(annotations))["rows"][0]["assessment"]["disposition"], "candidate")
        with self.assertRaisesRegex(ValueError, "exactly one of"):
            self.run_build("--annotations", str(annotations), "--row-reviews", str(self.receipts))


class BindRowReviewTests(PacketFixture):
    def test_the_reviewed_item_needs_an_explicit_receipt_over_its_exact_values(self):
        self.row_receipt("item-a", decision="clear")
        receipts = sr.load_row_receipts(self.receipts)
        item = {"item_key": "item-a", "gold_value": "3977.75", "actual_parent_outcome_value": "3977.7500000000005"}
        assessment = bind.row_review_for(item, receipts)
        self.assertEqual(assessment["disposition"], "candidate")
        self.assertEqual(assessment["v7_category"], "held_numeric_representation")
        with self.assertRaisesRegex(ValueError, "no row-review receipt"):
            bind.row_review_for({**item, "item_key": "item-b"}, receipts)
        with self.assertRaisesRegex(ValueError, "different gold/actual values"):
            bind.row_review_for({**item, "gold_value": "3977.76"}, receipts)

    def test_an_undecided_receipt_does_not_bind(self):
        self.row_receipt("item-a")
        item = {"item_key": "item-a", "gold_value": "3977.75", "actual_parent_outcome_value": "3977.7500000000005"}
        with self.assertRaisesRegex(sr.ReviewError, "no explicit hold or clear decision"):
            bind.row_review_for(item, sr.load_row_receipts(self.receipts))


if __name__ == "__main__":
    unittest.main()
