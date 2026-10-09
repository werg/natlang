import json
import sys
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import source_review as sr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXED = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
SOURCE_SHA = "a" * 64


def reviewer(name="reviewSourceItem", executor="scripted-model"):
    inputs = {"definition_source_sha256": SOURCE_SHA, "compiler_version": "natlang-ts-host/0.1.0", "executor": executor}
    return {"kind": "natlang", "function": name, "definition_key": "k" * 32, "call_id": None, "hash_inputs": inputs,
            "reviewer_hash": sr.reviewer_hash(SOURCE_SHA, inputs["compiler_version"], executor)}


def item_input(identity="item-1"):
    item = {"dataset": "sst2", "id": identity, "visible": "The film is fine. Nothing else is said.", "annotated_label": "negative",
            "contract": "Label the sentiment.", "answer_format": None}
    return {"item": item, "precedents": []}


def hold_recommendation():
    return {"recommendation": "hold", "concern": "label-disagrees-with-source", "reason": "The text says the film is fine.",
            "evidence": [{"quote": "The film is fine."}], "proposed_entry": {"reason": "The text says the film is fine."}, "confidence": "high"}


def item_output(identity="item-1", recommendation=None):
    return {"kind": "item", "input": item_input(identity), "reviewer": reviewer(),
            "recommendation": recommendation or hold_recommendation()}


def row_input(identifier="row-1", gold="3977.75", actual="3977.7500000000005", precheck=None):
    row = {"id": identifier, "dataset": "qa", "split": "test", "source_groups": ["g"], "question": "What is the mean?",
           "answer_format": "number", "evidence": "The mean is 3977.75 across four values.", "gold": gold, "actual": actual}
    return {"row": row, "precheck": precheck or {"exact_match": False, "actual_is_exact_source_span": False, "numerically_equal": True}}


def row_output(identifier="row-1", status="equivalent", **kwargs):
    return {"kind": "row", "input": row_input(identifier, **kwargs), "reviewer": reviewer("reviewSourceRow"),
            "recommendation": {"status": status, "rationale": "The mean is stated in the evidence.",
                               "evidence": [{"quote": "The mean is 3977.75"}], "confidence": "medium"}}


class RegistryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = self.root / "source-reviews"
        shutil.copytree(sr.REGISTRY, self.registry, ignore=shutil.ignore_patterns("receipts"))
        self.manifests = self.root / "manifests"
        self.manifests.mkdir()
        self.receipts = self.registry / "receipts"

    def receipt(self, output):
        built = sr.build_receipt(output, self.registry)
        sr.save_receipt(built, self.receipts)
        return built

    def decide(self, receipt, value, by="owner", **kwargs):
        return sr.decide(receipt["id"], value, by, "note", registry=self.registry, receipts=self.receipts,
                         manifests=self.manifests, now=FIXED, **kwargs)


class RegistryTests(unittest.TestCase):
    def test_reviewer_hash_is_pinned_so_typescript_and_python_agree(self):
        self.assertEqual(sr.reviewer_hash(SOURCE_SHA, "natlang-ts-host/0.1.0", "scripted-model"),
                         "natlang@" + "3dbe1b19e71bf333")

    def test_committed_manifest_describes_the_migrated_registry(self):
        manifest = json.loads((sr.MANIFESTS / "source-reviews-20261009-v1.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["record_count"], manifest["identity_count"], manifest["family_hold_count"]), (184, 255, 3))
        self.assertFalse(manifest["training_admission"])
        for entry in manifest["files"]:
            self.assertTrue((ROOT / entry["path"]).is_file())
        newest = sorted(path.stem for path in sr.MANIFESTS.glob("source-reviews-*-v*.json"))[-1]
        current = json.loads((sr.MANIFESTS / f"{newest}.json").read_text(encoding="utf-8"))
        rebuilt = sr.build_manifest(current["id"], sr.REGISTRY, current["previous"])
        self.assertEqual(rebuilt["files"], current["files"], "the newest manifest matches the registry bytes")

    def test_manifests_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = sr.build_manifest("source-reviews-20261009-v1")
            sr.write_manifest(manifest, Path(tmp))
            with self.assertRaisesRegex(sr.ReviewError, "immutable"):
                sr.write_manifest(manifest, Path(tmp))


class ReceiptTests(RegistryCase):
    def test_an_item_receipt_binds_reviewer_hash_subject_and_no_admission(self):
        built = self.receipt(item_output())
        self.assertEqual(built["schema"], "natlang.source-review-receipt/1")
        self.assertIs(built["training_admission"], False)
        self.assertIsNone(built["decision"])
        self.assertEqual(built["agreement"], "single")
        self.assertEqual(built["subject"]["subject_sha256"], sr.canonical_sha256(item_input()))
        self.assertTrue(built["reviewer"]["reviewer_hash"].startswith("natlang@"))
        self.assertEqual(built["registry"]["before_sha256"], built["registry"]["after_sha256"])
        sr.validate_receipt(built)

    def test_a_recommendation_alone_never_changes_the_registry(self):
        before = (self.registry / "holds.jsonl").read_bytes()
        self.receipt(item_output())
        self.assertEqual((self.registry / "holds.jsonl").read_bytes(), before)
        self.assertEqual(list(self.manifests.iterdir()), [])

    def test_the_typed_contract_is_checked_with_messages_that_say_what_to_return(self):
        bad = hold_recommendation() | {"proposed_entry": None}
        with self.assertRaisesRegex(sr.ReviewError, "proposed_entry with a reason is present exactly when"):
            sr.build_receipt(item_output(recommendation=bad), self.registry)
        bad = hold_recommendation() | {"concern": "none"}
        with self.assertRaisesRegex(sr.ReviewError, "concern is none exactly when"):
            sr.build_receipt(item_output(recommendation=bad), self.registry)
        bad = hold_recommendation() | {"evidence": [{"quote": "Not in the text."}]}
        with self.assertRaisesRegex(sr.ReviewError, "must occur in the item's visible text"):
            sr.build_receipt(item_output(recommendation=bad), self.registry)
        forged = item_output()
        forged["reviewer"]["reviewer_hash"] = "natlang@" + "0" * 16
        with self.assertRaisesRegex(sr.ReviewError, "reviewer_hash must equal"):
            sr.build_receipt(forged, self.registry)
        exact = row_output(gold="a", actual="a", status="mismatch", precheck={"exact_match": True, "actual_is_exact_source_span": False, "numerically_equal": None})
        with self.assertRaisesRegex(sr.ReviewError, "equivalent whenever precheck.exact_match"):
            sr.build_receipt(exact, self.registry)

    def test_a_second_reviewer_gives_agreement_or_disagreement(self):
        output = item_output()
        output["second_reviewer"] = reviewer(executor="other-model")
        output["second_recommendation"] = hold_recommendation()
        self.assertEqual(sr.build_receipt(output, self.registry)["agreement"], "agree")
        output["second_recommendation"] = {"recommendation": "admit", "concern": "none", "reason": "Fine.", "evidence": [],
                                           "proposed_entry": None, "confidence": "low"}
        self.assertEqual(sr.build_receipt(output, self.registry)["agreement"], "disagree")

    def test_tampering_with_a_saved_receipt_is_found(self):
        built = self.receipt(item_output())
        path = self.receipts / f"{built['id']}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["subject"]["input"]["item"]["visible"] = "Changed."
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(sr.ReviewError, "subject_sha256 differs"):
            sr.load_receipt(built["id"], self.receipts)


class DecisionTests(RegistryCase):
    def test_hold_appends_a_pending_entry_and_writes_a_new_manifest(self):
        built = self.receipt(item_output())
        before = sr.registry_sha256(self.registry)
        decided = self.decide(built, "hold", by="agent:session-1")
        lines = (self.registry / "holds.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 185)
        entry = json.loads(lines[-1])
        self.assertEqual((entry["dataset"], entry["id"], entry["status"], entry["aliases"]), ("sst2", "item-1", "pending", []))
        self.assertEqual(entry["annotatedLabel"], "negative")
        self.assertEqual(entry["reason"], "The text says the film is fine.")
        self.assertEqual(decided["registry"]["before_sha256"], before)
        self.assertNotEqual(decided["registry"]["after_sha256"], before)
        self.assertEqual(decided["registry"]["manifest"], "source-reviews-20261009-v1")
        manifest = json.loads((self.manifests / "source-reviews-20261009-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["record_count"], 185)
        self.assertIs(decided["training_admission"], False)
        self.assertEqual(decided["decision"]["by"], "agent:session-1")

    def test_clear_and_defer_leave_the_registry_alone_and_defer_stays_open(self):
        built = self.receipt(item_output())
        before = (self.registry / "holds.jsonl").read_bytes()
        self.assertEqual(self.decide(built, "defer")["decision"]["value"], "defer")
        cleared = self.decide(built, "clear")
        self.assertEqual(cleared["decision"]["value"], "clear")
        self.assertEqual((self.registry / "holds.jsonl").read_bytes(), before)
        self.assertEqual(list(self.manifests.iterdir()), [])
        with self.assertRaisesRegex(sr.ReviewError, "was decided"):
            self.decide(built, "hold")

    def test_the_signer_and_the_decision_are_explicit(self):
        built = self.receipt(item_output())
        with self.assertRaisesRegex(sr.ReviewError, "owner or agent"):
            self.decide(built, "clear", by="somebody")
        with self.assertRaisesRegex(sr.ReviewError, "hold, clear or defer"):
            self.decide(built, "admit")

    def test_hold_refuses_an_identity_that_already_has_an_entry(self):
        existing = json.loads((self.registry / "holds.jsonl").read_text(encoding="utf-8").splitlines()[0])
        output = item_output(existing["id"])
        output["input"]["item"]["dataset"] = existing["dataset"]
        built = self.receipt(output)
        with self.assertRaisesRegex(sr.ReviewError, "already has a registry entry"):
            self.decide(built, "hold")

    def test_hold_needs_a_hold_recommendation(self):
        admit = {"recommendation": "admit", "concern": "none", "reason": "Supported.", "evidence": [], "proposed_entry": None, "confidence": "high"}
        built = self.receipt(item_output(recommendation=admit))
        with self.assertRaisesRegex(sr.ReviewError, "recommended admit"):
            self.decide(built, "hold")

    def test_the_loader_accepts_the_registry_after_a_hold(self):
        built = self.receipt(item_output())
        self.decide(built, "hold")
        names = [json.loads(line)["id"] for line in (self.registry / "holds.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(names), len(set(names)))


class RowReceiptTests(RegistryCase):
    def test_categories_follow_the_v7_mapping(self):
        cases = {"equivalent": "receipts", "normalization-candidate": "semantic_normalization_candidate",
                 "mismatch": "strict_semantic_mismatches", "ambiguous": "semantic_ambiguities"}
        for status, category in cases.items():
            built = sr.build_receipt(row_output(status=status, precheck={"exact_match": False, "actual_is_exact_source_span": False, "numerically_equal": False}), self.registry)
            self.assertEqual(sr.v7_category(built), category)
        numeric = sr.build_receipt(row_output(status="normalization-candidate"), self.registry)
        self.assertEqual(sr.v7_category(numeric), "held_numeric_representation")
        transport = sr.build_receipt({**row_output(), "recommendation": None, "excluded": "transport_error"}, self.registry)
        self.assertEqual(sr.v7_category(transport), "transport_error")
        with self.assertRaisesRegex(sr.ReviewError, "transport_error"):
            sr.build_receipt({**row_output(), "recommendation": None}, self.registry)

    def test_a_packet_disposition_needs_an_explicit_decision(self):
        built = self.receipt(row_output(status="mismatch"))
        with self.assertRaisesRegex(sr.ReviewError, "no explicit hold or clear decision"):
            sr.row_assessment(built)
        self.decide(built, "defer")
        with self.assertRaisesRegex(sr.ReviewError, "no explicit hold or clear decision"):
            sr.row_assessment(sr.load_receipt(built["id"], self.receipts))
        self.decide(built, "clear", by="agent:s1")
        assessment = sr.row_assessment(sr.load_receipt(built["id"], self.receipts))
        self.assertEqual(assessment["disposition"], "candidate")
        self.assertEqual(assessment["reviewer"], reviewer("reviewSourceRow")["reviewer_hash"])
        self.assertIs(assessment["training_admission"], False)
        held = self.receipt(row_output("row-2", status="mismatch"))
        self.decide(held, "hold")
        self.assertEqual(sr.annotations_from_row_receipts(self.receipts)["rows"]["row-2"]["disposition"], "hold")

    def test_loading_refuses_item_receipts_and_duplicates(self):
        self.receipt(item_output())
        with self.assertRaisesRegex(sr.ReviewError, "item receipt"):
            sr.load_row_receipts(self.receipts)

    def test_exact_precheck_receipts_need_no_natlang_hash(self):
        exact = {"kind": "row", "input": row_input("row-3", gold="a", actual="a", precheck={"exact_match": True, "actual_is_exact_source_span": False, "numerically_equal": None}),
                 "reviewer": {"kind": "crisp", "function": "precheck"},
                 "recommendation": {"status": "equivalent", "rationale": "Equal.", "evidence": [], "confidence": "high"}}
        self.assertEqual(sr.v7_category(sr.build_receipt(exact, self.registry)), "receipts")


if __name__ == "__main__":
    unittest.main()
