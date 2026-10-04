import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_scifact_research_tasks import prepare_scifact_bytes, write_packet


def jsonl(rows):
    return ("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n").encode()


def doc(doc_id, *sentences):
    return {"doc_id": doc_id, "title": f"Paper {doc_id}", "abstract": list(sentences), "structured": False}


def claim(claim_id, claim_text, evidence, cited):
    return {"id": claim_id, "claim": claim_text, "evidence": evidence, "cited_doc_ids": cited}


class SciFactPreparationTests(unittest.TestCase):
    def setUp(self):
        self.corpus = jsonl([
            doc(10, "Sentence zero.", "Sentence one.", "Sentence two."),
            doc(11, "Another abstract."),
            doc(12, "Validation abstract."),
            doc(13, "Conflict abstract."),
            doc(14, "Bad annotation abstract."),
            doc(15, "Cross-role document."),
        ])
        self.train = jsonl([
            claim(1, "A supported claim.", {"10": [
                {"label": "SUPPORT", "sentences": [0]},
                {"label": "SUPPORT", "sentences": [1, 2]},
            ]}, [10, 11]),
            claim(2, "A contradictory claim.", {"11": [{"label": "CONTRADICT", "sentences": [0]}]}, [11]),
            claim(3, "Repeated source row.", {"10": [{"label": "SUPPORT", "sentences": [0]}]}, [10]),
            claim(3, "Repeated source row.", {"10": [{"label": "SUPPORT", "sentences": [0]}]}, [10]),
            claim(9, "No supplied citation supports or contradicts this claim.", {}, [10]),
            claim(4, "Conflicting rationale labels.", {
                "13": [{"label": "SUPPORT", "sentences": [0]}],
                "14": [{"label": "CONTRADICT", "sentences": [0]}],
            }, [13, 14]),
            claim(5, "Out-of-range rationale.", {"14": [{"label": "SUPPORT", "sentences": [9]}]}, [14]),
            claim(6, "Shares a paper with dev.", {"15": [{"label": "SUPPORT", "sentences": [0]}]}, [15]),
        ])
        self.dev = jsonl([claim(7, "Validation claim.", {"12": [{"label": "CONTRADICT", "sentences": [0]}]}, [12]),
                          claim(8, "Connected validation claim.", {"15": [{"label": "SUPPORT", "sentences": [0]}]}, [15])])

    def prepare(self, **kwargs):
        return prepare_scifact_bytes(self.corpus, self.train, source_revision="68b98a56d93e0f9da0d2aab4e6c3294699a0f72e",
                                     corpus_path="corpus.jsonl", claims_train_path="claims_train.jsonl", **kwargs)

    def test_default_is_train_only_and_keeps_labels_out_of_visible_task(self):
        packet = self.prepare()
        self.assertEqual({row["role"] for row in packet["candidates"]}, {"train"})
        self.assertNotIn("validation", [row["role"] for row in packet["candidates"]])
        first = next(row for row in packet["candidates"] if row["candidate_id"].startswith("scifact:train:1:"))
        self.assertEqual(first["host_only_oracle"]["label"], "SUPPORT")
        self.assertEqual([item["sentence_ids"] for item in first["host_only_oracle"]["accepted_evidence_sets"]], [[0], [1, 2]])
        self.assertEqual(first["task"]["documents"][0]["abstract_sentences"], [
            {"sentence_id": 0, "text": "Sentence zero."},
            {"sentence_id": 1, "text": "Sentence one."},
            {"sentence_id": 2, "text": "Sentence two."},
        ])
        self.assertNotIn("label", first["task"])
        self.assertNotIn("evidence", first["task"])
        self.assertEqual(first["collection_status"], "candidate_not_admitted")
        nei = next(row for row in packet["candidates"] if row["candidate_id"].startswith("scifact:train:9:"))
        self.assertEqual(nei["host_only_oracle"]["label"], "NOT_ENOUGH_INFO")
        self.assertEqual(nei["host_only_oracle"]["accepted_evidence_sets"], [])
        self.assertEqual(nei["provenance"]["label_derivation"], "empty_evidence_to_NOT_ENOUGH_INFO_via_official_oracle")
        self.assertEqual(packet["manifest"]["inputs"]["corpus"]["sha256"], __import__("hashlib").sha256(self.corpus).hexdigest())
        self.assertEqual(packet["manifest"]["licenses"], {
            "claims_and_evidence_annotations": "CC-BY-4.0", "abstracts": "ODC-By-1.0"})

    def test_dev_requires_explicit_validation_and_connected_roles_are_held(self):
        with self.assertRaisesRegex(ValueError, "include_dev_validation"):
            self.prepare(claims_dev_bytes=self.dev, claims_dev_path="claims_dev.jsonl")
        packet = self.prepare(claims_dev_bytes=self.dev, claims_dev_path="claims_dev.jsonl", include_dev_validation=True)
        self.assertIn("validation", {row["role"] for row in packet["candidates"]})
        leaked = [row for row in packet["held"] if "connected_component_crosses_train_validation" in row["reasons"]]
        self.assertEqual({row["source_claim_id"] for row in leaked}, {6, 8})
        self.assertTrue(all(row["source_groups"] == leaked[0]["source_groups"] for row in leaked))

    def test_conflicting_labels_and_invalid_sentence_references_are_held(self):
        packet = self.prepare()
        held = {row["source_claim_id"]: set(row["reasons"]) for row in packet["held"]}
        self.assertIn("conflicting_rationale_labels", held[4])
        self.assertIn("evidence_sentence_id_out_of_range", held[5])
        self.assertNotIn("scifact:train:4:row", {row["candidate_id"] for row in packet["candidates"]})

    def test_exact_duplicate_claim_rows_are_retained_and_share_group(self):
        packet = self.prepare()
        duplicates = [row for row in packet["candidates"] if row["candidate_id"].startswith("scifact:train:3:")]
        self.assertEqual(len(duplicates), 2)
        self.assertNotEqual(duplicates[0]["candidate_id"], duplicates[1]["candidate_id"])
        self.assertEqual(duplicates[0]["source_groups"], duplicates[1]["source_groups"])
        self.assertEqual({row["provenance"]["exact_duplicate_ordinal"] for row in duplicates}, {1, 2})

    def test_conflicting_duplicate_identity_and_conflicting_document_id_fail_closed(self):
        altered = json.loads(self.train.splitlines()[0])
        altered["claim"] = "Different claim under the same ID."
        train = self.train + (json.dumps(altered, separators=(",", ":")) + "\n").encode()
        packet = prepare_scifact_bytes(self.corpus, train, source_revision="rev")
        held_ids = {row["source_claim_id"] for row in packet["held"] if "conflicting_duplicate_claim_id" in row["reasons"]}
        self.assertEqual(held_ids, {1})

        duplicate_corpus = self.corpus + jsonl([doc(10, "Different text for same ID.")])
        packet = prepare_scifact_bytes(duplicate_corpus, self.train, source_revision="rev")
        self.assertTrue(any("conflicting_corpus_document_id" in row["reasons"] for row in packet["held"] if row["source_claim_id"] == 1))

        identical_duplicate_corpus = self.corpus + jsonl([doc(10, "Sentence zero.", "Sentence one.", "Sentence two.")])
        packet = prepare_scifact_bytes(identical_duplicate_corpus, self.train, source_revision="rev")
        first = next(row for row in packet["candidates"] if row["candidate_id"].startswith("scifact:train:1:"))
        self.assertEqual(len(first["provenance"]["document_source_rows"][0]["corpus_rows"]), 2)

    def test_missing_documents_and_malformed_rationales_are_held(self):
        extra = [
            claim(30, "Missing evidence and citation document.", {"999": [{"label": "SUPPORT", "sentences": [0]}]}, [999]),
            claim(31, "Evidence document not cited.", {"10": [{"label": "SUPPORT", "sentences": [0]}]}, [11]),
            claim(32, "Empty rationale list.", {"10": []}, [10]),
        ]
        packet = prepare_scifact_bytes(self.corpus, jsonl(extra), source_revision="rev")
        reasons = {row["source_claim_id"]: set(row["reasons"]) for row in packet["held"]}
        self.assertIn("missing_cited_document", reasons[30])
        self.assertIn("missing_evidence_document", reasons[30])
        self.assertIn("evidence_document_not_cited", reasons[31])
        self.assertIn("missing_rationales", reasons[32])

    def test_output_directory_publication_is_exclusive(self):
        packet = self.prepare()
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "candidate"
            write_packet(output, packet)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["counts"]["candidate_rows"], len(packet["candidates"]))
            with self.assertRaises(FileExistsError):
                write_packet(output, packet)

    def test_archive_checksum_format_is_pinned_when_provided(self):
        with self.assertRaisesRegex(ValueError, "lowercase 64-character"):
            prepare_scifact_bytes(self.corpus, self.train, source_revision="rev", source_archive_sha256="not-a-sha")


if __name__ == "__main__":
    unittest.main()
