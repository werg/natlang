import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_lfm_test_runtime_packet import audit, canonical


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in rows))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LfmPacketAuditTests(unittest.TestCase):
    def test_preserves_distinct_ir_variants_and_checks_rendered_train_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "joint.jsonl"
            first = {"id": "case", "task_modality": "primitive", "source_groups": ["g"],
                     "semantics": {"root": "main.nl", "expected": "ok"}}
            second = {"id": "case", "task_modality": "directory", "source_groups": ["g"],
                      "semantics": {"root": "main.nl", "expected": "ok"}}
            write_jsonl(source, [
                {"id": "t", "split": "train", "program_id": "train-p", "source_groups": ["train-g"]},
                {"id": "e1", "split": "test", "program_id": "case", "source_groups": ["g"],
                 "task": {"program_ir": first}},
                {"id": "e2", "split": "test", "program_id": "case", "source_groups": ["g"],
                 "task": {"program_ir": second}},
            ])
            source_manifest = root / "joint.manifest.json"
            source_manifest.write_text(json.dumps({"sha256": sha(source)}))
            packet = root / "packet"
            packet.mkdir()
            write_jsonl(packet / "selection.jsonl", [{"id": "case"}])
            (packet / "packet-receipt.json").write_text(json.dumps({"source": {"sha256": sha(source)}}) + "\n")
            ready = root / "ready.jsonl"
            write_jsonl(ready, [
                {"id": "rt", "split": "train", "program_id": "train-p", "source_groups": ["train-g"],
                 "token_counts": {"total_tokens": 17}},
                {"id": "re", "split": "test", "program_id": "case", "source_groups": ["g"],
                 "token_counts": {"total_tokens": 11}},
            ])
            receipt = audit(source, source_manifest, ready, packet, root / "out")
            with (root / "out/cases.ir.jsonl").open() as stream:
                cases = [json.loads(line) for line in stream]
            with (root / "out/selection.jsonl").open() as stream:
                selection = [json.loads(line) for line in stream]
            self.assertEqual(len(cases), 2)
            self.assertEqual(len({x["evaluation_case_id"] for x in selection}), 2)
            self.assertEqual(receipt["source_and_alias_closure"]["semantic_or_contract_variant_ids"], 0)
            self.assertEqual(receipt["actual_lfm_ready"]["split_rows"], {"train": 1, "test": 1})

    def test_rejects_overlap_with_actual_rendered_train_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "joint.jsonl"
            ir = {"id": "case", "source_groups": ["shared"], "semantics": {"expected": 1}}
            write_jsonl(source, [{"id": "e", "split": "test", "program_id": "case", "source_groups": ["shared"],
                                 "task": {"program_ir": ir}}])
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"sha256": sha(source)}))
            packet = root / "packet"
            packet.mkdir()
            write_jsonl(packet / "selection.jsonl", [{"id": "case"}])
            (packet / "packet-receipt.json").write_text(json.dumps({"source": {"sha256": sha(source)}}) + "\n")
            ready = root / "ready.jsonl"
            write_jsonl(ready, [{"id": "train", "split": "train", "program_id": "train",
                                 "source_groups": ["shared"], "token_counts": {"total_tokens": 2}}])
            with self.assertRaisesRegex(ValueError, "overlap actual LFM train"):
                audit(source, manifest, ready, packet, root / "out")


if __name__ == "__main__":
    unittest.main()
